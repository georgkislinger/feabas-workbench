"""
Structure-guided coarse matching.

For each neighbouring section pair, match the detected structures (from
yolo_detect) between the two thumbnails: every object gets a descriptor made
of its size, its intensity patch (rotation-normalised by the object's own
orientation) and a shape signature; candidates are paired by descriptor
distance and verified with a RANSAC affine model. The inlier centroid pairs
are written as FEABAS thumbnail matches (HDF5: xy0, xy1, weight, resolution)
either replacing FEABAS's own feature matches for that pair (mode 'replace')
or merged into them with a weight factor (mode 'augment'). A BigWarp CSV copy
goes to thumbnail_align/manual_matches for inspection/editing in Fiji.

Only numpy/cv2/h5py are needed (runs in any environment).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def load_dets(struct_dir: Path, sec: str):
    p = struct_dir / f"{sec}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def describe(img: np.ndarray, dets: list[dict], patch: int = 48) -> np.ndarray:
    """Descriptor per detection: normalised, orientation-aligned intensity patch + log-area."""
    import cv2
    H, W = img.shape[:2]
    out = np.zeros((len(dets), patch * patch + 2), np.float32)
    f = img.astype(np.float32)
    for i, d in enumerate(dets):
        cx, cy = d["cx"], d["cy"]
        x0, y0, x1, y1 = d["box"]
        size = max(8.0, 1.6 * max(x1 - x0, y1 - y0))
        # orientation from the box aspect is weak; use image moments of the crop
        sx0, sy0 = int(max(0, cx - size)), int(max(0, cy - size))
        sx1, sy1 = int(min(W, cx + size)), int(min(H, cy + size))
        crop = f[sy0:sy1, sx0:sx1]
        if crop.size < 16:
            continue
        m = cv2.moments(crop - crop.mean())
        ang = 0.5 * math.degrees(math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"] + 1e-9))
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), ang, patch / (2 * size))
        M[0, 2] += patch / 2 - cx; M[1, 2] += patch / 2 - cy
        warped = cv2.warpAffine(f, M, (patch, patch), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        v = warped.ravel()
        v = (v - v.mean()) / (v.std() + 1e-6)
        out[i, :patch * patch] = v
        out[i, -2] = math.log(max(1.0, d["area"])) * 4.0
        out[i, -1] = float(d.get("cls", 0)) * 50.0
    return out


def _similarity_from_two(a0, a1, b0, b1):
    """Similarity transform (scale, rotation, translation) mapping a0->b0, a1->b1. Returns 2x3 or None."""
    da = a1 - a0
    db = b1 - b0
    na = float(np.hypot(*da)); nb = float(np.hypot(*db))
    if na < 1e-6 or nb < 1e-6:
        return None
    s = nb / na
    ang = math.atan2(db[1], db[0]) - math.atan2(da[1], da[0])
    c, si = s * math.cos(ang), s * math.sin(ang)
    M = np.array([[c, -si, 0.0], [si, c, 0.0]], np.float64)
    t = b0 - (M[:, :2] @ a0)
    M[:, 2] = t
    return M


def _apply(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def _one_to_one_inliers(M, c0, c1, tol):
    """Indices (i, j) of mutually nearest pairs within tol under transform M."""
    proj = _apply(M, c0)
    d = np.sqrt(((proj[:, None, :] - c1[None, :, :]) ** 2).sum(-1))
    j = d.argmin(axis=1)
    i_back = d.argmin(axis=0)
    ok = (d[np.arange(len(c0)), j] <= tol) & (i_back[j] == np.arange(len(c0)))
    return np.nonzero(ok)[0], j[ok]


def match_pair(img0, dets0, img1, dets1, spec) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Point-set registration of structure centroids that does not trust the
    descriptors: descriptors only shortlist candidate partners; the transform
    is found by RANSAC over pairs of candidate correspondences (2 points ->
    similarity), scored by one-to-one inliers, then refined as an affine.
    """
    if len(dets0) < 4 or len(dets1) < 4:
        return np.zeros((0, 2)), np.zeros((0, 2)), {"reason": "too few structures"}
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    c0 = np.array([[d["cx"], d["cy"]] for d in dets0], np.float64)
    c1 = np.array([[d["cx"], d["cy"]] for d in dets1], np.float64)
    D0 = describe(img0, dets0); D1 = describe(img1, dets1)
    d2 = ((D0[:, None, :] - D1[None, :, :]) ** 2).sum(-1)
    k = int(spec.get("candidates", 5))
    nn = np.argsort(d2, axis=1)[:, :min(k, len(dets1))]
    cand = [(i, int(j)) for i in range(len(dets0)) for j in nn[i]]
    tol = float(spec.get("ransac_tol", 6.0))
    smin, smax = float(spec.get("scale_min", 0.8)), float(spec.get("scale_max", 1.25))
    max_rot = math.radians(float(spec.get("max_rotation_deg", 60.0)))
    iters = int(spec.get("ransac_iters", 20000))
    best = None
    best_n = 0
    min_sep = 0.05 * max(img0.shape)
    for _ in range(iters):
        a, b = cand[rng.integers(len(cand))], cand[rng.integers(len(cand))]
        if a[0] == b[0] or a[1] == b[1]:
            continue
        if np.hypot(*(c0[a[0]] - c0[b[0]])) < min_sep:
            continue
        M = _similarity_from_two(c0[a[0]], c0[b[0]], c1[a[1]], c1[b[1]])
        if M is None:
            continue
        sc = math.hypot(M[0, 0], M[1, 0])
        if not (smin <= sc <= smax):
            continue
        rot = math.atan2(M[1, 0], M[0, 0])
        if abs(rot) > max_rot:
            continue
        ii, jj = _one_to_one_inliers(M, c0, c1, tol * 2)
        if len(ii) > best_n:
            best_n = len(ii); best = (M, ii, jj)
    if best is None or best_n < int(spec.get("min_inliers", 6)):
        return np.zeros((0, 2)), np.zeros((0, 2)), {"reason": f"only {best_n} consistent structures"}
    M, ii, jj = best
    # refine: least-squares affine on inliers, then recollect with the tight tolerance
    for _ in range(2):
        A = np.hstack([c0[ii], np.ones((len(ii), 1))])
        sol, *_ = np.linalg.lstsq(A, c1[jj], rcond=None)      # 3x2
        M = sol.T
        ii, jj = _one_to_one_inliers(M, c0, c1, tol)
        if len(ii) < 3:
            return np.zeros((0, 2)), np.zeros((0, 2)), {"reason": "refinement lost the inliers"}
    xy0 = c0[ii]; xy1 = c1[jj]
    scale = math.sqrt(abs(np.linalg.det(M[:, :2])))
    info = {"n_pairs": int(len(xy0)), "ransac_inliers": int(best_n), "scale": scale, "shift_px": float(np.linalg.norm(M[:, 2])),
            "rotation_deg": math.degrees(math.atan2(M[1, 0], M[0, 0])), "n0": len(dets0), "n1": len(dets1)}
    max_shift = float(spec.get("max_shift_px", 0)) or max(img0.shape) * 0.75
    if info["shift_px"] > max_shift:
        info["reason"] = "implausible shift"
        return np.zeros((0, 2)), np.zeros((0, 2)), info
    return xy0.astype(np.float32), xy1.astype(np.float32), info


def write_h5(path: Path, xy0, xy1, weight, resolution, name0=None, name1=None, strain=None):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("xy0", data=np.asarray(xy0, np.float32), compression="gzip")
        f.create_dataset("xy1", data=np.asarray(xy1, np.float32), compression="gzip")
        f.create_dataset("weight", data=np.asarray(weight, np.float32), compression="gzip")
        f.create_dataset("resolution", data=float(resolution))
        if strain is not None:
            f.create_dataset("strain", data=float(strain))
        if name0 is not None:
            f.create_dataset("name0", data=np.frombuffer(name0.encode("ascii"), np.uint8))
            f.create_dataset("name1", data=np.frombuffer(name1.encode("ascii"), np.uint8))


def read_h5(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        xy0 = f["xy0"][()]; xy1 = f["xy1"][()]; w = f["weight"][()]; res = float(np.asarray(f["resolution"][()]).item())
        strain = float(np.asarray(f["strain"][()]).item()) if "strain" in f else None
    return xy0, xy1, w, res, strain


def write_bigwarp_csv(path: Path, xy0, xy1) -> None:
    lines = [f'"Pt-{i}","true",{a[0]:.2f},{a[1]:.2f},{b[0]:.2f},{b[1]:.2f}' for i, (a, b) in enumerate(zip(xy0, xy1))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    spec = load_spec("structure matching")
    from feabas_workbench.core.images import imread
    struct_dir = Path(spec["structures_dir"])
    thumb_dir = Path(spec["thumbnail_dir"])
    match_dir = Path(spec["match_dir"]); match_dir.mkdir(parents=True, exist_ok=True)
    manual_dir = Path(spec["manual_dir"]); manual_dir.mkdir(parents=True, exist_ok=True)
    delim = spec.get("delimiter", "__to__")
    thumb_res = float(spec["thumbnail_resolution"])
    mode = spec.get("mode", "augment")            # replace | augment
    weight = float(spec.get("weight", 1.0))
    pairs = spec["pairs"]                          # [[sec0, sec1], ...]
    progress(0, len(pairs), "starting")
    report = {}
    n_written = 0
    for i, (s0, s1) in enumerate(pairs, 1):
        d0 = load_dets(struct_dir, s0); d1 = load_dets(struct_dir, s1)
        out = match_dir / f"{s0}{delim}{s1}.h5"
        if d0 is None or d1 is None:
            report[f"{s0}{delim}{s1}"] = {"reason": "no detections"}
            progress(i, len(pairs), f"{s0}-{s1}: no detections")
            continue
        img0 = imread(_thumb(thumb_dir, s0)); img1 = imread(_thumb(thumb_dir, s1))
        sc0 = img0.shape[1] / d0["shape"][1]; sc1 = img1.shape[1] / d1["shape"][1]     # detections may be at another mip
        det0 = [dict(d, cx=d["cx"] * sc0, cy=d["cy"] * sc0, box=[v * sc0 for v in d["box"]], area=d["area"] * sc0 * sc0) for d in d0["detections"]]
        det1 = [dict(d, cx=d["cx"] * sc1, cy=d["cy"] * sc1, box=[v * sc1 for v in d["box"]], area=d["area"] * sc1 * sc1) for d in d1["detections"]]
        xy0, xy1, info = match_pair(img0, det0, img1, det1, spec)
        report[f"{s0}{delim}{s1}"] = info
        if len(xy0) == 0:
            progress(i, len(pairs), f"{s0}-{s1}: {info.get('reason')}")
            continue
        write_bigwarp_csv(manual_dir / f"{s0}{delim}{s1}.csv", xy0, xy1)
        w = np.full(len(xy0), weight, np.float32)
        if mode == "augment" and out.is_file():
            x0, x1, w0, res, strain = read_h5(out)
            if abs(res - thumb_res) > 1e-6:
                sc = res / thumb_res
                xy0s, xy1s = xy0 / sc, xy1 / sc
            else:
                xy0s, xy1s = xy0, xy1
            write_h5(out, np.concatenate([x0, xy0s]), np.concatenate([x1, xy1s]), np.concatenate([w0, w]), res, s0, s1, strain)
        else:
            write_h5(out, xy0, xy1, w, thumb_res, s0, s1)
        n_written += 1
        progress(i, len(pairs), f"{s0}-{s1}: {len(xy0)} structure matches")
    (match_dir / "structure_match_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"{n_written}/{len(pairs)} pairs got structure matches")
    result({"n_pairs": len(pairs), "n_written": n_written})
    return 0


def _thumb(thumb_dir: Path, sec: str) -> Path:
    for ext in ("png", "jpg", "tif"):
        p = thumb_dir / f"{sec}.{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"no thumbnail for {sec}")


if __name__ == "__main__":
    run(main)
