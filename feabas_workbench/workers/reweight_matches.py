"""
Re-weight FEABAS fine matches by structure masks.

After align_main.py --mode matching, every align/matches/<a>__to__<b>.h5 holds
match points (xy0 in section a, xy1 in section b, at 'resolution' nm/px) and
weights. This worker multiplies the weight of matches whose points fall inside
the structure masks (both sides) by `inside_factor` and the others by
`outside_factor`, so the optimisation is driven by the chosen structures.
The original file is kept once as <name>.h5.orig so the operation can be undone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def main() -> int:
    spec = load_spec("match re-weighting")
    import cv2
    import h5py
    match_dir = Path(spec["match_dir"])
    struct_dir = Path(spec["structures_dir"])
    struct_res = float(spec["structure_resolution"])     # nm/px of the structure masks
    f_in = float(spec.get("inside_factor", 1.0)); f_out = float(spec.get("outside_factor", 0.1))
    undo = bool(spec.get("undo", False))
    delim = spec.get("delimiter", "__to__")
    files = sorted(match_dir.glob("*.h5"))
    progress(0, len(files), "starting")
    masks = {}

    def mask_for(sec):
        if sec not in masks:
            p = struct_dir / f"{sec}.png"
            masks[sec] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
        return masks[sec]

    stats = {}
    for i, f in enumerate(files, 1):
        orig = f.with_suffix(".h5.orig")
        if undo:
            if orig.is_file():
                shutil.move(str(orig), str(f))
            progress(i, len(files), f.name)
            continue
        if not orig.is_file():
            shutil.copyfile(f, orig)
        with h5py.File(orig, "r") as h:
            xy0 = h["xy0"][()]; xy1 = h["xy1"][()]; w = h["weight"][()].astype(np.float32)
            res = float(np.asarray(h["resolution"][()]).item())
            extra = {k: h[k][()] for k in h.keys() if k not in ("xy0", "xy1", "weight")}
        name = f.stem
        s0, s1 = name.split(delim)[:2]
        m0, m1 = mask_for(s0), mask_for(s1)
        if m0 is None or m1 is None:
            progress(i, len(files), f"{name}: no structure mask")
            continue
        sc = res / struct_res
        p0 = np.round(xy0 * sc).astype(int); p1 = np.round(xy1 * sc).astype(int)
        in0 = np.zeros(len(p0), bool); in1 = np.zeros(len(p1), bool)
        ok0 = (p0[:, 0] >= 0) & (p0[:, 1] >= 0) & (p0[:, 0] < m0.shape[1]) & (p0[:, 1] < m0.shape[0])
        ok1 = (p1[:, 0] >= 0) & (p1[:, 1] >= 0) & (p1[:, 0] < m1.shape[1]) & (p1[:, 1] < m1.shape[0])
        in0[ok0] = m0[p0[ok0, 1], p0[ok0, 0]] > 0
        in1[ok1] = m1[p1[ok1, 1], p1[ok1, 0]] > 0
        inside = in0 | in1
        w_new = w * np.where(inside, f_in, f_out).astype(np.float32)
        with h5py.File(f, "w") as h:
            h.create_dataset("xy0", data=xy0, compression="gzip")
            h.create_dataset("xy1", data=xy1, compression="gzip")
            h.create_dataset("weight", data=w_new, compression="gzip")
            for k, v in extra.items():
                h.create_dataset(k, data=v)
        stats[name] = {"n": int(len(w)), "inside": int(inside.sum())}
        progress(i, len(files), f"{name}: {int(inside.sum())}/{len(w)} inside structures")
    if not undo:
        (match_dir / "reweight_report.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log("undo done" if undo else f"re-weighted {len(stats)} match files")
    result({"n_files": len(files), "undo": undo})
    return 0


if __name__ == "__main__":
    run(main)
