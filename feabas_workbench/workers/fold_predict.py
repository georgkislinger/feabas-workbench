"""
Fold (wrinkle) detection with the segmentation-models-pytorch U-Net.

Input per section: either a single image file (thumbnail) or a FEABAS tiled
section folder at a given mip. Output: probability map (uint8) and binary mask
PNGs in an output folder, named <section>.png / <section>_prob.png.

Checkpoints: Lightning checkpoints from the original training script
(state_dict with 'model.' prefix, smp.Unet resnet34, 1 in / 1 out channel,
img/255 normalisation) and this workbench's own fold_train output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def build_model(encoder: str = "resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)


def load_checkpoint(path: Path, device):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
        meta = ck.get("workbench_meta", {}) or {}
    elif isinstance(ck, dict) and "model_state" in ck:
        sd = ck["model_state"]
        meta = ck.get("meta", {})
    else:
        sd = ck
    encoder = meta.get("encoder", "resnet34")
    # strip Lightning prefix
    clean = {}
    for k, v in sd.items():
        kk = k
        for pre in ("model.", "net.", "module."):
            if kk.startswith(pre):
                kk = kk[len(pre):]
        clean[kk] = v
    tried = []
    for enc in [encoder] + [e for e in ("resnet34", "resnet18", "resnet50", "efficientnet-b0") if e != encoder]:
        try:
            m = build_model(enc)
            missing, unexpected = m.load_state_dict(clean, strict=False)
            if len(missing) == 0 and len(unexpected) == 0:
                m.to(device).eval()
                return m, {"encoder": enc, **meta}
            tried.append(f"{enc}: {len(missing)} missing / {len(unexpected)} unexpected")
        except Exception as e:  # noqa: BLE001
            tried.append(f"{enc}: {e}")
    raise RuntimeError("checkpoint does not fit a supported U-Net: " + "; ".join(tried))


def _hann(h: int, w: int) -> np.ndarray:
    wy = np.hanning(h + 2)[1:-1] if h > 1 else np.ones(1)
    wx = np.hanning(w + 2)[1:-1] if w > 1 else np.ones(1)
    win = np.outer(wy, wx).astype(np.float32)
    return np.maximum(win, 1e-3)


def predict_map(model, img: np.ndarray, device, tile: int = 1024, overlap: int = 128, batch: int = 4,
                normalization: str = "div255") -> np.ndarray:
    """Tiled inference with Hann blending; reflect padding; returns float32 probability HxW."""
    import torch
    if normalization == "div255":
        x = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else float(np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0))
    else:
        lo, hi = np.percentile(img, (1, 99))
        x = np.clip((img.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
    H, W = x.shape
    if H <= tile and W <= tile:
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        xp = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
        with torch.no_grad():
            t = torch.from_numpy(xp)[None, None].to(device)
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        return p[:H, :W].astype(np.float32)
    step = tile - overlap
    pad = overlap
    xp = np.pad(x, ((pad, pad + tile), (pad, pad + tile)), mode="reflect")
    acc = np.zeros(xp.shape, np.float32)
    wacc = np.zeros(xp.shape, np.float32)
    win = _hann(tile, tile)
    coords = []
    for y0 in range(0, H + pad, step):
        for x0 in range(0, W + pad, step):
            coords.append((y0, x0))
    with torch.no_grad():
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            arr = np.stack([xp[y0:y0 + tile, x0:x0 + tile] for y0, x0 in chunk])[:, None]
            t = torch.from_numpy(arr).to(device)
            p = torch.sigmoid(model(t))[:, 0].cpu().numpy()
            for (y0, x0), pm in zip(chunk, p):
                acc[y0:y0 + tile, x0:x0 + tile] += pm * win
                wacc[y0:y0 + tile, x0:x0 + tile] += win
    prob = acc / np.maximum(wacc, 1e-6)
    return prob[pad:pad + H, pad:pad + W].astype(np.float32)


def postprocess(prob: np.ndarray, thr: float, min_area: int, dilate: int, close: int) -> np.ndarray:
    import cv2
    m = (prob >= thr).astype(np.uint8)
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[lab].astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def load_section_image(item: dict) -> np.ndarray:
    from feabas_workbench.core.images import imread, TiledSectionSource
    if "image" in item:
        return imread(item["image"])
    src = TiledSectionSource(Path(item["tiled_base"]), item["section"])
    mip = int(item.get("mip", 0))
    lvl = [l for l in src.levels if l.mip == mip]
    if not lvl:
        raise RuntimeError(f"{item['section']}: mip{mip} not rendered")
    l = lvl[0]
    return src.read(mip, 0, 0, l.width, l.height)


def main() -> int:
    spec = load_spec("fold detection")
    import cv2
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not spec.get("cpu") else "cpu")
    model, meta = load_checkpoint(Path(spec["checkpoint"]), device)
    log(f"model loaded ({meta.get('encoder')}), device {device}")
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    items = spec["items"]
    thr = float(spec.get("threshold", 0.5))
    tile = int(spec.get("tile", 1024)); overlap = int(spec.get("overlap", 128)); batch = int(spec.get("batch", 4))
    min_area = int(spec.get("min_area", 50)); dilate = int(spec.get("dilate", 0)); close = int(spec.get("close", 2))
    norm = spec.get("normalization", meta.get("normalization", "div255"))
    progress(0, len(items), "starting")
    n_fail = 0
    stats = {}
    for i, item in enumerate(items, 1):
        sec = item["section"]
        try:
            img = load_section_image(item)
            if img.ndim == 3:
                img = img[..., 0]
            prob = predict_map(model, img, device, tile, overlap, batch, norm)
            mask = postprocess(prob, thr, min_area, dilate, close)
            cv2.imwrite(str(out_dir / f"{sec}_prob.png"), np.clip(prob * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(str(out_dir / f"{sec}.png"), (mask * 255).astype(np.uint8))
            stats[sec] = {"fold_pct": float(100.0 * mask.mean()), "shape": list(mask.shape), "mip": int(item.get("mip", 0))}
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log(f"FAILED {sec}: {e}")
        progress(i, len(items), sec)
    (out_dir / "fold_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    result({"n_done": len(items) - n_fail, "n_failed": n_fail, "out_dir": str(out_dir)})
    return 1 if n_fail else 0


if __name__ == "__main__":
    run(main)
