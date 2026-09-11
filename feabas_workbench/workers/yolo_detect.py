"""
Detect biological structures (nuclei, mitochondria, vessels, ...) with an
ultralytics YOLO segmentation model on section images, tile by tile.

Per section it writes
    <out_dir>/<section>.png     union of instance masks (255)
    <out_dir>/<section>.json    detections: centroid, area, box, class, confidence
and a structures.json summary. Coordinates are in pixels of the image that was
processed (thumbnail mip or a stitched mip level).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def _nms(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    if not dets:
        return dets
    boxes = np.array([d["box"] for d in dets], dtype=np.float32)
    scores = np.array([d["conf"] for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / np.maximum(a_i + a_o - inter, 1e-6)
        order = order[1:][iou < iou_thr]
    return [dets[i] for i in keep]


def detect_image(model, img: np.ndarray, tile: int, overlap: int, conf: float, iou: float, classes, device, max_det: int):
    import cv2
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img if img.dtype == np.uint8 else (img / 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), step)) + ([max(0, H - tile)] if H > tile else [0])
    xs = list(range(0, max(1, W - tile + 1), step)) + ([max(0, W - tile)] if W > tile else [0])
    ys = sorted(set(ys)); xs = sorted(set(xs))
    dets: list[dict] = []
    mask = np.zeros((H, W), np.uint8)
    for y0 in ys:
        for x0 in xs:
            crop = rgb[y0:y0 + tile, x0:x0 + tile]
            res = model.predict(crop, imgsz=tile, conf=conf, iou=iou, classes=classes or None, device=device,
                                verbose=False, max_det=max_det, retina_masks=True)
            if not res:
                continue
            r = res[0]
            if r.masks is None or r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy()
            mdata = r.masks.data.cpu().numpy()   # N x h x w (crop size)
            ch, cw = crop.shape[:2]
            for k in range(len(boxes)):
                m = mdata[k]
                if m.shape != (ch, cw):
                    m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_NEAREST)
                mb = m > 0.5
                area = float(mb.sum())
                if area <= 0:
                    continue
                yy, xx = np.nonzero(mb)
                bx = boxes[k]
                # skip detections touching the crop border unless the crop is at the image border
                touches = (bx[0] <= 1 and x0 > 0) or (bx[1] <= 1 and y0 > 0) or (bx[2] >= cw - 1 and x0 + cw < W) or (bx[3] >= ch - 1 and y0 + ch < H)
                if touches and len(xs) * len(ys) > 1:
                    continue
                dets.append({"cx": float(xx.mean() + x0), "cy": float(yy.mean() + y0), "area": area,
                             "box": [float(bx[0] + x0), float(bx[1] + y0), float(bx[2] + x0), float(bx[3] + y0)],
                             "cls": int(cls[k]), "conf": float(confs[k]), "_mask": (y0, x0, mb)})
    dets = _nms(dets, 0.5)
    for d in dets:
        y0, x0, mb = d.pop("_mask")
        h, w = mb.shape
        mask[y0:y0 + h, x0:x0 + w] |= mb.astype(np.uint8) * 255
    return dets, mask


def main() -> int:
    spec = load_spec("YOLO-seg structure detection")
    import cv2
    import torch
    from ultralytics import YOLO
    from feabas_workbench.workers.fold_predict import load_section_image

    model = YOLO(spec["model"])
    device = 0 if torch.cuda.is_available() and not spec.get("cpu") else "cpu"
    names = model.names if hasattr(model, "names") else {}
    classes = spec.get("classes") or []
    class_ids = []
    for c in classes:
        if isinstance(c, int) or str(c).isdigit():
            class_ids.append(int(c))
        else:
            for k, v in names.items():
                if v == c:
                    class_ids.append(int(k))
    out_dir = Path(spec["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    items = spec["items"]
    tile = int(spec.get("tile", 1024)); overlap = int(spec.get("overlap", 128))
    conf = float(spec.get("conf", 0.25)); iou = float(spec.get("iou", 0.5)); max_det = int(spec.get("max_det", 3000))
    log(f"model {spec['model']} classes {names}; using {class_ids or 'all'}; device {device}")
    progress(0, len(items), "starting")
    summary = {}
    n_fail = 0
    for i, item in enumerate(items, 1):
        sec = item["section"]
        try:
            img = load_section_image(item)
            if img.ndim == 3:
                img = img[..., 0]
            dets, mask = detect_image(model, img, tile, overlap, conf, iou, class_ids, device, max_det)
            cv2.imwrite(str(out_dir / f"{sec}.png"), mask)
            (out_dir / f"{sec}.json").write_text(json.dumps({"section": sec, "mip": int(item.get("mip", 0)), "shape": list(mask.shape),
                                                              "names": {int(k): v for k, v in names.items()}, "detections": dets}), encoding="utf-8")
            summary[sec] = {"n": len(dets), "mip": int(item.get("mip", 0)), "mask_pct": float(100 * (mask > 0).mean())}
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log(f"FAILED {sec}: {e}")
        progress(i, len(items), f"{sec}: {summary.get(sec, {}).get('n', 0)} objects")
    (out_dir / "structures.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    result({"n_done": len(items) - n_fail, "n_failed": n_fail, "out_dir": str(out_dir)})
    return 1 if n_fail else 0


if __name__ == "__main__":
    run(main)
