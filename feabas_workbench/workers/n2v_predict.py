"""Denoise tiles with a trained CAREamics model, one file at a time (low memory), preserving dtype and folder layout."""

from __future__ import annotations

from pathlib import Path

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def main() -> int:
    spec = load_spec("N2V prediction")
    import numpy as np
    import tifffile
    from careamics.careamist import CAREamist

    ck = Path(spec["checkpoint"])
    in_root = Path(spec["in_root"])
    out_root = Path(spec["out_root"])
    files = [Path(p) for p in spec.get("files", [])]
    if not files:
        from feabas_workbench.core.tiles import list_image_files
        files = list_image_files(in_root, spec.get("ext", "tif"), spec.get("recursive", True))
    skip = bool(spec.get("skip_existing", True))
    ts = int(spec.get("tile_size", 512))
    ov = int(spec.get("tile_overlap", 64))
    bs = int(spec.get("batch_size", 4))
    import inspect
    if "checkpoint_path" in inspect.signature(CAREamist.__init__).parameters:
        careamist = CAREamist(checkpoint_path=ck, work_dir=ck.parent)
    else:
        careamist = CAREamist(ck, work_dir=ck.parent)
    log(f"model {ck.name}; {len(files)} files -> {out_root}")
    todo = []
    for f in files:
        try:
            rel = f.resolve().relative_to(in_root.resolve())
        except ValueError:
            rel = Path(f.name)
        dst = out_root / rel
        if skip and dst.is_file():
            continue
        todo.append((f, dst))
    progress(0, len(todo), "starting")
    n_fail = 0
    for i, (src, dst) in enumerate(todo, 1):
        try:
            img = tifffile.imread(str(src)) if src.suffix.lower() in (".tif", ".tiff") else np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(src))
            if img.ndim == 3:
                img = img[..., 0]
            dtype = img.dtype
            pred = careamist.predict(img.astype(np.float32), data_type="array", axes="YX",
                                     tile_size=(ts, ts), tile_overlap=(ov, ov), batch_size=bs, num_workers=0)
            arr = np.squeeze(np.asarray(pred[0] if isinstance(pred, (list, tuple)) else pred)).astype(np.float32)
            if arr.shape != img.shape:
                raise RuntimeError(f"prediction shape {arr.shape} != input {img.shape}")
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                arr = np.clip(np.rint(arr), info.min, info.max).astype(dtype)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.suffix.lower() in (".tif", ".tiff"):
                tifffile.imwrite(str(dst), arr, photometric="minisblack")
            else:
                from PIL import Image
                Image.fromarray(arr).save(dst)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            log(f"FAILED {src}: {e}")
        progress(i, len(todo), src.name)
    result({"n_done": len(todo) - n_fail, "n_failed": n_fail, "out_root": str(out_root)})
    return 1 if n_fail else 0


if __name__ == "__main__":
    run(main)
