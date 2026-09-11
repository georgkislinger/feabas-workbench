"""Train / fine-tune an ultralytics YOLO segmentation model on a YOLO-format dataset."""

from __future__ import annotations

from pathlib import Path

import yaml

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def ensure_data_yaml(dataset: Path, names: list[str]) -> Path:
    if dataset.is_file():
        return dataset
    y = dataset / "data.yaml"
    if y.is_file():
        return y
    imgs = dataset / "images"
    if not imgs.is_dir():
        raise RuntimeError(f"{dataset}: expected data.yaml or an images/ folder")
    train = imgs / "train" if (imgs / "train").is_dir() else imgs
    val = imgs / "val" if (imgs / "val").is_dir() else train
    if not names:
        raise RuntimeError("no data.yaml found; give the class names")
    data = {"path": str(dataset), "train": str(train), "val": str(val), "names": {i: n for i, n in enumerate(names)}}
    y.write_text(yaml.safe_dump(data), encoding="utf-8")
    log(f"wrote {y}")
    return y


def main() -> int:
    spec = load_spec("YOLO-seg training")
    from ultralytics import YOLO
    import torch
    data_yaml = ensure_data_yaml(Path(spec["dataset"]), spec.get("names") or [])
    base = spec.get("base", "yolo11n-seg.pt")
    epochs = int(spec.get("epochs", 100)); imgsz = int(spec.get("imgsz", 640))
    project_dir = Path(spec["project_dir"]); name = spec.get("name", "run")
    model = YOLO(base)

    def on_epoch_end(trainer):
        try:
            progress(trainer.epoch + 1, epochs, f"epoch {trainer.epoch + 1}/{epochs}")
        except Exception:  # noqa: BLE001
            pass

    model.add_callback("on_train_epoch_end", on_epoch_end)
    progress(0, epochs, "starting")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project_dir), name=name, exist_ok=True,
                device=0 if torch.cuda.is_available() else "cpu", workers=0, verbose=False, plots=True)
    best = project_dir / name / "weights" / "best.pt"
    log(f"best weights: {best}")
    result({"weights": str(best)})
    return 0


if __name__ == "__main__":
    run(main)
