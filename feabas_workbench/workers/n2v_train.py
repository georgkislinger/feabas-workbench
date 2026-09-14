"""Train a CAREamics N2V / N2V2 / StructN2V model on selected tiles (deep-learning environment)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def main() -> int:
    spec = load_spec("N2V training")
    import numpy as np
    import tifffile
    import torch
    from careamics.careamist import CAREamist
    from careamics.config.factories import create_advanced_n2v_config

    files = [Path(p) for p in spec["training_tiles"]]
    files = [f for f in files if f.is_file()]
    if len(files) < 4:
        raise RuntimeError("need at least 4 training tiles")
    rng = random.Random(int(spec.get("seed", 42)))
    rng.shuffle(files)
    n_val = max(1, int(round(0.2 * len(files))))
    val_files, train_files = files[:n_val], files[n_val:]
    method = spec.get("method", "n2v")
    work = Path(spec["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    (work / "training_files.json").write_text(json.dumps({"train": [str(f) for f in train_files], "val": [str(f) for f in val_files],
                                                          "spec": spec}, indent=2), encoding="utf-8")
    # dtype/range info for prediction-time conversion
    sample = tifffile.imread(str(train_files[0]))
    if sample.ndim != 2:
        raise RuntimeError("tiles must be single-plane 2D grey images")
    (work / "input_info.json").write_text(json.dumps({"dtype": str(sample.dtype), "shape": list(sample.shape)}), encoding="utf-8")
    est_gb = sum(np.prod(tifffile.TiffFile(str(f)).pages[0].shape) * sample.dtype.itemsize for f in files) / 1024 ** 3
    in_memory = est_gb <= float(spec.get("in_memory_limit_gb", 16))
    log(f"{len(train_files)} train / {len(val_files)} val tiles, ~{est_gb:.2f} GB, in_memory={in_memory}, "
        f"GPU={'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
    if method == "structn2v":
        struct_axes = spec.get("struct_axes", "horizontal")
        augmentations = []
        use_n2v2 = False
    else:
        struct_axes = "none"
        augmentations = None
        use_n2v2 = method == "n2v2"
    ps = int(spec.get("patch_size", 128))
    config = create_advanced_n2v_config(
        experiment_name=spec.get("experiment", work.name),
        data_type="tiff", axes="YX", patch_size=[ps, ps],
        batch_size=int(spec.get("batch_size", 12)),
        num_epochs=int(spec.get("epochs", 100)),
        num_steps=int(spec.get("steps_per_epoch", 100)),
        augmentations=augmentations, n_val_patches=8, in_memory=in_memory,
        use_n2v2=use_n2v2, roi_size=int(spec.get("roi_size", 11)),
        masked_pixel_percentage=float(spec.get("masked_pixel_percentage", 0.2)),
        struct_n2v_axes=struct_axes, struct_n2v_span=int(spec.get("struct_span", 5)),
        num_workers=0,
        # enable_progress_bar=False: the tqdm bar writes \r-terminated partial lines to stdout, the same
        # stream as our ##PROGRESS lines; an epoch end while the bar is mid-refresh glues the JSON onto
        # the partial line and the GUI reader no longer recognises it -> "no progress" in the UI.
        trainer_params={"accelerator": "gpu" if torch.cuda.is_available() else "cpu", "devices": 1,
                        "precision": spec.get("precision", "32-true"), "log_every_n_steps": 10,
                        "enable_progress_bar": False},
        logger="none", seed=int(spec.get("seed", 42)),
    )
    careamist = CAREamist(config, work_dir=work)
    total_epochs = int(spec.get("epochs", 100))

    # progress via a lightning callback
    try:
        import pytorch_lightning as pl

        class _Prog(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                m = trainer.callback_metrics
                vl = m.get("val_loss")
                progress(trainer.current_epoch + 1, total_epochs,
                         f"epoch {trainer.current_epoch + 1}/{total_epochs}" + (f" val_loss {float(vl):.4f}" if vl is not None else ""))
        careamist.trainer.callbacks.append(_Prog())
    except Exception as e:  # noqa: BLE001
        log(f"(no epoch progress: {e})")
    progress(0, total_epochs, "training")
    careamist.train(train_data=train_files, val_data=val_files)
    ckpts = [str(c) for c in careamist.get_checkpoints()] if hasattr(careamist, "get_checkpoints") else []
    if not ckpts:
        ckpts = [str(c) for c in work.rglob("*.ckpt")]
    try:
        losses = careamist.get_losses()
        (work / "losses.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in losses.items()}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    log(f"checkpoints: {ckpts}")
    result({"checkpoints": ckpts, "work_dir": str(work)})
    return 0


if __name__ == "__main__":
    run(main)
