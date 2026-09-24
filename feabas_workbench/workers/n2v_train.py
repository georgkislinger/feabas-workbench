"""Train a CAREamics N2V / N2V2 / StructN2V model on selected tiles (deep-learning environment)."""

from __future__ import annotations

import json
from pathlib import Path

from feabas_workbench.workers.common import load_spec, progress, result, log, run


def main() -> int:
    spec = load_spec("N2V training")
    import torch
    from careamics.careamist import CAREamist
    from careamics.config.factories import create_advanced_n2v_config

    from feabas_workbench.core.n2v_training import prepare_data
    from feabas_workbench.workers.n2v_monitor import TrainingMonitor

    method = spec.get("method", "n2v")
    work = Path(spec["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    if (work / "training_status.json").exists() or any(work.rglob("*.ckpt")):
        raise ValueError("This training folder already contains a run. Choose a new run name.")
    ps = int(spec.get("patch_size", 128))
    batch = int(spec.get("batch_size", 12))
    train_data, val_data, manifest = prepare_data(spec["training_tiles"], ps, batch,
                                                 int(spec.get("seed", 42)), log)
    (work / "training_files.json").write_text(json.dumps(dict(manifest, spec=spec), indent=2), encoding="utf-8")
    log(f"Sampled {train_data.shape[0]} training / {val_data.shape[0]} validation patches; "
        f"GPU={'yes: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'}")
    if method == "structn2v":
        struct_axes = spec.get("struct_axes", "horizontal")
        augmentations = []
        use_n2v2 = False
    else:
        struct_axes = "none"
        augmentations = None
        use_n2v2 = method == "n2v2"
    config = create_advanced_n2v_config(
        experiment_name=spec.get("experiment", work.name),
        data_type="array", axes="SYX", patch_size=[ps, ps],
        batch_size=int(spec.get("batch_size", 12)),
        num_epochs=int(spec.get("epochs", 100)),
        num_steps=int(spec.get("steps_per_epoch", 100)),
        augmentations=augmentations, n_val_patches=1, in_memory=True,
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
    careamist = CAREamist(config, work_dir=work, enable_progress_bar=False)
    total_epochs = int(spec.get("epochs", 100))

    monitor = TrainingMonitor(work, total_epochs)
    careamist.trainer.callbacks.append(monitor)
    progress(0, total_epochs, "training; preview appears after the first completed epoch")
    try:
        careamist.train(train_data=train_data, val_data=val_data)
        if monitor.state["best_epoch"] is None:
            raise RuntimeError("Training produced no finite validation loss. No best model was saved; check the data and job log.")
    except BaseException:
        monitor.finish(failed=True)
        raise
    monitor.finish()
    ckpts = [str(c) for c in work.rglob("*.ckpt")]
    log(f"checkpoints: {ckpts}")
    result({"checkpoints": ckpts, "work_dir": str(work), "best_checkpoint": str(work / "best.ckpt"),
            "best_epoch": monitor.state["best_epoch"], "state": monitor.state["state"]})
    return 0


if __name__ == "__main__":
    run(main)
