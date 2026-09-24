"""Small atomic training snapshots; imported only in the deep-learning worker."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from feabas_workbench.workers.common import progress


def replace_snapshot(temporary, target):
    # A Windows reader may briefly hold the old snapshot open without delete sharing.
    for attempt in range(10):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    replace_snapshot(temp, path)


class TrainingMonitor(pl.Callback):
    def __init__(self, work, epochs):
        self.work, self.epochs = Path(work), epochs
        self.fixed = None
        self.batch_losses = []
        self.state = dict(state="training", epochs=epochs, history=[], best_epoch=None,
                          best_loss=None, checkpoint=None, preview=None)
        self.publish()

    def publish(self):
        atomic_json(self.work / "training_status.json", self.state)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.fixed is None:
            # Capture BEFORE N2V masking. Keep this exact held-out patch for every preview.
            item = batch[0]
            data = item if isinstance(item, torch.Tensor) else item.data
            self.fixed = data[:1].detach().clone().cpu()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None:
            value = float(loss)
            if math.isfinite(value):
                self.batch_losses.append(value)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        value = trainer.callback_metrics.get("val_loss")
        val = float(value) if value is not None else None
        if val is not None and not math.isfinite(val):
            val = None
        train = sum(self.batch_losses) / len(self.batch_losses) if self.batch_losses else None
        self.batch_losses.clear()
        self.state["history"].append(dict(epoch=epoch, train_loss=train, val_loss=val))
        best = self.state["best_loss"]
        if val is not None and (best is None or val < best):
            target = self.work / "best.ckpt"
            temporary = self.work / "best.pending.ckpt"
            trainer.save_checkpoint(temporary)
            replace_snapshot(temporary, target)
            self.state.update(best_loss=val, best_epoch=epoch, checkpoint="best.ckpt")
            if self.fixed is not None:
                training = pl_module.training
                try:
                    pl_module.eval()
                    with torch.inference_mode():
                        prediction = pl_module(self.fixed.to(pl_module.device)).detach().float().cpu().numpy().squeeze()
                    raw = self.fixed.float().numpy().squeeze()
                    # Same fixed normalization/contrast for both images; no per-image auto contrast.
                    temp = self.work / "training_preview.tmp.npz"
                    np.savez_compressed(temp, original=raw, denoised=prediction, epoch=epoch)
                    replace_snapshot(temp, self.work / "training_preview.npz")
                    self.state["preview"] = "training_preview.npz"
                finally:
                    pl_module.train(training)
        stop = (self.work / "stop_after_epoch.request").is_file()
        if stop:
            trainer.should_stop = True
            self.state["state"] = "stopping"
        self.publish()
        progress(epoch, self.epochs, f"epoch {epoch}/{self.epochs}; loss {train}; val_loss {val}; best epoch {self.state['best_epoch']}")

    def finish(self, failed=False):
        self.state["state"] = "failed" if failed else ("stopped" if self.state["state"] == "stopping" else "completed")
        self.publish()
        atomic_json(self.work / "losses.json", self.state["history"])
