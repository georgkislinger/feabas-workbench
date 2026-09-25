"""Window 2: optional preprocessing of the raw tiles: histogram matching and N2V-family denoising."""

from __future__ import annotations

import sys
import json
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPushButton, QRadioButton, QVBoxLayout, QWidget, QSplitter)

from ...core import histmatch as H
from ...core.images import imread, downsample, to_uint8
from ...core.testruns import retarget_stitch_coords, parse_stitch_coord
from ..widgets import PathPicker, TileGridWidget, ImageView, card, hint, form_row, spin, dspin, combo, labelled, row_widget, Collapsible
from .base import Page
from ..threads import ThreadRunner
from ..widgets.loss_plot import LossPlot
from ...core.n2v_training import inspect_selection


class PreprocessPage(Page):
    title = "Preprocessing (optional)"
    subtitle = ("Match all tiles to one template histogram and/or denoise them with CAREamics Noise2Void. "
                "Outputs mirror the raw folder structure, so the pipeline can switch between raw and preprocessed tiles.")
    key = "preprocess"

    def build(self) -> None:
        self._plan = None
        # --- source selection --------------------------------------------
        f, lay = card("Tiles used by the pipeline")
        r = QHBoxLayout()
        self.src_raw = QRadioButton("raw tiles")
        self.src_hm = QRadioButton("histogram-matched tiles")
        self.src_dn = QRadioButton("denoised tiles")
        self.src_raw.setChecked(True)
        r.addWidget(self.src_raw); r.addWidget(self.src_hm); r.addWidget(self.src_dn)
        self.apply_src = QPushButton("Apply to coordinate files")
        self.apply_src.setObjectName("Primary")
        r.addWidget(self.apply_src); r.addStretch(1)
        lay.addLayout(r)
        self.src_info = QLabel("")
        self.src_info.setObjectName("Hint")
        lay.addWidget(self.src_info)
        self.apply_src.clicked.connect(self._apply_source)
        self.body.addWidget(f)

        # --- histogram matching ------------------------------------------
        f, lay = card("Histogram matching")
        lay.addWidget(hint("Every tile's grey-level histogram is mapped onto the template's (monotonic CDF matching), "
                           "which removes brightness/contrast drift between tiles and sections before stitching. "
                           "Black (0) and/or white pixels can be excluded and kept as they are."))
        self.hm_template = PathPicker("file", "template image (a representative tile)", "Images (*.tif *.tiff *.png)")
        lay.addWidget(form_row("Template", self.hm_template))
        r = QHBoxLayout()
        self.hm_black = QCheckBox("ignore black (0)"); self.hm_black.setChecked(True)
        self.hm_white = QCheckBox("ignore white (max)")
        self.hm_workers = spin(1, 128, 8, suffix=" workers")
        r.addWidget(self.hm_black); r.addWidget(self.hm_white); r.addWidget(self.hm_workers); r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.hm_preview = QPushButton("Preview on a tile")
        self.hm_run = QPushButton("Match all tiles")
        self.hm_run.setObjectName("Primary")
        self.hm_status = QLabel("")
        self.hm_status.setObjectName("Hint")
        r.addWidget(self.hm_preview); r.addWidget(self.hm_run); r.addWidget(self.hm_status, 1)
        lay.addLayout(r)
        self.hm_view = ImageView()
        self.hm_view.setMinimumHeight(260)
        self.hm_view.setVisible(False)
        lay.addWidget(self.hm_view)
        self.hm_preview.clicked.connect(self._hm_preview)
        self.hm_run.clicked.connect(self._hm_run)
        self.body.addWidget(f)

        # --- denoising -----------------------------------------------------
        f, lay = card("Denoising with CAREamics (N2V / N2V2 / StructN2V)")
        lay.addWidget(hint("Self-supervised: no clean images needed. Select representative tiles; the minimum is checked "
                           "by usable pixels, so one large image can be enough. Include varied tissue and noise: "
                           "pixel quantity alone does not ensure a good model. StructN2V removes "
                           "line-structured scan noise. Runs in the deep-learning environment on the GPU."))
        split = QSplitter()
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        r.addWidget(QLabel("Section")); self.dn_sec = QComboBox(); r.addWidget(self.dn_sec, 1)
        self.dn_add = QPushButton("Add selected tiles →")
        r.addWidget(self.dn_add)
        ll.addLayout(r)
        self.dn_grid = TileGridWidget(selectable=True)
        ll.addWidget(self.dn_grid, 1)
        split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Training tiles"))
        self.dn_list = QListWidget()
        rl.addWidget(self.dn_list, 1)
        r = QHBoxLayout()
        rm = QPushButton("Remove"); clr = QPushButton("Clear"); rnd = QPushButton("Auto-select by pixels")
        r.addWidget(rm); r.addWidget(clr); r.addWidget(rnd); r.addStretch(1)
        rl.addLayout(r)
        self.dn_pixels = QLabel("Select images to calculate usable pixels.")
        self.dn_pixels.setWordWrap(True)
        rl.addWidget(self.dn_pixels)
        split.addWidget(right)
        split.setSizes([600, 300])
        lay.addWidget(split)
        # what most people touch: method, run name, train; model, preview, denoise all
        r = QHBoxLayout()
        self.dn_method = combo([("N2V", "n2v"), ("N2V2", "n2v2"), ("StructN2V (line noise)", "structn2v")], "n2v")
        self.dn_method.setToolTip("N2V: the classic blind-spot denoiser. N2V2: improved variant without checkerboard "
                                  "artefacts (recommended). StructN2V: also removes line-structured scan noise along "
                                  "one axis.")
        self.dn_axes = combo([("horizontal", "horizontal"), ("vertical", "vertical"), ("cross", "cross")], "horizontal")
        self.dn_span = spin(3, 15, 5)
        self.dn_struct_row = row_widget(
            labelled("struct axes", self.dn_axes, "StructN2V: direction of the line noise to remove."),
            labelled("span px", self.dn_span, "StructN2V: length of the blind stripe along that axis."), stretch=False)
        self.dn_name = QLineEdit("n2v_run1"); self.dn_name.setMaximumWidth(200)
        self.dn_train = QPushButton("Train model"); self.dn_train.setObjectName("Primary")
        self.dn_train.setToolTip("Train on the training tiles listed above, in the deep-learning environment. "
                                 "The result appears in the model list.")
        r.addWidget(labelled("Method", self.dn_method)); r.addWidget(self.dn_struct_row)
        r.addWidget(labelled("Run name", self.dn_name, "Folder name under models/n2v/.")); r.addWidget(self.dn_train)
        r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.dn_models = QComboBox(); self.dn_models.setMinimumWidth(240)
        self.dn_models.setToolTip("Trained models found under models/n2v/.")
        self.dn_preview = QPushButton("Preview on a tile")
        self.dn_preview.setToolTip("Denoise one tile (the selected one, or the first of the section) and show raw vs. denoised.")
        self.dn_run = QPushButton("Denoise all tiles"); self.dn_run.setObjectName("Primary")
        self.dn_run.setToolTip("Denoise every tile of the project into preprocessed/denoised (skips tiles already done).")
        r.addWidget(labelled("Model", self.dn_models)); r.addWidget(self.dn_preview); r.addWidget(self.dn_run)
        r.addStretch(1)
        lay.addLayout(r)
        # everything else keeps its defaults for most data
        self.dn_adv = Collapsible("Advanced settings")
        self.dn_patch = spin(32, 1024, 128, 32); self.dn_batch = spin(1, 128, 12); self.dn_epochs = spin(1, 5000, 100)
        self.dn_steps = spin(1, 5000, 100); self.dn_roi = spin(3, 31, 11); self.dn_mask = dspin(0.01, 20, 0.2, 0.05, 2)
        self.dn_adv.addWidget(row_widget(
            QLabel("training:"),
            labelled("patch px", self.dn_patch, "Size of the random crops the network is trained on."),
            labelled("batch", self.dn_batch, "Crops per training step; lower on small GPUs."),
            labelled("epochs", self.dn_epochs, "Training epochs."),
            labelled("steps/epoch", self.dn_steps, "Training steps per epoch."),
            labelled("ROI px", self.dn_roi, "Neighbourhood from which a masked pixel's replacement is drawn."),
            labelled("masked %", self.dn_mask, "Percentage of pixels masked per patch.")))
        self.dn_tile = spin(128, 4096, 512, 64); self.dn_overlap = spin(0, 512, 64, 16); self.dn_pbatch = spin(1, 64, 4)
        self.dn_adv.addWidget(row_widget(
            QLabel("prediction:"),
            labelled("tile px", self.dn_tile, "Tiles the image is cut into for prediction; lower on small GPUs."),
            labelled("overlap px", self.dn_overlap, "Overlap between prediction tiles, so no seams show."),
            labelled("batch", self.dn_pbatch, "Prediction tiles per batch.")))
        lay.addWidget(self.dn_adv)
        self.dn_status = QLabel("")
        self.dn_status.setObjectName("Hint")
        lay.addWidget(self.dn_status)
        self.dn_live = QWidget()
        live = QVBoxLayout(self.dn_live); live.setContentsMargins(0, 0, 0, 0)
        live.addWidget(QLabel("Training progress · best model uses lowest validation loss"))
        self.dn_loss = LossPlot(); live.addWidget(self.dn_loss)
        self.dn_best_label = QLabel("The same held-out patch is used throughout; both images share the same contrast.")
        self.dn_best_label.setWordWrap(True); live.addWidget(self.dn_best_label)
        self.dn_best_view = ImageView(); self.dn_best_view.setMinimumHeight(260)
        self.dn_best_view.setVisible(False); live.addWidget(self.dn_best_view)
        self.dn_stop = QPushButton("Stop after this epoch")
        self.dn_stop.setToolTip("Finish this epoch and keep the checkpoint with the lowest validation loss.")
        self.dn_stop.setEnabled(False); self.dn_stop.clicked.connect(self._dn_stop_training)
        live.addWidget(self.dn_stop)
        self.dn_live.setVisible(False); lay.addWidget(self.dn_live)
        self._dn_training_work = None
        self._dn_preview_epoch = None
        self._dn_live_timer = QTimer(self); self._dn_live_timer.setInterval(1000)
        self._dn_live_timer.timeout.connect(self._dn_poll_training)
        self.runner = ThreadRunner(self)
        self._dn_check_timer = QTimer(self); self._dn_check_timer.setSingleShot(True)
        self._dn_check_timer.timeout.connect(self._dn_assess)
        self.dn_patch.valueChanged.connect(self._dn_schedule_check)
        self.dn_batch.valueChanged.connect(self._dn_schedule_check)
        self.dn_list.model().rowsInserted.connect(self._dn_schedule_check)
        self.dn_list.model().rowsRemoved.connect(self._dn_schedule_check)
        self.dn_list.model().modelReset.connect(self._dn_schedule_check)
        self.dn_view = ImageView()
        self.dn_view.setMinimumHeight(260)
        self.dn_view.setVisible(False)
        lay.addWidget(self.dn_view)
        self.body.addWidget(f)
        self.dn_sec.currentIndexChanged.connect(self._dn_show_section)
        self.dn_add.clicked.connect(self._dn_add)
        rm.clicked.connect(self._dn_remove)
        clr.clicked.connect(self._dn_clear)
        rnd.clicked.connect(self._dn_random)
        self.dn_train.clicked.connect(self._dn_train)
        self.dn_preview.clicked.connect(self._dn_preview)
        self.dn_run.clicked.connect(self._dn_run)
        self.dn_method.currentIndexChanged.connect(self._dn_method_changed)
        self._dn_method_changed()
        self.ctx.jobs.job_finished.connect(self._job_finished)

    def shutdown(self):
        self._dn_check_timer.stop()
        self._dn_live_timer.stop()
        super().shutdown()

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        self._plan = None
        self._dn_training_work = None
        self._dn_live_timer.stop()
        self.dn_live.setVisible(False)
        self.dn_sec.clear()
        self.dn_list.clear()
        if project is None:
            return
        pp = project.state.preprocessing
        hm, dn = pp.histmatch, pp.denoise
        self.hm_template.setText(hm.get("template", ""))
        self.hm_black.setChecked(hm.get("ignore_black", True))
        self.hm_white.setChecked(hm.get("ignore_white", False))
        self.hm_workers.setValue(int(hm.get("workers", 8)))
        for p in dn.get("training_tiles", []):
            self.dn_list.addItem(p)
        self.dn_method.setCurrentIndex(max(0, self.dn_method.findData(dn.get("method", "n2v"))))
        self.dn_patch.setValue(int(dn.get("patch_size", 128))); self.dn_batch.setValue(int(dn.get("batch_size", 12)))
        self.dn_epochs.setValue(int(dn.get("epochs", 100))); self.dn_steps.setValue(int(dn.get("steps_per_epoch", 100)))
        self.dn_roi.setValue(int(dn.get("roi_size", 11))); self.dn_mask.setValue(float(dn.get("masked_pixel_percentage", 0.2)))
        self.dn_axes.setCurrentIndex(max(0, self.dn_axes.findData(dn.get("struct_axes", "horizontal"))))
        self.dn_span.setValue(int(dn.get("struct_span", 5)))
        self.dn_tile.setValue(int(dn.get("tile_size", 512))); self.dn_overlap.setValue(int(dn.get("tile_overlap", 64)))
        {"raw": self.src_raw, "histmatch": self.src_hm, "denoise": self.src_dn}.get(pp.active_source, self.src_raw).setChecked(True)
        self._refresh_models()
        self._refresh_source_info()
        self._load_sections()

    def on_shown(self) -> None:
        if self.project and not self._plan:
            self._load_sections()

    def _load_sections(self) -> None:
        p = self.project
        if not p:
            return
        names = p.section_names()
        self.dn_sec.blockSignals(True)
        self.dn_sec.clear()
        self.dn_sec.addItems(names)
        self.dn_sec.blockSignals(False)
        if names:
            self._dn_show_section()

    def _section_tiles(self, name: str):
        """(rel, abs path, x, y) for a section from its coordinate file (raw root)."""
        p = self.project
        f = p.stitch_coord_dir / f"{name}.txt"
        if not f.is_file():
            return None, []
        info = parse_stitch_coord(f)
        raw_root = Path(p.state.source.root_dir) if p.state.source.root_dir else Path(info["root"] or ".")
        out = []
        for rel, x, y in info["tiles"]:
            out.append((rel, raw_root / rel, x, y))
        return info, out

    def _dn_show_section(self) -> None:
        name = self.dn_sec.currentText()
        if not name:
            return
        info, tl = self._section_tiles(name)
        if not tl:
            self.dn_grid.set_tiles([])
            return
        th, tw = info["tile_size"] or (self.project.state.volume.tile_h, self.project.state.volume.tile_w)
        self.dn_grid.set_tiles([(str(ap), x, y, tw, th, "") for rel, ap, x, y in tl])

    def _dn_add(self) -> None:
        existing = {self.dn_list.item(i).text() for i in range(self.dn_list.count())}
        for k in sorted(self.dn_grid.selected):
            if k not in existing:
                self.dn_list.addItem(k)
        self.dn_grid.set_selected(set())
        self._save_dn()

    def _dn_remove(self) -> None:
        for it in self.dn_list.selectedItems():
            self.dn_list.takeItem(self.dn_list.row(it))
        self._save_dn()

    def _dn_clear(self):
        self.dn_list.clear()
        self._save_dn()

    def _dn_schedule_check(self, *args):
        self._dn_check_timer.start(300)

    def _dn_random(self):
        self._dn_assess("auto")

    def _dn_assess(self, action="check"):
        if not self.project:
            return
        if self.runner.running:
            self._dn_check_timer.start(300)
            return
        settings = self._dn_settings()
        paths = settings["training_tiles"]
        if action == "auto":
            paths = [str(ap) for name in self.project.section_names()
                     for _, ap, _, _ in self._section_tiles(name)[1]]
        project = self.project
        self.dn_train.setEnabled(False)
        self.dn_pixels.setText("Reading image dimensions…")
        def done(info, error):
            self.dn_train.setEnabled(not self.ctx.jobs.running)
            if self.project is not project:
                return
            now = self._dn_settings()
            if any(now[k] != settings[k] for k in ("training_tiles", "patch_size", "batch_size")):
                self._dn_schedule_check()
                return
            if error:
                self.dn_pixels.setText(error.splitlines()[0])
                return
            if not info:
                return
            self.dn_pixels.setText(info["message"])
            if action == "auto":
                self.dn_list.clear()
                self.dn_list.addItems([tile.path for tile in info["tiles"]])
                self._save_dn()
            elif action == "train" and info["valid"]:
                self._dn_launch_training()
        self.runner.start(inspect_selection, (paths, settings["patch_size"], settings["batch_size"]),
                          {"auto": action == "auto"}, on_done=done)

    def _dn_method_changed(self) -> None:
        self.dn_struct_row.setVisible(self.dn_method.currentData() == "structn2v")

    def _dn_settings(self) -> dict:
        return {
            "method": self.dn_method.currentData(), "patch_size": self.dn_patch.value(), "batch_size": self.dn_batch.value(),
            "epochs": self.dn_epochs.value(), "steps_per_epoch": self.dn_steps.value(), "roi_size": self.dn_roi.value(),
            "masked_pixel_percentage": self.dn_mask.value(), "struct_axes": self.dn_axes.currentData(),
            "struct_span": self.dn_span.value(), "tile_size": self.dn_tile.value(), "tile_overlap": self.dn_overlap.value(),
            "training_tiles": [self.dn_list.item(i).text() for i in range(self.dn_list.count())],
        }

    def _save_dn(self) -> None:
        if self.project:
            self.project.state.preprocessing.denoise.update(self._dn_settings())
            self.project.save()

    def _save_hm(self) -> None:
        if self.project:
            self.project.state.preprocessing.histmatch.update({
                "template": self.hm_template.text(), "ignore_black": self.hm_black.isChecked(),
                "ignore_white": self.hm_white.isChecked(), "workers": self.hm_workers.value()})
            self.project.save()

    # -- histogram matching ---------------------------------------------
    def _hm_preview(self) -> None:
        if not self.require_project():
            return
        tmpl = self.hm_template.path()
        if not tmpl or not tmpl.is_file():
            QMessageBox.information(self, "Template", "Choose a template image first.")
            return
        keys = sorted(self.dn_grid.selected)
        if keys:
            tile = Path(keys[0])
        else:
            _, tl = self._section_tiles(self.dn_sec.currentText())
            if not tl:
                return
            tile = tl[0][1]
        try:
            t = H.read_gray(tmpl)
            s = H.read_gray(tile)
            cT = H.template_cdf(t, self.hm_black.isChecked(), self.hm_white.isChecked())
            out = H.match_image(s, cT, self.hm_black.isChecked(), self.hm_white.isChecked())
        except Exception as e:  # noqa: BLE001
            self.error(f"preview failed: {e}")
            return
        f = 1
        while max(s.shape) / f > 1200:
            f *= 2
        a = to_uint8(downsample(s, f)); b = to_uint8(downsample(out, f))
        both = np.concatenate([a, np.full((a.shape[0], 8), 255, np.uint8), b], axis=1)
        self.hm_view.setVisible(True)
        self.hm_view.set_image(both)
        self.hm_status.setText(f"left: {tile.name} raw, right: matched to {tmpl.name}")
        self._save_hm()

    def _hm_run(self) -> None:
        if not self.require_project():
            return
        p = self.project
        tmpl = self.hm_template.path()
        if not tmpl or not tmpl.is_file():
            QMessageBox.information(self, "Template", "Choose a template image first.")
            return
        self._save_hm()
        raw = Path(p.state.source.root_dir)
        out = p.preprocessed_dir / "histmatch"
        rule = p.state.source.naming_rule()
        payload = {"in_root": str(raw), "out_root": str(out), "template": str(tmpl), "ext": rule.ext,
                   "recursive": rule.recursive, "ignore_black": self.hm_black.isChecked(),
                   "ignore_white": self.hm_white.isChecked(), "workers": self.hm_workers.value()}
        spec = self.ctx.worker_spec("histmatch_worker", payload, "Histogram matching", python=sys.executable)
        self.submit(spec)

    # -- denoising ---------------------------------------------------------
    def _refresh_models(self) -> None:
        self.dn_models.clear()
        p = self.project
        if not p:
            return
        d = p.models_dir / "n2v"
        if d.is_dir():
            for run in sorted(d.iterdir()):
                ck = [c for c in run.rglob("*.ckpt") if not c.name.endswith(".pending.ckpt")]
                if ck:
                    best = [c for c in ck if "last" not in c.name] or ck
                    checkpoint = run / "best.ckpt"
                    self.dn_models.addItem(run.name, str(checkpoint if checkpoint.is_file() else sorted(best)[-1]))

    def _dn_train(self) -> None:
        if not self.require_project():
            return
        if self.ctx.jobs.running:
            return
        self._dn_assess("train")

    def _dn_launch_training(self):
        if self.ctx.jobs.running:
            return
        if not self.ctx.dl_python():
            QMessageBox.information(self, "Environment", "Configure the deep-learning Python on the Setup page.")
            return
        self._save_dn()
        name = self.dn_name.text().strip() or "n2v_run"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            QMessageBox.information(self, "Run name", "Use letters, numbers, dashes and underscores for the run name.")
            return
        work = self.project.models_dir / "n2v" / name
        if work.exists():
            QMessageBox.information(self, "Run exists", "Choose a new run name to preserve the previous model and loss history.")
            return
        payload = dict(self._dn_settings(), work_dir=str(work), experiment=name, seed=42)
        spec = self.ctx.worker_spec("n2v_train", payload, f"N2V training '{name}'")
        self._dn_training_work = work
        self._dn_preview_epoch = None
        self.dn_loss.set_history([]); self.dn_best_view.setVisible(False)
        self.dn_live.setVisible(True)
        self.dn_best_label.setText("Waiting for the first completed epoch. Left: original held-out patch; right: best denoised patch.")
        self.dn_stop.setEnabled(not self.ctx.cluster_enabled)
        self.dn_stop.setText("Stop after this epoch")
        if self.ctx.cluster_enabled:
            self.dn_best_label.setText("Live preview and stop-after-epoch are available for local training. Cluster models return after the job finishes.")
        else:
            self._dn_live_timer.start()
        self.submit(spec)

    def _dn_poll_training(self):
        work = self._dn_training_work
        if work is None:
            return
        try:
            state = json.loads((work / "training_status.json").read_text(encoding="utf-8"))
            self.dn_loss.set_history(state["history"])
            epoch = state.get("best_epoch")
            if epoch and epoch != self._dn_preview_epoch and state.get("preview"):
                with np.load(work / "training_preview.npz", allow_pickle=False) as preview:
                    if int(preview["epoch"]) != epoch:
                        return  # reader crossed an atomic snapshot update; retry next tick
                    raw, den = preview["original"], preview["denoised"]
                lo, hi = np.percentile(raw[np.isfinite(raw)], (.5, 99.5))
                a, b = to_uint8(raw, lo, hi), to_uint8(den, lo, hi)
                self.dn_best_view.set_image(np.concatenate([a, np.full((a.shape[0], 8), 255, np.uint8), b], axis=1))
                self.dn_best_view.setVisible(True)
                self._dn_preview_epoch = epoch
            loss = f"{state['best_loss']:.5g}" if state.get("best_loss") is not None else "—"
            self.dn_best_label.setText(f"{state['state'].capitalize()} · best epoch {epoch or '—'} · "
                f"validation loss {loss}\nLeft: original held-out patch · right: best denoised patch · same contrast")
            if state["state"] in {"completed", "stopped", "failed"}:
                self.dn_stop.setEnabled(False)
                self._dn_live_timer.stop()
        except (OSError, ValueError, KeyError):
            pass  # no complete snapshot yet

    def _dn_stop_training(self):
        if not self._dn_training_work or self.ctx.cluster_enabled or not self.ctx.jobs.running:
            return
        try:
            self._dn_training_work.mkdir(parents=True, exist_ok=True)
            (self._dn_training_work / "stop_after_epoch.request").touch()
        except OSError as error:
            self.error(f"Could not request a graceful stop: {error}")
            return
        self.dn_stop.setEnabled(False)
        self.dn_stop.setText("Finishing this epoch; keeping the best model…")

    def on_running_changed(self, running):
        self.dn_train.setEnabled(not running and not self.runner.running)
        if not running:
            self.dn_stop.setEnabled(False)

    def _dn_preview(self) -> None:
        ck = self.dn_models.currentData()
        if not ck:
            QMessageBox.information(self, "Model", "Train a model first (or none found under models/n2v).")
            return
        keys = sorted(self.dn_grid.selected)
        if keys:
            tile = Path(keys[0])
        else:
            _, tl = self._section_tiles(self.dn_sec.currentText())
            if not tl:
                return
            tile = tl[0][1]
        out = self.project.models_dir / "n2v" / "previews"
        payload = {"checkpoint": ck, "files": [str(tile)], "in_root": str(tile.parent), "out_root": str(out),
                   "tile_size": self.dn_tile.value(), "tile_overlap": self.dn_overlap.value(), "batch_size": self.dn_pbatch.value(),
                   "skip_existing": False}
        self._preview_pair = (tile, out / tile.name)
        spec = self.ctx.worker_spec("n2v_predict", payload, "N2V preview")
        self.submit(spec)

    def _dn_run(self) -> None:
        if not self.require_project():
            return
        ck = self.dn_models.currentData()
        if not ck:
            QMessageBox.information(self, "Model", "Train a model first.")
            return
        p = self.project
        src_root = p.preprocessed_dir / "histmatch" if self.src_hm.isChecked() and (self.ctx.cluster_enabled or (p.preprocessed_dir / "histmatch").is_dir()) else Path(p.state.source.root_dir)
        rule = p.state.source.naming_rule()
        out = p.preprocessed_dir / "denoised"
        payload = {"checkpoint": ck, "in_root": str(src_root), "out_root": str(out), "ext": rule.ext, "recursive": rule.recursive,
                   "tile_size": self.dn_tile.value(), "tile_overlap": self.dn_overlap.value(), "batch_size": self.dn_pbatch.value(),
                   "skip_existing": True}
        p.state.preprocessing.denoise["checkpoint"] = ck
        p.save()
        spec = self.ctx.worker_spec("n2v_predict", payload, "N2V denoising of all tiles")
        self.submit(spec)

    def _job_finished(self, res) -> None:
        if not self.project:
            return
        name = res.spec.name
        if name.startswith("N2V training"):
            self._dn_poll_training()
            self._dn_live_timer.stop()
            self.dn_stop.setEnabled(False)
            self._refresh_models()
            if self._dn_training_work:
                index = self.dn_models.findText(self._dn_training_work.name)
                if index >= 0:
                    self.dn_models.setCurrentIndex(index)
            if res.ok:
                self.dn_status.setText(f"Training {res.result.get('state', 'finished')}. Best epoch: {res.result.get('best_epoch', 'see model')}.")
            else:
                self.dn_status.setText("Training stopped or failed. Any previously saved best model is preserved; see the job log.")
        elif name == "N2V preview" and res.ok and getattr(self, "_preview_pair", None):
            raw, den = self._preview_pair
            if den.is_file():
                a = imread(raw); b = imread(den)
                f = 1
                while max(a.shape) / f > 1200:
                    f *= 2
                a8 = to_uint8(downsample(a, f)); b8 = to_uint8(downsample(b, f))
                both = np.concatenate([a8, np.full((a8.shape[0], 8), 255, np.uint8), b8], axis=1)
                self.dn_view.setVisible(True)
                self.dn_view.set_image(both)
                self.dn_status.setText(f"left: raw {raw.name}, right: denoised")
        elif name.startswith("N2V denoising") and res.ok:
            self.project.state.preprocessing.denoise["done"] = True
            self.project.save()
        elif name == "Histogram matching" and res.ok:
            self.project.state.preprocessing.histmatch["done"] = True
            self.project.save()
        self._refresh_source_info()

    # -- source switch -----------------------------------------------------
    def _refresh_source_info(self) -> None:
        p = self.project
        if not p:
            return
        hm = p.preprocessed_dir / "histmatch"
        dn = p.preprocessed_dir / "denoised"
        n_hm = sum(1 for _ in hm.rglob("*.*")) if hm.is_dir() else 0
        n_dn = sum(1 for _ in dn.rglob("*.*")) if dn.is_dir() else 0
        if self.ctx.cluster_enabled:
            counts = self.ctx.cluster.snapshot.get("preprocessing", {})
            n_hm, n_dn = counts.get("histmatch", 0), counts.get("denoised", 0)
        self.src_hm.setEnabled(n_hm > 0); self.src_dn.setEnabled(n_dn > 0)
        cur = p.state.preprocessing.active_source
        self.src_info.setText(f"raw: {p.state.volume.n_tiles} tiles · histogram-matched: {n_hm} files · denoised: {n_dn} files · "
                              f"currently active: {cur}")

    def _apply_source(self) -> None:
        if not self.require_project():
            return
        p = self.project
        choice = "histmatch" if self.src_hm.isChecked() else ("denoise" if self.src_dn.isChecked() else "raw")
        p.state.preprocessing.active_source = choice
        root = p.active_tile_root()
        if root is None or (not self.ctx.cluster_enabled and not root.is_dir()):
            self.error(f"tile folder not found: {root}")
            return
        n = retarget_stitch_coords(p, root)
        p.save()
        self.info(f"{n} coordinate files now point at {root}")
        if any((p.root / d).exists() for d in ("stitch/match_h5", "stitch/tform")):
            self.warn("stitching outputs exist; clear 'Match tiles' (Stitching page) to re-run with the new tiles")
        self._refresh_source_info()
        self.ctx.state_changed.emit()
