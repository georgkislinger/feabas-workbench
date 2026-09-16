"""Window 2: optional preprocessing of the raw tiles: histogram matching and N2V-family denoising."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QPushButton, QRadioButton, QVBoxLayout, QWidget, QSplitter)

from ...core import histmatch as H
from ...core.images import imread, downsample, to_uint8
from ...core.testruns import retarget_stitch_coords, parse_stitch_coord
from ..widgets import PathPicker, TileGridWidget, ImageView, card, hint, form_row, spin, dspin, combo, labelled, row_widget, Collapsible
from .base import Page


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
        lay.addWidget(hint("Self-supervised: no clean images needed. Pick a handful of representative tiles as training "
                           "data (10–40 is plenty), train, check the preview, then denoise all tiles. StructN2V removes "
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
        rm = QPushButton("Remove"); clr = QPushButton("Clear"); rnd = QPushButton("Random 24")
        r.addWidget(rm); r.addWidget(clr); r.addWidget(rnd); r.addStretch(1)
        rl.addLayout(r)
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
        self.dn_view = ImageView()
        self.dn_view.setMinimumHeight(260)
        self.dn_view.setVisible(False)
        lay.addWidget(self.dn_view)
        self.body.addWidget(f)
        self.dn_sec.currentIndexChanged.connect(self._dn_show_section)
        self.dn_add.clicked.connect(self._dn_add)
        rm.clicked.connect(self._dn_remove)
        clr.clicked.connect(self.dn_list.clear)
        rnd.clicked.connect(self._dn_random)
        self.dn_train.clicked.connect(self._dn_train)
        self.dn_preview.clicked.connect(self._dn_preview)
        self.dn_run.clicked.connect(self._dn_run)
        self.dn_method.currentIndexChanged.connect(self._dn_method_changed)
        self._dn_method_changed()
        self.ctx.jobs.job_finished.connect(self._job_finished)

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        self._plan = None
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

    def _dn_random(self) -> None:
        import random
        p = self.project
        if not p:
            return
        allt = []
        for name in p.section_names():
            _, tl = self._section_tiles(name)
            allt += [str(ap) for _, ap, _, _ in tl]
        if not allt:
            return
        random.seed(42)
        self.dn_list.clear()
        for t in random.sample(allt, min(24, len(allt))):
            self.dn_list.addItem(t)
        self._save_dn()

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
                ck = list(run.rglob("*.ckpt"))
                if ck:
                    best = [c for c in ck if "last" not in c.name] or ck
                    self.dn_models.addItem(run.name, str(sorted(best)[-1]))

    def _dn_train(self) -> None:
        if not self.require_project():
            return
        s = self._dn_settings()
        if len(s["training_tiles"]) < 4:
            QMessageBox.information(self, "Training tiles", "Add at least 4 training tiles (10–40 recommended).")
            return
        if not self.ctx.dl_python():
            QMessageBox.information(self, "Environment", "Configure the deep-learning Python on the Setup page.")
            return
        self._save_dn()
        name = self.dn_name.text().strip() or "n2v_run"
        work = self.project.models_dir / "n2v" / name
        payload = dict(s, work_dir=str(work), experiment=name, seed=42)
        spec = self.ctx.worker_spec("n2v_train", payload, f"N2V training '{name}'")
        self.submit(spec)

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
        src_root = p.preprocessed_dir / "histmatch" if self.src_hm.isChecked() and (p.preprocessed_dir / "histmatch").is_dir() else Path(p.state.source.root_dir)
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
            self._refresh_models()
            if res.ok:
                self.dn_status.setText(f"training finished: {res.result}")
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
        if root is None or not root.is_dir():
            self.error(f"tile folder not found: {root}")
            return
        n = retarget_stitch_coords(p, root)
        p.save()
        self.info(f"{n} coordinate files now point at {root}")
        if any((p.root / d).exists() for d in ("stitch/match_h5", "stitch/tform")):
            self.warn("stitching outputs exist; clear 'Match tiles' (Stitching page) to re-run with the new tiles")
        self._refresh_source_info()
        self.ctx.state_changed.emit()
