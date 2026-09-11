"""Window 5: coarse (thumbnail) and fine alignment, with optional structure-guided matching."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget, QPlainTextEdit)

from ...core.steps import PipelineScan
from ...core.testruns import create_align_test, list_test_runs, delete_test_run, TestRun
from ...core.maskstore import MaskStore
from ...core.images import imread, to_uint8, compose_two_color, downsample
from ...core.configs import suggest_working_mip
from ..widgets import PathPicker, ImageView, SectionPicker, ConfigEditor, card, hint, form_row, spin, dspin, combo
from ..widgets.steps_panel import StepsPanel
from .base import Page
from .. import theme


def _read_match_h5(path: Path):
    import h5py
    with h5py.File(path, "r") as f:
        xy0 = f["xy0"][()]; xy1 = f["xy1"][()]; w = f["weight"][()]
        res = float(np.asarray(f["resolution"][()]).item())
    return xy0, xy1, w, res


class AlignPage(Page):
    title = "Alignment"
    subtitle = ("Coarse alignment on thumbnails, then fine alignment with finite-element meshes. Optionally let a "
                "structure detector (YOLO-seg: nuclei, mitochondria, vessels…) decide where the alignment should be driven from.")
    key = "align"

    def build(self) -> None:
        self.store: MaskStore | None = None
        self.tabs = QTabWidget()
        self.body.addWidget(self.tabs)
        self._build_coarse()
        self._build_structure()
        self._build_fine()
        self._build_test()
        self._build_qc()
        self.editor_t = ConfigEditor(); self.editor_t.changed.connect(lambda: self._editor_changed("thumbnail"))
        self.editor_a = ConfigEditor(); self.editor_a.changed.connect(lambda: self._editor_changed("alignment"))
        self.tabs.addTab(self.editor_t, "Thumbnail settings (all)")
        self.tabs.addTab(self.editor_a, "Alignment settings (all)")
        self.ctx.jobs.job_finished.connect(self._job_finished)

    # ---------------------------------------------------------------- coarse
    def _build_coarse(self) -> None:
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Coarse alignment settings")
        r = QHBoxLayout()
        self.c_dist = spin(1, 10, 2); self.c_mode = combo([("feature matching (general)", "feature"), ("template/block matching (block-face style)", "template")], "feature")
        self.c_feat = spin(0, 100000, 5000, 500); self.c_workers = spin(1, 256, 15)
        r.addWidget(QLabel("compare distance")); r.addWidget(self.c_dist)
        r.addWidget(QLabel("match mode")); r.addWidget(self.c_mode, 1)
        r.addWidget(QLabel("max keypoints")); r.addWidget(self.c_feat)
        r.addWidget(QLabel("workers")); r.addWidget(self.c_workers)
        b = QPushButton("Apply"); b.setObjectName("Primary"); r.addWidget(b)
        lay.addLayout(r)
        lay.addWidget(hint("compare distance 2 matches each section to its two neighbours on either side, which makes the "
                           "stack robust to one bad section. If thumbnail matching fails for a pair (see log warnings), you can "
                           "add manual BigWarp matches in Fiji or use structure-guided matching on the next tab."))
        b.clicked.connect(self._apply_coarse)
        tl.addWidget(f)
        f, lay = card("Steps")
        self.c_steps = StepsPanel(self.ctx, ["thumbnail.matching", "thumbnail.optimization", "thumbnail.render"], compact=True)
        self.c_steps.inspect_requested.connect(lambda s: self._inspect_coarse(s))
        lay.addWidget(self.c_steps)
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "Coarse alignment")

    # ---------------------------------------------------------------- structure-guided
    def _build_structure(self) -> None:
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        tl.addWidget(hint("Instead of anonymous texture features, align on biological structures you care about. A YOLO-seg "
                          "model finds them in every section; then (1) their centroids are matched between neighbouring "
                          "sections as coarse matches, and/or (2) fine matching is concentrated on them. Both are optional "
                          "and leave the normal FEABAS workflow untouched when switched off."))
        f, lay = card("1. Detect structures")
        self.y_model = PathPicker("file", "YOLO-seg weights (.pt), e.g. a model trained on nuclei", "Weights (*.pt)")
        lay.addWidget(form_row("Model", self.y_model))
        r = QHBoxLayout()
        self.y_classes = QLineEdit(""); self.y_classes.setPlaceholderText("class names or ids to use, comma separated (empty = all)")
        self.y_conf = dspin(0.01, 0.99, 0.25, 0.05, 2); self.y_tile = spin(256, 4096, 1024, 128); self.y_src = combo([("thumbnails", "thumb")], "thumb")
        r.addWidget(QLabel("classes")); r.addWidget(self.y_classes, 1); r.addWidget(QLabel("confidence")); r.addWidget(self.y_conf)
        r.addWidget(QLabel("tile")); r.addWidget(self.y_tile); r.addWidget(QLabel("run on")); r.addWidget(self.y_src)
        lay.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("Detect on checked sections"); b.setObjectName("Primary"); r.addWidget(b)
        self.y_info = QLabel(""); self.y_info.setObjectName("Hint"); r.addWidget(self.y_info, 1)
        lay.addLayout(r)
        b.clicked.connect(self._detect_structures)
        tl.addWidget(f)

        f, lay = card("2. Coarse matches from structures")
        r = QHBoxLayout()
        self.s_mode = combo([("augment FEABAS matches (add structure matches with weight)", "augment"),
                             ("replace: structure matches only for pairs where they succeed", "replace")], "augment")
        self.s_weight = dspin(0.1, 50, 3.0, 0.5, 1); self.s_tol = dspin(1, 50, 6, 1, 1); self.s_min = spin(3, 500, 8)
        r.addWidget(self.s_mode, 1); r.addWidget(QLabel("weight")); r.addWidget(self.s_weight)
        r.addWidget(QLabel("RANSAC tol px")); r.addWidget(self.s_tol); r.addWidget(QLabel("min inliers")); r.addWidget(self.s_min)
        lay.addLayout(r)
        r = QHBoxLayout()
        b = QPushButton("Match structures between neighbouring sections"); b.setObjectName("Primary"); r.addWidget(b)
        r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("'augment' runs after 'Match thumbnails' and adds the structure matches to FEABAS's. 'replace' runs "
                           "before it: pairs that got structure matches are skipped by FEABAS, the rest are matched normally. "
                           "A BigWarp CSV of every pair is saved to thumbnail_align/manual_matches for inspection."))
        b.clicked.connect(self._match_structures)
        tl.addWidget(f)

        f, lay = card("3. Fine alignment driven by structures")
        r = QHBoxLayout()
        self.m_mode = combo([("off: FEABAS matches everywhere in the tissue", "off"),
                             ("restrict: no fine matching outside structures (background becomes a low-weight material)", "restrict")], "off")
        self.m_dil = spin(0, 200, 10)
        r.addWidget(self.m_mode, 1); r.addWidget(QLabel("grow structures by px")); r.addWidget(self.m_dil)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.rw_in = dspin(0, 100, 1.0, 0.5, 2); self.rw_out = dspin(0, 100, 0.1, 0.05, 2)
        b1 = QPushButton("Re-weight fine matches by structures"); b2 = QPushButton("Undo re-weighting")
        r.addWidget(QLabel("after fine matching: weight inside")); r.addWidget(self.rw_in); r.addWidget(QLabel("outside")); r.addWidget(self.rw_out)
        r.addWidget(b1); r.addWidget(b2); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("'restrict' is applied when material masks are composed (Masks page) and adds a 'background_lowweight' "
                           "material to the project's material table. Re-weighting is a softer alternative applied after "
                           "'Fine matching': matches inside structures keep their weight, the others are damped; run "
                           "'Optimize stack' afterwards."))
        b1.clicked.connect(lambda: self._reweight(False)); b2.clicked.connect(lambda: self._reweight(True))
        self.m_mode.currentIndexChanged.connect(self._save_structure)
        self.m_dil.valueChanged.connect(self._save_structure)
        tl.addWidget(f)

        f, lay = card("Train a YOLO-seg model")
        self.y_data = PathPicker("file", "data.yaml of a YOLO-format dataset (or a folder with images/ and labels/)", "YOLO dataset (*.yaml *.yml);;All (*)")
        lay.addWidget(form_row("Dataset", self.y_data))
        r = QHBoxLayout()
        self.y_base = combo([("yolo11n-seg (fast)", "yolo11n-seg.pt"), ("yolo11s-seg", "yolo11s-seg.pt"), ("yolo11m-seg", "yolo11m-seg.pt"),
                             ("yolov8n-seg", "yolov8n-seg.pt"), ("yolov8m-seg", "yolov8m-seg.pt")], "yolo11n-seg.pt")
        self.y_epochs = spin(1, 1000, 100); self.y_imgsz = spin(256, 2048, 640, 32); self.y_name = QLineEdit("nuclei_run1"); self.y_name.setMaximumWidth(160)
        self.y_names = QLineEdit(""); self.y_names.setPlaceholderText("class names if the folder has no data.yaml, e.g. nucleus,mito")
        r.addWidget(QLabel("base")); r.addWidget(self.y_base); r.addWidget(QLabel("epochs")); r.addWidget(self.y_epochs)
        r.addWidget(QLabel("imgsz")); r.addWidget(self.y_imgsz); r.addWidget(QLabel("run")); r.addWidget(self.y_name)
        b = QPushButton("Train"); b.setObjectName("Primary"); r.addWidget(b); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(self.y_names)
        r = QHBoxLayout()
        self.y_models = QComboBox(); self.y_models.setMinimumWidth(260); b3 = QPushButton("Use selected model")
        r.addWidget(QLabel("trained models")); r.addWidget(self.y_models); r.addWidget(b3); r.addStretch(1)
        lay.addLayout(r)
        b.clicked.connect(self._train_yolo)
        b3.clicked.connect(lambda: self.y_model.setText(self.y_models.currentData() or ""))
        tl.addWidget(f)

        # viewer
        f, lay = card("Inspect")
        split = QSplitter()
        self.s_sections = SectionPicker(checkable=True)
        split.addWidget(self.s_sections)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        self.s_show = combo([("structures on thumbnail", "struct"), ("coarse matches to next section (red/green)", "pair")], "struct")
        r.addWidget(self.s_show, 1); fb = QPushButton("fit"); r.addWidget(fb)
        rl.addLayout(r)
        self.s_view = ImageView(); self.s_view.setMinimumHeight(420)
        rl.addWidget(self.s_view, 1)
        self.s_vinfo = QLabel(""); self.s_vinfo.setObjectName("Hint"); self.s_vinfo.setWordWrap(True)
        rl.addWidget(self.s_vinfo)
        split.addWidget(right); split.setSizes([220, 900])
        lay.addWidget(split)
        self.s_sections.current_changed.connect(lambda _n: self._show_structure())
        self.s_show.currentIndexChanged.connect(self._show_structure)
        fb.clicked.connect(self.s_view.fit)
        tl.addWidget(f)
        self.tabs.addTab(t, "Structure-guided (optional)")

    # ---------------------------------------------------------------- fine
    def _build_fine(self) -> None:
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Fine alignment settings")
        r = QHBoxLayout()
        self.f_mip = spin(0, 10, 2); self.f_mesh = spin(50, 20000, 600, 50); self.f_conf = dspin(0, 1, 0.35, 0.05, 2)
        self.f_workers = spin(1, 256, 15); self.f_optw = spin(1, 256, 5)
        r.addWidget(QLabel("working mip")); r.addWidget(self.f_mip); r.addWidget(QLabel("mesh size (mip0 px)")); r.addWidget(self.f_mesh)
        r.addWidget(QLabel("match confidence")); r.addWidget(self.f_conf)
        r.addWidget(QLabel("workers: matching")); r.addWidget(self.f_workers); r.addWidget(QLabel("optimization")); r.addWidget(self.f_optw)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.f_chunk = spin(0, 4, 0); self.f_chunksize = spin(2, 1000, 16); self.f_window = spin(4, 2000, 64); self.f_buffer = spin(1, 500, 16)
        r.addWidget(QLabel("chunked depth (0 = sliding window)")); r.addWidget(self.f_chunk); r.addWidget(QLabel("chunk size")); r.addWidget(self.f_chunksize)
        r.addWidget(QLabel("window")); r.addWidget(self.f_window); r.addWidget(QLabel("buffer")); r.addWidget(self.f_buffer)
        b = QPushButton("Apply"); b.setObjectName("Primary"); r.addWidget(b); r.addStretch(1)
        lay.addLayout(r)
        self.f_hint = QLabel(""); self.f_hint.setObjectName("Hint"); self.f_hint.setWordWrap(True)
        lay.addWidget(self.f_hint)
        b.clicked.connect(self._apply_fine)
        tl.addWidget(f)
        f, lay = card("Steps")
        self.f_steps = StepsPanel(self.ctx, ["align.meshing", "align.matching", "align.optimization"], compact=True)
        self.f_steps.inspect_requested.connect(self._inspect_fine)
        lay.addWidget(self.f_steps)
        r = QHBoxLayout()
        b = QPushButton("Make match-coverage figures (FEABAS tool)")
        r.addWidget(b); r.addStretch(1)
        lay.addLayout(r)
        b.clicked.connect(self._coverage)
        lay.addWidget(hint("Match-coverage figures overlay match locations on each thumbnail: red = matches to the previous "
                           "section, green = to the next; areas without yellow have no matches and will only follow the mesh."))
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "Fine alignment")

    # ---------------------------------------------------------------- test
    def _build_test(self) -> None:
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Test alignment settings on a subset of sections")
        lay.addWidget(hint("Creates a sandbox that links the stitched sections and copies thumbnails/masks of the chosen "
                           "sections, so coarse and fine alignment can be tried with different settings without touching the project."))
        self.t_sections = SectionPicker(checkable=True)
        self.t_sections.setMaximumHeight(220)
        lay.addWidget(self.t_sections)
        r = QHBoxLayout()
        self.t_name = QLineEdit("align_test1"); self.t_name.setMaximumWidth(220)
        b = QPushButton("Create test run"); b.setObjectName("Primary")
        r.addWidget(QLabel("Name")); r.addWidget(self.t_name); r.addWidget(b); r.addStretch(1)
        lay.addLayout(r)
        b.clicked.connect(self._create_test)
        tl.addWidget(f)
        f, lay = card("Test runs")
        r = QHBoxLayout()
        self.t_list = QComboBox(); self.t_list.setMinimumWidth(260)
        b2 = QPushButton("Delete"); b2.setObjectName("Danger"); b3 = QPushButton("Edit this test's settings…")
        r.addWidget(QLabel("Test run")); r.addWidget(self.t_list); r.addWidget(b3); r.addWidget(b2); r.addStretch(1)
        lay.addLayout(r)
        self.t_steps = StepsPanel(self.ctx, ["thumbnail.downsample", "thumbnail.matching", "thumbnail.optimization", "thumbnail.render",
                                             "align.meshing", "align.matching", "align.optimization"], compact=True)
        self.t_steps.inspect_requested.connect(lambda s: self._inspect_test(s))
        lay.addWidget(self.t_steps)
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "Test on subset")
        self.t_list.currentIndexChanged.connect(self._test_changed)
        b2.clicked.connect(self._delete_test)
        b3.clicked.connect(self._edit_test_settings)

    # ---------------------------------------------------------------- QC
    def _build_qc(self) -> None:
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        r = QHBoxLayout()
        r.addWidget(QLabel("Source")); self.q_source = QComboBox(); r.addWidget(self.q_source, 1)
        r.addWidget(QLabel("What")); self.q_what = combo([("aligned thumbnails: this section", "single"),
                                                          ("aligned thumbnails: this (red) vs next (green)", "pair"),
                                                          ("aligned thumbnails: checkerboard with next", "checker"),
                                                          ("match coverage figure", "cover")], "pair")
        r.addWidget(self.q_what, 1)
        r.addWidget(QLabel("Section")); self.q_sec = QComboBox(); self.q_sec.setMinimumWidth(120); r.addWidget(self.q_sec)
        pb = QPushButton("◀"); nb = QPushButton("▶"); pb.setFixedWidth(32); nb.setFixedWidth(32); r.addWidget(pb); r.addWidget(nb)
        fb = QPushButton("fit"); r.addWidget(fb)
        tl.addLayout(r)
        self.q_view = ImageView(); self.q_view.setMinimumHeight(520)
        tl.addWidget(self.q_view, 1)
        self.q_info = QLabel(""); self.q_info.setObjectName("Hint"); self.q_info.setWordWrap(True)
        tl.addWidget(self.q_info)
        tl.addWidget(hint("Red/green overlay: grey means the two sections agree, coloured fringes are residual misalignment. "
                          "Some colour is expected from real biological change between sections; systematic shifts or "
                          "distortions in one region point at missing matches (check the coverage figure) or a mask problem."))
        self.tabs.addTab(t, "Quality check")
        self._qc_tab = t
        # the source list must pick up test runs created while the page is open
        self.tabs.currentChanged.connect(
            lambda i: self._qc_sources() if self.tabs.widget(i) is self._qc_tab else None)
        self.q_source.currentIndexChanged.connect(self._qc_sections)
        self.q_what.currentIndexChanged.connect(self._qc_show)
        self.q_sec.currentIndexChanged.connect(self._qc_show)
        pb.clicked.connect(lambda: self.q_sec.setCurrentIndex(max(0, self.q_sec.currentIndex() - 1)))
        nb.clicked.connect(lambda: self.q_sec.setCurrentIndex(min(self.q_sec.count() - 1, self.q_sec.currentIndex() + 1)))
        fb.clicked.connect(self.q_view.fit)

    # ================================================================ state
    def on_project_changed(self, project) -> None:
        if project is None:
            self.store = None
            self.editor_t.set_doc(None); self.editor_a.set_doc(None)
            return
        self.store = MaskStore(project)
        cs = self.ctx.configs
        self.editor_t.set_doc(cs["thumbnail"]); self.editor_a.set_doc(cs["alignment"])
        self._load_settings()
        st = project.state.structure
        self.y_model.setText(st.get("model", "")); self.y_classes.setText(st.get("classes", ""))
        self.y_conf.setValue(float(st.get("conf", 0.25))); self.y_tile.setValue(int(st.get("tile", 1024)))
        self.s_mode.setCurrentIndex(max(0, self.s_mode.findData(st.get("coarse_mode", "augment"))))
        self.s_weight.setValue(float(st.get("weight", 3.0)))
        self.m_mode.blockSignals(True); self.m_mode.setCurrentIndex(max(0, self.m_mode.findData(st.get("material_mode", "off")))); self.m_mode.blockSignals(False)
        self.m_dil.blockSignals(True); self.m_dil.setValue(int(st.get("material_dilate", 10))); self.m_dil.blockSignals(False)
        self.rw_in.setValue(float(st.get("rw_in", 1.0))); self.rw_out.setValue(float(st.get("rw_out", 0.1)))
        names = project.section_names()
        self.s_sections.set_sections(names); self.t_sections.set_sections(names)
        self._refresh_yolo_models()
        self._refresh_tests()
        self._qc_sources()
        self._refresh_struct_sources()

    def on_state_changed(self) -> None:
        scan = self.ctx.scan()
        self.c_steps.refresh(scan); self.f_steps.refresh(scan)
        self._refresh_test_scan()
        self._refresh_struct_sources()
        if self.tabs.currentIndex() == 4:
            self._qc_sources()

    def on_running_changed(self, running: bool) -> None:
        self.on_state_changed()

    def _load_settings(self) -> None:
        cs = self.ctx.configs
        g = lambda k, d: cs.get("thumbnail", k, d)
        self.c_dist.setValue(int(g("alignment.compare_distance", 2)))
        self.c_mode.setCurrentIndex(max(0, self.c_mode.findData(g("alignment.match_mode", "feature"))))
        self.c_feat.setValue(int(g("alignment.feature_matching.detect_settings.num_features", 5000)))
        self.c_workers.setValue(int(g("alignment.num_workers", 15)))
        a = lambda k, d: cs.get("alignment", k, d)
        self.f_mip.setValue(int(a("matching.working_mip_level", 2)))
        self.f_mesh.setValue(int(a("meshing.mesh_size", 600)))
        self.f_conf.setValue(float(a("matching.matcher_config.conf_thresh", 0.35)))
        self.f_workers.setValue(int(a("matching.matcher_config.num_workers", 15)))
        self.f_optw.setValue(int(a("optimization.num_workers", 5)))
        self.f_chunk.setValue(int(a("optimization.chunk_settings.chunked_to_depth", 0)))
        self.f_chunksize.setValue(int(a("optimization.chunk_settings.default_chunk_size", 16)))
        self.f_window.setValue(int(a("optimization.slide_window.window_size", 64)))
        self.f_buffer.setValue(int(a("optimization.slide_window.buffer_size", 16)))
        v = self.project.state.volume
        wm = suggest_working_mip(v.pixel_size_nm, v.section_thickness_nm)
        self.f_hint.setText(f"Suggested working mip for {v.pixel_size_nm:g} nm pixels and {v.section_thickness_nm:g} nm sections: "
                            f"mip{wm} ({v.pixel_size_nm * 2 ** wm:g} nm/px). Mesh size 600 px at mip0 is a good start; finer "
                            f"meshes follow local distortion but may fit real z-changes as if they were deformation.")

    def _apply_coarse(self) -> None:
        cs = self.ctx.configs
        cs.set("thumbnail", "alignment.compare_distance", self.c_dist.value())
        cs.set("thumbnail", "alignment.match_mode", self.c_mode.currentData())
        cs.set("thumbnail", "alignment.feature_matching.detect_settings.num_features", self.c_feat.value())
        cs.set("thumbnail", "alignment.num_workers", self.c_workers.value())
        cs.save("thumbnail"); self.editor_t.rebuild()
        self.info("coarse alignment settings saved"); self.ctx.state_changed.emit()

    def _apply_fine(self) -> None:
        cs = self.ctx.configs
        cs.set("alignment", "matching.working_mip_level", self.f_mip.value())
        cs.set("alignment", "meshing.mesh_size", self.f_mesh.value())
        cs.set("alignment", "matching.matcher_config.conf_thresh", float(self.f_conf.value()))
        cs.set("alignment", "matching.matcher_config.num_workers", self.f_workers.value())
        cs.set("alignment", "optimization.num_workers", self.f_optw.value())
        cs.set("alignment", "optimization.chunk_settings.chunked_to_depth", self.f_chunk.value())
        cs.set("alignment", "optimization.chunk_settings.default_chunk_size", self.f_chunksize.value())
        cs.set("alignment", "optimization.slide_window.window_size", self.f_window.value())
        cs.set("alignment", "optimization.slide_window.buffer_size", self.f_buffer.value())
        cs.save("alignment"); self.editor_a.rebuild()
        self.info("fine alignment settings saved"); self.ctx.state_changed.emit()

    def _editor_changed(self, kind: str) -> None:
        if self.ctx.configs:
            self.ctx.configs.save(kind)
            self._load_settings()
            self.ctx.state_changed.emit()

    # ================================================================ structures
    def _save_structure(self) -> None:
        if not self.project:
            return
        st = self.project.state.structure
        st.update({"model": self.y_model.text(), "classes": self.y_classes.text(), "conf": self.y_conf.value(), "tile": self.y_tile.value(),
                   "coarse_mode": self.s_mode.currentData(), "weight": self.s_weight.value(),
                   "material_mode": self.m_mode.currentData(), "material_dilate": self.m_dil.value(),
                   "rw_in": self.rw_in.value(), "rw_out": self.rw_out.value()})
        self.project.save()
        self._ensure_material_table()

    def _ensure_material_table(self) -> None:
        """Add/remove the low-weight background material in the project's material table."""
        from ...core.masks import STRUCTURE_MATERIAL_YAML
        import yaml
        p = self.project.configs_dir / "material_table.yaml"
        data = {}
        if p.is_file():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        want = self.m_mode.currentData() == "restrict"
        has = "background_lowweight" in data
        if want and not has:
            data.update(yaml.safe_load(STRUCTURE_MATERIAL_YAML))
            p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            self.info("material 'background_lowweight' (label 150) added to configs/material_table.yaml; re-compose masks")
        elif not want and has:
            del data["background_lowweight"]
            if data:
                p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
            else:
                p.unlink()
            self.info("material 'background_lowweight' removed; re-compose masks")
        self.ctx.reload_configs()

    def _refresh_struct_sources(self) -> None:
        cur = self.y_src.currentData()
        self.y_src.blockSignals(True); self.y_src.clear(); self.y_src.addItem("thumbnails", "thumb")
        if self.project:
            base = self.project.root / "stitched_sections"
            for d in sorted(base.glob("mip*")) if base.is_dir() else []:
                if d.name[3:].isdigit():
                    self.y_src.addItem(f"stitched sections {d.name}", int(d.name[3:]))
        self.y_src.blockSignals(False)
        self.y_src.setCurrentIndex(max(0, self.y_src.findData(cur)))
        if self.store:
            sj = self.store.structures_dir / "structures.json"
            if sj.is_file():
                try:
                    d = json.loads(sj.read_text(encoding="utf-8"))
                    n = sum(v.get("n", 0) for v in d.values())
                    self.y_info.setText(f"{len(d)} sections with detections, {n} objects in total")
                except ValueError:
                    pass

    def _detect_structures(self) -> None:
        if not self.store:
            return
        m = self.y_model.path()
        if not m or not m.is_file():
            QMessageBox.information(self, "Model", "Choose YOLO-seg weights first (or train a model below).")
            return
        if not self.ctx.dl_python():
            QMessageBox.information(self, "Environment", "Configure the deep-learning Python on the Setup page.")
            return
        self._save_structure()
        src = self.y_src.currentData()
        items = []
        for s in self.s_sections.checked():
            if src == "thumb":
                p = self.store.thumbnail_path(s)
                if p:
                    items.append({"section": s, "image": str(p), "mip": self.store.thumbnail_mip()})
            else:
                items.append({"section": s, "tiled_base": str(self.project.root / "stitched_sections"), "mip": int(src)})
        if not items:
            QMessageBox.information(self, "Sections", "No thumbnails for the checked sections (run the Masks → Thumbnails step).")
            return
        classes = [c.strip() for c in self.y_classes.text().split(",") if c.strip()]
        payload = {"model": str(m), "items": items, "out_dir": str(self.store.structures_dir), "classes": classes,
                   "conf": self.y_conf.value(), "tile": self.y_tile.value(), "overlap": max(64, self.y_tile.value() // 8)}
        self.submit(self.ctx.worker_spec("yolo_detect", payload, f"Structure detection ({len(items)} sections)",
                                         count_outputs=lambda: len(list(self.store.structures_dir.glob("*.json"))) - 1, expected=len(items)))

    def _pairs(self) -> list[list[str]]:
        names = self.project.section_names()
        cd = self.ctx.configs.get("thumbnail", "alignment.compare_distance", 1)
        steps = list(cd) if isinstance(cd, (list, tuple)) else list(range(1, int(cd) + 1))
        pairs = []
        for k in steps:
            for i in range(len(names) - k):
                pairs.append([names[i], names[i + k]])
        return pairs

    def _match_structures(self) -> None:
        if not self.store:
            return
        self._save_structure()
        v = self.project.state.volume
        tres = v.pixel_size_nm * 2 ** self.store.thumbnail_mip()
        payload = {"structures_dir": str(self.store.structures_dir), "thumbnail_dir": str(self.store.thumb_dir),
                   "match_dir": str(self.project.root / "thumbnail_align" / "matches"),
                   "manual_dir": str(self.project.root / "thumbnail_align" / "manual_matches"),
                   "delimiter": self.ctx.configs.get("thumbnail", "alignment.match_name_delimiter", "__to__"),
                   "thumbnail_resolution": tres, "mode": self.s_mode.currentData(), "weight": self.s_weight.value(),
                   "ransac_tol": self.s_tol.value(), "min_inliers": self.s_min.value(), "pairs": self._pairs()}
        py = self.ctx.dl_python() or __import__("sys").executable
        self.submit(self.ctx.worker_spec("structure_match", payload, "Structure matching (coarse)", python=py))

    def _reweight(self, undo: bool) -> None:
        if not self.store:
            return
        self._save_structure()
        sj = self.store.structures_dir / "structures.json"
        mip = None
        if sj.is_file():
            try:
                d = json.loads(sj.read_text(encoding="utf-8"))
                mip = next(iter(d.values())).get("mip") if d else None
            except (ValueError, StopIteration):
                mip = None
        if mip is None and not undo:
            QMessageBox.information(self, "Structures", "Detect structures first.")
            return
        v = self.project.state.volume
        payload = {"match_dir": str(self.project.root / "align" / "matches"), "structures_dir": str(self.store.structures_dir),
                   "structure_resolution": v.pixel_size_nm * 2 ** int(mip or 0), "inside_factor": self.rw_in.value(),
                   "outside_factor": self.rw_out.value(), "undo": undo,
                   "delimiter": self.ctx.configs.get("thumbnail", "alignment.match_name_delimiter", "__to__")}
        py = self.ctx.dl_python() or __import__("sys").executable
        self.submit(self.ctx.worker_spec("reweight_matches", payload, "Undo match re-weighting" if undo else "Re-weight fine matches", python=py))

    def _train_yolo(self) -> None:
        if not self.require_project():
            return
        d = self.y_data.path()
        if not d or not d.exists():
            QMessageBox.information(self, "Dataset", "Choose a YOLO dataset (data.yaml or folder).")
            return
        name = self.y_name.text().strip() or "yolo_run"
        payload = {"dataset": str(d), "base": self.y_base.currentData(), "epochs": self.y_epochs.value(), "imgsz": self.y_imgsz.value(),
                   "project_dir": str(self.project.models_dir / "yolo"), "name": name,
                   "names": [c.strip() for c in self.y_names.text().split(",") if c.strip()]}
        self.submit(self.ctx.worker_spec("yolo_train", payload, f"YOLO-seg training '{name}'"))

    def _refresh_yolo_models(self) -> None:
        self.y_models.clear()
        if not self.project:
            return
        d = self.project.models_dir / "yolo"
        for run in sorted(d.iterdir()) if d.is_dir() else []:
            best = run / "weights" / "best.pt"
            if best.is_file():
                self.y_models.addItem(run.name, str(best))

    def _show_structure(self) -> None:
        sec = self.s_sections.current()
        if not self.store or not sec:
            self.s_view.clear()
            return
        img = self.store.thumbnail(sec)
        if img is None:
            self.s_view.clear()
            self.s_vinfo.setText("no thumbnail yet")
            return
        what = self.s_show.currentData()
        self.s_view.clear_overlays(); self.s_view.clear_points()
        if what == "struct":
            self.s_view.set_image(to_uint8(img))
            s = self.store.structure_mask(sec)
            n = 0
            if s is not None:
                from ...core.masks import resize_mask
                s = resize_mask(s.astype(np.uint8), img.shape[:2]) > 0
                rgba = np.zeros((*img.shape[:2], 4), np.uint8); rgba[s] = (60, 220, 120, 120)
                self.s_view.set_overlay("s", rgba)
                dj = self.store.structures_dir / f"{sec}.json"
                if dj.is_file():
                    d = json.loads(dj.read_text(encoding="utf-8"))
                    sc = img.shape[1] / d["shape"][1]
                    pts = np.array([[x["cx"] * sc, x["cy"] * sc] for x in d["detections"]]) if d["detections"] else None
                    n = len(d["detections"])
                    self.s_view.set_points(pts, color=theme.OK, radius=max(2, img.shape[1] / 600))
            self.s_vinfo.setText(f"{sec}: {n} structures" if s is not None else f"{sec}: no structure detections yet")
        else:
            names = self.project.section_names()
            i = names.index(sec) if sec in names else -1
            if i < 0 or i + 1 >= len(names):
                self.s_vinfo.setText("no next section")
                return
            nxt = names[i + 1]
            img1 = self.store.thumbnail(nxt)
            self.s_view.set_image(compose_two_color(img, img1))
            delim = self.ctx.configs.get("thumbnail", "alignment.match_name_delimiter", "__to__")
            h5 = self.project.root / "thumbnail_align" / "matches" / f"{sec}{delim}{nxt}.h5"
            if h5.is_file():
                xy0, xy1, w, res = _read_match_h5(h5)
                v = self.project.state.volume
                sc = res / (v.pixel_size_nm * 2 ** self.store.thumbnail_mip())
                self.s_view.set_points(xy0 * sc, color=theme.WARN, radius=max(2, img.shape[1] / 500), lines_to=xy1 * sc)
                self.s_vinfo.setText(f"{sec} → {nxt}: {len(xy0)} coarse matches (lines: displacement to the green section), mean weight {w.mean():.2f}")
            else:
                self.s_vinfo.setText(f"{sec} → {nxt}: no coarse matches yet")

    # ================================================================ tests
    def _create_test(self) -> None:
        if not self.require_project():
            return
        secs = self.t_sections.checked()
        if len(secs) < 2:
            QMessageBox.information(self, "Sections", "Check at least two sections.")
            return
        tr = create_align_test(self.project, self.t_name.text(), secs)
        self.info(f"alignment test run created: {tr.root}")
        self._refresh_tests(select=tr.name)
        self._qc_sources()

    def _refresh_tests(self, select: str | None = None) -> None:
        self.t_list.blockSignals(True); self.t_list.clear()
        if self.project:
            for tr in list_test_runs(self.project, "align"):
                self.t_list.addItem(f"{tr.name}  ({len(tr.sections)} sections)", str(tr.root))
        self.t_list.blockSignals(False)
        if select:
            for i in range(self.t_list.count()):
                if self.t_list.itemText(i).startswith(select + " "):
                    self.t_list.setCurrentIndex(i)
        self._test_changed()

    def _current_test(self) -> TestRun | None:
        d = self.t_list.currentData()
        return TestRun.load(Path(d)) if d else None

    def _test_changed(self) -> None:
        tr = self._current_test()
        self.t_steps.root_override = tr.root if tr else None
        self.t_steps.tag = f"test {tr.name}" if tr else ""
        self._refresh_test_scan()

    def _refresh_test_scan(self) -> None:
        tr = self._current_test()
        if tr is None:
            self.t_steps.refresh(None)
            return
        from ...core.configs import ConfigStore
        self.t_steps.refresh(PipelineScan(tr.root, len(tr.sections), ConfigStore(tr.root / "configs")))

    def _delete_test(self) -> None:
        tr = self._current_test()
        if tr and self.confirm("Delete test run", f"Delete {tr.root}?"):
            delete_test_run(tr)
            self._refresh_tests()
            self._qc_sources()

    def _edit_test_settings(self) -> None:
        tr = self._current_test()
        if not tr:
            return
        from ...core.configs import ConfigStore
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        cs = ConfigStore(tr.root / "configs")
        dlg = QDialog(self); dlg.setWindowTitle(f"Settings of test run {tr.name}"); dlg.resize(1000, 750)
        lay = QVBoxLayout(dlg)
        tabs = QTabWidget()
        e1 = ConfigEditor(); e1.set_doc(cs["thumbnail"]); e2 = ConfigEditor(); e2.set_doc(cs["alignment"])
        tabs.addTab(e1, "thumbnail"); tabs.addTab(e2, "alignment")
        lay.addWidget(tabs, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        copy_btn = bb.addButton("Save and copy to project", QDialogButtonBox.ActionRole)
        lay.addWidget(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); copy_btn.clicked.connect(lambda: dlg.done(2))
        rc = dlg.exec()
        if rc:
            cs.save()
            self.info(f"test run settings saved ({tr.name})")
        if rc == 2 and self.ctx.configs:
            for k in ("thumbnail", "alignment"):
                self.ctx.configs[k].overrides = dict(cs[k].overrides)
            self.ctx.configs.save()
            self._load_settings(); self.editor_t.rebuild(); self.editor_a.rebuild()
            self.info("settings copied to the project"); self.ctx.state_changed.emit()

    # ================================================================ QC
    def _qc_sources(self) -> None:
        cur = self.q_source.currentData()
        self.q_source.blockSignals(True); self.q_source.clear()
        if self.project:
            self.q_source.addItem("project", str(self.project.root))
            for tr in list_test_runs(self.project, "align"):
                self.q_source.addItem(f"test run {tr.name}", str(tr.root))
        self.q_source.blockSignals(False)
        if cur:
            i = self.q_source.findData(cur)
            if i >= 0:
                self.q_source.setCurrentIndex(i)
        self._qc_sections()

    def _qc_root(self) -> Path | None:
        d = self.q_source.currentData()
        return Path(d) if d else None

    def _aligned_thumb_dir(self, root: Path) -> Path | None:
        ds = sorted((root / "thumbnail_align").glob("aligned_thumbnails_*"))
        return ds[-1] if ds else None

    def _qc_sections(self) -> None:
        root = self._qc_root()
        cur = self.q_sec.currentText()
        self.q_sec.blockSignals(True); self.q_sec.clear()
        if root:
            d = self._aligned_thumb_dir(root)
            names = sorted(p.stem for p in d.glob("*.*")) if d else []
            if not names:
                cov = root / "align" / "matches" / "match_cover"
                names = sorted(p.stem for p in cov.glob("*.*")) if cov.is_dir() else []
            self.q_sec.addItems(names)
        self.q_sec.blockSignals(False)
        if cur:
            i = self.q_sec.findText(cur)
            if i >= 0:
                self.q_sec.setCurrentIndex(i)
        self._qc_show()

    def _qc_show(self) -> None:
        root = self._qc_root(); sec = self.q_sec.currentText(); what = self.q_what.currentData()
        if not root or not sec:
            self.q_view.clear()
            return
        try:
            if what == "cover":
                cov = root / "align" / "matches" / "match_cover"
                hits = list(cov.glob(f"{sec}.*")) if cov.is_dir() else []
                if not hits:
                    self.q_info.setText("no coverage figure yet (Fine alignment → Make match-coverage figures)")
                    self.q_view.clear()
                    return
                import cv2
                img = cv2.imread(str(hits[0]), cv2.IMREAD_COLOR)
                self.q_view.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.q_info.setText(f"{hits[0].name}: red = matches to previous, green = to next")
                return
            d = self._aligned_thumb_dir(root)
            if not d:
                self.q_info.setText("no aligned thumbnails yet (Coarse alignment → Render coarse stack)")
                self.q_view.clear()
                return
            hits = list(d.glob(f"{sec}.*"))
            a = imread(hits[0])
            if what == "single":
                self.q_view.set_image(to_uint8(a)); self.q_info.setText(f"{sec}: {a.shape[1]}×{a.shape[0]} px ({d.name})")
                return
            names = [self.q_sec.itemText(i) for i in range(self.q_sec.count())]
            i = names.index(sec)
            if i + 1 >= len(names):
                self.q_view.set_image(to_uint8(a)); self.q_info.setText("last section")
                return
            b = imread(list(d.glob(f"{names[i + 1]}.*"))[0])
            if what == "pair":
                self.q_view.set_image(compose_two_color(a, b))
            else:
                from ...core.images import checkerboard
                self.q_view.set_image(checkerboard(a, b, block=max(32, a.shape[1] // 24)))
            self.q_info.setText(f"{sec} (red) vs {names[i + 1]} (green), {d.name}")
        except Exception as e:  # noqa: BLE001
            self.error(f"cannot show: {e}", dialog=False)

    def _inspect_coarse(self, step) -> None:
        self.tabs.setCurrentIndex(4)
        self._qc_sources()
        if step.key == "thumbnail.matching":
            self.tabs.setCurrentIndex(1)
            self.s_show.setCurrentIndex(1)
            self._show_structure()

    def _inspect_fine(self, step) -> None:
        self.tabs.setCurrentIndex(4)
        self.q_what.setCurrentIndex(self.q_what.findData("cover") if step.key == "align.matching" else 1)
        self._qc_sources()

    def _inspect_test(self, step) -> None:
        self.tabs.setCurrentIndex(4)
        self._qc_sources()
        tr = self._current_test()
        if tr:
            i = self.q_source.findData(str(tr.root))
            if i >= 0:
                self.q_source.setCurrentIndex(i)

    def _coverage(self) -> None:
        if not self.require_project():
            return
        try:
            spec = self.ctx.feabas_tool_spec("visualize_align_match_coverage.py", ["--mode", "fine"], "match coverage figures")
        except RuntimeError as e:
            self.error(str(e))
            return
        self.submit(spec)

    def _job_finished(self, res) -> None:
        n = res.spec.name
        if n.startswith("Structure detection"):
            self._refresh_struct_sources(); self._show_structure()
        elif n.startswith("YOLO-seg training"):
            self._refresh_yolo_models()
        elif n.startswith("Structure matching"):
            rep = self.project.root / "thumbnail_align" / "matches" / "structure_match_report.json" if self.project else None
            if rep and rep.is_file():
                d = json.loads(rep.read_text(encoding="utf-8"))
                ok = sum(1 for v in d.values() if v.get("n_pairs"))
                self.info(f"structure matching: {ok}/{len(d)} pairs matched; details in {rep}")
        elif n == "match coverage figures":
            self.tabs.setCurrentIndex(4); self.q_what.setCurrentIndex(self.q_what.findData("cover")); self._qc_sources()
