"""Window 3: stitching (xy montage) with settings, subset test runs and quality control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
                               QTabWidget, QVBoxLayout, QWidget, QCheckBox)

from ...core.steps import STEPS_BY_KEY, PipelineScan
from ...core.testruns import create_montage_test, list_test_runs, delete_test_run, parse_stitch_coord, TestRun
from ...core.images import TiledSectionSource, TensorStoreSource
from ..widgets import ConfigEditor, ImageView, SectionPicker, TileGridWidget, card, hint, form_row, spin, dspin, combo
from ..widgets.steps_panel import StepsPanel
from ..widgets.imageview import RectSpec
from .base import Page
from .. import theme


class QuickStitchSettings(QWidget):
    """The handful of stitching settings people actually change."""

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        self.w_match = spin(1, 256, 15); self.w_opt = spin(1, 256, 3); self.w_render = spin(1, 256, 15)
        r.addWidget(QLabel("workers: matching")); r.addWidget(self.w_match)
        r.addWidget(QLabel("optimization")); r.addWidget(self.w_opt)
        r.addWidget(QLabel("rendering")); r.addWidget(self.w_render); r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.driver = combo([("PNG tiles (simple, VAST-like)", "image"), ("Neuroglancer precomputed volume (TensorStore)", "neuroglancer_precomputed")], "image")
        self.clahe = QCheckBox("CLAHE"); self.inverse = QCheckBox("invert grey")
        self.conf = dspin(0, 1, 0.1, 0.05, 2); self.margin = spin(0, 20000, 1000, 100)
        r.addWidget(QLabel("render as")); r.addWidget(self.driver, 1); r.addWidget(self.clahe); r.addWidget(self.inverse)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.blend = combo([("PYRAMID – detail from one tile, brightness blended (default)", "PYRAMID"),
                            ("NEAREST – every pixel from one tile, hard seam", "NEAREST"),
                            ("LINEAR – weighted average across the overlap", "LINEAR"),
                            ("MAX – brightest of the overlapping tiles", "MAX"),
                            ("MIN – darkest of the overlapping tiles", "MIN"),
                            ("NONE – whichever tile is drawn last", "NONE")], "PYRAMID")
        r.addWidget(QLabel("overlap blending")); r.addWidget(self.blend, 1)
        lay.addLayout(r)
        self.blend.setToolTip(
            "How pixels covered by more than one tile are combined. Each tile carries a weight that grows with the "
            "distance from its own edge.\n"
            "PYRAMID: fine detail from the tile with the largest weight, low frequencies blended linearly – no ghosting, "
            "no visible brightness step.\n"
            "NEAREST: the whole pixel comes from the tile with the largest weight – nothing is averaged, so a brightness "
            "difference between the two exposures shows as a hard edge.\n"
            "LINEAR: weighted average of all covering tiles.\n"
            "MAX / MIN: brightest / darkest value of the covering tiles.\n"
            "NONE: no selection at all – the last tile processed overwrites the others.")
        r = QHBoxLayout()
        r.addWidget(QLabel("match confidence threshold")); r.addWidget(self.conf)
        r.addWidget(QLabel("search margin (px)")); r.addWidget(self.margin)
        self.cache = spin(1, 10000, 150)
        r.addWidget(QLabel("image cache (tiles)")); r.addWidget(self.cache); r.addStretch(1)
        lay.addLayout(r)
        self.margin.setToolTip("Extra width searched around the nominal overlap to absorb stage-position errors.")
        self.conf.setToolTip("Cross-correlations below this confidence are rejected (0.1 default; raise for noisy false matches).")
        self.cache.setToolTip("Tiles kept in RAM per worker. Lower it if matching runs out of memory.")

    def load(self) -> None:
        cs = self.ctx.configs
        if not cs:
            return
        g = lambda k, d: cs.get("stitching", k, d)
        self.w_match.setValue(int(g("matching.num_workers", 15)))
        self.w_opt.setValue(int(g("optimization.num_workers", 3)))
        self.w_render.setValue(int(g("rendering.num_workers", 15)))
        self.driver.setCurrentIndex(max(0, self.driver.findData(g("rendering.driver", "image"))))
        self.clahe.setChecked(bool(g("rendering.loader_settings.apply_CLAHE", True)))
        self.inverse.setChecked(bool(g("rendering.loader_settings.inverse", True)))
        blend = g("rendering.render_settings.blend", "PYRAMID")
        i = self.blend.findData(str(blend).upper() if blend is not None else "NONE")
        self.blend.setCurrentIndex(max(0, i))
        self.conf.setValue(float(g("matching.matcher_config.conf_thresh", 0.1)))
        self.margin.setValue(int(g("matching.margin", 1000)))
        self.cache.setValue(int(g("matching.loader_config.cache_size", 150)))

    def apply(self) -> None:
        cs = self.ctx.configs
        if not cs:
            return
        s = lambda k, v: cs.set("stitching", k, v)
        s("matching.num_workers", self.w_match.value()); s("optimization.num_workers", self.w_opt.value())
        s("rendering.num_workers", self.w_render.value()); s("rendering.driver", self.driver.currentData())
        s("rendering.loader_settings.apply_CLAHE", self.clahe.isChecked()); s("rendering.loader_settings.inverse", self.inverse.isChecked())
        s("rendering.render_settings.blend", self.blend.currentData())
        s("matching.matcher_config.conf_thresh", float(self.conf.value())); s("matching.margin", self.margin.value())
        s("matching.loader_config.cache_size", self.cache.value())
        cs.save("stitching")


class StitchPage(Page):
    title = "Stitching"
    subtitle = ("Match tile overlaps, relax each section into a montage and render it. Try settings on a subset "
                "first (test runs live in their own sandbox and never touch the real results).")
    key = "stitch"

    def build(self) -> None:
        self.tabs = QTabWidget()
        self.body.addWidget(self.tabs)

        # --- run tab ------------------------------------------------------
        run_tab = QWidget(); rl = QVBoxLayout(run_tab); rl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Settings")
        self.quick = QuickStitchSettings(self.ctx)
        lay.addWidget(self.quick)
        r = QHBoxLayout()
        apply_btn = QPushButton("Apply settings"); apply_btn.setObjectName("Primary")
        r.addWidget(apply_btn); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("Changing settings after a step ran marks it stale; clear and re-run the affected steps. "
                           "All other settings are in the 'All settings' tab. PNG tiles are the output every later step and "
                           "viewer has been validated with; the precomputed (TensorStore) driver is faster for very large "
                           "volumes, and on Windows the workbench patches FEABAS's file-URL handling for it at run time."))
        apply_btn.clicked.connect(self._apply_quick)
        rl.addWidget(f)
        f, lay = card("Steps")
        self.steps = StepsPanel(self.ctx, ["stitch.matching", "stitch.optimization", "stitch.rendering"])
        self.steps.inspect_requested.connect(self._inspect)
        lay.addWidget(self.steps)
        rl.addWidget(f)
        rl.addStretch(1)
        self.tabs.addTab(run_tab, "Run")

        # --- test tab -----------------------------------------------------
        test_tab = QWidget(); tl = QVBoxLayout(test_tab); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Create a test run on a subset")
        lay.addWidget(hint("Pick sections and optionally drag-select a tile region below. A sandbox working directory with "
                           "a copy of the current settings is created under <project>/tests; run the steps there and inspect "
                           "the montage. When happy, apply the settings to the project and run for real."))
        split = QSplitter()
        self.test_sections = SectionPicker(checkable=True)
        split.addWidget(self.test_sections)
        right = QWidget(); rl2 = QVBoxLayout(right); rl2.setContentsMargins(0, 0, 0, 0)
        r = QHBoxLayout()
        r.addWidget(QLabel("Tile subset of section")); self.test_sec = QComboBox(); r.addWidget(self.test_sec, 1)
        rl2.addLayout(r)
        self.test_grid = TileGridWidget(selectable=True)
        rl2.addWidget(self.test_grid, 1)
        split.addWidget(right)
        split.setSizes([260, 700])
        lay.addWidget(split)
        r = QHBoxLayout()
        self.test_name = QLineEdit("montage_test1"); self.test_name.setMaximumWidth(220)
        create_btn = QPushButton("Create test run"); create_btn.setObjectName("Primary")
        r.addWidget(QLabel("Name")); r.addWidget(self.test_name); r.addWidget(create_btn); r.addStretch(1)
        lay.addLayout(r)
        tl.addWidget(f)
        f, lay = card("Test runs")
        r = QHBoxLayout()
        self.test_list = QComboBox(); self.test_list.setMinimumWidth(260)
        del_btn = QPushButton("Delete"); del_btn.setObjectName("Danger")
        self.test_edit_btn = QPushButton("Edit this test's settings…")
        r.addWidget(QLabel("Test run")); r.addWidget(self.test_list); r.addWidget(self.test_edit_btn); r.addWidget(del_btn); r.addStretch(1)
        lay.addLayout(r)
        self.test_steps = StepsPanel(self.ctx, ["stitch.matching", "stitch.optimization", "stitch.rendering",
                                                "thumbnail.downsample"], compact=True,
                                     labels={"thumbnail.downsample": "Mipmaps of the montage (to zoom out in Quality check)"})
        self.test_steps.inspect_requested.connect(lambda step: self._inspect(step, test=True))
        lay.addWidget(self.test_steps)
        lay.addWidget(hint("Rendering writes mip0 only. The mipmap step downsamples it up to "
                           "thumbnail_mip_level − 1 (and at least the fine-alignment working mip), which is what the "
                           "viewer needs to show a whole section; it also writes the sandbox's thumbnails."))
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(test_tab, "Test on subset")
        self.test_sec.currentIndexChanged.connect(self._test_show_section)
        create_btn.clicked.connect(self._create_test)
        del_btn.clicked.connect(self._delete_test)
        self.test_list.currentIndexChanged.connect(self._test_changed)
        self.test_edit_btn.clicked.connect(self._edit_test_settings)

        # --- QC tab -------------------------------------------------------
        qc_tab = QWidget(); ql = QVBoxLayout(qc_tab); ql.setContentsMargins(6, 6, 6, 6)
        r = QHBoxLayout()
        r.addWidget(QLabel("Source")); self.qc_source = QComboBox(); r.addWidget(self.qc_source, 1)
        r.addWidget(QLabel("Section")); self.qc_sec = QComboBox(); self.qc_sec.setMinimumWidth(120); r.addWidget(self.qc_sec)
        prev_b = QPushButton("◀"); next_b = QPushButton("▶"); prev_b.setFixedWidth(32); next_b.setFixedWidth(32)
        r.addWidget(prev_b); r.addWidget(next_b)
        self.qc_outlines = QCheckBox("tile outlines (nominal)"); r.addWidget(self.qc_outlines)
        fit_b = QPushButton("fit"); r.addWidget(fit_b)
        self.fiji_btn = QPushButton("Open section in Fiji"); r.addWidget(self.fiji_btn)
        ql.addLayout(r)
        self.qc_view = ImageView()
        self.qc_view.setMinimumHeight(520)
        ql.addWidget(self.qc_view, 1)
        self.qc_info = QLabel("")
        self.qc_info.setObjectName("Hint")
        ql.addWidget(self.qc_info)
        self.qc_note = QLabel("")
        self.qc_note.setWordWrap(True)
        self.qc_note.setVisible(False)
        self.qc_note.setStyleSheet(f"color: {theme.WARN};")
        ql.addWidget(self.qc_note)
        ql.addWidget(hint("Zoom into tile borders: seams should be invisible. Visible duplication or blur along a seam means "
                          "the match there was rejected or wrong; check the log for that section and consider a larger "
                          "search margin or lower confidence threshold. Rendering writes mip0 only: to see a whole section "
                          "at once, run the mipmap step (test runs) or 'Make thumbnails' on the Masks page (project)."))
        self.tabs.addTab(qc_tab, "Quality check")
        self._qc_tab = qc_tab
        # the source list must pick up test runs created while the page is open
        self.tabs.currentChanged.connect(
            lambda i: self._qc_refresh_sources() if self.tabs.widget(i) is self._qc_tab else None)
        self.qc_source.currentIndexChanged.connect(self._qc_refresh_sections)
        self.qc_sec.currentIndexChanged.connect(self._qc_show)
        prev_b.clicked.connect(lambda: self.qc_sec.setCurrentIndex(max(0, self.qc_sec.currentIndex() - 1)))
        next_b.clicked.connect(lambda: self.qc_sec.setCurrentIndex(min(self.qc_sec.count() - 1, self.qc_sec.currentIndex() + 1)))
        self.qc_outlines.toggled.connect(self._qc_show)
        fit_b.clicked.connect(self.qc_view.fit)
        self.qc_view.note.connect(self._qc_note)
        self.fiji_btn.clicked.connect(self._open_fiji)
        self.qc_view.hovered.connect(lambda x, y: self.qc_info.setText(f"x={x:.0f} y={y:.0f} (mip0 px)   mip shown: {self.qc_view.current_mip}"))

        # --- all settings tab ----------------------------------------------
        self.editor = ConfigEditor()
        self.editor.changed.connect(self._editor_changed)
        self.tabs.addTab(self.editor, "All settings")

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        if project is None:
            self.editor.set_doc(None)
            return
        self.quick.load()
        self.editor.set_doc(self.ctx.configs["stitching"])
        names = project.section_names()
        self.test_sections.set_sections(names)
        self.test_sec.blockSignals(True); self.test_sec.clear(); self.test_sec.addItems(names); self.test_sec.blockSignals(False)
        self._test_show_section()
        self._refresh_tests()
        self._qc_refresh_sources()

    def on_state_changed(self) -> None:
        scan = self.ctx.scan()
        self.steps.refresh(scan)
        self._refresh_test_scan()
        if self.tabs.currentIndex() == 2:
            self._qc_refresh_sources()

    def on_running_changed(self, running: bool) -> None:
        self.on_state_changed()

    def _apply_quick(self) -> None:
        self.quick.apply()
        self.editor.rebuild()
        self.info("stitching settings saved")
        self.ctx.state_changed.emit()

    def _editor_changed(self) -> None:
        if self.ctx.configs:
            self.ctx.configs.save("stitching")
            self.quick.load()
            self.ctx.state_changed.emit()

    # -- test runs ----------------------------------------------------------
    def _test_show_section(self) -> None:
        p = self.project
        name = self.test_sec.currentText()
        if not p or not name:
            self.test_grid.set_tiles([])
            return
        f = p.stitch_coord_dir / f"{name}.txt"
        if not f.is_file():
            return
        info = parse_stitch_coord(f)
        th, tw = info["tile_size"] or (p.state.volume.tile_h, p.state.volume.tile_w)
        self.test_grid.set_tiles([((x, y), x, y, tw, th, "") for rel, x, y in info["tiles"]])

    def _create_test(self) -> None:
        if not self.require_project():
            return
        secs = self.test_sections.checked()
        if not secs:
            QMessageBox.information(self, "Sections", "Check at least one section.")
            return
        if len(secs) > 5 and not self.confirm("Many sections", f"{len(secs)} sections in a test run; continue?"):
            return
        bbox = None
        if self.test_grid.selected:
            tw, th = self.project.state.volume.tile_w, self.project.state.volume.tile_h
            xs = [k[0] for k in self.test_grid.selected]; ys = [k[1] for k in self.test_grid.selected]
            bbox = (min(xs) + 1, min(ys) + 1, max(xs) + tw - 1, max(ys) + th - 1)
        self.quick.apply()
        tr = create_montage_test(self.project, self.test_name.text(), secs, bbox)
        self.info(f"test run created: {tr.root}")
        self._refresh_tests(select=tr.name)
        self._qc_refresh_sources()

    def _refresh_tests(self, select: str | None = None) -> None:
        self.test_list.blockSignals(True)
        self.test_list.clear()
        if self.project:
            for tr in list_test_runs(self.project, "montage"):
                self.test_list.addItem(f"{tr.name}  ({len(tr.sections)} sections)", str(tr.root))
        self.test_list.blockSignals(False)
        if select:
            for i in range(self.test_list.count()):
                if self.test_list.itemText(i).startswith(select + " "):
                    self.test_list.setCurrentIndex(i)
        self._test_changed()

    def _current_test(self) -> TestRun | None:
        d = self.test_list.currentData()
        return TestRun.load(Path(d)) if d else None

    def _test_changed(self) -> None:
        tr = self._current_test()
        self.test_steps.root_override = tr.root if tr else None
        self.test_steps.tag = f"test {tr.name}" if tr else ""
        self._refresh_test_scan()

    def _refresh_test_scan(self) -> None:
        tr = self._current_test()
        if tr is None:
            self.test_steps.refresh(None)
            return
        n = len(list((tr.root / "stitch" / "stitch_coord").glob("*.txt")))
        self.test_steps.refresh(PipelineScan(tr.root, n, None))

    def _delete_test(self) -> None:
        tr = self._current_test()
        if tr and self.confirm("Delete test run", f"Delete {tr.root}?"):
            delete_test_run(tr)
            self._refresh_tests()
            self._qc_refresh_sources()

    def _edit_test_settings(self) -> None:
        tr = self._current_test()
        if not tr:
            return
        from ...core.configs import ConfigStore
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        cs = ConfigStore(tr.root / "configs")
        dlg = QDialog(self); dlg.setWindowTitle(f"Stitching settings of test run {tr.name}"); dlg.resize(900, 700)
        lay = QVBoxLayout(dlg)
        ed = ConfigEditor(); ed.set_doc(cs["stitching"]); lay.addWidget(ed, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        copy_btn = bb.addButton("Save and copy to project", QDialogButtonBox.ActionRole)
        lay.addWidget(bb)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        copy_btn.clicked.connect(lambda: dlg.done(2))
        rc = dlg.exec()
        if rc:
            cs.save("stitching")
            self.info(f"test run settings saved ({tr.name})")
        if rc == 2 and self.ctx.configs:
            self.ctx.configs["stitching"].overrides = dict(cs["stitching"].overrides)
            self.ctx.configs.save("stitching")
            self.quick.load(); self.editor.rebuild()
            self.info("settings copied to the project")
            self.ctx.state_changed.emit()

    # -- QC -------------------------------------------------------------
    def _qc_refresh_sources(self) -> None:
        cur = self.qc_source.currentData()
        self.qc_source.blockSignals(True)
        self.qc_source.clear()
        if self.project:
            self.qc_source.addItem("project", str(self.project.root))
            for tr in list_test_runs(self.project, "montage"):
                self.qc_source.addItem(f"test run {tr.name}", str(tr.root))
        self.qc_source.blockSignals(False)
        if cur:
            i = self.qc_source.findData(cur)
            if i >= 0:
                self.qc_source.setCurrentIndex(i)
        self._qc_refresh_sections()

    def _qc_root(self) -> Path | None:
        d = self.qc_source.currentData()
        return Path(d) if d else None

    def _qc_refresh_sections(self) -> None:
        root = self._qc_root()
        cur = self.qc_sec.currentText()
        self.qc_sec.blockSignals(True)
        self.qc_sec.clear()
        if root:
            base = root / "stitched_sections"
            names = set()
            if (base / "mip0").is_dir():
                names |= {p.name for p in (base / "mip0").iterdir() if p.is_dir()}
            for p in base.glob("*/info") if base.is_dir() else []:
                names.add(p.parent.name)
            self.qc_sec.addItems(sorted(names))
        self.qc_sec.blockSignals(False)
        if cur:
            i = self.qc_sec.findText(cur)
            if i >= 0:
                self.qc_sec.setCurrentIndex(i)
        self._qc_show()
        if self.qc_sec.count() == 0:
            where = f"'{root.name}'" if root else "this project"
            self._qc_note(f"No rendered sections in {where} yet: run 'Render montages' there. "
                          f"Matching and optimization produce no images to look at.")

    def _qc_note(self, text: str) -> None:
        self.qc_note.setText(text)
        self.qc_note.setVisible(bool(text))

    def _qc_show(self) -> None:
        root = self._qc_root()
        name = self.qc_sec.currentText()
        if not root or not name:
            self.qc_view.clear()
            return
        base = root / "stitched_sections"
        try:
            if (base / "mip0" / name).is_dir():
                src = TiledSectionSource(base, name)
            elif (base / name / "info").is_file():
                src = TensorStoreSource(base / name)
            else:
                self.qc_view.clear()
                return
        except Exception as e:  # noqa: BLE001
            self.error(f"cannot open {name}: {e}", dialog=False)
            return
        if not src.levels:
            # the section folder exists but holds no readable tiles yet: it is being rendered right now
            self.qc_view.clear()
            self.qc_info.setText("")
            self._qc_note(f"{name} has no readable tiles yet — it is probably still being rendered. "
                          f"Pick another section, or come back when the step reports it as finished.")
            return
        self._qc_note("")
        self.qc_view.set_source(src)
        self.qc_view.clear_rects()
        if self.qc_outlines.isChecked():
            f = root / "stitch" / "stitch_coord" / f"{name}.txt"
            if f.is_file():
                info = parse_stitch_coord(f)
                th, tw = info["tile_size"] or (0, 0)
                self.qc_view.set_rects([RectSpec(x, y, tw, th, Path(rel).stem[-8:], theme.WARN, rel) for rel, x, y in info["tiles"]], show_labels=False)
        l0 = src.levels[0]
        self.qc_info.setText(f"{name}: {l0.width * 2 ** l0.mip} × {l0.height * 2 ** l0.mip} px, {len(src.levels)} mip level(s)")

    def _inspect(self, step, test: bool = False) -> None:
        self.tabs.setCurrentIndex(2)
        self._qc_refresh_sources()
        # inspect always shows the run the step card belongs to, not whatever was selected before
        target = self._current_test() if test else None
        want = str(target.root) if target else (str(self.project.root) if self.project else "")
        i = self.qc_source.findData(want) if want else -1
        if i >= 0:
            self.qc_source.setCurrentIndex(i)
            self._qc_refresh_sections()
        if step.key != "stitch.rendering":
            self.info("matching/optimization have no images to show; inspect after rendering. Log lines show residues per section.")

    def _open_fiji(self) -> None:
        root = self._qc_root(); name = self.qc_sec.currentText()
        if not root or not name:
            return
        from ..external import open_in_fiji_folder
        d = root / "stitched_sections" / "mip0" / name
        if d.is_dir():
            open_in_fiji_folder(self.ctx.settings, d, self.info)
        else:
            self.warn("Fiji can open PNG tile folders; precomputed volumes are best viewed via the Export page.")
