"""Window 1: project, raw data, naming rule, tile layout, voxel size -> stitch_coord files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit)

from ...core import tiles as T
from ...core.project import Project
from ...core.configs import suggest_working_mip, suggest_thumbnail_mip
from ..widgets import PathPicker, TileGridWidget, card, hint, form_row, spin, dspin, combo
from .base import Page


class _ScanWorker(QObject):
    done = Signal(object, object)   # plan or None, error text

    def __init__(self, root, rule, layout, resolution):
        super().__init__()
        self.args = (root, rule, layout, resolution)

    def run(self) -> None:
        try:
            plan = T.build_plan(*self.args)
            self.done.emit(plan, "")
        except Exception as e:  # noqa: BLE001
            self.done.emit(None, str(e))


class ProjectPage(Page):
    title = "Project & data"
    subtitle = ("Point the workbench at your raw tiles, tell it how filenames encode row, column and section, "
                "check voxel size and tile layout, and write the FEABAS stitch coordinate files.")
    key = "project"

    def build(self) -> None:
        self.plan: T.CoordPlan | None = None
        self._thread: QThread | None = None

        # project card
        f, lay = card("Project")
        row = QHBoxLayout()
        self.proj_path = QLabel("no project open")
        self.proj_path.setObjectName("Mono")
        new_btn = QPushButton("New project…")
        new_btn.setObjectName("Primary")
        open_btn = QPushButton("Open…")
        row.addWidget(self.proj_path, 1)
        row.addWidget(open_btn)
        row.addWidget(new_btn)
        lay.addLayout(row)
        self.proj_name = QLineEdit()
        lay.addWidget(form_row("Project name", self.proj_name))
        lay.addWidget(hint("A project folder is the FEABAS working directory: everything the pipeline produces "
                           "lives inside it. Choose an empty folder on a fast, large disk."))
        new_btn.clicked.connect(self._new)
        open_btn.clicked.connect(self._open)
        self.proj_name.editingFinished.connect(self._name_changed)
        self.body.addWidget(f)

        # raw data card
        f, lay = card("Raw tiles")
        self.src = PathPicker("dir", "folder that contains the tiles (subfolders allowed)")
        lay.addWidget(form_row("Tile folder", self.src))
        r = QHBoxLayout()
        self.ext = QLineEdit("tif"); self.ext.setMaximumWidth(80)
        self.recursive = QCheckBox("include subfolders"); self.recursive.setChecked(True)
        self.preset = combo([(v, k) for k, v in T.PRESET_LABELS.items()], "thermo")
        r.addWidget(QLabel("Extension")); r.addWidget(self.ext)
        r.addWidget(self.recursive)
        r.addSpacing(16)
        r.addWidget(QLabel("Naming rule")); r.addWidget(self.preset, 1)
        lay.addLayout(r)
        self.order = QLineEdit("yxz")
        self.order.setPlaceholderText("letters for the numbers in the name, e.g. 'yxz' or 'xyrz' (r = skip)")
        self.template = QLineEdit("")
        self.template.setPlaceholderText("e.g. Tile_#y#-#x#-#r#_0-000.s#z#_e00   (#r# = ignored number)")
        self.order_row = form_row("Number order", self.order)
        self.template_row = form_row("Pattern", self.template)
        lay.addWidget(self.order_row)
        lay.addWidget(self.template_row)
        gr = QHBoxLayout()
        self.guess_btn = QPushButton("Guess rule from filenames")
        self.guesses = QComboBox()
        self.guesses.setMinimumWidth(320)
        use_guess = QPushButton("Use")
        gr.addWidget(self.guess_btn); gr.addWidget(self.guesses, 1); gr.addWidget(use_guess)
        lay.addLayout(gr)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.guess_btn.clicked.connect(self._guess)
        use_guess.clicked.connect(self._use_guess)
        lay.addWidget(hint("Row (y) and column (x) indices come from the filename; the section number gives z. "
                           "Thermo Maps names are Tile_row-col-id_0-000.sZZZZ_e00; use 'Guess' for anything else."))
        self.body.addWidget(f)

        # layout card
        f, lay = card("Voxel size & tile layout")
        g = QGridLayout()
        g.setHorizontalSpacing(14)
        self.pix = dspin(0.1, 10000, 4.0, 0.5, 3, " nm"); self.pix.setToolTip("xy pixel size at full resolution (mip0)")
        self.thick = dspin(1, 100000, 50.0, 5, 1, " nm"); self.thick.setToolTip("section thickness (z step)")
        self.tile_w = spin(0, 100000, 0); self.tile_h = spin(0, 100000, 0)
        self.meta_btn = QPushButton("Read from image metadata")
        self.mode = combo([("Regular grid from indices", "grid"), ("Microscope stage positions (Thermo or Zeiss metadata)", "stage")], "grid")
        self.ov_x = dspin(0, 100000, 10, 1, 2); self.ov_y = dspin(0, 100000, 10, 1, 2)
        self.ov_unit = combo([("% of tile", "percent"), ("pixels", "pixels")], "percent")
        self.est_btn = QPushButton("Estimate overlap from stage")
        self.flip_x = QCheckBox("flip columns"); self.flip_y = QCheckBox("flip rows")
        self.z_off = spin(-100000, 100000, 0)
        self.path_mode = combo([("relative to tile folder", "relative"), ("absolute", "absolute")], "relative")
        g.addWidget(QLabel("Pixel size"), 0, 0); g.addWidget(self.pix, 0, 1)
        g.addWidget(QLabel("Section thickness"), 0, 2); g.addWidget(self.thick, 0, 3)
        g.addWidget(self.meta_btn, 0, 4)
        g.addWidget(QLabel("Tile width (px)"), 1, 0); g.addWidget(self.tile_w, 1, 1)
        g.addWidget(QLabel("Tile height (px)"), 1, 2); g.addWidget(self.tile_h, 1, 3)
        g.addWidget(QLabel("Placement"), 2, 0); g.addWidget(self.mode, 2, 1, 1, 3)
        g.addWidget(QLabel("Overlap x"), 3, 0); g.addWidget(self.ov_x, 3, 1)
        g.addWidget(QLabel("Overlap y"), 3, 2); g.addWidget(self.ov_y, 3, 3)
        g.addWidget(self.ov_unit, 3, 4); g.addWidget(self.est_btn, 3, 5)
        g.addWidget(self.flip_x, 4, 0); g.addWidget(self.flip_y, 4, 1)
        g.addWidget(QLabel("Section number offset"), 4, 2); g.addWidget(self.z_off, 4, 3)
        g.addWidget(QLabel("Paths in coord files"), 5, 0); g.addWidget(self.path_mode, 5, 1, 1, 3)
        lay.addLayout(g)
        lay.addWidget(hint("Grid placement puts tile (col, row) at (col·(width−overlap), row·(height−overlap)); FEABAS "
                           "refines the exact positions during matching, so ±10 % errors are fine. Stage placement reads the "
                           "microscope stage coordinates from the TIFF metadata and rotates them into image space."))
        self.meta_btn.clicked.connect(self._read_meta)
        self.est_btn.clicked.connect(self._estimate_overlap)
        self.body.addWidget(f)

        # scan + preview card
        f, lay = card("Scan & preview")
        r = QHBoxLayout()
        self.scan_btn = QPushButton("Scan tiles")
        self.scan_btn.setObjectName("Primary")
        self.scan_info = QLabel("")
        self.scan_info.setObjectName("Hint")
        self.scan_info.setWordWrap(True)
        r.addWidget(self.scan_btn); r.addWidget(self.scan_info, 1)
        lay.addLayout(r)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Preview section"))
        self.sec_combo = QComboBox()
        self.sec_combo.setMinimumWidth(140)
        r2.addWidget(self.sec_combo)
        self.overlap_btn = QPushButton("Show overlap of two neighbouring tiles")
        r2.addWidget(self.overlap_btn)
        r2.addStretch(1)
        lay.addLayout(r2)
        self.grid = TileGridWidget(selectable=False)
        self.grid.setMinimumHeight(320)
        lay.addWidget(self.grid)
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMaximumHeight(70)
        self.warnings.setVisible(False)
        lay.addWidget(self.warnings)
        self.scan_btn.clicked.connect(self._scan)
        self.sec_combo.currentIndexChanged.connect(self._show_section)
        self.overlap_btn.clicked.connect(self._show_overlap)
        self.body.addWidget(f)

        # write card
        f, lay = card("Write FEABAS coordinate files")
        r = QHBoxLayout()
        self.write_btn = QPushButton("Write stitch_coord files & save volume info")
        self.write_btn.setObjectName("Primary")
        self.write_btn.setEnabled(False)
        self.write_info = QLabel("")
        self.write_info.setObjectName("Hint")
        self.write_info.setWordWrap(True)
        r.addWidget(self.write_btn); r.addWidget(self.write_info, 1)
        lay.addLayout(r)
        self.suggest = QLabel("")
        self.suggest.setObjectName("Hint")
        self.suggest.setWordWrap(True)
        lay.addWidget(self.suggest)
        self.write_btn.clicked.connect(self._write)
        self.body.addWidget(f)

        # tools card
        f, lay = card("Tools")
        r = QHBoxLayout()
        self.rename_level = combo([("parent folder", "parent"), ("grandparent folder", "grandparent"), ("great-grandparent", "greatgrand")], "parent")
        self.rename_btn = QPushButton("Prefix filenames with folder name…")
        r.addWidget(QLabel("Prefix with")); r.addWidget(self.rename_level); r.addWidget(self.rename_btn); r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(hint("For acquisitions where each section sits in its own folder with repeating filenames: "
                           "prefixes every tile with its folder name so the section number ends up in the filename. "
                           "Shows a preview before renaming anything."))
        self.rename_btn.clicked.connect(self._rename)
        self.body.addWidget(f)

        self._preset_changed()

    # ------------------------------------------------------------------
    def on_project_changed(self, project: Project | None) -> None:
        if project is None:
            self.proj_path.setText("no project open")
            return
        st = project.state
        self.proj_path.setText(str(project.root))
        self.proj_name.setText(st.name)
        s = st.source
        self.src.setText(s.root_dir)
        rule = s.naming_rule()
        self.ext.setText(rule.ext)
        self.recursive.setChecked(rule.recursive)
        i = self.preset.findData(rule.preset)
        self.preset.setCurrentIndex(max(0, i))
        self.order.setText(rule.order)
        self.template.setText(rule.template)
        lp = s.layout_params()
        self.mode.setCurrentIndex(max(0, self.mode.findData(lp.mode)))
        self.ov_x.setValue(lp.overlap_x); self.ov_y.setValue(lp.overlap_y)
        self.ov_unit.setCurrentIndex(max(0, self.ov_unit.findData(lp.overlap_unit)))
        self.flip_x.setChecked(lp.flip_x); self.flip_y.setChecked(lp.flip_y)
        self.z_off.setValue(lp.z_offset)
        self.path_mode.setCurrentIndex(max(0, self.path_mode.findData(s.path_mode)))
        v = st.volume
        self.pix.setValue(v.pixel_size_nm); self.thick.setValue(v.section_thickness_nm)
        self.tile_w.setValue(v.tile_w); self.tile_h.setValue(v.tile_h)
        self.plan = None
        self.grid.set_tiles([])
        self.sec_combo.clear()
        n = len(project.section_names())
        self.write_info.setText(f"{n} coordinate file(s) currently in stitch/stitch_coord." if n else "")
        self._update_suggestions()

    # -- project ---------------------------------------------------------
    def _new(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new project")
        if p:
            self.window().open_project(Path(p))

    def _open(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Open project folder")
        if p:
            self.window().open_project(Path(p))

    def _name_changed(self) -> None:
        if self.project:
            self.project.state.name = self.proj_name.text().strip() or self.project.root.name
            self.project.save()
            self.ctx.project_changed.emit(self.project)

    # -- rule ------------------------------------------------------------
    def _preset_changed(self) -> None:
        p = self.preset.currentData()
        self.order_row.setVisible(p == "sequential")
        self.template_row.setVisible(p == "custom")

    def _rule(self) -> T.NamingRule:
        return T.NamingRule(preset=self.preset.currentData(), order=self.order.text().strip() or "z",
                            template=self.template.text().strip(), ext=self.ext.text().strip().lstrip("."),
                            recursive=self.recursive.isChecked())

    def _layout(self) -> T.LayoutParams:
        return T.LayoutParams(tile_w=self.tile_w.value(), tile_h=self.tile_h.value(),
                              overlap_x=self.ov_x.value(), overlap_y=self.ov_y.value(),
                              overlap_unit=self.ov_unit.currentData(), mode=self.mode.currentData(),
                              z_offset=self.z_off.value(), flip_x=self.flip_x.isChecked(), flip_y=self.flip_y.isChecked())

    def _files(self, limit=None):
        root = self.src.path()
        if not root or not root.is_dir():
            QMessageBox.information(self, "Tile folder", "Choose an existing tile folder first.")
            return None
        files = T.list_image_files(root, self.ext.text().strip().lstrip("."), self.recursive.isChecked(), limit=limit)
        if not files:
            QMessageBox.information(self, "Tile folder", f"No *.{self.ext.text().strip()} files found under {root}.")
            return None
        return files

    def _guess(self) -> None:
        files = self._files(limit=400)
        if not files:
            return
        guesses = T.guess_rules(files, self.ext.text().strip().lstrip("."))
        self.guesses.clear()
        self._guess_rules = []
        for g in guesses[:12]:
            self.guesses.addItem(g.label())
            self._guess_rules.append(g.rule)
        if guesses:
            self.info(f"best naming rule guess: {guesses[0].label()}")
        else:
            self.warn("no naming rule matched the filenames; enter a custom pattern")

    def _use_guess(self) -> None:
        i = self.guesses.currentIndex()
        if i < 0 or i >= len(getattr(self, "_guess_rules", [])):
            return
        r = self._guess_rules[i]
        self.preset.setCurrentIndex(max(0, self.preset.findData(r.preset)))
        self.order.setText(r.order)
        self.template.setText(r.template)

    # -- metadata --------------------------------------------------------
    def _read_meta(self) -> None:
        files = self._files(limit=5)
        if not files:
            return
        info = T.read_tile_info(files[0])
        self.tile_w.setValue(info.width); self.tile_h.setValue(info.height)
        msg = f"{files[0].name}: {info.width}×{info.height} {info.dtype}"
        if info.pixel_size_nm:
            self.pix.setValue(info.pixel_size_nm)
            msg += f", pixel size {info.pixel_size_nm:g} nm"
        else:
            msg += ", no pixel size in metadata (enter it manually)"
        if info.stage_xy_m:
            msg += f", stage coordinates present ({info.vendor or 'generic'} metadata)"
        if info.vendor == "zeiss":
            msg += "; pixel size derived from field of view / image size"
        self.info(msg)
        self.scan_info.setText(msg)

    def _estimate_overlap(self) -> None:
        files = self._files()
        if not files:
            return
        rule = self._rule()
        tiles, _ = T.scan_tiles(self.src.path(), rule)
        if not tiles:
            self.warn("no tiles matched the naming rule")
            return
        z0 = tiles[0].z
        sec = [t for t in tiles if t.z == z0]
        T.load_positions(sec)
        if self.tile_w.value() == 0:
            self._read_meta()
        lp = self._layout()
        if len(sec) < 2:
            self.info("one tile per section: there is no overlap to estimate (the overlap fields are ignored for single-tile sections)")
            return
        est = T.estimate_overlap_from_stage(sec, lp, self.pix.value())
        if est is None:
            self.warn("no usable stage coordinates (or no neighbouring tiles) in the metadata of this section")
            return
        import math
        ox, oy = est
        self.ov_unit.setCurrentIndex(self.ov_unit.findData("pixels"))
        bits = []
        if not math.isnan(ox):
            self.ov_x.setValue(round(ox)); bits.append(f"x {ox:.0f} px ({100 * ox / max(1, lp.tile_w):.1f} %)")
        else:
            bits.append("x: no horizontal neighbours, kept as is")
        if not math.isnan(oy):
            self.ov_y.setValue(round(oy)); bits.append(f"y {oy:.0f} px ({100 * oy / max(1, lp.tile_h):.1f} %)")
        else:
            bits.append("y: no vertical neighbours, kept as is")
        self.info("overlap estimated from stage positions: " + ", ".join(bits))

    # -- scan ------------------------------------------------------------
    def _scan(self) -> None:
        root = self.src.path()
        if not root or not root.is_dir():
            QMessageBox.information(self, "Tile folder", "Choose an existing tile folder first.")
            return
        if self.tile_w.value() == 0 or self.tile_h.value() == 0:
            self._read_meta()
        self.scan_btn.setEnabled(False)
        self.scan_info.setText("scanning…")
        self._worker = _ScanWorker(root, self._rule(), self._layout(), self.pix.value())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._scanned)
        self._thread.start()

    def _scanned(self, plan, err: str) -> None:
        self._thread.quit()
        self._thread.wait(2000)
        self.scan_btn.setEnabled(True)
        if plan is None:
            self.error(f"scan failed: {err}")
            self.scan_info.setText("")
            return
        self.plan = plan
        rows, cols = plan.grid_shape()
        names = list(plan.sections)
        txt = (f"{plan.n_tiles} tiles in {len(names)} sections ({names[0]} … {names[-1]}), grid {rows} rows × {cols} cols, "
               f"tiles {plan.tile_w}×{plan.tile_h} px, {plan.resolution_nm:g} nm/px, "
               f"placement by {'stage coordinates' if plan.placed_by_stage else 'grid indices'}.") if names else "nothing matched"
        self.scan_info.setText(txt)
        self.info(txt)
        self.warnings.setVisible(bool(plan.warnings))
        self.warnings.setPlainText("\n".join(plan.warnings))
        for w in plan.warnings:
            self.warn(w)
        self.sec_combo.blockSignals(True)
        self.sec_combo.clear()
        self.sec_combo.addItems(names)
        self.sec_combo.blockSignals(False)
        if names:
            self.sec_combo.setCurrentIndex(0)
            self._show_section()
        self.write_btn.setEnabled(bool(names))
        self._update_suggestions()

    def _show_section(self) -> None:
        if not self.plan:
            return
        name = self.sec_combo.currentText()
        tl = self.plan.sections.get(name, [])
        self.grid.set_tiles([(t.path, t.x_px, t.y_px, self.plan.tile_w, self.plan.tile_h, f"r{t.iy} c{t.ix}") for t in tl])

    def _show_overlap(self) -> None:
        if not self.plan:
            return
        name = self.sec_combo.currentText()
        tl = self.plan.sections.get(name, [])
        by = {(t.ix, t.iy): t for t in tl}
        pair = None
        for t in tl:
            n = by.get((t.ix + 1, t.iy))
            if n is not None:
                pair = (t, n, "x")
                break
        if pair is None:
            for t in tl:
                n = by.get((t.ix, t.iy + 1))
                if n is not None:
                    pair = (t, n, "y")
                    break
        if pair is None:
            self.warn("no neighbouring tiles in this section")
            return
        from ..dialogs import OverlapDialog
        OverlapDialog(pair[0], pair[1], self.plan.tile_w, self.plan.tile_h, self).exec()

    # -- write -----------------------------------------------------------
    def _update_suggestions(self) -> None:
        pix, thick = self.pix.value(), self.thick.value()
        wm = suggest_working_mip(pix, thick)
        txt = f"Suggested fine-alignment working mip for {pix:g} nm pixels and {thick:g} nm sections: mip{wm} ({pix * 2 ** wm:g} nm/px)."
        if self.plan and self.plan.sections:
            name = next(iter(self.plan.sections))
            w, h = self.plan.section_bbox(name)
            tm = suggest_thumbnail_mip(w, h)
            txt += f" Section ≈ {w:.0f}×{h:.0f} px → suggested thumbnail mip{tm} ({max(w, h) / 2 ** tm:.0f} px thumbnails)."
        self.suggest.setText(txt)

    def _write(self) -> None:
        if not self.require_project() or not self.plan:
            return
        p = self.project
        existing = list(p.stitch_coord_dir.glob("*.txt"))
        downstream = any((p.root / d).exists() for d in ("stitch/match_h5", "stitch/tform", "stitched_sections"))
        if existing and downstream:
            if not self.confirm("Overwrite coordinate files",
                                "Coordinate files exist and stitching outputs were already produced from them. "
                                "Rewriting the coordinates makes those outputs stale (use Pipeline → Clear to remove them). Continue?"):
                return
        for old in existing:
            old.unlink()
        # persist settings
        st = p.state
        st.source.root_dir = str(self.src.path())
        st.source.rule = self._rule().to_dict()
        st.source.layout = self._layout().to_dict()
        st.source.path_mode = self.path_mode.currentData()
        v = st.volume
        v.pixel_size_nm = float(self.plan.resolution_nm if self.pix.value() <= 0 else self.pix.value())
        v.section_thickness_nm = float(self.thick.value())
        v.tile_w, v.tile_h = self.plan.tile_w, self.plan.tile_h
        v.n_sections = len(self.plan.sections)
        v.n_tiles = self.plan.n_tiles
        v.grid_rows, v.grid_cols = self.plan.grid_shape()
        v.section_names = list(self.plan.sections)
        try:
            v.dtype = T.read_tile_info(next(iter(self.plan.sections.values()))[0].path).dtype
        except Exception:  # noqa: BLE001
            pass
        self.plan.resolution_nm = v.pixel_size_nm
        # if a preprocessed source is active, point ROOT_DIR at it
        root_dir = p.active_tile_root() or Path(st.source.root_dir)
        plan = self.plan
        if root_dir != Path(st.source.root_dir):
            plan = _retarget_plan(plan, Path(st.source.root_dir), root_dir)
        files = T.write_plan(plan, p.stitch_coord_dir, st.source.path_mode)
        p.write_general_config()
        p.save()
        # default config suggestions
        cs = self.ctx.configs
        if cs is not None:
            wm = suggest_working_mip(v.pixel_size_nm, v.section_thickness_nm)
            name = next(iter(plan.sections))
            w, h = plan.section_bbox(name)
            tm = suggest_thumbnail_mip(w, h)
            cs.set("alignment", "matching.working_mip_level", int(wm))
            cs.set("thumbnail", "thumbnail_mip_level", int(tm))
            cs.set("alignment", "meshing.mask_mip_level", int(tm))
            cs.set("stitching", "section_thickness", float(v.section_thickness_nm))
            cs.save()
        self.write_info.setText(f"wrote {len(files)} coordinate files to {p.stitch_coord_dir}")
        self.info(f"wrote {len(files)} stitch coordinate files; tile root {root_dir}")
        self.ctx.project_changed.emit(p)
        self.ctx.state_changed.emit()

    # -- rename ----------------------------------------------------------
    def _rename(self) -> None:
        root = self.src.path()
        if not root or not root.is_dir():
            return
        pairs = T.plan_prefix_rename(root, self.ext.text().strip().lstrip("."), self.rename_level.currentData(), self.recursive.isChecked())
        if not pairs:
            QMessageBox.information(self, "Rename", "Nothing to rename (files already prefixed or none found).")
            return
        preview = "\n".join(f"{a.name}  →  {b}" for a, b in pairs[:15])
        if len(pairs) > 15:
            preview += f"\n… and {len(pairs) - 15} more"
        if not self.confirm("Rename files", f"Rename {len(pairs)} files in place?\n\n{preview}"):
            return
        n = 0
        for a, b in pairs:
            try:
                a.rename(a.with_name(b))
                n += 1
            except OSError as e:
                self.warn(f"could not rename {a}: {e}")
        self.info(f"renamed {n} files")


def _retarget_plan(plan: T.CoordPlan, old_root: Path, new_root: Path) -> T.CoordPlan:
    """Point every tile path from the raw folder to the mirrored preprocessed folder."""
    import copy
    out = copy.deepcopy(plan)
    out.root_dir = new_root
    for tl in out.sections.values():
        for t in tl:
            try:
                rel = Path(t.path).resolve().relative_to(old_root.resolve())
            except ValueError:
                rel = Path(t.path.name)
            t.path = new_root / rel
    return out
