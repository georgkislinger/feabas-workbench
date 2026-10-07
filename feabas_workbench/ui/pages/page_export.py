"""Window 6: render the aligned stack, export for viewers, open viewers."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from ...core import segmentation as seg
from ...core.images import TiledSectionSource
from ...core.local_parallel import run_budget
from ...core.project import repair_working_directory
from ...core.steps import aligned_dir, aligned_render_mip, is_link, tensorstore_dir
from ...core.httpserve import VolumeServer
from ..widgets import PathPicker, ImageView, ConfigEditor, card, hint, spin, combo
from ..widgets.steps_panel import StepsPanel
from ..external import open_in_vast, open_in_fiji_folder
from .base import Page


class ExportPage(Page):
    title = "Export & view"
    subtitle = ("Render the aligned stack at full resolution (PNG tiles for VASTlite or a Neuroglancer precomputed "
                "volume), build mipmaps, and hand the result to a viewer.")
    key = "export"

    def build(self) -> None:
        self._server: VolumeServer | None = None
        self.tabs = QTabWidget()
        self.body.addWidget(self.tabs)
        self.cluster_download = QPushButton("Download rendered stack from LRZ for local viewing")
        self.cluster_download.clicked.connect(lambda: self.ctx.cluster.download_rendered())
        self.cluster_download.setVisible(self.ctx.cluster_enabled)
        self.body.addWidget(self.cluster_download)

        # ---- render tab
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Render settings")
        r = QHBoxLayout()
        self.r_tile = spin(256, 16384, 4096, 256); self.r_mip = spin(0, 8, 0); self.r_workers = spin(1, 256, 15)
        self.r_maxmip = spin(1, 10, 7); self.r_interp = combo(["LANCZOS", "CUBIC", "LINEAR", "NEAREST"], "LANCZOS")
        r.addWidget(QLabel("PNG tile size")); r.addWidget(self.r_tile); r.addWidget(QLabel("mip")); r.addWidget(self.r_mip)
        r.addWidget(QLabel("mipmaps up to")); r.addWidget(self.r_maxmip); r.addWidget(QLabel("interpolation")); r.addWidget(self.r_interp)
        r.addWidget(QLabel("workers")); r.addWidget(self.r_workers)
        b = QPushButton("Apply"); b.setObjectName("Primary"); r.addWidget(b); r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.r_outdir = PathPicker("dir", "leave empty to render inside the project (aligned_stack / aligned_tensorstore)")
        r.addWidget(QLabel("output folder")); r.addWidget(self.r_outdir, 1)
        lay.addLayout(r)
        lay.addWidget(hint("For ~1 TB volumes the precomputed volume is the efficient path (chunked, viewable in Neuroglancer "
                           "directly). PNG tiles are simpler and what VASTlite reads; both can be produced from the same "
                           "transforms. Rendering can be distributed with the subset controls on several machines."))
        b.clicked.connect(self._apply)
        tl.addWidget(f)
        f, lay = card("PNG tile stack (VASTlite)")
        self.png_steps = StepsPanel(self.ctx, ["align.rendering", "align.downsample"], compact=True)
        self.png_steps.inspect_requested.connect(lambda _s: self.tabs.setCurrentIndex(2))
        lay.addWidget(self.png_steps)
        tl.addWidget(f)
        f, lay = card("Precomputed volume (Neuroglancer / TensorStore)")
        self.ts_steps = StepsPanel(self.ctx, ["align.tsr", "align.tsd"], compact=True)
        self.ts_steps.inspect_requested.connect(lambda _s: self.tabs.setCurrentIndex(2))
        lay.addWidget(self.ts_steps)
        tl.addWidget(f)
        f, lay = card("Segmentation masks (labels)")
        r = QHBoxLayout()
        self.s_dir = PathPicker("dir", "folder with one 8- or 16-bit greyscale label image (PNG or TIFF) per input image, "
                                       "at full resolution or exported at one mip level")
        self.s_match = combo([("by file name, else in order", "auto"), ("by file name", "name"),
                              ("in section order", "order")], "auto")
        r.addWidget(QLabel("masks")); r.addWidget(self.s_dir, 1); r.addWidget(QLabel("match")); r.addWidget(self.s_match)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.s_name = QLineEdit("masks"); self.s_name.setMaximumWidth(180)
        self.s_out = PathPicker("dir", "leave empty to write inside the project (segmentation/<name>)")
        r.addWidget(QLabel("name")); r.addWidget(self.s_name); r.addWidget(QLabel("output folder")); r.addWidget(self.s_out, 1)
        lay.addLayout(r)
        r = QHBoxLayout()
        b1 = QPushButton("Render aligned masks"); b1.setObjectName("Primary")
        b2 = QPushButton("Clear…")
        self.s_status = QLabel(""); self.s_status.setObjectName("Hint"); self.s_status.setWordWrap(True)
        r.addWidget(b1); r.addWidget(b2); r.addWidget(self.s_status, 1)
        lay.addLayout(r)
        lay.addWidget(hint("For masks drawn on an image stack: import the stack as one image per section, align it, then "
                           "render the masks here. They take exactly the images' path - placed like their image, moved by "
                           "the section's alignment onto the same canvas - sampled nearest-neighbour so labels are never "
                           "mixed, with mipmaps that keep each 2x2 block's majority label. Masks exported at a mip level "
                           "(1/2, 1/4, 1/8 ... of their images' size) are recognised by their size; their aligned stack "
                           "starts at that level, and coarse masks render faster. The result has the aligned PNG stack's layout: "
                           "export it on the next tab (VAST tiles, OME-Zarr or one image per section) and check it over "
                           "the images under View aligned sections."))
        b1.clicked.connect(self._render_masks); b2.clicked.connect(self._clear_masks)
        self.s_name.editingFinished.connect(self._refresh_masks)
        self.s_dir.changed.connect(lambda _t: self._refresh_masks())
        self.s_out.changed.connect(lambda _t: self._refresh_masks())
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "Render")

        # ---- export tab
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Export a stack")
        r = QHBoxLayout()
        self.e_stack = QComboBox(); self.e_stack.setMinimumWidth(180)
        self.e_stack.addItem("aligned images", "")
        self.e_name = QLineEdit("aligned"); self.e_name.setMaximumWidth(200)
        self.e_what = combo([("VASTlite (.vsvi + tile pyramid, hard-linked, no copy)", "vast"), ("OME-Zarr 0.4 (uncompressed chunks)", "omezarr"),
                             ("both", "both"), ("one image per section", "slices")], "vast")
        self.e_chunk = spin(64, 2048, 256, 64)
        r.addWidget(QLabel("stack")); r.addWidget(self.e_stack); r.addWidget(QLabel("name")); r.addWidget(self.e_name)
        r.addWidget(self.e_what, 1); r.addWidget(QLabel("zarr chunk")); r.addWidget(self.e_chunk)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.e_slice_mip = spin(0, 10, 0); self.e_slice_fmt = combo(["png", "tif"], "png")
        r.addWidget(QLabel("images per section: mip")); r.addWidget(self.e_slice_mip); r.addWidget(QLabel("format"))
        r.addWidget(self.e_slice_fmt); r.addStretch(1)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.e_out = PathPicker("dir", "export folder (default: <project>/exports)")
        r.addWidget(QLabel("to")); r.addWidget(self.e_out, 1)
        b = QPushButton("Export"); b.setObjectName("Primary"); r.addWidget(b)
        lay.addLayout(r)
        lay.addWidget(hint("VAST export re-links the FEABAS tiles into VAST's folder layout and writes the .vsvi with the "
                           "voxel size, so it is instant and needs no extra disk space on the same drive. OME-Zarr writes "
                           "uncompressed chunks and is meant for moderate volumes (zarr, tensorstore or dask read it in "
                           "Python). One image per section writes whole sections at the chosen mip level, 8- or 16-bit like "
                           "the stack, for tools that import image sequences. Aligned segmentation masks export the same way "
                           "and line up with the images."))
        self.e_stack.currentIndexChanged.connect(self._stack_chosen)
        b.clicked.connect(self._export)
        tl.addWidget(f)
        f, lay = card("Open in a viewer")
        r = QHBoxLayout()
        self.v_vsvi = QComboBox(); self.v_vsvi.setMinimumWidth(320)
        b1 = QPushButton("Open in VASTlite"); b2 = QPushButton("Refresh list")
        r.addWidget(QLabel(".vsvi")); r.addWidget(self.v_vsvi, 1); r.addWidget(b1); r.addWidget(b2)
        lay.addLayout(r)
        r = QHBoxLayout()
        b3 = QPushButton("Serve precomputed volume & open in Neuroglancer")
        self.ng_info = QLabel(""); self.ng_info.setObjectName("Hint"); self.ng_info.setWordWrap(True)
        r.addWidget(b3); r.addWidget(self.ng_info, 1)
        lay.addLayout(r)
        r = QHBoxLayout()
        b4 = QPushButton("Open aligned section folder in Fiji"); r.addWidget(b4); r.addStretch(1)
        lay.addLayout(r)
        b1.clicked.connect(self._open_vast); b2.clicked.connect(self._refresh_vsvi); b3.clicked.connect(self._neuroglancer); b4.clicked.connect(self._fiji)
        tl.addWidget(f)
        tl.addStretch(1)
        self.tabs.addTab(t, "Export & viewers")

        # ---- viewer tab
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        r = QHBoxLayout()
        r.addWidget(QLabel("Section")); self.q_sec = QComboBox(); self.q_sec.setMinimumWidth(180); r.addWidget(self.q_sec)
        pb = QPushButton("◀"); nb = QPushButton("▶"); pb.setFixedWidth(32); nb.setFixedWidth(32); r.addWidget(pb); r.addWidget(nb)
        self.q_pair = QCheckBox("overlay next section (red/green)"); r.addWidget(self.q_pair)
        self.q_masks = QComboBox(); self.q_masks.addItem("no masks", "")
        r.addWidget(QLabel("masks")); r.addWidget(self.q_masks)
        fb = QPushButton("fit"); r.addWidget(fb); r.addStretch(1)
        tl.addLayout(r)
        self.q_view = ImageView(); self.q_view.setMinimumHeight(560)
        tl.addWidget(self.q_view, 1)
        self.q_info = QLabel(""); self.q_info.setObjectName("Hint")
        tl.addWidget(self.q_info)
        self.tabs.addTab(t, "View aligned sections")
        self.q_sec.currentIndexChanged.connect(self._show)
        self.q_pair.toggled.connect(self._show)
        self.q_masks.currentIndexChanged.connect(self._show)
        pb.clicked.connect(lambda: self.q_sec.setCurrentIndex(max(0, self.q_sec.currentIndex() - 1)))
        nb.clicked.connect(lambda: self.q_sec.setCurrentIndex(min(self.q_sec.count() - 1, self.q_sec.currentIndex() + 1)))
        fb.clicked.connect(self.q_view.fit)
        self.q_view.hovered.connect(lambda x, y: self.q_info.setText(f"x={x:.0f} y={y:.0f}   mip {self.q_view.current_mip}"))

        self.editor = ConfigEditor(); self.editor.changed.connect(self._editor_changed)
        self.tabs.addTab(self.editor, "Alignment settings")
        self.ctx.jobs.job_finished.connect(self._job_finished)
        self.ctx.export_downloaded.connect(self._download_finished)

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
        self.cluster_download.setVisible(self.ctx.cluster_enabled)
        if project is None:
            self.editor.set_doc(None)
            return
        self.editor.set_doc(self.ctx.configs["alignment"])
        cs = self.ctx.configs
        ts = cs.get("alignment", "rendering.tile_size", [4096, 4096])
        self.r_tile.setValue(int(ts[0]) if isinstance(ts, (list, tuple)) else int(ts))
        self.r_mip.setValue(int(cs.get("alignment", "rendering.mip_level", 0)))
        self.r_workers.setValue(int(cs.get("alignment", "rendering.num_workers", 15)))
        self.r_maxmip.setValue(int(cs.get("alignment", "downsample.max_mip", 7)))
        self.r_interp.setCurrentIndex(max(0, self.r_interp.findData(cs.get("alignment", "rendering.remap_interp", "LANCZOS"))))
        self.r_outdir.setText(cs.get("alignment", "rendering.out_dir", None) or "")
        self.e_out.setText(project.state.export.get("out_dir", ""))
        saved = project.state.export.get("segmentation", {})
        self.s_dir.setText(saved.get("masks_dir", ""))
        self.s_match.setCurrentIndex(max(0, self.s_match.findData(saved.get("match", "auto"))))
        self.s_name.setText(saved.get("name", "masks"))
        self.s_out.setText(saved.get("out_dir", ""))
        self._refresh_vsvi()
        self._refresh_sections()
        self._refresh_masks()

    def on_state_changed(self) -> None:
        scan = self.ctx.scan()
        self.png_steps.refresh(scan); self.ts_steps.refresh(scan)
        self._refresh_masks()
        if self.tabs.currentIndex() == 2:
            self._refresh_sections()

    def on_running_changed(self, running: bool) -> None:
        self.on_state_changed()

    def _apply(self) -> None:
        cs = self.ctx.configs
        t = self.r_tile.value()
        cs.set("alignment", "rendering.tile_size", [t, t])
        cs.set("alignment", "rendering.mip_level", self.r_mip.value())
        cs.set("alignment", "rendering.num_workers", self.r_workers.value())
        cs.set("alignment", "rendering.remap_interp", self.r_interp.currentData())
        cs.set("alignment", "downsample.max_mip", self.r_maxmip.value())
        cs.set("alignment", "tensorstore_rendering.num_workers", self.r_workers.value())
        out = self.r_outdir.text().strip() or None
        cs.set("alignment", "rendering.out_dir", out)
        cs.set("alignment", "tensorstore_rendering.out_dir", (str(Path(out) / "aligned_tensorstore") if out else None))
        cs.save("alignment"); self.editor.rebuild()
        self.info("render settings saved"); self.ctx.state_changed.emit()

    def _editor_changed(self) -> None:
        if self.ctx.configs:
            self.ctx.configs.save("alignment"); self.ctx.state_changed.emit()

    def _aligned_base(self) -> Path | None:
        if not self.project:
            return None
        return aligned_dir(self.project.root, self.ctx.configs)

    def _render_mip(self) -> int:
        """The mip level the PNG stack was rendered at: the export's full resolution."""
        return aligned_render_mip(self.ctx.configs) if self.ctx.configs else 0

    def _ts_dir(self) -> Path | None:
        if not self.project:
            return None
        return tensorstore_dir(self.project.root, self.ctx.configs)

    # -- segmentation masks -------------------------------------------------
    def _mask_base(self) -> Path | None:
        if not self.project:
            return None
        try:
            return seg.stack_dir(self.project.root, self.s_name.text(), self.s_out.path())
        except ValueError:
            return None

    def _refresh_masks(self) -> None:
        """Status of the named mask stack and the stack lists, from the files on disk."""
        if not self.project:
            self.s_status.setText("")
            return
        base = self._mask_base()
        if base is None:
            self.s_status.setText("give the masks a name")
        else:
            cs = self.ctx.configs
            mip, top = self._render_mip(), int(cs.get("alignment", "downsample.max_mip", 7)) if cs else 0
            folder = self.s_dir.path()
            masks = seg.list_mask_files(folder) if folder else []
            st = seg.stack_status(self.project.root, base, self.project.section_names(), mip, top,
                                  masks if len(masks) <= 5000 else None)
            if not st.rendered:
                text = f"not rendered yet; they will go to {base}"
            else:
                start = f" from mip {st.first_mip} (no finer levels)" if st.first_mip > mip else ""
                text = (f"{st.rendered} of {st.sections} sections rendered{start}, {st.mipmapped} with mipmaps up to "
                        f"mip {max(st.first_mip, top)}, in {base}")
            if st.stale:
                text += f" - stale: {st.stale}; Clear and render again"
            self.s_status.setText(text)
        stacks = seg.list_stacks(self.project.root, [self.s_out.path()] if self.s_out.path() else None)
        for box, first in ((self.e_stack, ("aligned images", "")), (self.q_masks, ("no masks", ""))):
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            box.addItem(*first)
            for d in stacks:
                box.addItem(("masks: " if box is self.e_stack else "") + d.name, str(d))
            box.setCurrentIndex(max(0, box.findData(current)))
            box.blockSignals(False)

    def _render_masks(self) -> None:
        if not self.require_project():
            return
        title = "Segmentation masks"
        if self.ctx.cluster_enabled:
            QMessageBox.information(self, title, "Masks are rendered on this PC from the local alignment: leave cluster mode first.")
            return
        folder = self.s_dir.path()
        if not folder:
            QMessageBox.information(self, title, "Choose the folder with the mask images first.")
            return
        try:
            name = seg.safe_name(self.s_name.text())
        except ValueError as e:
            QMessageBox.information(self, title, str(e))
            return
        root, sections = self.project.root, self.project.section_names()
        if repair_working_directory(root):
            self.info(f"{root}: configs/general_configs.yaml named another folder (a copied or moved project); "
                      f"FEABAS now works here again")
        unaligned = [s for s in sections if not (root / "align" / "tform" / f"{s}.h5").is_file()]
        if unaligned:
            QMessageBox.information(self, title, f"{len(unaligned)} of {len(sections)} sections are not aligned yet "
                                    f"(e.g. {unaligned[0]}): run the alignment first.")
            return
        try:
            plan = seg.plan_masks(root, sections, folder, self.s_match.currentData())
            python = self.ctx.require_feabas_python()
        except (ValueError, OSError, RuntimeError) as e:
            QMessageBox.warning(self, title, str(e))
            return
        base = seg.stack_dir(root, name, self.s_out.path())
        first = seg.first_level(plan.mip, self._render_mip())
        existing = seg.stack_start(base)
        if seg.legacy_problem(base):
            QMessageBox.information(self, title, f"The aligned masks '{name}' were {seg.legacy_problem(base)}. "
                                    "Clear them first, then render them again.")
            return
        if existing is not None and existing != first:
            QMessageBox.information(self, title, f"The aligned masks '{name}' start at mip {existing}; these would start "
                                    f"at mip {first}. Clear them first, or give the new masks another name.")
            return
        cpu, ram = run_budget(self.project)
        workers = max(1, min(len(plan.sections), cpu, int(ram // (1.5 if plan.bits == 8 else 2.0))))
        self.project.state.export["segmentation"] = {"masks_dir": str(folder), "match": self.s_match.currentData(),
                                                     "name": name, "out_dir": self.s_out.text().strip()}
        self.project.save()
        for note in plan.notes:
            self.info(f"masks: {note}")
        self.info(f"aligned masks: {plan.n_masks} {plan.bits}-bit mask file(s)"
                  + (f" at mip {plan.mip}" if plan.mip else "")
                  + f" for {len(sections)} sections, {workers} at once -> {base}, from mip {first}")
        payload = {"root": str(root), "out_dir": str(base), "plan": plan.to_dict(), "workers": workers,
                   "source": {"masks_dir": str(folder), "match": plan.mode}}
        self.submit(self.ctx.worker_spec("segmentation_render", payload, f"Aligned masks ({name})", python=python,
                                         expected=len(sections)))

    def _clear_masks(self) -> None:
        base = self._mask_base()
        if not self.project or base is None or not base.exists():
            QMessageBox.information(self, "Clear aligned masks", "There are no aligned masks of this name to clear.")
            return
        if self.ctx.jobs.running:
            QMessageBox.information(self, "Busy", "Wait for the running job to finish first.")
            return
        if is_link(base):
            QMessageBox.warning(self, "Clear aligned masks", f"{base} is a link; remove it yourself if you mean to.")
            return
        if not self.confirm("Clear aligned masks", f"Delete the aligned masks in\n{base}?\n\nYour mask files and "
                            "the images stay untouched."):
            return
        shutil.rmtree(base, ignore_errors=True)
        self.info(f"removed {base}")
        self._refresh_masks()

    # -- export ------------------------------------------------------------
    def _stack_chosen(self) -> None:
        d = self.e_stack.currentData()
        self.e_name.setText(Path(d).name if d else "aligned")

    def _export(self) -> None:
        if not self.require_project():
            return
        masks = self.e_stack.currentData()
        base = Path(masks) if masks else self._aligned_base()
        mip = render_mip = self._render_mip()
        if masks:                           # masks exported at a coarse mip start at that level
            first = seg.stack_start(base)
            mip = mip if first is None else first
        if masks and self.ctx.cluster_enabled:
            QMessageBox.information(self, "Export", "Aligned masks are exported on this PC: leave cluster mode first.")
            return
        rendered = (self.ctx.scan()["align.rendering"].done > 0 if self.ctx.cluster_enabled and not masks
                    else bool(base and (base / f"mip{mip}").is_dir()))
        if not rendered:
            QMessageBox.information(self, "Export", "Render the masks first (Render tab)." if masks else
                                    "Render the PNG tile stack first (Render tab).")
            return
        out = self.e_out.path() or self.project.exports_dir
        if self.ctx.cluster_enabled and out.is_relative_to(self.project.root):
            out = self.ctx.local_project.exports_dir / "cluster"
        self.project.state.export["out_dir"] = str(out); self.project.save()
        v = self.project.state.volume
        name = self.e_name.text().strip() or ("masks" if masks else "aligned")
        payload = {"aligned_stack": str(base), "out_dir": str(out), "name": name,
                   "voxel_nm": [v.pixel_size_nm, v.pixel_size_nm, v.section_thickness_nm], "what": self.e_what.currentData(),
                   "chunk": self.e_chunk.value(), "base_mip": mip,
                   # images keep exports/vast; each mask stack gets its own VAST folder next to it
                   "vast_dir": f"vast_{name}" if masks else "vast",
                   "slice_mip": max(mip, self.e_slice_mip.value()), "slice_format": self.e_slice_fmt.currentData()}
        if masks and mip > render_mip:
            if self.e_what.currentData() == "slices":
                if self.e_slice_mip.value() < mip:
                    self.info(f"{base.name} starts at mip {mip}: its images per section are written at mip {mip}")
            else:
                self.info(f"{base.name} starts at mip {mip}: level 0 of this export is mip {mip} of the images (voxel "
                          f"size x{2 ** (mip - render_mip)}). For a VAST segmentation layer of the image volume, export "
                          f"one image per section at mip {mip} and import those with VAST's 'Import Segmentation from "
                          f"Images'.")
        import sys
        self.submit(self.ctx.worker_spec("export_vast", payload, f"Export ({self.e_what.currentData()})", python=sys.executable))

    def _refresh_vsvi(self) -> None:
        self.v_vsvi.clear()
        if not self.project:
            return
        roots = [self.project.exports_dir]
        if self.e_out.path():
            roots.append(self.e_out.path())
        seen = set()
        for r in roots:
            for p in sorted(r.rglob("*.vsvi")) if r.is_dir() else []:
                if p not in seen:
                    seen.add(p); self.v_vsvi.addItem(str(p), str(p))

    def _open_vast(self) -> None:
        p = self.v_vsvi.currentData()
        if not p:
            QMessageBox.information(self, "VAST", "Export a VAST layout first.")
            return
        if not open_in_vast(self.ctx.settings, Path(p), self.info):
            QMessageBox.information(self, "VAST", "Set the VASTlite path on the Setup page.")

    def _neuroglancer(self) -> None:
        d = self._ts_dir()
        if not d or not (d / "info").is_file():
            QMessageBox.information(self, "Neuroglancer", "Render the precomputed volume first (Render tab).")
            return
        if self._server is None or self._server.directory != d:
            if self._server:
                self._server.stop()
            self._server = VolumeServer(d).start()
        url = self._server.neuroglancer_url("", self.project.state.name)
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(url))
        self.ng_info.setText(f"serving {d} at {self._server.url} while the workbench is open")
        self.info(f"neuroglancer: {url[:120]}…")

    def _fiji(self) -> None:
        base = self._aligned_base(); sec = self.q_sec.currentText()
        d = base / f"mip{self._render_mip()}" / sec if base and sec else None
        if d is not None and d.is_dir():
            open_in_fiji_folder(self.ctx.settings, d, self.info)

    # -- viewer ------------------------------------------------------------
    def _refresh_sections(self) -> None:
        base = self._aligned_base()
        cur = self.q_sec.currentText()
        self.q_sec.blockSignals(True); self.q_sec.clear()
        level = base / f"mip{self._render_mip()}" if base else None
        if level is not None and level.is_dir():
            self.q_sec.addItems(sorted(p.name for p in level.iterdir() if p.is_dir()))
        self.q_sec.blockSignals(False)
        i = self.q_sec.findText(cur)
        if i >= 0:
            self.q_sec.setCurrentIndex(i)
        self._show()

    def _show(self) -> None:
        base = self._aligned_base(); sec = self.q_sec.currentText()
        if not base or not sec:
            self.q_view.clear()
            return
        try:
            src = TiledSectionSource(base, sec)
            masks = self.q_masks.currentData()
            if masks:
                self.q_view.set_source(_MaskOverlaySource(src, TiledSectionSource(Path(masks), sec)))
            elif self.q_pair.isChecked() and self.q_sec.currentIndex() + 1 < self.q_sec.count():
                nxt = self.q_sec.itemText(self.q_sec.currentIndex() + 1)
                src2 = TiledSectionSource(base, nxt)
                self.q_view.set_source(_PairSource(src, src2))
            else:
                self.q_view.set_source(src)
        except Exception as e:  # noqa: BLE001
            self.error(f"cannot open {sec}: {e}", dialog=False)

    def _job_finished(self, res) -> None:
        if res.spec.name.startswith("Aligned masks"):
            self._refresh_masks()
            if res.ok and res.result.get("out_dir"):
                self.info(f"aligned masks: {res.result['out_dir']}")
        if res.spec.name.startswith("Export") and res.ok:
            if self.ctx.cluster_enabled:
                self.info("Export completed at LRZ. The verified download is tracked in the top bar.")
                return
            self._refresh_vsvi()
            v = res.result.get("vast", {}).get("vsvi")
            if v:
                self.info(f"VAST file: {v}")

    def _download_finished(self, path):
        self.e_out.setText(path)
        self._refresh_vsvi()
        self.info("Export downloaded and verified: " + path)


class _MaskOverlaySource:
    """An aligned section with an aligned mask stack's labels laid over it in colour. Below the
    level a mask stack starts at (masks exported at a coarse mip) its first level is enlarged, each
    label pixel covering exactly the block of pixels it stands for."""

    def __init__(self, images, masks):
        self.images, self.masks = images, masks
        self.levels = images.levels
        self.name = images.name

    def read(self, mip, x0, y0, w, h):
        from ...core.images import label_overlay
        return label_overlay(self.images.read(mip, x0, y0, w, h), self.labels(mip, x0, y0, w, h))

    def labels(self, mip, x0, y0, w, h):
        import numpy as np
        have = [lv.mip for lv in self.masks.levels]
        if not have or mip >= min(have):
            return self.masks.read(mip, x0, y0, w, h)
        first = min(have)
        f = 2 ** (first - mip)
        cx0, cy0 = x0 // f, y0 // f
        coarse = self.masks.read(first, cx0, cy0, -(-(x0 + w) // f) - cx0, -(-(y0 + h) // f) - cy0)
        big = np.repeat(np.repeat(coarse, f, axis=0), f, axis=1)
        return big[y0 - cy0 * f:y0 - cy0 * f + h, x0 - cx0 * f:x0 - cx0 * f + w]


class _PairSource:
    """Two tiled sections composed in red/green on the fly."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.levels = a.levels
        self.name = f"{a.name}+{b.name}"

    def read(self, mip, x0, y0, w, h):
        from ...core.images import compose_two_color
        ia = self.a.read(mip, x0, y0, w, h)
        ib = self.b.read(mip, x0, y0, w, h)
        return compose_two_color(ia, ib)
