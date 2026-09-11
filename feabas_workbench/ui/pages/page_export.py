"""Window 6: render the aligned stack, export for viewers, open viewers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QTabWidget, QVBoxLayout, QWidget)

from ...core.images import TiledSectionSource, TensorStoreSource, find_precomputed_scales
from ...core.httpserve import VolumeServer
from ..widgets import PathPicker, ImageView, ConfigEditor, card, hint, form_row, spin, combo
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
        tl.addStretch(1)
        self.tabs.addTab(t, "Render")

        # ---- export tab
        t = QWidget(); tl = QVBoxLayout(t); tl.setContentsMargins(6, 6, 6, 6)
        f, lay = card("Export the PNG stack")
        r = QHBoxLayout()
        self.e_name = QLineEdit("aligned"); self.e_name.setMaximumWidth(200)
        self.e_what = combo([("VASTlite (.vsvi + tile pyramid, hard-linked, no copy)", "vast"), ("OME-Zarr 0.4 (uncompressed chunks)", "omezarr"), ("both", "both")], "vast")
        self.e_chunk = spin(64, 2048, 256, 64)
        r.addWidget(QLabel("name")); r.addWidget(self.e_name); r.addWidget(self.e_what, 1); r.addWidget(QLabel("zarr chunk")); r.addWidget(self.e_chunk)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.e_out = PathPicker("dir", "export folder (default: <project>/exports)")
        r.addWidget(QLabel("to")); r.addWidget(self.e_out, 1)
        b = QPushButton("Export"); b.setObjectName("Primary"); r.addWidget(b)
        lay.addLayout(r)
        lay.addWidget(hint("VAST export re-links the FEABAS tiles into VAST's folder layout and writes the .vsvi with the "
                           "voxel size, so it is instant and needs no extra disk space on the same drive. OME-Zarr writes "
                           "uncompressed chunks and is meant for moderate volumes."))
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
        fb = QPushButton("fit"); r.addWidget(fb); r.addStretch(1)
        tl.addLayout(r)
        self.q_view = ImageView(); self.q_view.setMinimumHeight(560)
        tl.addWidget(self.q_view, 1)
        self.q_info = QLabel(""); self.q_info.setObjectName("Hint")
        tl.addWidget(self.q_info)
        self.tabs.addTab(t, "View aligned sections")
        self.q_sec.currentIndexChanged.connect(self._show)
        self.q_pair.toggled.connect(self._show)
        pb.clicked.connect(lambda: self.q_sec.setCurrentIndex(max(0, self.q_sec.currentIndex() - 1)))
        nb.clicked.connect(lambda: self.q_sec.setCurrentIndex(min(self.q_sec.count() - 1, self.q_sec.currentIndex() + 1)))
        fb.clicked.connect(self.q_view.fit)
        self.q_view.hovered.connect(lambda x, y: self.q_info.setText(f"x={x:.0f} y={y:.0f}   mip {self.q_view.current_mip}"))

        self.editor = ConfigEditor(); self.editor.changed.connect(self._editor_changed)
        self.tabs.addTab(self.editor, "Alignment settings (all)")
        self.ctx.jobs.job_finished.connect(self._job_finished)

    # ------------------------------------------------------------------
    def on_project_changed(self, project) -> None:
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
        self._refresh_vsvi()
        self._refresh_sections()

    def on_state_changed(self) -> None:
        scan = self.ctx.scan()
        self.png_steps.refresh(scan); self.ts_steps.refresh(scan)
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
        out = self.ctx.configs.get("alignment", "rendering.out_dir", None)
        return Path(out) if out else self.project.root / "aligned_stack"

    def _ts_dir(self) -> Path | None:
        if not self.project:
            return None
        out = self.ctx.configs.get("alignment", "tensorstore_rendering.out_dir", None)
        return Path(out) if out else self.project.root / "aligned_tensorstore"

    # -- export ------------------------------------------------------------
    def _export(self) -> None:
        if not self.require_project():
            return
        base = self._aligned_base()
        if not base or not (base / "mip0").is_dir():
            QMessageBox.information(self, "Export", "Render the PNG tile stack first (Render tab).")
            return
        out = self.e_out.path() or self.project.exports_dir
        self.project.state.export["out_dir"] = str(out); self.project.save()
        v = self.project.state.volume
        payload = {"aligned_stack": str(base), "out_dir": str(out), "name": self.e_name.text().strip() or "aligned",
                   "voxel_nm": [v.pixel_size_nm, v.pixel_size_nm, v.section_thickness_nm], "what": self.e_what.currentData(),
                   "chunk": self.e_chunk.value()}
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
        if base and sec and (base / "mip0" / sec).is_dir():
            open_in_fiji_folder(self.ctx.settings, base / "mip0" / sec, self.info)

    # -- viewer ------------------------------------------------------------
    def _refresh_sections(self) -> None:
        base = self._aligned_base()
        cur = self.q_sec.currentText()
        self.q_sec.blockSignals(True); self.q_sec.clear()
        if base and (base / "mip0").is_dir():
            self.q_sec.addItems(sorted(p.name for p in (base / "mip0").iterdir() if p.is_dir()))
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
            if self.q_pair.isChecked() and self.q_sec.currentIndex() + 1 < self.q_sec.count():
                nxt = self.q_sec.itemText(self.q_sec.currentIndex() + 1)
                src2 = TiledSectionSource(base, nxt)
                self.q_view.set_source(_PairSource(src, src2))
            else:
                self.q_view.set_source(src)
        except Exception as e:  # noqa: BLE001
            self.error(f"cannot open {sec}: {e}", dialog=False)

    def _job_finished(self, res) -> None:
        if res.spec.name.startswith("Export") and res.ok:
            self._refresh_vsvi()
            v = res.result.get("vast", {}).get("vsvi")
            if v:
                self.info(f"VAST file: {v}")


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
