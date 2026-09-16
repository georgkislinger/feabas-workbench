"""Small dialogs shared by pages."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from ..core.images import imread, downsample, compose_two_color, to_uint8
from .widgets.imageview import ImageView


class OverlapDialog(QDialog):
    """Two neighbouring raw tiles overlaid in red/green at their nominal offset (downsampled)."""

    def __init__(self, tile_a, tile_b, tile_w: int, tile_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neighbouring tiles overlap (red: first, green: second)")
        self.resize(1100, 800)
        lay = QVBoxLayout(self)
        self.view = ImageView()
        lay.addWidget(self.view, 1)
        info = QLabel("")
        info.setObjectName("Hint")
        lay.addWidget(info)
        f = 1
        while max(tile_w, tile_h) / f > 1500:
            f *= 2
        a = downsample(imread(tile_a.path), f)
        b = downsample(imread(tile_b.path), f)
        dx = int(round((tile_b.x_px - tile_a.x_px) / f))
        dy = int(round((tile_b.y_px - tile_a.y_px) / f))
        H = max(a.shape[0], b.shape[0] + dy) - min(0, dy)
        W = max(a.shape[1], b.shape[1] + dx) - min(0, dx)
        ox, oy = -min(0, dx), -min(0, dy)
        ca = np.zeros((H, W), np.uint8)
        cb = np.zeros((H, W), np.uint8)
        ca[oy:oy + a.shape[0], ox:ox + a.shape[1]] = to_uint8(a)
        cb[oy + dy:oy + dy + b.shape[0], ox + dx:ox + dx + b.shape[1]] = to_uint8(b)
        rgb = compose_two_color(ca, cb)
        self.view.set_image(rgb)
        info.setText(f"{tile_a.path.name}  +  {tile_b.path.name}   nominal offset {tile_b.x_px - tile_a.x_px:.0f}, "
                     f"{tile_b.y_px - tile_a.y_px:.0f} px (shown at 1/{f}). Grey/yellow where they agree; coloured fringes are the "
                     f"stage error FEABAS will correct during matching.")
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn, 0, Qt.AlignRight)


class StandardPipelineDialog(QDialog):
    """
    Pipeline > Run standard pipeline: every step of a plain run (stitch, thumbnails, coarse and
    fine alignment) with its current state, and what will be queued - the steps that are not
    done yet, in order. The queue stops at the first failure, like 'Run all steps' on a page.
    """

    def __init__(self, ctx, parent=None):
        from PySide6.QtWidgets import QCheckBox, QDialogButtonBox, QTreeWidget, QTreeWidgetItem
        from ..core.steps import STEPS_BY_KEY, STANDARD_PIPELINE, RENDER_PIPELINE, State
        from . import theme
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Run the standard pipeline")
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        intro = QLabel("Queues every step of a plain run that is not done yet, in order, and stops at the first "
                       "failure. Settings are taken as they are on each page.")
        intro.setWordWrap(True); intro.setObjectName("Hint")
        lay.addWidget(intro)
        self.mips = QLabel(self._mip_text()); self.mips.setWordWrap(True)
        lay.addWidget(self.mips)
        masks = QLabel("Masks: FEABAS's default is used - everything the tiles cover is tissue, no fold detection. "
                       "For sections with folds, tears or empty areas tick the option below: the run stops after the "
                       "thumbnails, you make the masks on the Masks page (tissue method, fold detection, Compose), "
                       "then run the pipeline again - it continues with your masks.")
        masks.setWordWrap(True); masks.setObjectName("Hint")
        lay.addWidget(masks)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["step", "state", "outputs", "will run"])
        self.tree.setRootIsDecorated(False)
        lay.addWidget(self.tree, 1)
        self.stop_for_masks = QCheckBox("stop after 'Make thumbnails' to make masks by hand (folded or partly empty sections)")
        self.stop_for_masks.setChecked(False)
        lay.addWidget(self.stop_for_masks)
        self.render = QCheckBox("also render the aligned stack as PNG tiles + mipmaps (large output, for VASTlite)")
        self.render.setChecked(False)
        lay.addWidget(self.render)
        self.summary = QLabel(""); self.summary.setObjectName("Hint"); self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Queue the steps")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.render.toggled.connect(self._refresh)
        self.stop_for_masks.toggled.connect(self._refresh)
        self._STEPS_BY_KEY, self._STANDARD, self._RENDER, self._State, self._theme = STEPS_BY_KEY, STANDARD_PIPELINE, RENDER_PIPELINE, State, theme
        self._QTreeWidgetItem = QTreeWidgetItem
        self._refresh()

    def keys(self) -> list[str]:
        keys = list(self._STANDARD) + (list(self._RENDER) if self.render.isChecked() else [])
        if self.stop_for_masks.isChecked():
            keys = keys[:keys.index("thumbnail.downsample") + 1]
        return keys

    def _mip_text(self) -> str:
        """The mip levels the run will use, from the project's configs (set when the coordinate files were written)."""
        cs, p = self.ctx.configs, self.ctx.project
        if cs is None or p is None:
            return ""
        v = p.state.volume
        tm = int(cs.get("thumbnail", "thumbnail_mip_level", 2) or 0)
        wm = int(cs.get("alignment", "matching.working_mip_level", 2) or 0)
        mm = int(cs.get("alignment", "meshing.mask_mip_level", tm) or 0)
        px = float(v.pixel_size_nm or 0)
        w = v.tile_w * max(1, v.grid_cols) * 0.95 / 2 ** tm
        h = v.tile_h * max(1, v.grid_rows) * 0.95 / 2 ** tm
        return (f"Mip levels from this project's settings: thumbnails and coarse alignment at mip {tm} "
                f"({px * 2 ** tm:g} nm/px, about {w:.0f} x {h:.0f} px), fine matching at mip {wm} ({px * 2 ** wm:g} nm/px), "
                f"masks at mip {mm}. Change them on the Masks page (thumbnail mip) and the Alignment page (working mip) "
                f"before running if they do not suit the data.")

    def steps_to_run(self) -> list:
        """The Step objects that will be queued, in order."""
        scan = self.ctx.scan()
        out = []
        for key in self.keys():
            step = self._STEPS_BY_KEY[key]
            if step.local:
                continue
            st = scan[key] if scan else None
            if st is None:
                continue
            # done and stale steps have all their outputs; FEABAS skips existing outputs, so
            # queuing them again would compute nothing - 'Clear...' on the step card is the way
            if st.state in (self._State.COMPLETE, self._State.STALE):
                continue
            out.append(step)
        return out

    def stale_steps(self) -> list:
        scan = self.ctx.scan()
        return [self._STEPS_BY_KEY[k] for k in self.keys()
                if not self._STEPS_BY_KEY[k].local and scan and scan[k].state is self._State.STALE]

    def _refresh(self) -> None:
        from PySide6.QtGui import QColor
        scan = self.ctx.scan()
        run = {s.key for s in self.steps_to_run()}
        self.tree.clear()
        for key in self.keys():
            step = self._STEPS_BY_KEY[key]
            if step.local:
                continue
            st = scan[key] if scan else None
            it = self._QTreeWidgetItem([step.label, st.state.value if st else "", st.summary() if st else "",
                                        "yes" if key in run else ""])
            if st:
                it.setForeground(1, QColor(self._theme.STATE_COLORS.get(st.state.value, self._theme.MUTED)))
            self.tree.addTopLevelItem(it)
        for i in range(4):
            self.tree.resizeColumnToContents(i)
        n = len(run)
        text = ("Nothing to do: every step is done." if n == 0 else
                f"{n} step(s) will be queued. The log shows progress; the status bar has a Cancel button.")
        if self.stop_for_masks.isChecked() and n:
            text += " The run stops after the thumbnails; make the masks, then run the pipeline again."
        stale = self.stale_steps()
        if stale:
            text += (f" {len(stale)} step(s) are marked stale (a config or an input changed after their outputs); they are "
                     f"left as they are, because FEABAS never recomputes existing outputs. To redo one, use 'Clear…' on "
                     f"its step card first (that also clears everything after it), then run the pipeline again.")
        self.summary.setText(text)
