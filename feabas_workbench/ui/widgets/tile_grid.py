"""
Clickable tile layout of one section (rows x columns) drawn to scale, used to
pick tiles for training data, montage test subsets and to preview overlaps.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .imageview import ImageView, RectSpec
from .. import theme


class TileGridWidget(QWidget):
    selection_changed = Signal()
    tile_clicked = Signal(object)     # key

    def __init__(self, parent=None, selectable: bool = True):
        super().__init__(parent)
        self.selectable = selectable
        self.tiles: dict[object, RectSpec] = {}
        self.selected: set = set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.info = QLabel("")
        self.info.setObjectName("Hint")
        top.addWidget(self.info, 1)
        if selectable:
            b_all = QPushButton("all"); b_all.setObjectName("Flat")
            b_none = QPushButton("none"); b_none.setObjectName("Flat")
            b_all.clicked.connect(lambda: self.set_selected(set(self.tiles)))
            b_none.clicked.connect(lambda: self.set_selected(set()))
            top.addWidget(b_all); top.addWidget(b_none)
        fit = QPushButton("fit"); fit.setObjectName("Flat")
        top.addWidget(fit)
        lay.addLayout(top)
        self.view = ImageView()
        self.view.setMinimumHeight(220)
        lay.addWidget(self.view, 1)
        fit.clicked.connect(self.view.fit)
        self.view.rect_clicked.connect(self._clicked)

    def set_tiles(self, tiles: list[tuple[object, float, float, float, float, str]], background=None,
                  keep_selection: bool = False) -> None:
        """tiles: (key, x, y, w, h, label)."""
        self.tiles = {k: RectSpec(x, y, w, h, label, theme.ACCENT, k) for k, x, y, w, h, label in tiles}
        if not keep_selection:
            self.selected = set()
        else:
            self.selected &= set(self.tiles)
        if background is not None:
            self.view.set_image(background, fit=False)
        else:
            self.view.set_image(None)
            if self.tiles:
                import numpy as np
                W = max(t.x + t.w for t in self.tiles.values())
                H = max(t.y + t.h for t in self.tiles.values())
                self.view._level0_size = (int(W), int(H))
                self.view.scene().setSceneRect(0, 0, W, H)
        self._redraw()
        self.view.fit()

    def _redraw(self) -> None:
        for k, spec in self.tiles.items():
            spec.selected = k in self.selected
        self.view.set_rects(list(self.tiles.values()))
        self.info.setText(f"{len(self.tiles)} tiles, {len(self.selected)} selected" if self.selectable else f"{len(self.tiles)} tiles")

    def _clicked(self, key) -> None:
        self.tile_clicked.emit(key)
        if not self.selectable:
            return
        if key in self.selected:
            self.selected.discard(key)
        else:
            self.selected.add(key)
        self.view.update_rect_selection(self.selected)
        self.info.setText(f"{len(self.tiles)} tiles, {len(self.selected)} selected")
        self.selection_changed.emit()

    def set_selected(self, keys: set) -> None:
        self.selected = set(keys) & set(self.tiles)
        self.view.update_rect_selection(self.selected)
        self.info.setText(f"{len(self.tiles)} tiles, {len(self.selected)} selected")
        self.selection_changed.emit()
