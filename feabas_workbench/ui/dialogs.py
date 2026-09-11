"""Small dialogs shared by pages."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout

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
