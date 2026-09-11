from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class PathPicker(QWidget):
    """Line edit + browse button. mode: 'dir' | 'file' | 'save'."""

    changed = Signal(str)

    def __init__(self, mode: str = "dir", placeholder: str = "", filter: str = "", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.filter = filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.btn = QPushButton("…")
        self.btn.setFixedWidth(30)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)
        self.btn.clicked.connect(self.browse)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.edit.text()))

    def text(self) -> str:
        return self.edit.text().strip()

    def path(self) -> Path | None:
        t = self.text()
        return Path(t) if t else None

    def setText(self, t: str) -> None:
        self.edit.setText(str(t) if t else "")

    def browse(self) -> None:
        start = self.text() or str(Path.home())
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif self.mode == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Choose file", start, self.filter)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", start, self.filter)
        if p:
            self.edit.setText(p)
            self.changed.emit(p)
