from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .. import theme

COLORS = {
    "info": theme.TEXT,
    "warn": theme.WARN,
    "error": theme.ERR,
    "job": "#B6C2D1",
    "job-err": theme.ERR,
}


class LogPanel(QWidget):
    """Application + subprocess log with a text filter. Keeps the last ~5000 lines."""

    MAX_LINES = 5000

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter (substring)")
        self.filter.setMaximumWidth(260)
        self.only_problems = QCheckBox("only warnings/errors")
        self.autoscroll = QCheckBox("autoscroll")
        self.autoscroll.setChecked(True)
        clear = QPushButton("Clear")
        clear.setObjectName("Flat")
        top.addWidget(self.filter)
        top.addWidget(self.only_problems)
        top.addWidget(self.autoscroll)
        top.addStretch(1)
        top.addWidget(clear)
        lay.addLayout(top)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(self.MAX_LINES)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(9)
        self.view.setFont(f)
        lay.addWidget(self.view, 1)
        self._all: list[tuple[str, str]] = []
        clear.clicked.connect(self.clear)
        self.filter.textChanged.connect(self._refilter)
        self.only_problems.toggled.connect(self._refilter)

    def append(self, level: str, text: str) -> None:
        self._all.append((level, text))
        if len(self._all) > self.MAX_LINES:
            del self._all[: len(self._all) - self.MAX_LINES]
        if self._passes(level, text):
            self._write(level, text)

    def _passes(self, level: str, text: str) -> bool:
        f = self.filter.text().strip().lower()
        if f and f not in text.lower():
            return False
        if self.only_problems.isChecked():
            low = text.lower()
            if level not in ("warn", "error", "job-err") and not any(k in low for k in ("warn", "error", "fail", "exception", "traceback")):
                return False
        return True

    def _write(self, level: str, text: str) -> None:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(COLORS.get(level, theme.TEXT)))
        cur = self.view.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(text if text.endswith("\n") else text + "\n", fmt)
        if self.autoscroll.isChecked():
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().maximum())

    def _refilter(self) -> None:
        self.view.clear()
        for level, text in self._all:
            if self._passes(level, text):
                self._write(level, text)

    def clear(self) -> None:
        self._all.clear()
        self.view.clear()
