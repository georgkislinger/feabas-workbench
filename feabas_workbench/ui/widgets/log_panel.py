from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .. import theme

COLORS = {
    "info": theme.TEXT,
    "warn": theme.WARN,
    "error": theme.ERR,
    "job": "#B6C2D1",
    "job-err": theme.ERR,
}

# How much of the log to show. The workbench's own lines (started, finished, wrote N files, its
# warnings and errors) are few and always shown; the levels differ in how much of what the
# FEABAS / worker processes print gets through.
DETAIL_LEVELS = (
    ("messages", "messages only", "The workbench's own lines and every error a process prints (tracebacks, "
                                       "'failed …'). FEABAS's progress lines and warnings stay hidden."),
    ("warnings", "messages + warnings", "Also the warning lines of FEABAS and the workers (Python warnings, "
                                       "'WARNING:' log lines, OpenCV notices)."),
    ("full", "full log", "Everything the processes print, progress lines included."),
)
_ERROR_WORDS = ("error", "fail", "exception", "traceback", "critical")


def severity(level: str, text: str) -> str:
    """'info' | 'warn' | 'error' for a workbench line; a process line is classed by content and
    otherwise 'job' (plain output)."""
    if level in ("error", "job-err"):
        return "error"
    if level in ("info", "warn"):
        return level
    low = text.lower()
    if any(k in low for k in _ERROR_WORDS):
        return "error"
    if "warn" in low:
        return "warn"
    return "job"


class LogPanel(QWidget):
    """Application + subprocess log with a detail level and a text filter. Keeps the last ~5000 lines."""

    MAX_LINES = 5000
    detail_changed = Signal(str)     # key from DETAIL_LEVELS

    def __init__(self, parent=None, detail: str = "full"):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter (substring)")
        self.filter.setMaximumWidth(260)
        self.detail = QComboBox()
        for key, label, tip in DETAIL_LEVELS:
            self.detail.addItem(label, key)
            self.detail.setItemData(self.detail.count() - 1, tip, Qt.ToolTipRole)
        self.detail.setToolTip("How much of the log to show. The workbench's own lines are always there; "
                               "the levels add what FEABAS and the workers print.")
        self.set_detail(detail)
        self.autoscroll = QCheckBox("autoscroll")
        self.autoscroll.setChecked(True)
        self.autoscroll.setToolTip("Follow new lines as they arrive. Untick to read while a job keeps writing; "
                                   "the view then stays where it is.")
        clear = QPushButton("Clear")
        clear.setObjectName("Flat")
        top.addWidget(self.filter)
        top.addWidget(self.detail)
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
        self.detail.currentIndexChanged.connect(self._detail_picked)

    # -- detail level ---------------------------------------------------
    def current_detail(self) -> str:
        return self.detail.currentData()

    def set_detail(self, key: str) -> None:
        i = self.detail.findData(key)
        self.detail.setCurrentIndex(i if i >= 0 else self.detail.count() - 1)

    def _detail_picked(self) -> None:
        self._refilter()
        self.detail_changed.emit(self.current_detail())

    # -- lines ----------------------------------------------------------
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
        detail = self.current_detail()
        if detail == "full":
            return True
        sev = severity(level, text)
        if level in ("info", "warn", "error") or sev == "error":
            return True
        return detail == "warnings" and sev == "warn"

    def _write(self, level: str, text: str) -> None:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(COLORS.get(level, theme.TEXT)))
        sb = self.view.verticalScrollBar()
        doc = self.view.document()
        before_value, before_blocks = sb.value(), doc.blockCount()
        # a cursor on the document, not the widget's own: moving the widget cursor makes the view
        # follow it, which is exactly the scrolling that 'autoscroll off' is meant to stop
        cur = QTextCursor(doc)
        cur.movePosition(QTextCursor.End)
        cur.insertText(text if text.endswith("\n") else text + "\n", fmt)
        if self.autoscroll.isChecked():
            sb.setValue(sb.maximum())
        else:
            # keep the same lines in view; when the block limit trims lines at the top, the
            # remaining lines move up by that many, so the scroll position follows them
            added = text.count("\n") or 1
            trimmed = max(0, before_blocks + added - doc.blockCount())
            sb.setValue(max(0, before_value - trimmed))

    def _refilter(self) -> None:
        self.view.clear()
        for level, text in self._all:
            if self._passes(level, text):
                self._write(level, text)

    def clear(self) -> None:
        self._all.clear()
        self.view.clear()
