"""Small layout helpers used by every page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QToolButton, QVBoxLayout, QWidget, QSizePolicy)


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #374151;")
    return f


def card(title: str | None = None, sub: bool = False) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("SubCard" if sub else "Card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 10)
    lay.setSpacing(6)
    if title:
        lay.addWidget(section_label(title))
    return f, lay


def hint(text: str, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Hint")
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


def section_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("Section")
    return lab


def form_row(label: str, widget: QWidget, tip: str = "", stretch: bool = True) -> QWidget:
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lab = QLabel(label)
    lab.setMinimumWidth(150)
    if tip:
        lab.setToolTip(tip)
        widget.setToolTip(tip)
    lay.addWidget(lab)
    lay.addWidget(widget, 1 if stretch else 0)
    if not stretch:
        lay.addStretch(1)
    return row


def spin(lo: int, hi: int, val: int, step: int = 1, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setMaximumWidth(140)
    return s


def dspin(lo: float, hi: float, val: float, step: float = 1.0, decimals: int = 2, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(val)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setMaximumWidth(160)
    return s


def combo(items: list[tuple[str, object]] | list[str], current=None) -> QComboBox:
    c = QComboBox()
    for it in items:
        if isinstance(it, tuple):
            c.addItem(it[0], it[1])
        else:
            c.addItem(it, it)
    if current is not None:
        idx = c.findData(current)
        if idx >= 0:
            c.setCurrentIndex(idx)
    return c


def labelled(label: str, widget: QWidget, tip: str = "") -> QWidget:
    """A label and its widget as one unit, so a row can show or hide them together."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lab = QLabel(label)
    lay.addWidget(lab)
    lay.addWidget(widget)
    if tip:
        lab.setToolTip(tip)
        widget.setToolTip(tip)
    return row


def row_widget(*items: QWidget, stretch: bool = True) -> QWidget:
    """Widgets side by side in a container, so the whole row can be shown/hidden per method."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    for it in items:
        lay.addWidget(it)
    if stretch:
        lay.addStretch(1)
    return w


class Collapsible(QWidget):
    """
    A header button that shows or hides a block of settings ("Advanced settings ▸").

    Most people keep the defaults, so the rarely touched settings of a card live in one of
    these: still one click away, but not in the way.
    """

    def __init__(self, title: str = "Advanced settings", expanded: bool = False, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.button = QToolButton()
        self.button.setObjectName("Collapser")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.button.setAutoRaise(True)
        self.button.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.button, 0, Qt.AlignLeft)
        self.content = QWidget()
        self.layout_ = QVBoxLayout(self.content)
        self.layout_.setContentsMargins(12, 0, 0, 0)
        self.layout_.setSpacing(6)
        self.content.setVisible(expanded)
        lay.addWidget(self.content)
        self.button.toggled.connect(self._toggle)

    def _toggle(self, on: bool) -> None:
        self.content.setVisible(on)
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    def addWidget(self, w: QWidget) -> None:      # noqa: N802 - mirrors QLayout
        self.layout_.addWidget(w)

    def addLayout(self, lay) -> None:             # noqa: N802
        self.layout_.addLayout(lay)

    def set_expanded(self, on: bool) -> None:
        self.button.setChecked(on)


def busy_button(text: str, primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    if primary:
        b.setObjectName("Primary")
    b.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return b
