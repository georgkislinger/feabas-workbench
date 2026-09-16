"""A row for one pipeline step: state, progress, run/clear/inspect controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

from ...core.steps import Step, StepStatus, State
from .. import theme


class StateBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fraction = 0.0
        self._color = QColor(theme.BLOCKED)
        self.setFixedHeight(4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_state(self, fraction: float, color: str) -> None:
        self._fraction = max(0.0, min(1.0, fraction))
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        w = int(self.width() * self._fraction)
        if w > 0:
            p.fillRect(0, 0, w, self.height(), self._color)
        p.end()


class StepCard(QFrame):
    run_requested = Signal(object, object, object, object)   # step, start, stop, stride
    clear_requested = Signal(object)
    inspect_requested = Signal(object)
    errors_clear_requested = Signal(object)

    def __init__(self, step: Step, compact: bool = False, parent=None, label: str | None = None):
        super().__init__(parent)
        self.step = step
        self.status: StepStatus | None = None
        self.setObjectName("SubCard")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(4)
        head = QHBoxLayout()
        self.name = QLabel(label or step.label)      # a step can read differently in another context
        self.name.setStyleSheet(f"font-weight: 600; color: {theme.TEXT};")
        self.badge = QLabel("")
        self.badge.setStyleSheet(f"color: {theme.MUTED};")
        self.count = QLabel("")
        self.count.setObjectName("Mono")
        self.count.setStyleSheet(f"color: {theme.MUTED};")
        head.addWidget(self.name)
        head.addWidget(self.badge)
        head.addStretch(1)
        head.addWidget(self.count)
        root.addLayout(head)
        self.bar = StateBar()
        root.addWidget(self.bar)
        if not compact:
            blurb = QLabel(step.blurb)
            blurb.setObjectName("Hint")
            blurb.setWordWrap(True)
            root.addWidget(blurb)
        ctl = QHBoxLayout()
        ctl.setSpacing(6)
        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("Primary")
        self.range_chk = QCheckBox("subset")
        self.range_chk.setToolTip("Run only sections [start, stop) with the given step (FEABAS --start/--stop/--step).")
        self.start = QSpinBox(); self.start.setRange(0, 999999); self.start.setPrefix("from "); self.start.setMaximumWidth(110)
        self.stop = QSpinBox(); self.stop.setRange(0, 999999); self.stop.setPrefix("to "); self.stop.setMaximumWidth(110)
        self.stride = QSpinBox(); self.stride.setRange(1, 999); self.stride.setPrefix("every "); self.stride.setMaximumWidth(100)
        for w in (self.start, self.stop, self.stride):
            w.setVisible(False)
        self.range_chk.toggled.connect(self._toggle_range)
        self.inspect_btn = QPushButton("Inspect")
        self.errors_btn = QPushButton("Remove error files")
        self.errors_btn.setVisible(False)
        self.clear_btn = QPushButton("Clear…")
        self.clear_btn.setObjectName("Danger")
        ctl.addWidget(self.run_btn)
        if step.supports_range:
            ctl.addWidget(self.range_chk)
            ctl.addWidget(self.start)
            ctl.addWidget(self.stop)
            ctl.addWidget(self.stride)
        ctl.addStretch(1)
        ctl.addWidget(self.errors_btn)
        ctl.addWidget(self.inspect_btn)
        ctl.addWidget(self.clear_btn)
        root.addLayout(ctl)
        self.reason = QLabel("")
        self.reason.setObjectName("Hint")
        self.reason.setWordWrap(True)
        self.reason.setVisible(False)
        root.addWidget(self.reason)
        self.run_btn.clicked.connect(self._run)
        self.clear_btn.clicked.connect(lambda: self.clear_requested.emit(self.step))
        self.inspect_btn.clicked.connect(lambda: self.inspect_requested.emit(self.step))
        self.errors_btn.clicked.connect(lambda: self.errors_clear_requested.emit(self.step))
        if step.local:
            self.run_btn.setText("Open")

    def _toggle_range(self, on: bool) -> None:
        for w in (self.start, self.stop, self.stride):
            w.setVisible(on)

    def _run(self) -> None:
        if self.range_chk.isChecked():
            self.run_requested.emit(self.step, self.start.value(), self.stop.value() or None, self.stride.value())
        else:
            self.run_requested.emit(self.step, None, None, None)

    def set_status(self, st: StepStatus | None, busy: bool = False) -> None:
        self.status = st
        if st is None:
            self.badge.setText("")
            self.count.setText("")
            self.bar.set_state(0, theme.BLOCKED)
            self.run_btn.setEnabled(False)
            return
        color = theme.STATE_COLORS.get(st.state.value, theme.BLOCKED)
        self.badge.setText(f"● {st.state.value}")
        self.badge.setStyleSheet(f"color: {color};")
        self.count.setText(st.summary())
        self.bar.set_state(st.fraction if st.step.cardinality.value != "single" else float(st.done > 0), color)
        self.run_btn.setEnabled(not busy and st.state is not State.BLOCKED)
        self.clear_btn.setEnabled(not busy and st.done > 0)
        self.inspect_btn.setEnabled(st.done > 0)
        self.errors_btn.setVisible(st.errors > 0)
        self.errors_btn.setEnabled(not busy)
        if st.reasons:
            self.reason.setText("; ".join(dict.fromkeys(st.reasons)))
            self.reason.setVisible(True)
        else:
            self.reason.setVisible(False)
        self.stop.setMaximum(max(1, st.expected))
        if self.stop.value() == 0:
            self.stop.setValue(st.expected)
