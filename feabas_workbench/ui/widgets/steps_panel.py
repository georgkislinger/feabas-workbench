"""A column of StepCards for a subset of pipeline steps, wired to the app context."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ...core.steps import STEPS_BY_KEY, Step, PipelineScan, clear_error_files
from .step_card import StepCard


class StepsPanel(QWidget):
    inspect_requested = Signal(object)      # Step
    open_local_requested = Signal(object)   # Step (local steps: the page handles it)

    def __init__(self, ctx, step_keys: list[str], compact: bool = False, parent=None,
                 labels: dict[str, str] | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.cards: dict[str, StepCard] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.run_all = QPushButton("Run all steps below in order")
        self.run_all.setObjectName("Primary")
        self.status = QLabel("")
        self.status.setObjectName("Hint")
        top.addWidget(self.run_all)
        top.addWidget(self.status, 1)
        lay.addLayout(top)
        for k in step_keys:
            step = STEPS_BY_KEY[k]
            card = StepCard(step, compact=compact, label=(labels or {}).get(k))
            card.run_requested.connect(self._run)
            card.clear_requested.connect(self._clear)
            card.inspect_requested.connect(self.inspect_requested.emit)
            card.errors_clear_requested.connect(self._clear_errors)
            self.cards[k] = card
            lay.addWidget(card)
        self.run_all.clicked.connect(self._run_all)
        self.root_override: Path | None = None
        self.tag = ""

    def refresh(self, scan: PipelineScan | None) -> None:
        busy = self.ctx.jobs.running
        for k, card in self.cards.items():
            card.set_status(scan[k] if scan else None, busy=busy)
        self.run_all.setEnabled(scan is not None and not busy)

    def _spec(self, step: Step, start=None, stop=None, stride=None):
        return self.ctx.feabas_step_spec(step, start, stop, stride, root=self.root_override, tag=self.tag)

    def _run(self, step: Step, start, stop, stride) -> None:
        if step.local:
            self.open_local_requested.emit(step)
            return
        if self.ctx.jobs.running:
            self.ctx.log("a job is already running", "warn")
            return
        try:
            spec = self._spec(step, start, stop, stride)
        except RuntimeError as e:
            self._cannot_run(e)
            return
        self.ctx.jobs.submit(spec)

    def _cannot_run(self, err: Exception) -> None:
        """A misconfigured environment is a setting to fix, so say so up front."""
        self.ctx.log(str(err), "error")
        QMessageBox.warning(self, "Cannot run this step", str(err))

    def _run_all(self) -> None:
        if self.ctx.jobs.running:
            return
        specs = []
        for k, card in self.cards.items():
            step = card.step
            if step.local:
                continue
            st = card.status
            if st is not None and st.state.value == "done":
                continue
            try:
                specs.append(self._spec(step))
            except RuntimeError as e:
                self._cannot_run(e)
                return
        if specs:
            self.ctx.jobs.submit(specs)
        else:
            self.ctx.log("nothing to run: all steps are done", "info")

    def _clear(self, step: Step) -> None:
        w = self.window()
        if hasattr(w, "confirm_clear"):
            w.confirm_clear(step)

    def _clear_errors(self, step: Step) -> None:
        root = self.root_override or self.ctx.project.root
        n = len(clear_error_files(root, step))
        self.ctx.log(f"removed {n} error file(s) of '{step.label}'; re-run the step to retry those sections")
        self.ctx.state_changed.emit()
