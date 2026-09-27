"""A column of StepCards for a subset of pipeline steps, wired to the app context."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from ...core.steps import STEPS_BY_KEY, Step, PipelineScan, State, clear_error_files
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
        from ...core.local_parallel import SECTION_STEPS
        self.parallel_keys = [k for k in step_keys if k in SECTION_STEPS]
        self.parallel_button = QPushButton("Local parallelism…")
        self.parallel_button.clicked.connect(self._parallelism)
        self.parallel_button.setVisible(bool(self.parallel_keys) and not ctx.cluster_enabled)
        top.addWidget(self.parallel_button)
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
        # called with the step(s) about to run, before their specs are built - a page can
        # write inputs the step reads (e.g. the fine-alignment match list) at the last moment
        self.before_run: Callable[[list[Step]], None] | None = None

    def refresh(self, scan: PipelineScan | None) -> None:
        busy = self.ctx.jobs.running
        self.parallel_button.setVisible(bool(self.parallel_keys) and not self.ctx.cluster_enabled)
        self.parallel_button.setEnabled(self.ctx.project is not None and not busy)
        for k, card in self.cards.items():
            card.set_status(scan[k] if scan else None, busy=busy)
        self.run_all.setEnabled(scan is not None and not busy)

    def _parallelism(self):
        if not self.ctx.project or self.ctx.cluster_enabled:
            return
        from ..local_parallel_dialog import LocalParallelDialog
        LocalParallelDialog(self.ctx, self.parallel_keys, self).exec()

    def _spec(self, step: Step, start=None, stop=None, stride=None):
        return self.ctx.feabas_step_spec(step, start, stop, stride, root=self.root_override, tag=self.tag)

    def _run(self, step: Step, start, stop, stride) -> None:
        if step.local:
            self.open_local_requested.emit(step)
            return
        if self.ctx.jobs.running:
            self.ctx.log("a job is already running", "warn")
            return
        if not self._handle_stale([step]):
            return
        if self.before_run:
            self.before_run([step])
        try:
            spec = self._spec(step, start, stop, stride)
        except RuntimeError as e:
            self._cannot_run(e)
            return
        self.ctx.jobs.submit(spec)

    def _handle_stale(self, steps: list[Step]) -> bool:
        """
        Offer to clear stale outputs before running: FEABAS skips every output that exists, so
        running a stale step as it is recomputes nothing. True means: go on and run.
        """
        stale = [s for s in steps if s.key in self.cards and self.cards[s.key].status is not None
                 and self.cards[s.key].status.state is State.STALE]
        if not stale:
            return True
        first = stale[0]
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Stale outputs")
        more = f" (and {len(stale) - 1} later step(s))" if len(stale) > 1 else ""
        box.setText(f"'{first.label}'{more} has stale outputs: "
                    + "; ".join(dict.fromkeys(self.cards[first.key].status.reasons)) + ".")
        box.setInformativeText("FEABAS skips outputs that already exist, so running now leaves the stale ones as "
                               "they are. Clear them first (this also clears every later step), then run?")
        clear_btn = box.addButton("Clear, then run", QMessageBox.AcceptRole)
        run_btn = box.addButton("Run without clearing", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(clear_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is run_btn:
            return True
        if clicked is not clear_btn:
            return False
        w = self.window()
        if not hasattr(w, "confirm_clear") or not w.confirm_clear(first, root=self.root_override):
            return False
        if self.ctx.cluster_enabled and self.root_override is None:
            self.ctx.log(f"clearing '{first.label}' at LRZ was queued; run the step again once it has finished")
            return False
        return True

    def _cannot_run(self, err: Exception) -> None:
        """A misconfigured environment is a setting to fix, so say so up front."""
        self.ctx.log(str(err), "error")
        QMessageBox.warning(self, "Cannot run this step", str(err))

    def _run_all(self) -> None:
        if self.ctx.jobs.running:
            return
        if not self._handle_stale([c.step for c in self.cards.values() if not c.step.local]):
            return
        if self.before_run:
            self.before_run([c.step for c in self.cards.values() if not c.step.local])
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
            # a test-run panel clears its own sandbox, never the project
            w.confirm_clear(step, root=self.root_override)

    def _clear_errors(self, step: Step) -> None:
        if self.ctx.cluster_enabled:
            self._clear(step)
            return
        root = self.root_override or self.ctx.project.root
        n = len(clear_error_files(root, step))
        self.ctx.log(f"removed {n} error file(s) of '{step.label}'; re-run the step to retry those sections")
        self.ctx.state_changed.emit()
