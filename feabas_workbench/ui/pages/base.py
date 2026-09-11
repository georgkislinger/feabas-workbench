from __future__ import annotations

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget, QMessageBox

from ..bridge import AppContext
from ..threads import ThreadRunner


class Page(QWidget):
    """One workflow window. Subclasses fill ``self.body`` in ``build()``."""

    title = ""
    subtitle = ""
    key = ""
    needs_project = True

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        # Pages stretch to the window and normally need no horizontal scrolling, but some
        # (Masks) have a minimum width of ~1170 px that does not fit a small screen. With
        # the bar switched off entirely that content was simply unreachable, so show one
        # only when the page really is wider than the viewport.
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(18, 14, 18, 14)
        self.body.setSpacing(10)
        t = QLabel(self.title)
        t.setObjectName("Title")
        s = QLabel(self.subtitle)
        s.setObjectName("Subtitle")
        s.setWordWrap(True)
        self.body.addWidget(t)
        self.body.addWidget(s)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll)
        self.build()
        self.body.addStretch(1)
        ctx.project_changed.connect(self._project_changed)
        ctx.state_changed.connect(self.on_state_changed)
        ctx.jobs.running_changed.connect(self.on_running_changed)

    # -- to override -------------------------------------------------------
    def build(self) -> None:
        pass

    def on_project_changed(self, project) -> None:
        pass

    def on_state_changed(self) -> None:
        pass

    def on_running_changed(self, running: bool) -> None:
        pass

    def on_shown(self) -> None:
        pass

    def shutdown(self) -> None:
        """Stop background threads before the application is torn down.

        Qt aborts the process if a QThread is still running when it is destroyed, so
        every page that starts one must be stopped on the way out. The default handles
        the two shapes used here - a ``ThreadRunner`` in ``self.runner`` and a bare
        ``QThread`` in ``self._thread``; override for anything else.
        """
        runner = getattr(self, "runner", None)
        if isinstance(runner, ThreadRunner):
            runner.stop()
        thread = getattr(self, "_thread", None)
        if isinstance(thread, QThread) and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                thread.terminate()
                thread.wait(1000)
        self._thread = None

    # -- helpers -----------------------------------------------------------
    def _project_changed(self, project) -> None:
        self.on_project_changed(project)
        self.on_state_changed()

    @property
    def project(self):
        return self.ctx.project

    def info(self, text: str) -> None:
        self.ctx.log(text, "info")

    def warn(self, text: str) -> None:
        self.ctx.log(text, "warn")

    def error(self, text: str, dialog: bool = True) -> None:
        self.ctx.log(text, "error")
        if dialog:
            QMessageBox.critical(self, "Error", text)

    def confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes

    def require_project(self) -> bool:
        if self.ctx.project is None:
            QMessageBox.information(self, "No project", "Create or open a project first (Project page).")
            return False
        return True

    def submit(self, specs) -> None:
        if self.ctx.jobs.running:
            QMessageBox.information(self, "Busy", "A job is already running. Wait for it to finish or cancel it.")
            return
        self.ctx.jobs.submit(specs)
