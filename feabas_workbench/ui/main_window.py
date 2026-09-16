"""Main window: step navigation on the left, one page at a time, log at the bottom."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget, QSplitter)

from .. import APP_NAME, __version__
from ..core.envs import Settings
from ..core.steps import STEPS, STEPS_BY_KEY, clear_targets, clear_step, create_snapshot, list_snapshots, restore_snapshot, estimate_snapshot_size
from .bridge import AppContext
from .widgets.log_panel import LogPanel
from . import theme


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.ctx = AppContext(settings, self)
        self._pages = []
        self._build()
        self._wire()
        self.ctx.log(f"{APP_NAME} {__version__} started")
        QTimer.singleShot(200, self._first_run)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        from .pages.page_setup import SetupPage
        from .pages.page_project import ProjectPage
        from .pages.page_preprocess import PreprocessPage
        from .pages.page_stitch import StitchPage
        from .pages.page_masks import MasksPage
        from .pages.page_align import AlignPage
        from .pages.page_export import ExportPage

        central = QWidget()
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.setFixedWidth(200)
        self.stack = QStackedWidget()
        for cls in (SetupPage, ProjectPage, PreprocessPage, StitchPage, MasksPage, AlignPage, ExportPage):
            page = cls(self.ctx)
            self._pages.append(page)
            self.stack.addWidget(page)
            it = QListWidgetItem(f"{len(self._pages) - 1}  {page.title}")
            it.setToolTip(page.subtitle)
            self.nav.addItem(it)
        self.nav.currentRowChanged.connect(self._nav_changed)

        left = QWidget()
        left.setFixedWidth(200)
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(0)
        self.project_label = QLabel("no project")
        self.project_label.setObjectName("Hint")
        self.project_label.setWordWrap(True)
        self.project_label.setContentsMargins(12, 10, 12, 10)
        self.project_label.setStyleSheet(f"background: {theme.PANEL}; border-bottom: 1px solid {theme.BORDER};")
        llay.addWidget(self.project_label)
        llay.addWidget(self.nav, 1)
        lay.addWidget(left)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        # log dock: hidden until something worth reading arrives, so the workflow pages
        # keep the whole window. View > Log (Ctrl+L) toggles it; it also opens on its
        # own for errors and while a job runs, and the status bar mirrors the last line.
        self.log_panel = LogPanel()
        self.ctx.add_log_sink(self.log_panel.append)
        self.ctx.add_log_sink(self._on_log)
        dock = QDockWidget("Log", self)
        dock.setObjectName("LogDock")
        dock.setWidget(self.log_panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.setMinimumHeight(160)
        dock.setVisible(False)
        self.log_dock = dock

        # status bar
        sb = self.statusBar()
        self.job_label = QLabel("idle")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(260)
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel job")
        self.cancel_btn.setObjectName("Danger")
        self.cancel_btn.setVisible(False)
        sb.addWidget(self.job_label, 1)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.cancel_btn)

        self._build_menu()
        self.nav.setCurrentRow(1)

    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&Project")
        a = QAction("&New project…", self); a.setShortcut(QKeySequence.New); a.triggered.connect(self.new_project); m.addAction(a)
        a = QAction("&Open project…", self); a.setShortcut(QKeySequence.Open); a.triggered.connect(self.choose_project); m.addAction(a)
        self.recent_menu = m.addMenu("Open &recent")
        m.addSeparator()
        a = QAction("&Save project", self); a.setShortcut(QKeySequence.Save); a.triggered.connect(self.ctx.save_project); m.addAction(a)
        a = QAction("Show project folder", self); a.triggered.connect(self._reveal); m.addAction(a)
        m.addSeparator()
        a = QAction("&Quit", self); a.setShortcut(QKeySequence.Quit); a.triggered.connect(self.close); m.addAction(a)

        p = mb.addMenu("&Pipeline")
        a = QAction("Create &snapshot of current state…", self); a.triggered.connect(self.make_snapshot); p.addAction(a)
        a = QAction("&Restore snapshot…", self); a.triggered.connect(self.restore_snapshot_dialog); p.addAction(a)
        p.addSeparator()
        a = QAction("&Clear a step and everything after it…", self); a.triggered.connect(self.clear_step_dialog); p.addAction(a)
        a = QAction("Re-read pipeline state", self); a.setShortcut("F5"); a.triggered.connect(self.ctx.state_changed.emit); p.addAction(a)

        v = mb.addMenu("&View")
        self.log_action = QAction("&Log", self)
        self.log_action.setShortcut("Ctrl+L")
        self.log_action.setCheckable(True)
        self.log_action.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(self.log_action.setChecked)
        v.addAction(self.log_action)

        h = mb.addMenu("&Help")
        a = QAction("FEABAS on GitHub", self); a.triggered.connect(lambda: self._open_url("https://github.com/YuelongWu/feabas")); h.addAction(a)
        a = QAction("About", self); a.triggered.connect(self._about); h.addAction(a)
        self._refresh_recent()

    def _wire(self) -> None:
        self.ctx.project_changed.connect(self._on_project)
        self.ctx.jobs.job_started.connect(self._job_started)
        self.ctx.jobs.progress.connect(self._job_progress)
        self.ctx.jobs.job_finished.connect(self._job_finished)
        self.ctx.jobs.running_changed.connect(self._running)
        self.cancel_btn.clicked.connect(self._cancel)

    # ------------------------------------------------------------------
    def page_count(self) -> int:
        return len(self._pages)

    def show_page(self, index: int) -> None:
        self.nav.setCurrentRow(max(0, min(index, len(self._pages) - 1)))

    def apply_interface_settings(self) -> None:
        """Global interface switches (Setup page) that pages read, e.g. optional tabs."""
        for page in self._pages:
            fn = getattr(page, "apply_interface_settings", None)
            if fn:
                fn()

    def _nav_changed(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        try:
            self._pages[row].on_shown()
        except Exception as e:  # noqa: BLE001
            self.ctx.log(f"page error: {e}", "error")

    def _first_run(self) -> None:
        # Fires 200 ms after start. If the window was already shut down by then (a test that
        # builds and closes a window quickly, or a user closing it at once), showing the Setup
        # page would start its environment-probe thread on a dead window and nothing would
        # stop it: Qt aborts the process when such a QThread is destroyed while running.
        if getattr(self, "_shut_down", False):
            return
        s = self.ctx.settings
        if self.ctx.project is not None:
            return
        if not s.feabas_python:
            self.ctx.log("No FEABAS environment configured yet: open the Setup page to detect or install one.", "warn")
            self.show_page(0)
        elif s.last_project and Path(s.last_project).is_dir():
            self.open_project(Path(s.last_project))

    # -- project -------------------------------------------------------
    def new_project(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Choose an (ideally empty) folder for the new project")
        if p:
            self.open_project(Path(p))

    def choose_project(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Open project folder")
        if p:
            self.open_project(Path(p))

    def open_project(self, path: Path) -> None:
        try:
            self.ctx.open_project(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open project", str(e))
            self.ctx.log(f"could not open {path}: {e}", "error")
            return
        self._refresh_recent()
        if self.nav.currentRow() == 0:
            self.show_page(1)

    def _on_project(self, project) -> None:
        if project is None:
            self.project_label.setText("no project")
            self.setWindowTitle(f"{APP_NAME} {__version__}")
        else:
            root = str(project.root)
            if len(root) > 34:
                root = "…" + root[-33:]
            self.project_label.setText(f"<b>{project.state.name}</b><br><span style='color:{theme.MUTED}'>{root}</span>")
            self.project_label.setToolTip(str(project.root))
            self.setWindowTitle(f"{APP_NAME} {__version__}  –  {project.state.name}")

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        for p in self.ctx.settings.recent_projects:
            a = QAction(p, self)
            a.triggered.connect(lambda _c=False, pp=p: self.open_project(Path(pp)))
            self.recent_menu.addAction(a)

    def _reveal(self) -> None:
        if self.ctx.project:
            self._open_url(self.ctx.project.root.as_uri())

    @staticmethod
    def _open_url(url: str) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(url))

    def _about(self) -> None:
        QMessageBox.about(self, "About", f"<b>{APP_NAME} {__version__}</b><br>A desktop workbench around FEABAS "
                                         f"(Yuelong Wu, MIT licence) for stitching and alignment of serial-section EM.<br>"
                                         f"Vendored FEABAS 3.0.5 scripts; steps run in your own FEABAS environment.")

    # -- jobs ----------------------------------------------------------
    def _on_log(self, level: str, text: str) -> None:
        """Keep one log line in the status bar and open the dock when it matters."""
        line = text.splitlines()[0] if text.splitlines() else ""
        if len(line) > 160:
            line = line[:157] + "…"
        self.statusBar().showMessage(line, 10000)
        if level == "error":
            self.log_dock.setVisible(True)

    def _job_started(self, spec) -> None:
        self.job_label.setText(f"running: {spec.name}")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.cancel_btn.setVisible(True)
        self.ctx.log(f"started: {spec.name}")
        self.log_dock.setVisible(True)   # a job's output lives in the log; show it while it runs

    def _job_progress(self, done: int, expected: int, msg: str) -> None:
        if expected > 0:
            self.progress.setRange(0, expected)
            self.progress.setValue(min(done, expected))
            self.progress.setFormat(f"{done}/{expected} {msg}".strip())
        else:
            self.progress.setRange(0, 0)
            if msg:
                self.progress.setFormat(msg)

    def _job_finished(self, res) -> None:
        status = "cancelled" if res.cancelled else ("finished" if res.ok else f"FAILED (exit {res.exit_code})")
        level = "info" if res.ok else "error"
        self.ctx.log(f"{status}: {res.spec.name} after {res.seconds:.0f}s", level)
        if not res.ok and not res.cancelled:
            tail = "\n".join(res.last_lines[-12:])
            QMessageBox.warning(self, "Job failed", f"{res.spec.name} failed (exit code {res.exit_code}).\n\nLast output:\n{tail}")

    def _running(self, running: bool) -> None:
        if not running:
            self.job_label.setText("idle")
            self.progress.setVisible(False)
            self.cancel_btn.setVisible(False)

    def _cancel(self) -> None:
        if QMessageBox.question(self, "Cancel", "Stop the running job (and skip queued ones)?") == QMessageBox.Yes:
            self.ctx.jobs.cancel_all()

    # -- pipeline utilities ------------------------------------------------
    def make_snapshot(self) -> None:
        if not self.ctx.project:
            return
        size_mb = estimate_snapshot_size(self.ctx.project.root) / 1e6
        label, ok = QInputDialog.getText(self, "Snapshot", f"Label for this snapshot (about {size_mb:.0f} MB will be copied):")
        if not ok:
            return
        d = create_snapshot(self.ctx.project.root, label, log=lambda t: self.ctx.log(t))
        self.ctx.log(f"snapshot created: {d}")

    def restore_snapshot_dialog(self) -> None:
        if not self.ctx.project:
            return
        snaps = list_snapshots(self.ctx.project.root)
        if not snaps:
            QMessageBox.information(self, "Snapshots", "No snapshots yet (Pipeline → Create snapshot).")
            return
        names = [f"{s['name']}  {s.get('label', '')}" for s in snaps]
        choice, ok = QInputDialog.getItem(self, "Restore snapshot", "Snapshot:", names, len(names) - 1, False)
        if not ok:
            return
        snap = snaps[names.index(choice)]
        if QMessageBox.question(self, "Restore", f"Replace the current matches/meshes/transforms/configs with snapshot {snap['name']}?\n"
                                                 f"Rendered images are not part of snapshots.") != QMessageBox.Yes:
            return
        restore_snapshot(self.ctx.project.root, snap["path"], log=lambda t: self.ctx.log(t))
        self.ctx.reload_configs()
        self.ctx.state_changed.emit()

    def clear_step_dialog(self, step=None) -> None:
        if not self.ctx.project:
            return
        if step is None:
            labels = [s.label for s in STEPS]
            choice, ok = QInputDialog.getItem(self, "Clear step", "Clear outputs of this step and every later step:", labels, 0, False)
            if not ok:
                return
            step = STEPS[labels.index(choice)]
        self.confirm_clear(step)

    def confirm_clear(self, step, cascade: bool = True) -> bool:
        targets = clear_targets(self.ctx.project.root, step, cascade)
        if not targets:
            QMessageBox.information(self, "Nothing to clear", f"No outputs of '{step.label}' found.")
            return False
        listing = "\n".join(str(t.relative_to(self.ctx.project.root)) for t in targets[:30])
        if len(targets) > 30:
            listing += f"\n… and {len(targets) - 30} more"
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Warning)
        dlg.setWindowTitle("Clear outputs")
        dlg.setText(f"Delete the outputs of '{step.label}'" + (" and all later steps" if cascade else "") + "?")
        dlg.setInformativeText("Consider creating a snapshot first (Pipeline menu). This removes:\n\n" + listing)
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setDefaultButton(QMessageBox.No)
        if dlg.exec() != QMessageBox.Yes:
            return False
        removed = clear_step(self.ctx.project.root, step, cascade, log=lambda t: self.ctx.log(t))
        self.ctx.log(f"cleared {len(removed)} item(s) for '{step.label}'")
        self.ctx.state_changed.emit()
        return True

    def shutdown(self) -> None:
        """Stop every page's background threads and persist state. Idempotent: it also
        runs on the --screenshot path, which quits without ever closing the window."""
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True
        self.ctx.jobs.cancel_all()
        for page in self._pages:
            try:
                page.shutdown()
            except Exception as e:  # noqa: BLE001 - never let teardown raise
                self.ctx.log(f"error stopping {type(page).__name__}: {e}", "warn")
        self.ctx.save_project()
        self.ctx.settings.save()

    def closeEvent(self, event) -> None:
        if self.ctx.jobs.running:
            if QMessageBox.question(self, "Quit", "A job is running. Stop it and quit?") != QMessageBox.Yes:
                event.ignore()
                return
        self.shutdown()
        event.accept()
