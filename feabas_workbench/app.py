"""Application entry point."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _excepthook_factory(window_getter):
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.__stderr__.write(text)
        try:
            w = window_getter()
            if w is not None:
                w.ctx.log("Unhandled error:\n" + text, "error")
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(w, "Unexpected error", text[-2000:])
        except Exception:  # noqa: BLE001
            pass
    return hook


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="feabas-workbench")
    ap.add_argument("--project", help="project folder to open")
    ap.add_argument("--page", type=int, default=None, help="page index to show")
    ap.add_argument("--screenshot", help="render every page to PNG files in this folder and exit (debug)")
    ap.add_argument("--offscreen", action="store_true", help="use the offscreen Qt platform (debug)")
    args = ap.parse_args(argv)

    if args.offscreen or args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from . import APP_NAME, ORG_NAME
    from .ui.theme import QSS
    from .ui.main_window import MainWindow
    from .core.envs import Settings

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)

    settings = Settings.load()
    holder = {"w": None}
    prev_excepthook = sys.excepthook
    sys.excepthook = _excepthook_factory(lambda: holder["w"])
    win = MainWindow(settings)
    holder["w"] = win
    win.resize(1500, 950)
    win.show()
    # Never open larger than the screen: a laptop panel (often HiDPI, so a small logical
    # size) can be narrower than 1500 px, and a window wider than its screen pushes the
    # right-hand controls out of reach. Only after show() does win.screen() report the
    # screen the window actually landed on, so shrink here and only when it is needed -
    # on a large monitor the full size is kept. The offscreen platform reports an
    # arbitrary 800x800 virtual screen, so leave --screenshot alone there.
    if app.platformName() != "offscreen":
        avail = win.screen().availableGeometry()
        if win.width() > avail.width() or win.height() > avail.height():
            win.resize(min(win.width(), avail.width()), min(win.height(), avail.height()))

    if args.project:
        win.open_project(Path(args.project))
    if args.page is not None:
        win.show_page(args.page)

    if args.screenshot:
        out = Path(args.screenshot)
        out.mkdir(parents=True, exist_ok=True)

        def shoot():
            for i in range(win.page_count()):
                win.show_page(i)
                app.processEvents()
                win.grab().save(str(out / f"page{i}.png"))
            app.quit()

        QTimer.singleShot(1500, shoot)

    code = app.exec()

    # Teardown, in this order:
    # 1. Pages start QThreads (environment probing, tile scanning). Qt aborts the process
    #    if one is still running when it is destroyed. closeEvent does not fire when the
    #    app quits without the window being closed (--screenshot), so stop them here.
    win.shutdown()
    # 2. Drop every reference that would outlive `app`. The excepthook closes over
    #    `holder`, so leaving it installed keeps the window alive past the end of this
    #    function - and a widget destroyed after its QApplication crashes the process.
    sys.excepthook = prev_excepthook
    holder["w"] = None
    # 3. Destroy the window while the QApplication is still there.
    win.close()
    win.deleteLater()
    app.processEvents()
    del win
    return code


if __name__ == "__main__":
    sys.exit(main())
