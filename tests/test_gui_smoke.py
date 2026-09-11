"""
Offscreen GUI smoke test: builds the main window, opens a temporary project on
the example data (if present), drives the page handlers that do not need
external environments, and fails on any error logged.

Run: python -m pytest tests/test_gui_smoke.py -q
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "Example_data_to_stitch_and_align_and_export"


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    a = QApplication.instance() or QApplication([])
    yield a


@pytest.fixture()
def window(app, tmp_path, monkeypatch):
    from feabas_workbench.core import envs
    monkeypatch.setattr(envs, "SETTINGS_FILE", tmp_path / "settings.json")
    from feabas_workbench.ui.main_window import MainWindow
    s = envs.Settings()
    s.feabas_python = ""
    w = MainWindow(s)
    errors: list[str] = []
    w.ctx.add_log_sink(lambda level, text: errors.append(text) if level == "error" else None)
    w.errors = errors
    w.show()
    app.processEvents()
    yield w
    w.close()
    app.processEvents()


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def test_pages_build_without_project(app, window):
    for i in range(window.page_count()):
        window.show_page(i)
        _pump(app)
    assert not window.errors


def test_project_page_scan_and_write(app, window, tmp_path):
    if not EXAMPLE.is_dir():
        pytest.skip("example data not present")
    proj = tmp_path / "proj"
    window.open_project(proj)
    _pump(app)
    page = window._pages[1]
    page.src.setText(str(EXAMPLE))
    page._guess()
    _pump(app)
    assert page.guesses.count() > 0
    page._use_guess()
    page._read_meta()
    assert page.tile_w.value() == 6144 and page.pix.value() == pytest.approx(10.0)
    page._estimate_overlap()
    assert page.ov_x.value() > 100
    page._scan()
    # wait for the scan thread
    import time
    t0 = time.time()
    while page.plan is None and time.time() - t0 < 120:
        _pump(app)
        time.sleep(0.05)
    assert page.plan is not None and page.plan.n_tiles == 400
    page._write()
    _pump(app)
    assert len(list(proj.glob("stitch/stitch_coord/*.txt"))) == 10
    assert not window.errors, window.errors
    # other pages react to the project
    for i in range(window.page_count()):
        window.show_page(i)
        _pump(app)
    assert not window.errors, window.errors
    # stitching test run creation
    st = window._pages[3]
    st.test_sections.set_checked({"s0035"})
    st.test_name.setText("smoke")
    st._create_test()
    _pump(app)
    assert (proj / "tests" / "smoke" / "stitch" / "stitch_coord" / "s0035.txt").is_file()
    assert not window.errors, window.errors
    # the quality-check tab must offer the new test run as a source right away
    assert st.qc_source.findData(str(proj / "tests" / "smoke")) >= 0
    # config editor edits persist
    st.editor.set_value("matching.num_workers", 3)
    from feabas_workbench.core.configs import ConfigStore
    assert ConfigStore(proj / "configs").get("stitching", "matching.num_workers") == 3
    # pipeline clear dialog machinery on an empty step reports nothing to clear (no dialog interaction needed)
    from feabas_workbench.core.steps import STEPS_BY_KEY, clear_targets
    assert clear_targets(proj, STEPS_BY_KEY["stitch.matching"]) == []
    shutil.rmtree(proj, ignore_errors=True)


def test_section_picker_keeps_the_selection(app):
    """A refresh happens after every mask computation; it must not re-check every section."""
    from feabas_workbench.ui.widgets.section_picker import SectionPicker
    p = SectionPicker()
    names = [f"s{i:04d}" for i in range(5)]
    p.set_sections(names)
    assert p.checked() == names
    p.set_checked({"s0001", "s0002"})
    p.set_sections(names, {"s0001": "thumb tissue"})
    assert p.checked() == ["s0001", "s0002"]
    p.set_sections(names + ["s0005"])          # new sections are checked, old ones keep their state
    assert p.checked() == ["s0001", "s0002", "s0005"]
    p.list.item(0).setSelected(True)
    p.check_selected(True)
    assert p.checked() == ["s0000", "s0001", "s0002", "s0005"]


def test_thread_runner_calls_back_on_the_gui_thread(app):
    """The callbacks touch widgets, so they must not run in the worker thread."""
    from PySide6.QtCore import QThread, QEventLoop, QTimer
    from feabas_workbench.ui.threads import ThreadRunner

    main = QThread.currentThread()
    seen: dict = {}
    loop = QEventLoop()

    def work(progress=None, cancelled=None):
        progress(1, 1, "")
        return "ok"

    runner = ThreadRunner()
    assert runner.start(work,
                        on_progress=lambda d, t, m: seen.update(progress_main=QThread.currentThread() is main),
                        on_done=lambda res, err: (seen.update(done_main=QThread.currentThread() is main,
                                                              result=res, error=err), loop.quit()))
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    assert seen.get("progress_main") is True and seen.get("done_main") is True
    assert seen.get("result") == "ok" and seen.get("error") == ""
    assert not runner.running
