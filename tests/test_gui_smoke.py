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
    """The Project page on real example data when present, else on a synthetic Thermo-named dataset."""
    if EXAMPLE.is_dir():
        src, tile_w, pix, n_tiles, n_sections, first = EXAMPLE, 6144, 10.0, 400, 10, "s0035"
    else:
        from feabas_workbench.core.synthetic import make_synthetic_tiles
        facts = make_synthetic_tiles(tmp_path / "tiles", n_sections=3, rows=2, cols=2, tile=128, first_section=35)
        src, tile_w, pix, n_tiles, n_sections, first = tmp_path / "tiles", 128, None, facts["n_tiles"], 3, "s0035"
    proj = tmp_path / "proj"
    window.open_project(proj)
    _pump(app)
    page = window._pages[1]
    page.src.setText(str(src))
    page._guess()
    _pump(app)
    assert page.guesses.count() > 0
    page._use_guess()
    page._read_meta()
    assert page.tile_w.value() == tile_w
    if pix is not None:
        assert page.pix.value() == pytest.approx(pix)
        page._estimate_overlap()
        assert page.ov_x.value() > 100
    page._scan()
    # wait for the scan thread
    import time
    t0 = time.time()
    while page.plan is None and time.time() - t0 < 120:
        _pump(app)
        time.sleep(0.05)
    assert page.plan is not None and page.plan.n_tiles == n_tiles
    page._write()
    _pump(app)
    assert len(list(proj.glob("stitch/stitch_coord/*.txt"))) == n_sections
    assert not window.errors, window.errors
    # other pages react to the project
    for i in range(window.page_count()):
        window.show_page(i)
        _pump(app)
    assert not window.errors, window.errors
    # stitching test run creation
    st = window._pages[3]
    st.test_sections.set_checked({first})
    st.test_name.setText("smoke")
    st._create_test()
    _pump(app)
    assert (proj / "tests" / "smoke" / "stitch" / "stitch_coord" / f"{first}.txt").is_file()
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


def test_masks_page_shows_only_the_chosen_methods_settings(app, window, tmp_path):
    """Each tissue method has its own row; the shared rows stay."""
    window.open_project(tmp_path / "proj_masks")
    _pump(app)
    page = next(p for p in window._pages if p.key == "masks")
    window.show_page(window._pages.index(page)); page.tabs.setCurrentIndex(1); _pump(app)
    vis = lambda w: not w.isHidden()
    page.ti_method.setCurrentIndex(page.ti_method.findData("all")); _pump(app)
    assert not vis(page.ti_border_row) and not vis(page.ti_manual_row) and not vis(page.ti_texture_row) and not vis(page.ti_cleanup_row)
    page.ti_method.setCurrentIndex(page.ti_method.findData("border")); _pump(app)
    assert vis(page.ti_border_row) and vis(page.ti_cleanup_row) and not vis(page.ti_manual_row)
    assert vis(page.ti_border_black_w) and vis(page.ti_border_white_w) and not vis(page.ti_border_tol_w)
    page.ti_border_mode.setCurrentIndex(page.ti_border_mode.findData("auto")); _pump(app)
    assert vis(page.ti_border_tol_w) and not vis(page.ti_border_black_w)
    page.ti_method.setCurrentIndex(page.ti_method.findData("manual")); _pump(app)
    assert vis(page.ti_manual_row) and not vis(page.ti_cleanup_row)
    page.ti_method.setCurrentIndex(page.ti_method.findData("intensity")); _pump(app)
    assert vis(page.ti_texture_row) and vis(page.ti_invert) and not vis(page.ti_window_w)
    page.ti_left.setValue(12); page.ti_bright.setChecked(True); page.ti_bright_min.setValue(240)
    tp = page._tissue_params()
    assert tp.method == "intensity" and tp.crop_left == 12 and tp.exclude_bright and tp.bright_min == 240
    page.fold_method.setCurrentIndex(page.fold_method.findData("unet")); _pump(app)
    assert vis(page.fold_unet_row) and vis(page.fold_adv) and not vis(page.fold_dark_w)
    assert page.tabs.tabText(0).startswith("1.") and page.tabs.tabText(1).startswith("2.")
    assert not window.errors


def test_structure_tab_follows_the_setup_switch(app, window):
    page = next(p for p in window._pages if p.key == "align")
    i = page.tabs.indexOf(page._structure_tab)
    assert i >= 0 and not page.tabs.isTabVisible(i)
    assert page.tabs.tabText(0).startswith("1.") and page.tabs.tabText(1).startswith("2.")
    setup = window._pages[0]
    setup.show_struct.setChecked(True); _pump(app)
    assert page.tabs.isTabVisible(i) and window.ctx.settings.show_structure_tab
    setup.show_struct.setChecked(False); _pump(app)
    assert not page.tabs.isTabVisible(i)
    assert not window.errors


def test_denoise_card_hides_advanced_settings(app, window):
    page = next(p for p in window._pages if p.key == "preprocess")
    assert page.dn_adv.content.isHidden() and page.dn_struct_row.isHidden()
    page.dn_adv.set_expanded(True); _pump(app)
    assert not page.dn_adv.content.isHidden()
    page.dn_method.setCurrentIndex(page.dn_method.findData("structn2v")); _pump(app)
    assert not page.dn_struct_row.isHidden()


def test_log_panel_stays_put_without_autoscroll(app):
    from feabas_workbench.ui.widgets.log_panel import LogPanel
    panel = LogPanel()
    panel.resize(400, 120); panel.show(); _pump(app)
    for i in range(300):
        panel.append("info", f"line {i}")
    _pump(app)
    sb = panel.view.verticalScrollBar()
    assert sb.value() == sb.maximum()
    panel.autoscroll.setChecked(False)
    sb.setValue(10); _pump(app)
    for i in range(300, 400):
        panel.append("info", f"line {i}")
    _pump(app)
    assert sb.value() == 10
    panel.autoscroll.setChecked(True)
    panel.append("info", "last"); _pump(app)
    assert sb.value() == sb.maximum()


def test_standard_pipeline_dialog_lists_what_is_left(app, window, tmp_path):
    """Pipeline > Run the standard pipeline: queues the steps that are not done, in order."""
    from feabas_workbench.core.synthetic import make_demo_project
    from feabas_workbench.ui.dialogs import StandardPipelineDialog
    p = make_demo_project(tmp_path / "demo", n_sections=2, rows=2, cols=2, tile=128)
    window.open_project(p.root)
    _pump(app)
    dlg = StandardPipelineDialog(window.ctx, window)
    steps = dlg.steps_to_run()
    assert [s.key for s in steps] == ["stitch.matching", "stitch.optimization", "stitch.rendering", "thumbnail.downsample",
                                      "thumbnail.matching", "thumbnail.optimization", "thumbnail.render",
                                      "align.meshing", "align.matching", "align.optimization"]
    assert dlg.tree.topLevelItemCount() == 10 and "10 step(s)" in dlg.summary.text()
    dlg.render.setChecked(True)
    assert [s.key for s in dlg.steps_to_run()][-2:] == ["align.rendering", "align.downsample"]
    assert "mip 0" in dlg.mips.text()                       # the tiny synthetic sections get thumbnail mip 0
    dlg.stop_for_masks.setChecked(True)
    assert [s.key for s in dlg.steps_to_run()][-1] == "thumbnail.downsample" and "run the pipeline again" in dlg.summary.text()
    dlg.stop_for_masks.setChecked(False)
    # a finished step is skipped: fake the outputs of 'Match tiles'
    for sec in p.section_names():
        (p.root / "stitch" / "match_h5").mkdir(parents=True, exist_ok=True)
        (p.root / "stitch" / "match_h5" / f"{sec}.h5").write_bytes(b"")
    dlg.render.setChecked(False)
    assert [s.key for s in dlg.steps_to_run()][0] == "stitch.optimization"
    # a stale step keeps its outputs and is not queued either; the summary explains how to redo it
    (p.root / "configs" / "stitching_configs.yaml").write_text("matching: {num_workers: 2}\n", encoding="utf-8")
    import os, time
    for sec in p.section_names():
        os.utime(p.root / "stitch" / "match_h5" / f"{sec}.h5", (time.time() - 100, time.time() - 100))
    window.ctx.reload_configs()
    dlg._refresh()
    assert [s.key for s in dlg.steps_to_run()][0] == "stitch.optimization" and "stale" in dlg.summary.text()
    dlg.close()
    assert not window.errors, window.errors


def test_help_menu_and_about_cite_feabas(app, window, monkeypatch):
    """Help links to the FEABAS repository, the FEABAS paper and the workbench; About cites the paper."""
    from PySide6.QtWidgets import QMenu, QMessageBox
    from feabas_workbench import FEABAS_REPO, FEABAS_PAPER, WORKBENCH_REPO
    help_menu = next(m for m in window.menuBar().findChildren(QMenu) if m.title() == "&Help")
    titles = [a.text() for a in help_menu.actions() if not a.isSeparator()]
    assert titles[:3] == ["FEABAS on GitHub", "FEABAS paper (Wu && Lichtman, 2026)", "Workbench on GitHub"]
    opened: list[str] = []
    monkeypatch.setattr(type(window), "_open_url", staticmethod(opened.append))
    for a in help_menu.actions()[:3]:
        a.trigger()
    assert opened == [FEABAS_REPO, FEABAS_PAPER, WORKBENCH_REPO]
    shown: list[str] = []
    monkeypatch.setattr(QMessageBox, "about", staticmethod(lambda parent, title, text: shown.append(text)))
    window._about()
    assert "10.64898/2026.06.07.730510" in shown[0] and "Lichtman" in shown[0] and WORKBENCH_REPO in shown[0]
