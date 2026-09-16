"""
End-to-end check through the GUI: open a synthetic project in the real MainWindow (offscreen),
accept 'Pipeline > Run the standard pipeline', and let the workbench's own job queue run
every step with the configured FEABAS interpreter.

    python tools/run_demo_pipeline_gui.py D:/demo_gui --feabas-python C:/envs/fw-feabas/python.exe

Exit code 0 only if the queue finished with every job ok and every step has its outputs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--feabas-python", default=sys.executable)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--timeout", type=float, default=2400, help="seconds for the whole queue")
    a = ap.parse_args(argv)

    from PySide6.QtWidgets import QApplication, QDialog
    from feabas_workbench.core import envs
    from feabas_workbench.core.synthetic import make_demo_project
    from feabas_workbench.core.steps import STEPS_BY_KEY, STANDARD_PIPELINE, RENDER_PIPELINE, count_outputs, expected_outputs
    from feabas_workbench.ui.main_window import MainWindow
    from feabas_workbench.ui import dialogs

    root = Path(a.project)
    if not (root / "workbench_project.json").is_file():
        make_demo_project(root, n_sections=4, tile=384)
    app = QApplication.instance() or QApplication([])
    settings = envs.Settings()                 # fresh, not the user's file
    settings.feabas_python = a.feabas_python
    envs.SETTINGS_FILE = root / "_settings.json"   # keep the run's settings out of %APPDATA%
    win = MainWindow(settings)
    win.show()
    win.open_project(root)
    app.processEvents()

    # accept the dialog as a user would, with the render option as requested
    orig_exec = dialogs.StandardPipelineDialog.exec

    def auto_exec(self):
        self.render.setChecked(a.render)
        self._refresh()
        print("dialog: " + self.summary.text())
        return QDialog.Accepted
    dialogs.StandardPipelineDialog.exec = auto_exec

    results = []
    win.ctx.jobs.job_finished.connect(lambda r: results.append(r))
    finished = {"done": False}
    win.ctx.jobs.queue_finished.connect(lambda _r: finished.update(done=True))
    win.run_standard_pipeline()
    t0 = time.time()
    while not finished["done"] and time.time() - t0 < a.timeout:
        app.processEvents()
        time.sleep(0.05)
    dialogs.StandardPipelineDialog.exec = orig_exec
    ok = finished["done"] and bool(results) and all(r.ok for r in results)
    print()
    for r in results:
        print(f"{r.spec.name:<40} {'ok' if r.ok else 'FAILED exit ' + str(r.exit_code):<18} {r.seconds:6.0f} s")
    keys = list(STANDARD_PIPELINE) + (list(RENDER_PIPELINE) if a.render else [])
    for k in keys:
        step = STEPS_BY_KEY[k]
        if step.local:
            continue
        done, exp = count_outputs(root, step), expected_outputs(root, step)
        print(f"   {step.label:<38} {done}/{exp} outputs")
        ok = ok and done >= exp
    print(f"\n{'GUI PIPELINE OK' if ok else 'GUI PIPELINE FAILED'} in {(time.time() - t0) / 60:.1f} min "
          f"({len(results)} jobs{'' if finished['done'] else ', queue did not finish'})")
    win.shutdown()
    win.close()
    app.processEvents()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
