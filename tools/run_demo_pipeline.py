"""
End-to-end check: build a small synthetic dataset and run the whole FEABAS pipeline on it.

    python tools/run_demo_pipeline.py D:/demo --feabas-python C:/envs/fw-feabas/python.exe
    python tools/run_demo_pipeline.py D:/demo --render          # also render the aligned PNG stack

Run it with any interpreter that has the workbench installed; --feabas-python is the one with
FEABAS (default: the same interpreter, which is how CI runs it). Exit code 0 only if every step
ran and produced what it should. A project that already exists is reused (finished steps are
skipped), so a failed run can be repeated after a fix.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feabas_workbench.core.pipeline import run_standard_pipeline, summary   # noqa: E402
from feabas_workbench.core.project import Project                              # noqa: E402
from feabas_workbench.core.synthetic import make_demo_project                  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="project folder (created with a synthetic dataset if it does not exist)")
    ap.add_argument("--feabas-python", default=sys.executable, help="interpreter with FEABAS installed")
    ap.add_argument("--render", action="store_true", help="also render the aligned stack as PNG tiles + mipmaps")
    ap.add_argument("--sections", type=int, default=4)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--timeout", type=float, default=1800, help="seconds allowed per step")
    a = ap.parse_args(argv)
    root = Path(a.project)
    if Project.exists(root):
        print(f"reusing project {root}")
    else:
        p = make_demo_project(root, n_sections=a.sections, tile=a.tile)
        v = p.state.volume
        print(f"created {root}: {v.n_sections} sections, {v.grid_rows}x{v.grid_cols} tiles of {v.tile_w} px")
    t0 = time.time()
    logfile = open(root / "workbench.log", "a", encoding="utf-8")

    def log(text: str) -> None:
        print(text, flush=True)
        logfile.write(text + "\n")

    runs = run_standard_pipeline(root, a.feabas_python, render=a.render, log=log, timeout_per_step=a.timeout)
    logfile.close()
    print()
    print(summary(runs))
    ok = bool(runs) and all(r.ok for r in runs) and len(runs) >= 10
    print(f"\n{'PIPELINE OK' if ok else 'PIPELINE FAILED'} in {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
