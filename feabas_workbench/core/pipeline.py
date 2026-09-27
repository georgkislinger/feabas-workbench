"""
Run the standard pipeline of a project without the GUI: stitch, thumbnails, coarse and fine
alignment (and optionally the aligned-stack render), one FEABAS step after another, each
checked against the number of outputs it should leave behind.

Used by ``tools/run_demo_pipeline.py`` (the end-to-end test on a synthetic dataset, also in
CI) and as the reference sequence for the GUI's one-click runner. Nothing here imports Qt.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .jobs import feabas_env, kill_tree
from .steps import (STEPS_BY_KEY, STANDARD_PIPELINE, RENDER_PIPELINE, Step, count_outputs, expected_outputs,
                    step_argv)


@dataclass
class StepRun:
    step: Step
    exit_code: int
    seconds: float
    done: int
    expected: int
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.skipped or (self.exit_code == 0 and self.done >= self.expected)


def run_step(root: Path, step: Step, python: str = sys.executable, log: Callable[[str], None] = print,
             timeout: float | None = None) -> StepRun:
    """Run one FEABAS step in *python* with the project as working directory; stream its output to *log*."""
    from .project import repair_working_directory
    root = Path(root)
    if repair_working_directory(root):
        log(f"== configs/general_configs.yaml pointed at another folder; now {root}")
    env = os.environ.copy()
    env.update(feabas_env())
    env["PYTHONUNBUFFERED"] = "1"
    argv = step_argv(python, step)
    t0 = time.time()
    log(f"== {step.label}: {' '.join(argv[1:])}")
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen(argv, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **options)
    timed_out = threading.Event()
    def expire():
        if proc.poll() is None:
            timed_out.set()
            kill_tree(proc.pid)
    timer = threading.Timer(timeout, expire) if timeout and timeout > 0 else None
    if timer:
        timer.daemon = True
        timer.start()
    tail: list[str] = []
    try:
        for line in proc.stdout:            # type: ignore[union-attr]
            line = line.rstrip("\n")
            tail.append(line)
            if len(tail) > 40:
                del tail[:-40]
            log("   " + line)
    finally:
        try:
            # EOF does not guarantee process exit: keep the watchdog active
            # when a child closes its output stream before finishing.
            code = proc.wait()
        finally:
            if timer:
                timer.cancel()
            if proc.stdout:
                proc.stdout.close()
    if timed_out.is_set():
        log(f"   process tree stopped after {timeout:.0f} s")
        code = code or 124
    done, expected = count_outputs(root, step), expected_outputs(root, step)
    run = StepRun(step, code, time.time() - t0, done, expected)
    log(f"== {step.label}: exit {code}, {done}/{expected} outputs, {run.seconds:.0f} s")
    return run


def run_standard_pipeline(root: Path, python: str = sys.executable, render: bool = False,
                          log: Callable[[str], None] = print, skip_done: bool = True,
                          timeout_per_step: float | None = None) -> list[StepRun]:
    """
    Every step of STANDARD_PIPELINE (plus RENDER_PIPELINE with *render*) in order; stops at the
    first step that fails or leaves fewer outputs than expected. Steps that already have all
    their outputs are skipped when *skip_done* is set, like the GUI's 'Run all steps'.
    """
    root = Path(root)
    keys = list(STANDARD_PIPELINE) + (list(RENDER_PIPELINE) if render else [])
    runs: list[StepRun] = []
    for key in keys:
        step = STEPS_BY_KEY[key]
        if step.local:
            continue
        expected = expected_outputs(root, step)
        if skip_done and expected and count_outputs(root, step) >= expected:
            log(f"== {step.label}: already done ({expected} outputs), skipped")
            runs.append(StepRun(step, 0, 0.0, expected, expected, skipped=True))
            continue
        run = run_step(root, step, python, log, timeout_per_step)
        runs.append(run)
        if not run.ok:
            log(f"!! {step.label} failed; stopping here")
            break
    return runs


def summary(runs: list[StepRun]) -> str:
    lines = []
    for r in runs:
        state = "skipped" if r.skipped else ("ok" if r.ok else f"FAILED (exit {r.exit_code})")
        lines.append(f"{r.step.label:<40} {state:<18} {r.done}/{r.expected} outputs  {r.seconds:6.0f} s")
    return "\n".join(lines)


def match_problems(root: Path, min_points: int = 5) -> list[str]:
    """
    What the output counts cannot see: match files that exist but hold (almost) nothing.

    A montage whose tile pairs all failed to match still leaves one file per section (FEABAS
    3.0.5's stitching_matcher bug did exactly that), and a fine alignment computed from a single
    point per section pair still leaves one file per pair. Each stitch match file must connect
    all tiles of its section into one montage, and each coarse and fine match file must hold at
    least *min_points* points. Returns one line per problem; empty when all is well.
    """
    import h5py
    root = Path(root)
    problems = []
    for f in sorted((root / "stitch" / "match_h5").glob("*.h5")):
        with h5py.File(f, "r") as h:
            pairs = len(h["matches"]) if "matches" in h else 0
            groups = len(set(h["connected_subsystem"][()].tolist())) if "connected_subsystem" in h else 0
        if pairs == 0 or groups != 1:
            problems.append(f"stitch/match_h5/{f.name}: {pairs} matched tile pairs, "
                            f"tiles fall into {groups} separate groups (1 expected)")
    for sub in ("thumbnail_align/matches", "align/matches"):
        for f in sorted((root / sub).glob("*.h5")):
            with h5py.File(f, "r") as h:
                n = int(h["xy0"].shape[0]) if "xy0" in h else 0
            if n < min_points:
                problems.append(f"{sub}/{f.name}: {n} match point(s), at least {min_points} expected")
    return problems
