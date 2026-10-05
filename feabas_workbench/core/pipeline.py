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


def _stream(argv: list[str], cwd: Path, env: dict, log: Callable[[str], None], timeout: float | None) -> int:
    """Run *argv*, stream its output to *log*; stop the whole process tree after *timeout* seconds."""
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
               if os.name == "nt" else {"start_new_session": True})
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
    try:
        for line in proc.stdout:            # type: ignore[union-attr]
            log("   " + line.rstrip("\n"))
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
    return code


def _automatic_mipmaps(root: Path, step: Step):
    """The GUI's default for the mipmap steps (Local parallelism 'Automatic'), so this runner runs them alike."""
    from .local_parallel import MIPMAP_STEPS, mipmap_plan, options
    from .project import Project
    if step.key not in MIPMAP_STEPS or not Project.exists(root):
        return None
    project = Project.load(root)
    return mipmap_plan(project, step.key, root) if options(project, step.key)["mode"] == "auto" else None


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
    try:
        tuning = _automatic_mipmaps(root, step)
    except Exception as e:  # noqa: BLE001 - the run itself does not depend on its tuning
        tuning = None
        log(f"== {step.label}: FEABAS's own settings this time ({e})")
    if tuning is not None:
        env.update(tuning.env())
        log(f"== {step.label}: {tuning.note}")
    code = _stream(argv, root, env, log, timeout)
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


def render_masks(root: Path, masks_dir: Path, name: str, python: str = sys.executable,
                 log: Callable[[str], None] = print, mode: str = "auto", workers: int = 4,
                 timeout: float | None = None) -> tuple[int, Path]:
    """Carry segmentation masks through the project's alignment the way the Export page does
    (core.segmentation.plan_masks, workers/segmentation_render). Returns (exit code, output folder)."""
    from .jobs import worker_env, write_spec_file
    from .project import Project, repair_working_directory
    from .segmentation import plan_masks, stack_dir
    # absolute: the worker runs in the project folder, so paths relative to here would not hold there
    root, masks_dir = Path(root).resolve(), Path(masks_dir).resolve()
    repair_working_directory(root)
    plan = plan_masks(root, Project.load(root).section_names(), masks_dir, mode)
    out = stack_dir(root, name)
    spec = write_spec_file(root / "logs" / "specs", "segmentation_render",
                           {"root": str(root), "out_dir": str(out), "plan": plan.to_dict(), "workers": workers,
                            "source": {"masks_dir": str(masks_dir), "match": plan.mode}})
    env = os.environ.copy()
    env.update(worker_env())
    env["PYTHONUNBUFFERED"] = "1"
    log(f"== Aligned masks '{name}': {plan.n_masks} {plan.bits}-bit mask file(s), matched "
        f"{'by name' if plan.mode == 'name' else 'in section order'}")
    t0 = time.time()
    code = _stream([python, "-m", "feabas_workbench.workers.segmentation_render", "--spec", str(spec)], root, env, log,
                   timeout)
    log(f"== Aligned masks '{name}': exit {code}, {time.time() - t0:.0f} s")
    return code, out


def mask_problems(images: Path, masks: Path, labels: set[int] | None = None, images_as_masks: bool = False,
                  max_shift: float = 1.0) -> list[str]:
    """
    What would make an aligned mask stack not match its aligned images: a different tile layout at
    any mip level (file names apart from the extension, and bounding boxes), label values that are
    not among *labels*, and - when the masks were made from the images' own files
    (*images_as_masks*) - a shift of more than *max_shift* px between the two at full resolution.
    """
    import numpy as np
    from .images import TiledSectionSource
    from .segmentation import read_tile_metadata
    images, masks = Path(images), Path(masks)
    problems = []
    levels = sorted(int(p.name[3:]) for p in images.glob("mip*") if p.name[3:].isdigit())
    if not levels:
        return [f"{images}: no rendered levels"]
    seen = set()
    for m in levels:
        for d in sorted(p for p in (images / f"mip{m}").iterdir() if p.is_dir()):
            meta = masks / f"mip{m}" / d.name / "metadata.txt"
            if not meta.is_file():
                problems.append(f"mip{m}/{d.name}: no aligned masks")
                continue
            want = {Path(k).stem: v for k, v in read_tile_metadata(d / "metadata.txt").items()}
            got = {Path(k).stem: v for k, v in read_tile_metadata(meta).items()}
            if not set(got) <= set(want) or any(got[k] != want[k] for k in got):
                problems.append(f"mip{m}/{d.name}: tile layout differs from the images'")
            if labels is not None and m in (levels[0], levels[-1]):
                import cv2
                for name in got:
                    a = cv2.imread(str(meta.parent / f"{name}.png"), cv2.IMREAD_UNCHANGED)
                    values = set(np.unique(a).tolist())
                    seen |= values
                    if not values <= labels | {0}:
                        problems.append(f"mip{m}/{d.name}/{name}: values that are not mask labels")
            if images_as_masks and m == levels[0]:
                import cv2
                src = TiledSectionSource(images, d.name)
                lvl = next((lv for lv in src.levels if lv.mip == m), None)
                if lvl is None:
                    continue
                a = src.read(m, 0, 0, lvl.width, lvl.height).astype(np.float32)
                b = TiledSectionSource(masks, d.name).read(m, 0, 0, lvl.width, lvl.height).astype(np.float32)
                valid = (a > 0) & (b > 0)
                if valid.sum() < 100:
                    problems.append(f"mip{m}/{d.name}: masks and images barely overlap")
                    continue
                # FEABAS may invert the images when it renders montages: compare in the orientation that fits
                if np.corrcoef(a[valid], b[valid])[0, 1] < 0:
                    a = np.where(a > 0, 255 - a, 0)
                win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
                if max(abs(dx), abs(dy)) > max_shift:
                    problems.append(f"mip{m}/{d.name}: masks shifted by ({dx:+.2f}, {dy:+.2f}) px against the images")
    if labels is not None and not (seen - {0}):
        problems.append("no labels in the aligned masks")
    return problems


def summary(runs: list[StepRun]) -> str:
    lines = []
    for r in runs:
        state = "skipped" if r.skipped else ("ok" if r.ok else f"FAILED (exit {r.exit_code})")
        lines.append(f"{r.step.label:<40} {state:<18} {r.done}/{r.expected} outputs  {r.seconds:6.0f} s")
    return "\n".join(lines)


def _single_tile(root: Path, section: str) -> bool:
    """One image per section (an image stack): there are no tile pairs to match."""
    coord = Path(root) / "stitch" / "stitch_coord" / f"{section}.txt"
    if not coord.is_file():
        return False
    lines = coord.read_text(encoding="utf-8", errors="replace").splitlines()
    return sum(1 for line in lines if line.strip() and not line.startswith("{")) == 1


def match_problems(root: Path, min_points: int = 5) -> list[str]:
    """
    What the output counts cannot see: match files that exist but hold (almost) nothing.

    A montage whose tile pairs all failed to match still leaves one file per section (FEABAS
    3.0.5's stitching_matcher bug did exactly that), and a fine alignment computed from a single
    point per section pair still leaves one file per pair. Each stitch match file must connect
    all tiles of its section into one montage (a section of one image has nothing to match),
    and each coarse and fine match file must hold at
    least *min_points* points. Returns one line per problem; empty when all is well.
    """
    import h5py
    root = Path(root)
    problems = []
    for f in sorted((root / "stitch" / "match_h5").glob("*.h5")):
        with h5py.File(f, "r") as h:
            pairs = len(h["matches"]) if "matches" in h else 0
            groups = len(set(h["connected_subsystem"][()].tolist())) if "connected_subsystem" in h else 0
        if (pairs == 0 or groups != 1) and not _single_tile(root, f.stem):
            problems.append(f"stitch/match_h5/{f.name}: {pairs} matched tile pairs, "
                            f"tiles fall into {groups} separate groups (1 expected)")
    for sub in ("thumbnail_align/matches", "align/matches"):
        for f in sorted((root / sub).glob("*.h5")):
            with h5py.File(f, "r") as h:
                n = int(h["xy0"].shape[0]) if "xy0" in h else 0
            if n < min_points:
                problems.append(f"{sub}/{f.name}: {n} match point(s), at least {min_points} expected")
    return problems
