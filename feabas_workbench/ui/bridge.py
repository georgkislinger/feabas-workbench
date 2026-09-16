"""
Qt glue for the Qt-free core: a job queue with signals, and an application
context shared by all pages.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal

from ..core.configs import ConfigStore
from ..core.envs import Settings, check_imports
from ..core.jobs import JobQueue, JobSpec, worker_env, feabas_env
from ..core.project import Project, VENDOR_DIR
from ..core.steps import PipelineScan, Step, count_outputs, thumbnail_progress, step_argv, expected_outputs


class QtJobQueue(QObject):
    """Wraps core.jobs.JobQueue; signals arrive on the GUI thread (queued)."""

    output = Signal(str, str)                # stream, text
    progress = Signal(int, int, str)         # done, expected, message
    job_started = Signal(object)             # JobSpec
    job_finished = Signal(object)            # JobResult
    queue_finished = Signal(object)          # list[JobResult]
    running_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = JobQueue(
            on_output=lambda s, t: self.output.emit(s, t),
            on_progress=lambda d, e, m: self.progress.emit(d, e, m),
            on_job_started=self._started,
            on_job_finished=lambda r: self.job_finished.emit(r),
            on_queue_finished=self._queue_done,
        )

    def _started(self, spec: JobSpec) -> None:
        self.job_started.emit(spec)
        self.running_changed.emit(True)

    def _queue_done(self, results) -> None:
        self.queue_finished.emit(results)
        self.running_changed.emit(False)

    @property
    def running(self) -> bool:
        return self.queue.running

    def submit(self, specs) -> None:
        self.queue.submit(specs)

    def cancel_all(self) -> None:
        self.queue.cancel_all()

    def current_spec(self) -> JobSpec | None:
        cur = self.queue.current
        return cur.spec if cur else None


class AppContext(QObject):
    """What every page needs: settings, the open project, its configs, the job queue, logging."""

    project_changed = Signal(object)         # Project | None
    state_changed = Signal()                 # pipeline state should be re-read
    message = Signal(str, str)               # level, text  (info|warn|error)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project: Project | None = None
        self.configs: ConfigStore | None = None
        self.jobs = QtJobQueue(self)
        self.jobs.output.connect(self._on_job_output)
        self.jobs.queue_finished.connect(lambda _r: self.state_changed.emit())
        self.jobs.job_finished.connect(lambda _r: self.state_changed.emit())
        self._log_sinks: list[Callable[[str, str], None]] = []
        self._feabas_env_ok: set[str] = set()   # interpreters already known to import feabas

    # -- logging ---------------------------------------------------------
    def add_log_sink(self, fn: Callable[[str, str], None]) -> None:
        self._log_sinks.append(fn)

    def log(self, text: str, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}\n"
        for s in self._log_sinks:
            s(level, line)
        if self.project is not None:
            try:
                with open(self.project.workbench_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d')} {line}")
            except OSError:
                pass

    def _on_job_output(self, stream: str, text: str) -> None:
        for s in self._log_sinks:
            s("job-err" if stream == "err" else "job", text)

    # -- project ---------------------------------------------------------
    def open_project(self, path: Path) -> Project:
        self.project = Project.load(path) if Project.exists(path) else Project.create(path)
        self.configs = ConfigStore(self.project.configs_dir)
        self.settings.remember_project(str(path))
        self.settings.save()
        self.project_changed.emit(self.project)
        self.log(f"opened project {self.project.root}")
        return self.project

    def close_project(self) -> None:
        self.project = None
        self.configs = None
        self.project_changed.emit(None)

    def save_project(self) -> None:
        if self.project:
            self.project.save()

    def reload_configs(self) -> None:
        if self.project:
            self.configs = ConfigStore(self.project.configs_dir)

    def scan(self) -> PipelineScan | None:
        if not self.project:
            return None
        n = len(self.project.section_names())
        return PipelineScan(self.project.root, n, self.configs)

    # -- interpreters ----------------------------------------------------
    def feabas_python(self) -> str:
        return self.project.python_for("feabas", self.settings.feabas_python) if self.project else self.settings.feabas_python

    def dl_python(self) -> str:
        return self.project.python_for("dl", self.settings.dl_python) if self.project else self.settings.dl_python

    def require_feabas_python(self) -> str:
        """
        The interpreter FEABAS steps run in, verified once per path.

        Without this a wrong environment only shows up as a ModuleNotFoundError from
        the subprocess, which looks like a broken step rather than a wrong setting.
        """
        py = self.feabas_python()
        if not py:
            raise RuntimeError("No FEABAS Python environment configured: Setup page → FEABAS Python.")
        if py in self._feabas_env_ok:
            return py
        problem = check_imports(py, ("feabas",))
        if problem:
            override = bool(self.project and self.project.state.envs.get("feabas_python"))
            source = ("this project's interpreter override (workbench_project.json)" if override
                      else "the 'FEABAS Python' setting on the Setup page")
            raise RuntimeError(
                f"FEABAS steps cannot run in {py}: {problem}. That interpreter comes from {source}. "
                f"Set it to the environment where FEABAS is installed, press 'Check selected' to verify, "
                f"then 'Save'.")
        self._feabas_env_ok.add(py)
        return py

    # -- job builders ----------------------------------------------------
    def feabas_step_spec(self, step: Step, start: int | None = None, stop: int | None = None,
                         stride: int | None = None, filt: str | None = None, root: Path | None = None,
                         tag: str = "", extra_args: list[str] | None = None) -> JobSpec:
        root = root or self.project.root
        py = self.require_feabas_python()
        argv = step_argv(py, step, start, stop, stride, filt, extra_args)
        n = len([p for p in (root / "stitch" / "stitch_coord").glob("*.txt")])
        expected = expected_outputs(root, step, start, stop, stride)
        full_run = start is None and stop is None
        progress_fn = None
        if step.key == "thumbnail.downsample" and full_run:
            # the slow part of this step is the mip-mapping, which leaves no thumbnail behind
            cfg = ConfigStore(root / "configs") if root != self.project.root else self.configs
            progress_fn = lambda: thumbnail_progress(root, cfg, n)
        return JobSpec(
            name=f"{step.label}" + (f" [{tag}]" if tag else ""),
            argv=argv, cwd=root, kind="feabas", step_key=step.key, tag=tag, env=feabas_env(),
            count_outputs=lambda: count_outputs(root, step), expected=expected, progress_fn=progress_fn,
            progress_absolute=full_run, log_file=root / "workbench.log",
        )

    def feabas_tool_spec(self, tool: str, args: list[str], name: str, root: Path | None = None) -> JobSpec:
        root = root or self.project.root
        py = self.require_feabas_python()
        argv = [py, str(VENDOR_DIR / "tools" / tool)] + list(args)
        return JobSpec(name=name, argv=argv, cwd=root, kind="feabas", env=feabas_env(), log_file=root / "workbench.log")

    def worker_spec(self, module: str, payload: dict, name: str, python: str | None = None,
                    count_outputs: Callable[[], int] | None = None, expected: int = 0,
                    step_key: str | None = None) -> JobSpec:
        """A workbench worker (feabas_workbench.workers.<module>) in the given interpreter."""
        from ..core.jobs import write_spec_file
        root = self.project.root if self.project else Path.cwd()
        import sys
        if python == sys.executable and getattr(sys, "frozen", False):
            python = None          # frozen exe cannot run "-m"; use one of the configured interpreters
        py = python or self.dl_python() or self.feabas_python()
        if not py:
            raise RuntimeError("No Python environment configured for workers (Setup page).")
        spec_file = write_spec_file(root / "logs" / "specs", module.split(".")[-1], payload)
        argv = [py, "-m", f"feabas_workbench.workers.{module}", "--spec", str(spec_file)]
        return JobSpec(name=name, argv=argv, cwd=root, env=worker_env(),
                       kind="worker", count_outputs=count_outputs, expected=expected, step_key=step_key,
                       log_file=root / "workbench.log")
