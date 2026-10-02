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
from ..core.project import Project, VENDOR_DIR, repair_working_directory
from ..core.steps import (PipelineScan, Step, count_outputs, drop_cut_short_tiles, drop_unreadable_outputs,
                          thumbnail_progress, step_argv, expected_outputs)


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
        self.context = parent
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
        if getattr(self.context, "cluster_enabled", False):
            return self.context.cluster.running or self.context.cluster.busy
        return self.queue.running

    def submit(self, specs) -> None:
        if getattr(self.context, "cluster_enabled", False):
            return self.context.cluster.submit(specs)
        self.queue.submit(specs)

    def cancel_all(self) -> None:
        if getattr(self.context, "cluster_enabled", False):
            return self.context.cluster.cancel_all()
        self.queue.cancel_all()

    def current_spec(self) -> JobSpec | None:
        cur = self.queue.current
        return cur.spec if cur else None


class AppContext(QObject):
    """What every page needs: settings, the open project, its configs, the job queue, logging."""

    project_changed = Signal(object)         # Project | None
    state_changed = Signal()                 # pipeline state should be re-read
    message = Signal(str, str)               # level, text  (info|warn|error)
    execution_changed = Signal()
    export_downloaded = Signal(str)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.project: Project | None = None
        self.configs: ConfigStore | None = None
        self.local_project: Project | None = None
        self.cluster_enabled = False
        self.cluster = None
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
    def open_project(self, path: Path, create: bool = True) -> Project:
        """Open the project in *path*; with *create*, a folder without one becomes a new project."""
        if self.jobs.running:
            raise RuntimeError("Wait for the current operation before changing projects. Cluster jobs continue if you close Workbench.")
        if not Project.exists(path) and not create:
            raise RuntimeError(f"There is no FEABAS Workbench project in {path}.")
        if self.cluster:
            self.cluster.shutdown()
            self.cluster = None
        self.cluster_enabled = False
        self.project = Project.load(path) if Project.exists(path) else Project.create(path)
        self.local_project = self.project
        self.configs = ConfigStore(self.project.configs_dir)
        self.settings.remember_project(str(path))
        self.settings.save()
        self.project_changed.emit(self.project)
        from ..core.cluster_workspace import profile_for
        profile = profile_for(self.local_project)
        if profile["mode"] == "cluster":
            self.use_cluster(profile)
        self.execution_changed.emit()
        self.log(f"opened project {self.project.root}")
        return self.project

    def close_project(self) -> None:
        if self.cluster:
            self.cluster.shutdown()
        self.cluster = None
        self.cluster_enabled = False
        self.local_project = None
        self.project = None
        self.configs = None
        self.project_changed.emit(None)
        self.execution_changed.emit()

    def cluster_setup(self):
        from ..core.cluster_workspace import profile_for
        from .cluster_backend import ClusterBackend
        if self.cluster is None:
            self.cluster = ClusterBackend(self, self.local_project, profile_for(self.local_project))
            self.cluster.changed.connect(self.execution_changed.emit)
        return self.cluster

    def use_cluster(self, profile):
        from ..core.cluster_workspace import save_profile, view_project
        backend = self.cluster_setup()
        if backend.busy or (backend.running and self.cluster_enabled) or self.jobs.queue.running:
            raise RuntimeError("Wait for the active cluster operation before changing its settings.")
        profile["mode"] = "cluster"
        save_profile(self.local_project, profile)
        backend.profile = profile
        self.project = view_project(self.local_project)
        self.configs = ConfigStore(self.local_project.configs_dir)
        self.cluster_enabled = True
        self.project_changed.emit(self.project)
        self.execution_changed.emit()

    def use_local(self):
        """Back to This PC. Submitted LRZ jobs and Globus transfers keep running; the backend
        pauses monitoring until cluster mode is chosen again."""
        from ..core.cluster_workspace import save_profile
        if self.cluster and self.cluster.busy:
            raise RuntimeError("Workbench is still busy with LRZ (" + self.cluster.message.rstrip("…") + "). "
                               "Switch when that has finished, or stop a copy with 'Stop transfer'.")
        if self.cluster:
            self.cluster.profile["mode"] = "local"
            save_profile(self.local_project, self.cluster.profile)
        self.cluster_enabled = False
        self.project = self.local_project
        self.configs = ConfigStore(self.project.configs_dir)
        self.project_changed.emit(self.project)
        self.execution_changed.emit()

    def save_project(self) -> None:
        if self.project:
            self.project.save()

    def reload_configs(self) -> None:
        if self.project:
            self.configs = ConfigStore((self.local_project if self.cluster_enabled else self.project).configs_dir)

    def scan(self) -> PipelineScan | None:
        if not self.project:
            return None
        if self.cluster_enabled:
            from ..core.cluster_workspace import remote_scan
            return remote_scan(self.project, self.configs, self.cluster.snapshot)
        n = len(self.project.section_names())
        return PipelineScan(self.project.root, n, self.configs)

    # -- interpreters ----------------------------------------------------
    def feabas_python(self) -> str:
        if self.cluster_enabled:
            return self.cluster.profile.get("remote_python", "")
        return self.project.python_for("feabas", self.settings.feabas_python) if self.project else self.settings.feabas_python

    def dl_python(self) -> str:
        if self.cluster_enabled:
            return self.cluster.profile.get("remote_dl_python", "")
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
        if self.cluster_enabled and root != self.project.root:
            raise RuntimeError("For cluster experiments use the section range on the main pipeline. Local test-run folders run in This PC mode.")
        py = "python" if self.cluster_enabled else self.require_feabas_python()
        if not self.cluster_enabled and repair_working_directory(root):
            self.log(f"{root}: configs/general_configs.yaml named another folder (a copied or moved project); "
                     f"FEABAS now works here again")
        argv = step_argv(py, step, start, stop, stride, filt, extra_args)
        n = len([p for p in (root / "stitch" / "stitch_coord").glob("*.txt")])
        expected = expected_outputs(root, step, start, stop, stride)
        env = feabas_env()
        if not self.cluster_enabled:
            from ..core.local_parallel import SECTION_STEPS, mipmap_plan, options, plan
            if step.key in SECTION_STEPS:
                settings = options(self.project, step.key)
                if settings["mode"] == "auto":
                    cfg = ConfigStore(root / "configs") if root != self.project.root else self.configs
                    try:
                        tuning = mipmap_plan(self.project, step.key, root, cfg, start=start, stop=stop or None,
                                             stride=stride, reverse="--reverse" in (extra_args or []))
                    except Exception as e:  # noqa: BLE001 - the run itself does not depend on its tuning
                        tuning = None
                        self.log(f"{step.label}: FEABAS's own settings this time ({e})")
                    if tuning is not None:
                        env.update(tuning.env())
                        self.log(f"{step.label}: {tuning.note}")
                elif settings["mode"] != "existing":
                    if extra_args and extra_args != ["--reverse"]:
                        raise RuntimeError("These extra arguments need Existing FEABAS settings for this stage.")
                    from ..core.jobs import write_spec_file
                    import uuid
                    allocation = plan(self.project, step.key, settings)
                    self.log(f"{step.label}: requested local {allocation.sections} sections × {allocation.workers} workers.")
                    payload = dict(root=str(root), project=str(self.project.root), step=step.key, settings=settings,
                                   start=start, stop=stop or None, stride=stride, filter=filt, reverse=bool(extra_args))
                    spec_file = write_spec_file(root / "logs/specs", "local_parallel_" + uuid.uuid4().hex, payload)
                    return JobSpec(name=step.label + (f" [{tag}]" if tag else ""),
                        argv=[py, "-m", "feabas_workbench.workers.local_parallel", "--spec", str(spec_file)],
                        cwd=root, env=worker_env(), kind="feabas", step_key=step.key, tag=tag,
                        expected=expected, log_file=root / "workbench.log",
                        after_cancel=self._after_cancel(root, step))
        full_run = start is None and stop is None
        progress_fn = None
        if step.key == "thumbnail.downsample" and full_run:
            # the slow part of this step is the mip-mapping, which leaves no thumbnail behind
            cfg = ConfigStore(root / "configs") if root != self.project.root else self.configs
            progress_fn = lambda: thumbnail_progress(root, cfg, n)
        return JobSpec(
            name=f"{step.label}" + (f" [{tag}]" if tag else ""),
            argv=argv, cwd=root, kind="feabas", step_key=step.key, tag=tag, env=env,
            count_outputs=lambda: count_outputs(root, step), expected=expected, progress_fn=progress_fn,
            progress_absolute=full_run, log_file=root / "workbench.log",
            after_cancel=None if self.cluster_enabled else self._after_cancel(root, step),
            remote=dict(kind="step", step=step.key, name=step.label, start=start, stop=stop, stride=stride,
                        filter=filt, extra_args=extra_args or []) if self.cluster_enabled else None,
        )

    @staticmethod
    def _after_cancel(root: Path, step: Step):
        """A killed FEABAS step may leave an .h5 file or image tile cut short, which FEABAS would keep as finished."""
        def clean(started: float) -> list[str]:
            removed = drop_unreadable_outputs(root, step, started) + drop_cut_short_tiles(root, step, started)
            return [f"-- removed {len(removed)} incomplete output(s) of '{step.label}' left by the cancelled run: "
                    + ", ".join(p.name for p in removed[:8]) + (" …" if len(removed) > 8 else "")] if removed else []
        return clean

    def feabas_tool_spec(self, tool: str, args: list[str], name: str, root: Path | None = None) -> JobSpec:
        root = root or self.project.root
        py = "python" if self.cluster_enabled else self.require_feabas_python()
        if not self.cluster_enabled:
            repair_working_directory(root)
        argv = [py, str(VENDOR_DIR / "tools" / tool)] + list(args)
        return JobSpec(name=name, argv=argv, cwd=root, kind="feabas", env=feabas_env(), log_file=root / "workbench.log",
                       remote=dict(kind="tool", tool=tool, args=args, name=name) if self.cluster_enabled else None)

    def worker_spec(self, module: str, payload: dict, name: str, python: str | None = None,
                    count_outputs: Callable[[], int] | None = None, expected: int = 0,
                    step_key: str | None = None) -> JobSpec:
        """A workbench worker (feabas_workbench.workers.<module>) in the given interpreter."""
        from ..core.jobs import write_spec_file
        root = self.project.root if self.project else Path.cwd()
        import sys
        if self.cluster_enabled:
            dl = module in {"fold_predict", "fold_train", "yolo_detect", "yolo_train", "n2v_train", "n2v_predict"}
            return JobSpec(name=name, argv=[], cwd=root, kind="worker", expected=expected, step_key=step_key,
                           remote=dict(kind="worker", module=module, payload=payload, name=name, dl=dl))
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
