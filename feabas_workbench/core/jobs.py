"""
Running external work: FEABAS steps and workbench workers as subprocesses.

Design:
* every heavy task is a subprocess in a chosen Python environment, so a broken
  environment shows up as a readable error instead of taking the GUI down, and
  the same commands can later be submitted to a cluster scheduler unchanged;
* the whole process *tree* is killed on cancel (FEABAS uses multiprocessing);
* progress comes from two sources: workbench workers print
  ``##PROGRESS {"done": i, "total": n, "msg": "..."}`` lines, and FEABAS steps
  are measured by counting their output files (or by a step-specific
  ``progress_fn`` when one count would not tell the story);
* the core is Qt-free (threads + callbacks); the UI wraps it in signals.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROGRESS_PREFIX = "##PROGRESS "
RESULT_PREFIX = "##RESULT "


@dataclass
class JobSpec:
    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    count_outputs: Callable[[], int] | None = None     # for file-count progress
    expected: int = 0
    # True: report count_outputs() as it is (a full run of a step that is already half done starts
    # the bar half full). False: report only what this job added, for subset runs and workers
    # whose expected count covers just their own items.
    progress_absolute: bool = False
    # optional richer progress: () -> (done, expected, message). Takes precedence over
    # count_outputs and reports absolute numbers (a step with several phases can say which one)
    progress_fn: Callable[[], tuple[int, int, str]] | None = None
    kind: str = "worker"                                # worker | feabas | shell
    step_key: str | None = None
    tag: str = ""                                       # free-form (e.g. test-run name)
    log_file: Path | None = None

    def cmdline(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


@dataclass
class JobResult:
    spec: JobSpec
    exit_code: int
    cancelled: bool
    seconds: float
    result: dict = field(default_factory=dict)          # from ##RESULT line, if any
    last_lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.cancelled


class Job:
    """One subprocess. Callbacks are invoked from the reader thread."""

    def __init__(self, spec: JobSpec,
                 on_output: Callable[[str, str], None] | None = None,
                 on_progress: Callable[[int, int, str], None] | None = None,
                 on_finished: Callable[[JobResult], None] | None = None):
        self.spec = spec
        self.on_output = on_output
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.proc: subprocess.Popen | None = None
        self.cancelled = False
        self.started_at = 0.0
        self.result_payload: dict = {}
        self._tail: list[str] = []
        self._thread: threading.Thread | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_poll = threading.Event()
        self._baseline = 0
        self._logfh = None

    # -- control --------------------------------------------------------
    def start(self) -> None:
        spec = self.spec
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"          # libraries that open text files without an encoding (Windows cp1252)
        if os.name == "posix" and "MALLOC_ARENA_MAX" not in env:
            env["MALLOC_ARENA_MAX"] = "2"   # documented mitigation for TensorStore RAM growth
        env.update(spec.env)
        creation = 0
        popen_kwargs = {}
        if os.name == "nt":
            creation = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            popen_kwargs["creationflags"] = creation
        else:
            popen_kwargs["start_new_session"] = True
        if spec.log_file:
            spec.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._logfh = open(spec.log_file, "a", encoding="utf-8")
            self._logfh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {spec.name}\n$ {spec.cmdline()}\n")
        self.started_at = time.time()
        self._emit("out", f"$ {spec.cmdline()}\n")
        try:
            self.proc = subprocess.Popen(
                spec.argv, cwd=str(spec.cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                **popen_kwargs)
        except OSError as e:
            self._emit("err", f"could not start process: {e}\n")
            self._finish(127)
            return
        if spec.progress_fn is not None or spec.count_outputs is not None:
            if spec.count_outputs is not None:
                try:
                    self._baseline = spec.count_outputs()
                except Exception:
                    self._baseline = 0
            self._poll_thread = threading.Thread(target=self._poll_outputs, daemon=True)
            self._poll_thread.start()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self.cancelled = True
        if self.proc is None or self.proc.poll() is not None:
            return
        self._emit("err", "-- cancelling: stopping the process tree --\n")
        kill_tree(self.proc.pid)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- internals ------------------------------------------------------
    def _emit(self, stream: str, text: str) -> None:
        if self._logfh:
            try:
                self._logfh.write(text)
                self._logfh.flush()
            except OSError:
                pass
        self._tail.append(text.rstrip("\n"))
        if len(self._tail) > 60:
            del self._tail[:-60]
        if self.on_output:
            self.on_output(stream, text)

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                # Some libraries (tqdm progress bars) write \r-terminated partial lines to stdout; a
                # worker's ##PROGRESS/##RESULT line can then be glued onto the tail of such a partial
                # line, so look for the marker anywhere on the line instead of only at column 0.
                prog_at = line.find(PROGRESS_PREFIX)
                res_at = line.find(RESULT_PREFIX)
                if prog_at >= 0 and (res_at < 0 or prog_at < res_at):
                    try:
                        d = json.loads(line[prog_at + len(PROGRESS_PREFIX):])
                        if self.on_progress:
                            self.on_progress(int(d.get("done", 0)), int(d.get("total", 0)), str(d.get("msg", "")))
                        continue
                    except (ValueError, TypeError):
                        pass  # fall through and log the raw line
                elif res_at >= 0:
                    try:
                        self.result_payload = json.loads(line[res_at + len(RESULT_PREFIX):])
                        continue
                    except ValueError:
                        pass
                low = line.lower()
                stream = "err" if ("error" in low or "traceback" in low or "exception" in low) else "out"
                self._emit(stream, line)
        except ValueError:
            pass
        code = self.proc.wait()
        self._finish(code)

    def _poll_outputs(self) -> None:
        while not self._stop_poll.wait(1.5):
            if self.spec.progress_fn is not None:
                try:
                    done, expected, msg = self.spec.progress_fn()
                except Exception:
                    continue
                if self.on_progress:
                    self.on_progress(max(0, int(done)), int(expected), str(msg))
                continue
            if self.spec.count_outputs is None:
                return
            try:
                n = progress_count(self.spec, self.spec.count_outputs(), self._baseline)
            except Exception:
                continue
            if self.on_progress:
                self.on_progress(n, self.spec.expected, "")

    def _finish(self, code: int) -> None:
        self._stop_poll.set()
        secs = time.time() - self.started_at
        if self._logfh:
            try:
                self._logfh.write(f"===== exit {code} after {secs:.0f}s\n")
                self._logfh.close()
            except OSError:
                pass
            self._logfh = None
        res = JobResult(self.spec, code, self.cancelled, secs, self.result_payload, list(self._tail))
        if self.on_finished:
            self.on_finished(res)


def progress_count(spec: JobSpec, count: int, baseline: int) -> int:
    """Outputs to report for a job: everything on disk, or only what appeared since it started."""
    n = int(count) if spec.progress_absolute else int(count) - int(baseline)
    return max(0, n)


def kill_tree(pid: int) -> None:
    """Kill a process and all of its descendants."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for c in children:
                try:
                    c.kill()
                except psutil.Error:
                    pass
            parent.kill()
            psutil.wait_procs(children + [parent], timeout=5)
            return
        except psutil.Error:
            pass
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass


# ----------------------------------------------------------------------
# queue
# ----------------------------------------------------------------------

class JobQueue:
    """Runs jobs one after another; stops on the first failure unless told otherwise."""

    def __init__(self, on_output=None, on_progress=None, on_job_finished=None, on_queue_finished=None,
                 on_job_started=None):
        self.on_output = on_output
        self.on_progress = on_progress
        self.on_job_finished = on_job_finished
        self.on_queue_finished = on_queue_finished
        self.on_job_started = on_job_started
        self._pending: list[JobSpec] = []
        self._current: Job | None = None
        self._lock = threading.Lock()
        self.stop_on_error = True
        self._results: list[JobResult] = []
        self.active = False

    @property
    def running(self) -> bool:
        return self.active

    @property
    def current(self) -> Job | None:
        return self._current

    def pending(self) -> list[JobSpec]:
        with self._lock:
            return list(self._pending)

    def submit(self, specs: list[JobSpec] | JobSpec) -> None:
        if isinstance(specs, JobSpec):
            specs = [specs]
        with self._lock:
            self._pending.extend(specs)
            if not self.active:
                self.active = True
                self._results = []
                self._start_next_locked()

    def _start_next_locked(self) -> None:
        if not self._pending:
            self.active = False
            self._current = None
            results = list(self._results)
            if self.on_queue_finished:
                threading.Thread(target=self.on_queue_finished, args=(results,), daemon=True).start()
            return
        spec = self._pending.pop(0)
        job = Job(spec, on_output=self.on_output, on_progress=self.on_progress, on_finished=self._job_done)
        self._current = job
        if self.on_job_started:
            self.on_job_started(spec)
        job.start()

    def _job_done(self, res: JobResult) -> None:
        with self._lock:
            self._results.append(res)
            if self.on_job_finished:
                self.on_job_finished(res)
            if (not res.ok) and self.stop_on_error:
                if self._pending:
                    if self.on_output:
                        self.on_output("err", f"-- skipping {len(self._pending)} queued job(s) after failure --\n")
                    self._pending.clear()
            self._start_next_locked()

    def cancel_all(self) -> None:
        with self._lock:
            self._pending.clear()
            cur = self._current
        if cur is not None:
            cur.cancel()


# ----------------------------------------------------------------------
# helpers for building specs
# ----------------------------------------------------------------------

def worker_argv(python: str, module: str, spec_file: Path | None = None, *extra: str) -> list[str]:
    argv = [python, "-m", module]
    if spec_file is not None:
        argv += ["--spec", str(spec_file)]
    argv += list(extra)
    return argv


def write_spec_file(directory: Path, name: str, payload: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{name}_{int(time.time())}.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def feabas_env() -> dict[str, str]:
    """
    Environment for FEABAS processes. The vendor/winfix folder is put first on PYTHONPATH:
    its sitecustomize.py applies the FEABAS 3.0.5 run-time fixes in every process,
    multiprocessing children included - the stitching_matcher 'phtm' UnboundLocalError on
    every platform, and on Windows the TensorStore file URLs (file://D:/ -> file:///D:/).
    """
    from .project import VENDOR_DIR
    winfix = VENDOR_DIR.parent / "winfix"
    existing = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": os.pathsep.join([str(winfix)] + ([existing] if existing else []))}


def python_env_for_package_root(package_root: Path) -> dict[str, str]:
    """PYTHONPATH so that `python -m feabas_workbench.workers.x` works from any environment."""
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(package_root)] + ([existing] if existing else [])
    return {"PYTHONPATH": os.pathsep.join(parts)}


def package_root() -> Path:
    """Folder that contains the feabas_workbench package: the repo for a development checkout,
    site-packages for a wheel install, the bundle folder for a frozen build."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent.parent


# what a worker process needs of the package: everything except the Qt UI
WORKER_PACKAGE_PARTS = ("__init__.py", "core", "workers", "vendor")


def worker_package_root() -> Path:
    """
    A folder whose only content is the feabas_workbench package, for the PYTHONPATH of
    workers that run in the FEABAS or deep-learning interpreter.

    package_root() is the repo for a development checkout, where it is safe to use directly.
    For a wheel install it is this environment's site-packages, and for a frozen build the
    PyInstaller bundle - both full of numpy, cv2, h5py ... compiled for *this* Python. Put at
    the front of another interpreter's path they shadow its own copies, and the worker dies
    importing numpy. So the pure-Python parts a worker needs are staged into a private
    folder under the settings directory and that is used instead. The copy is under 1 MB
    and is refreshed whenever the installed files change.
    """
    from .envs import settings_dir
    src = package_root() / "feabas_workbench"
    dst_root = settings_dir() / "worker_pkg"
    dst = dst_root / "feabas_workbench"
    stamp_file = dst_root / "stamp.json"
    files: list[Path] = []
    for part in WORKER_PACKAGE_PARTS:
        p = src / part
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    stamp = {"src": str(src), "n": len(files),
             "newest": max((f.stat().st_mtime for f in files), default=0.0)}
    try:
        if dst.is_dir() and json.loads(stamp_file.read_text(encoding="utf-8")) == stamp:
            return dst_root
    except (OSError, ValueError):
        pass
    shutil.rmtree(dst, ignore_errors=True)
    for f in files:
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    return dst_root


def worker_env() -> dict[str, str]:
    """Environment for `python -m feabas_workbench.workers.x` in any interpreter."""
    return python_env_for_package_root(worker_package_root())
