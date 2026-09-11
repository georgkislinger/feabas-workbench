"""Run a Python callable in a QThread with progress/finished signals (for in-process batch work)."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal


class _Worker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, str)     # result, error text

    def __init__(self, fn: Callable, args: tuple, kwargs: dict):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancelled = False

    def run(self) -> None:
        try:
            res = self.fn(*self.args, progress=self._prog, cancelled=lambda: self.cancelled, **self.kwargs)
            self.finished.emit(res, "")
        except Exception as e:  # noqa: BLE001
            import traceback
            self.finished.emit(None, f"{e}\n{traceback.format_exc()[-800:]}")

    def _prog(self, done: int, total: int, msg: str = "") -> None:
        self.progress.emit(done, total, msg)


class ThreadRunner(QObject):
    """
    runner = ThreadRunner(self)
    runner.start(fn, args, on_progress=..., on_done=...)
    fn must accept progress(done,total,msg) and cancelled() keyword callbacks.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._on_progress: Callable | None = None
        self._on_done: Callable | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(self, fn: Callable, args: tuple = (), kwargs: dict | None = None,
              on_progress: Callable | None = None, on_done: Callable | None = None) -> bool:
        if self.running:
            return False
        self._on_progress, self._on_done = on_progress, on_done
        self._worker = _Worker(fn, args, kwargs or {})
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # The worker's signals go to bound methods of this QObject, which lives in the GUI
        # thread, so Qt queues them there. A plain function or lambda would be connected
        # directly instead and would run the callbacks - and every widget call inside them -
        # in the worker thread.
        self._worker.progress.connect(self._progressed)
        self._worker.finished.connect(self._finished)
        self._thread.start()
        return True

    def _progressed(self, done: int, total: int, msg: str) -> None:
        if self._on_progress:
            self._on_progress(done, total, msg)

    def _finished(self, res, err: str) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.quit()
            thread.wait(3000)          # on the GUI thread: not the thread waiting on itself
        self._worker = None
        cb, self._on_done = self._on_done, None
        self._on_progress = None
        if cb:
            cb(res, err)

    def cancel(self) -> None:
        if self._worker:
            self._worker.cancelled = True

    def stop(self, msec: int = 5000) -> None:
        """Cancel and wait for the thread to end. Call before the owner is destroyed:
        a QThread still running when Qt tears it down aborts the process."""
        self.cancel()
        thread, self._thread = self._thread, None
        self._worker = None
        self._on_progress = self._on_done = None
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(msec):
                thread.terminate()
                thread.wait(1000)
