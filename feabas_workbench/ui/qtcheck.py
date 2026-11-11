"""
A check of the Qt binding itself, run once at start.

PySide6 6.12.0 gives away one reference to ``True`` on every ``Signal.emit()`` (PYSIDE-3477). From
Python 3.12 on ``True`` is immortal and nothing happens; on 3.10 and 3.11 the interpreter aborts
("Fatal Python error: bool_dealloc") once the references run out - for the workbench after a few
thousand job output lines and progress updates, in the middle of a run. ``pyproject.toml`` keeps that
release out of new installations; this module is for the ones that already have it.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QObject, Signal


class _Probe(QObject):
    fired = Signal(int)


def lost_references(n: int = 200) -> int:
    """How many references to ``True`` *n* signal emissions gave away: 0 with a healthy binding."""
    probe = _Probe()
    spare = [True] * (2 * n)         # the measurement itself must not use up what is left
    before = sys.getrefcount(True)
    for i in range(n):
        probe.fired.emit(i)
    lost = before - sys.getrefcount(True)
    del spare
    return max(0, lost)


def _pad(count: int = 1 << 40) -> None:
    """Give True, False and None so many references that lost ones never matter - what Python 3.12
    does for good by making them immortal. The reference count is the first field of every object."""
    import ctypes
    for obj in (True, False, None):
        field = ctypes.c_ssize_t.from_address(id(obj))
        field.value += count


def guard_signal_references() -> str:
    """Measure once; with a binding that loses references, compensate and return a note for the log
    (empty when all is well)."""
    if sys.version_info >= (3, 12):      # True is immortal there: nothing to lose, nothing to pad
        return ""
    try:
        lost = lost_references()
    except Exception:  # noqa: BLE001 - a start-up check must never keep the workbench from starting
        return ""
    if not lost:
        return ""
    import PySide6
    try:
        _pad()
        outcome = "The workbench compensates for it in this session"
    except Exception:  # noqa: BLE001
        outcome = "The workbench could not compensate for it, so a long session may end abruptly"
    return (f"PySide6 {PySide6.__version__} on Python {sys.version_info.major}.{sys.version_info.minor} gives away a "
            f"reference on every Qt signal (PYSIDE-3477), which ends in 'Fatal Python error: bool_dealloc'. {outcome}. "
            f"Install another PySide6 to be rid of it: run tools\\install.bat again, or "
            f"'python -m pip install \"PySide6<6.12\"' in the workbench's environment.")
