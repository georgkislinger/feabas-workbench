"""Launching external viewers (Fiji, VASTlite)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _popen(argv: list[str]) -> None:
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(argv, cwd=str(Path(argv[0]).parent), **kw)


def open_in_fiji(settings, path: Path, log=None) -> bool:
    exe = settings.fiji_path
    if not exe or not Path(exe).is_file():
        if log:
            log("Fiji path not set (Setup page)")
        return False
    _popen([exe, str(path)])
    if log:
        log(f"opened in Fiji: {path}")
    return True


def open_in_fiji_folder(settings, folder: Path, log=None) -> bool:
    """Open a folder of tiles as a virtual stack via an ImageJ macro."""
    exe = settings.fiji_path
    if not exe or not Path(exe).is_file():
        if log:
            log("Fiji path not set (Setup page)")
        return False
    macro = f'run("Image Sequence...", "open=[{folder.as_posix()}/] sort use");'
    _popen([exe, "-eval", macro])
    if log:
        log(f"opened folder in Fiji: {folder}")
    return True


def open_in_vast(settings, vsvi: Path, log=None) -> bool:
    exe = settings.vast_path
    if not exe or not Path(exe).is_file():
        if log:
            log("VASTlite path not set (Setup page)")
        return False
    _popen([exe, str(vsvi)])
    if log:
        log(f"opened in VASTlite: {vsvi}")
    return True
