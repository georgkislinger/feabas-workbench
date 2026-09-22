"""Distinguish DSS allocations from filesystems merely mounted on a login node, and
estimate whether a project fits the chosen storage."""
from __future__ import annotations

import os
import re
from pathlib import PurePosixPath

from .cluster_bundle import remote_path

# A full run keeps several full-resolution copies next to the uploaded raw tiles (1x): the
# stitched sections and their mip pyramid (~1.33x), the aligned stack and its pyramid
# (~1.33x) and a VAST export (~1.33x). Rounded up that is five times the raw data; every
# preprocessing variant (histogram matching, denoising) adds one more copy.
STORAGE_FACTOR = 5
# Python 3.11 with FEABAS and its dependencies, plus pip's download cache.
ENVIRONMENT_BYTES = 3 * 10**9
_UNITS = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}


def project_storage_path(value: str, home: str = "") -> str:
    path = remote_path(value.strip())
    if home:
        home = remote_path(home)
        if path == home or path.startswith(home + "/"):
            return path
    parts = PurePosixPath(path).parts
    if (len(parts) < 4 or parts[1] != "dss" or parts[2].startswith("dsshome")
            or "scratch" in parts[2].lower()):
        raise ValueError("Choose your home folder or an assigned DSS container folder, such as /dss/dssfs02/<container>. "
                         "A mounted filesystem root, another user's home or temporary scratch is not project storage.")
    return path


def _bytes(number: str, unit: str) -> int:
    # LRZ prints quotas as "GB"; decimal units keep the free-space figure on the safe side.
    return int(float(number.replace(",", ".")) * _UNITS.get(unit[:1].upper(), 1))


def parse_dss_storage(output: str) -> dict:
    """Only read the accessible-container section; unknown output is not 'no quota'."""
    active = complete = False
    entries = []
    home = ""
    home_used = home_limit = None
    section = ""
    for raw in output.splitlines():
        line = raw.strip().strip("*").strip()
        lower = line.lower()
        if "you have access to the following dss containers" in lower:
            active = True
            section = "containers"
            continue
        if re.match(r"DSS (?:Container usage|usage|Homedir info)", line, re.I):
            if active:
                complete = True
            active = False
            section = "home" if "homedir" in lower else "usage"
            continue
        if active:
            if line:
                entries.append(line)
        elif section == "home":
            m = re.search(r"home directory is at (/\S+)", line)
            if m:
                home = m.group(1)
            m = re.fullmatch(r"([\d.,]+)\s+of\s+([\d.,]+)\s+([KMGTP]?i?B)\s+used", line, re.I)
            if m:
                home_used, home_limit = _bytes(m.group(1), m.group(3)), _bytes(m.group(2), m.group(3))
    directories = set()
    for entry in entries:
        for value in re.findall(r"/dss/[A-Za-z0-9_./+-]+", entry):
            try:
                directories.add(project_storage_path(value))
            except ValueError:
                pass
    state = "found" if directories else "none" if complete and not entries else "unknown"
    return dict(directories=sorted(directories), storage_state=state, home=home,
                home_used=home_used, home_limit=home_limit)


def storage_message(info: dict) -> str:
    state = info.get("storage_state", "unknown")
    home = ""
    if info.get("home"):
        limit = info.get("home_limit")
        home = f"your home folder{f' ({limit / 1e9:.0f} GB)' if limit else ''} can hold smaller projects"
    if state == "found":
        return "Signed in · choose a DSS project container" + (" or your home folder" if home else "")
    if state == "none":
        return "Signed in · no DSS project container assigned" + (" · " + home if home else "; request project storage")
    return ("Signed in · storage discovery could not identify allocations; check the report or enter a confirmed DSS folder"
            + (" · " + home if home else ""))


def is_home(path: str, info: dict) -> bool:
    home = info.get("home") or ""
    return bool(home) and (path == home or path.startswith(home.rstrip("/") + "/"))


def raw_image_bytes(root) -> int:
    """Size of the source tiles from file metadata, without reading image contents."""
    total = 0
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".workbench-cluster"]
        for name in files:
            try:
                total += os.stat(os.path.join(directory, name)).st_size
            except OSError:
                pass
    return total


def preprocessing_copies(state) -> int:
    pp = state.preprocessing
    copies = {key for key in ("histmatch", "denoise") if (getattr(pp, key, None) or {}).get("enabled")}
    if pp.active_source in ("histmatch", "denoise"):
        copies.add(pp.active_source)
    return len(copies)


def estimate_project_bytes(raw_bytes: int, copies: int = 0) -> int:
    """Rough storage need of a complete run, raw copy included (see STORAGE_FACTOR)."""
    return raw_bytes * (STORAGE_FACTOR + copies)


def fit_warning(estimate: int, used: int | None, limit: int | None) -> str:
    """Empty if the estimate fits into the free quota (or the quota is unknown)."""
    if limit is None or used is None:
        return ""
    free = max(0, limit - used)
    if estimate <= free:
        return ""
    return (f"This project may need about {estimate / 1e9:.0f} GB, but only {free / 1e9:.0f} GB of "
            f"{limit / 1e9:.0f} GB are free here. Runs can fail once the quota is full.")
