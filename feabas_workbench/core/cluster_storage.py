"""Distinguish DSS allocations from filesystems merely mounted on a login node."""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from .cluster_bundle import remote_path


def project_storage_path(value: str) -> str:
    path = remote_path(value.strip())
    parts = PurePosixPath(path).parts
    if (len(parts) < 4 or parts[1] != "dss" or parts[2].startswith("dsshome")
            or "scratch" in parts[2].lower()):
        raise ValueError("Choose your assigned DSS container folder, such as /dss/dssfs02/<container>. "
                         "A mounted filesystem root, home directory or temporary scratch is not project storage.")
    return path


def parse_dss_storage(output: str) -> dict:
    """Only read the accessible-container section; unknown output is not 'no quota'."""
    active = complete = False
    entries = []
    for raw in output.splitlines():
        line = raw.strip().strip("*").strip()
        if "you have access to the following dss containers" in line.lower():
            active = True
            continue
        if active:
            if re.match(r"DSS (?:Container usage|usage|Homedir info)", line, re.I):
                complete = True
                break
            if line:
                entries.append(line)
    directories = set()
    for entry in entries:
        for value in re.findall(r"/dss/[A-Za-z0-9_./+-]+", entry):
            try:
                directories.add(project_storage_path(value))
            except ValueError:
                pass
    state = "found" if directories else "none" if active and complete and not entries else "unknown"
    return dict(directories=sorted(directories), storage_state=state)


def storage_message(info: dict) -> str:
    state = info.get("storage_state", "unknown")
    if state == "found":
        return "Signed in · select your assigned DSS project folder"
    if state == "none":
        return "Signed in · no DSS project containers assigned; request project storage (home is 100 GB)"
    return "Signed in · storage discovery could not identify allocations; check the report or enter a confirmed DSS folder"
