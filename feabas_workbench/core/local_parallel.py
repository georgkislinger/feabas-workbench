"""Opt-in workstation section scheduling and conservative resource planning."""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .configs import ConfigStore, load_yaml
from .steps import aligned_dir, aligned_render_mip, stitched_dir

# Configuration kind, worker key, supports work within one section.
SECTION_STEPS = {
    "stitch.matching": ("stitching", "matching", True),
    "stitch.optimization": ("stitching", "optimization", False),
    "stitch.rendering": ("stitching", "rendering", True),
    "thumbnail.downsample": ("thumbnail", "downsample", True),
    "align.rendering": ("alignment", "rendering", True),
    "align.downsample": ("alignment", "downsample", True),
}
MODES = {"existing", "within", "across", "both"}
GIB = 1024 ** 3


def hardware():
    import psutil
    cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    try:
        cores = min(cores, len(psutil.Process().cpu_affinity()))
    except (AttributeError, psutil.Error):
        pass
    return cores, psutil.virtual_memory().available / GIB


def hardware_threads():
    """Logical CPUs (hardware threads) this process may run on: twice the cores with hyper-threading."""
    import psutil
    threads = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    try:
        threads = min(threads, len(psutil.Process().cpu_affinity()))
    except (AttributeError, psutil.Error):
        pass
    return threads


def options(project, key):
    saved = project.state.local_execution.get("steps", {}).get(key, {})
    kind, field, _ = SECTION_STEPS[key]
    workers = ConfigStore(project.configs_dir).get(kind, field + ".num_workers", 1)
    return dict(mode=saved.get("mode", "existing"), workers=max(1, int(saved.get("workers", workers))),
                sections=max(1, int(saved.get("sections", 2))),
                measured_gib=max(0., float(saved.get("measured_gib", 0))))


def memory_estimate(project, key, workers):
    """Planning allowance, not a measured peak or an OOM guarantee."""
    v = project.state.volume
    if v.tile_w <= 0 or v.tile_h <= 0:
        return None
    import numpy as np
    try:
        itemsize = np.dtype(v.dtype).itemsize
    except TypeError:
        return None
    tile = v.tile_w * v.tile_h * itemsize / GIB
    tiles = max(v.grid_rows * v.grid_cols, math.ceil(v.n_tiles / max(1, v.n_sections)), 1)
    # Working arrays/meshes/caches have stage- and data-dependent multipliers.
    # Keep a full-section allowance even when a downsampled stage may use less.
    return max(0.5, 4 * tile * tiles + 2 * tile * workers + 0.25 * workers)


@dataclass(frozen=True)
class LocalPlan:
    sections: int
    workers: int
    cpu_budget: int
    ram_budget_gib: float
    per_section_gib: float | None
    requested_sections: int
    note: str = ""

    def to_dict(self):
        return asdict(self)


def plan(project, key, settings=None, *, count=None, available=None, cpu_override=None, threads=None):
    """
    Sections at once and workers per section for *key*. *available* is (physical cores, free RAM
    in GiB) and *threads* the logical CPUs; both are read from this machine when not given.
    Without a configured CPU budget the physical cores are the budget (FEABAS's own default); a
    budget that was set may also use hyper-threads, up to the logical CPUs.
    """
    setting = options(project, key) if settings is None else settings
    mode = setting["mode"]
    if mode not in MODES:
        raise ValueError("Unknown local parallelism mode.")
    physical, free_gib = available if available is not None else hardware()
    if threads is None:
        threads = hardware_threads() if available is None else physical
    threads = max(int(physical), int(threads))
    general = load_yaml(project.general_config_path())
    configured = (project.state.local_execution.get("cpu_budget", general.get("cpu_budget"))
                  if cpu_override is None else cpu_override)
    try:
        configured = int(configured or 0)
    except (TypeError, ValueError):
        configured = 0
    cpu_budget = max(1, min(threads, configured)) if configured > 0 else max(1, int(physical))
    requested = max(1, int(setting["sections"])) if mode in {"across", "both"} else 1
    workers = 1 if mode == "across" or not SECTION_STEPS[key][2] else max(1, int(setting["workers"]))
    workers = min(cpu_budget, workers)
    configured_ram = float(project.state.local_execution.get("ram_budget_gib", 0))
    ram = min(configured_ram, free_gib * .8) if configured_ram > 0 else free_gib * .8
    estimated = float(setting.get("measured_gib", 0)) or memory_estimate(project, key, workers)
    cpu_limit = max(1, cpu_budget // workers)
    ram_limit = max(1, int(ram // estimated)) if estimated else 1
    count_limit = max(1, count if count is not None else len(project.section_names()))
    sections = min(requested, cpu_limit, ram_limit, count_limit)
    note = ""
    if estimated is None:
        note = "Image dimensions unknown: one section until you enter a measured RAM allowance."
    elif estimated > ram:
        note = "Even one section may exceed available RAM. Use fewer workers or a larger-memory machine."
    elif sections < requested:
        # say which limit it was, and what would lift it
        reasons = []
        if cpu_limit < requested:
            budget = f"{cpu_budget} physical cores" if configured <= 0 else f"{cpu_budget} CPUs"
            reasons.append(f"CPU budget {budget} ÷ {workers} workers = {cpu_limit} section(s)")
            if threads > cpu_budget:
                reasons.append(f"this PC has {threads} logical CPUs (hyper-threads): set the total CPU budget up to "
                               f"{threads} to use them, or use fewer workers per section")
        if ram_limit < requested:
            reasons.append(f"RAM budget {ram:.0f} GiB ÷ ~{estimated:.1f} GiB per section = {ram_limit} section(s)")
        if count_limit < requested:
            reasons.append(f"only {count_limit} section(s) to process")
        note = f"{sections} of {requested} sections at once: " + "; ".join(reasons) + "."
    if not SECTION_STEPS[key][2]:
        note = "Montage optimization uses one worker per section; parallelism is across sections. " + note
    return LocalPlan(sections, workers, cpu_budget, ram, estimated, requested, note)


def input_sections(root, key, configs=None):
    """Use the exact input ordering of each FEABAS driver, including partial runs."""
    root = Path(root)
    cs = configs or ConfigStore(root / "configs")
    if key == "stitch.matching":
        paths = sorted((root / "stitch/stitch_coord").glob("*.txt"))
        return paths or sorted((root / "stitch/stitch_coord").glob("*.tsv"))
    if key == "stitch.optimization":
        return sorted((root / "stitch/match_h5").glob("*.h5"))
    if key == "stitch.rendering":
        return sorted((root / "stitch/tform").glob("*.h5"))
    if key == "align.rendering":
        return sorted((root / "align/tform").glob("*.h5"))
    if key == "thumbnail.downsample":
        if cs.get("stitching", "rendering.driver", "image") != "image":
            return sorted((root / "stitch/ts_specs").glob("*.json"))
        base = stitched_dir(root, cs)
        mip = int(cs.get("thumbnail", "downsample.min_mip", 0))
    elif key == "align.downsample":
        base = aligned_dir(root, cs)
        mip = aligned_render_mip(cs)
    else:
        raise ValueError("This stage cannot be split into independent sections.")
    return sorted((base / f"mip{mip}").rglob("metadata.txt"))


def selected_indices(paths, key, start=None, stop=None, stride=None, filt=None, reverse=False):
    if stride is not None and stride < 1:
        raise ValueError("The section stride must be positive.")
    if (start is not None and start < 0) or (stop is not None and stop < 0):
        raise ValueError("Section indices must be nonnegative.")
    filtered = [p for p in paths if not filt or filt in p.name] if key.startswith("stitch.") else paths
    if filt and not key.startswith("stitch."):
        raise ValueError("Section filtering is only supported for stitching in local parallel mode.")
    # Stitch/thumbnail drivers use 0 as 'to the end'; align uses literal stop=0.
    end = None if stop == 0 and not key.startswith("align.") else stop
    if reverse and key.startswith("thumbnail."):
        indices = list(range(len(filtered)))[slice(end, start or None, -(stride or 1))]
    else:
        indices = list(range(len(filtered)))[slice(start, end, stride)]
    if reverse and not key.startswith("thumbnail."):
        indices.reverse()
    return filtered, indices
