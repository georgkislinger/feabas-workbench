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
MODES = {"existing", "within", "across", "both", "auto"}
GIB = 1024 ** 3
# FEABAS's mipmap stages for PNG/JPEG tiles, where 'Automatic' is the default (see mipmap_plan).
MIPMAP_STEPS = ("thumbnail.downsample", "align.downsample")
# FEABAS reads with a cache of downsample + 2 = 4 tiles, which re-reads each source tile several
# times while one output tile is made; 16 keeps them (+~0.3 GB per worker for 4096² 8-bit tiles).
MIPMAP_CACHE_TILES = 16


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
    default = "auto" if key in MIPMAP_STEPS else "existing"
    return dict(mode=saved.get("mode", default), workers=max(1, int(saved.get("workers", workers))),
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


def _cpu_budget(project, physical, threads=None, cpu_override=None, available=None):
    """(logical CPUs, the budget that was set or 0, the budget in effect). Without a set budget the
    physical cores are the budget; a set one may use hyper-threads, up to the logical CPUs."""
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
    return threads, configured, max(1, min(threads, configured)) if configured > 0 else max(1, int(physical))


def _ram_budget(project, free_gib):
    configured = float(project.state.local_execution.get("ram_budget_gib", 0))
    return min(configured, free_gib * .8) if configured > 0 else free_gib * .8


def run_budget(project, available=None, threads=None):
    """(CPU budget, RAM budget in GiB) for work on this PC, as the Local parallelism dialog sets them."""
    physical, free_gib = available if available is not None else hardware()
    return _cpu_budget(project, physical, threads, None, available)[2], _ram_budget(project, free_gib)


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
    threads, configured, cpu_budget = _cpu_budget(project, physical, threads, cpu_override, available)
    requested = max(1, int(setting["sections"])) if mode in {"across", "both"} else 1
    workers = 1 if mode == "across" or not SECTION_STEPS[key][2] else max(1, int(setting["workers"]))
    workers = min(cpu_budget, workers)
    ram = _ram_budget(project, free_gib)
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


@dataclass(frozen=True)
class MipmapPlan:
    """'Automatic' for one run of a FEABAS mipmap stage; applied through the job environment only."""
    kind: str
    field: str
    sections: int
    workers: int
    across: bool
    values: dict
    cpu_budget: int | None
    note: str

    def env(self):
        """FW_STAGE_SETTINGS for vendor/winfix/sitecustomize.py: this run's values for the stage."""
        import json
        setting = dict(kind=self.kind, field=self.field, values=self.values)
        if self.cpu_budget:
            setting["cpu_budget"] = self.cpu_budget
        return {"FW_STAGE_SETTINGS": json.dumps(setting)}


def tiles_per_section(metadata_files, sample=16):
    """Average number of tiles listed in FEABAS metadata.txt files (a spread sample of them)."""
    files = list(metadata_files)
    if len(files) > sample:
        files = [files[round(i * (len(files) - 1) / (sample - 1))] for i in range(sample)]
    counts = []
    for f in files:
        try:
            lines = Path(f).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        counts.append(sum(1 for line in lines if line.strip() and not line.startswith("{")))
    return round(sum(counts) / len(counts)) if counts else 0


def mipmap_worker_gib(project, configs, key, cache=MIPMAP_CACHE_TILES):
    """Peak RAM of one mipmap worker: measured 1.2-1.7 GB for 4096² 8-bit tiles with a cache of 4-32
    tiles; it grows with the tile size, so unknown sample sizes count as 16-bit."""
    import numpy as np
    size = configs.get("alignment" if key == "align.downsample" else "stitching", "rendering.tile_size", [4096, 4096])
    size = list(size) if isinstance(size, (list, tuple)) else [size, size]
    try:
        itemsize = np.dtype(project.state.volume.dtype).itemsize
    except TypeError:
        itemsize = 2
    return 0.5 + (cache + 60) * int(size[0]) * int(size[-1]) * itemsize / GIB


def mipmap_plan(project, key, root=None, configs=None, *, start=None, stop=None, stride=None, reverse=False,
                available=None, threads=None, cpu_override=None):
    """
    'Automatic' for FEABAS's mipmap stages (PNG/JPEG tiles). FEABAS's default mipmaps one section
    at a time and starts a new pool of worker processes for every mip level, each with only as many
    jobs as that level has tiles (25, 9, 4, 4, ... for 64 tiles at mip 0), so a workstation mostly
    waits for processes to start. When there are at least as many sections as FEABAS workers, or
    at least 4 sections of at most 16 tiles per worker, each worker takes whole sections instead -
    FEABAS's own parallel_within_section: false, same output - one per core of the CPU budget as
    far as RAM allows. A larger read cache (MIPMAP_CACHE_TILES) stops the re-reading of source
    tiles either way. Settings the project sets itself are kept. None when the stage does not
    mipmap image tiles (a TensorStore stitch render keeps FEABAS's settings).
    """
    if key not in MIPMAP_STEPS:
        return None
    root = Path(root or project.root)
    cs = configs or ConfigStore(root / "configs")
    if key == "thumbnail.downsample" and cs.get("stitching", "rendering.driver", "image") != "image":
        return None
    kind, field, _ = SECTION_STEPS[key]
    try:
        paths, indices = selected_indices(input_sections(root, key, cs), key, start, stop, stride, None, reverse)
    except ValueError:
        return None
    selected = [paths[i] for i in indices]
    n = len(selected)
    physical, free_gib = available if available is not None else hardware()
    _, _, budget = _cpu_budget(project, physical, threads, cpu_override, available)
    doc = cs[kind]
    own = doc.is_overridden
    general = load_yaml(root / "configs" / "general_configs.yaml")
    try:
        feabas_budget = int(general.get("cpu_budget") or physical)   # FEABAS's own cap on its workers
    except (TypeError, ValueError):
        feabas_budget = int(physical)
    configured = max(1, int(cs.get(kind, field + ".num_workers", 1) or 1))
    within = max(1, min(configured, feabas_budget))
    if own(field + ".parallel_within_section"):
        across = not cs.get(kind, field + ".parallel_within_section", True)
    else:
        across = n > 1 and within > 1 and (n >= within or (n >= 4 and tiles_per_section(selected) <= 16 * within))
    cache = cs.get(kind, field + ".cache_size") if own(field + ".cache_size") else MIPMAP_CACHE_TILES
    values = {} if own(field + ".cache_size") else {"cache_size": cache}
    plural = "s" if n != 1 else ""
    if not across:
        note = (f"Automatic: {n} section{plural}, one at a time with FEABAS's {within} workers each"
                f"{'' if n < 2 else ' (few or large sections)'}; {cache}-tile read cache.")
        return MipmapPlan(kind, field, n, within, False, values, None, note)
    cap = configured if own(field + ".num_workers") else budget
    per_worker = mipmap_worker_gib(project, cs, key, int(cache or 4))
    ram = _ram_budget(project, free_gib)
    ram_limit = max(1, int(ram // per_worker))
    workers = max(1, min(n, cap, budget, ram_limit))
    values.update(parallel_within_section=False, num_workers=workers)
    limit = (f" (RAM budget {ram:.0f} GiB ÷ ~{per_worker:.1f} GiB per worker)"
             if ram_limit < min(n, cap, budget) else "")
    note = (f"Automatic: {n} section{plural}, {workers} at once with one worker each{limit}; "
            f"{cache}-tile read cache.")
    # FEABAS caps its workers at its CPU budget and gives each floor(budget / workers) threads
    return MipmapPlan(kind, field, n, workers, True, values, max(workers, budget), note)


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
