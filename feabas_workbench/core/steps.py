"""
Pipeline steps: definitions, on-disk state, clearing and snapshots.

FEABAS caches by file presence -- a step is done for a section when its
output exists, and re-running skips existing outputs. So the working directory
is a state machine that the workbench reads back from disk:

* per step: how many sections have outputs, whether error markers exist,
  whether the outputs are older than the config or the upstream outputs (stale)
* clear: delete a step's outputs and (optionally) everything downstream
* snapshot/restore: copy the small, expensive-to-recompute state
  (matches, meshes, transforms, configs) aside so a failed re-run can be undone

Output folders follow FEABAS: the rendered sections, the aligned PNG stack and the precomputed
volume go wherever ``rendering.out_dir`` / ``tensorstore_rendering.out_dir`` say (relative paths
are relative to the working directory), at the configured mip level.

Nothing here imports Qt.
"""

from __future__ import annotations

import datetime as _dt
import filecmp
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .configs import CHANGES_FILE, CONFIG_FILES, ConfigError, ConfigStore, dump_yaml


class State(str, Enum):
    EMPTY = "not started"
    PARTIAL = "partly done"
    COMPLETE = "done"
    STALE = "stale"
    ERROR = "errors"
    BLOCKED = "blocked"


class Cardinality(str, Enum):
    PER_SECTION = "per_section"
    PER_PAIR = "per_pair"
    SINGLE = "single"
    PER_LEVEL = "per_level"     # mip levels of one volume (the precomputed volume's mipmaps)


@dataclass(frozen=True)
class Step:
    key: str
    stage: str                  # stitch | thumbnail | masks | align | export
    label: str
    blurb: str
    script: str = ""            # vendored feabas script relative to vendor dir ('' for local steps)
    mode: str = ""              # --mode value
    out_subdir: str = ""        # main output folder relative to project root (the default for out_base steps)
    out_glob: str | tuple[str, ...] = "*.h5"   # may use {mip} / {max_mip} (aligned stack levels)
    err_glob: str | None = None
    cardinality: Cardinality = Cardinality.PER_SECTION
    config_kind: str | None = None          # stitching | thumbnail | alignment | material
    requires: tuple[str, ...] = ()
    supports_range: bool = True
    supports_filter: bool = True
    local: bool = False                     # performed by the workbench, not FEABAS
    optional: bool = False
    clear_paths: tuple[str, ...] = ()       # extra folders/files removed on clear (relative to root)
    snapshot_paths: tuple[str, ...] = ()    # folders worth snapshotting (relative to root)
    out_base: str = ""                      # "stitched" | "aligned": outputs live in FEABAS's configured render folder
    # the settings of config_kind that decide this step's results (dotted prefixes; all when
    # empty) and the parts of them that do not; worker counts and caches never count
    config_keys: tuple[str, ...] = ()
    config_exclude: tuple[str, ...] = ()


STEPS: tuple[Step, ...] = (
    Step("stitch.matching", "stitch", "Match tiles",
         "Find correspondences in the overlaps between neighbouring tiles of each section.",
         script="scripts/stitch_main.py", mode="matching",
         out_subdir="stitch/match_h5", out_glob="*.h5", err_glob="*.h5_err",
         config_kind="stitching", config_keys=("section_thickness", "matching"), snapshot_paths=("stitch/match_h5",)),
    Step("stitch.optimization", "stitch", "Optimize montage",
         "Elastically relax the tiles of each section into a consistent montage.",
         script="scripts/stitch_main.py", mode="optimization",
         out_subdir="stitch/tform", config_kind="stitching", config_keys=("optimization",), requires=("stitch.matching",),
         snapshot_paths=("stitch/tform",)),
    Step("stitch.rendering", "stitch", "Render montages",
         "Write each stitched section as PNG tiles or a TensorStore volume.",
         script="scripts/stitch_main.py", mode="rendering",
         # one output per finished section, whichever driver rendered it: PNG tiles write
         # mip0/<section>/metadata.txt last, the precomputed driver writes <section>/info
         out_subdir="stitched_sections", out_glob=("mip0/*/metadata.txt", "*/info"), out_base="stitched",
         config_kind="stitching", config_keys=("rendering",), requires=("stitch.optimization",),
         clear_paths=("stitch/ts_specs", "stitch/hist_tf")),
    Step("thumbnail.downsample", "thumbnail", "Make thumbnails",
         "Downsample the stitched sections to the coarse-alignment mip level and create default masks.",
         script="scripts/thumbnail_main.py", mode="downsample",
         # thumbnail_format in the thumbnail config picks the extension (png by default)
         out_subdir="thumbnail_align/thumbnails", out_glob=("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"),
         config_kind="thumbnail", config_keys=("thumbnail_mip_level", "downsample"), requires=("stitch.rendering",),
         clear_paths=("thumbnail_align/material_masks", "thumbnail_align/region_masks")),
    Step("masks", "masks", "Material masks",
         "Tissue/background and fold masks that tell FEABAS where the section is and where it is folded.",
         out_subdir="thumbnail_align/material_masks", out_glob="*.png",
         config_kind="material", requires=("thumbnail.downsample",),
         supports_range=False, local=True,
         clear_paths=("align/material_masks",)),
    Step("thumbnail.matching", "thumbnail", "Match thumbnails",
         "Coarse correspondences between neighbouring sections. The most failure-prone step; read its log.",
         script="scripts/thumbnail_main.py", mode="matching",
         out_subdir="thumbnail_align/matches", out_glob="*.h5", err_glob="*.h5_err",
         cardinality=Cardinality.PER_PAIR, config_kind="thumbnail", requires=("masks",),
         config_keys=("alignment",), config_exclude=("alignment.optimization", "alignment.render"),
         clear_paths=("thumbnail_align/feature_matches",), snapshot_paths=("thumbnail_align/matches",)),
    Step("thumbnail.optimization", "thumbnail", "Optimize coarse stack",
         "Solve the coarse stack; its result seeds the fine alignment.",
         script="scripts/thumbnail_main.py", mode="optimization",
         out_subdir="thumbnail_align/tform", config_kind="thumbnail", config_keys=("alignment.optimization",),
         requires=("thumbnail.matching",), supports_range=False,
         clear_paths=("thumbnail_align/mesh",), snapshot_paths=("thumbnail_align/tform",)),
    Step("thumbnail.render", "thumbnail", "Render coarse stack",
         "Aligned thumbnails to check the coarse result by eye (optional).",
         script="scripts/thumbnail_main.py", mode="render",
         out_subdir="thumbnail_align", out_glob="aligned_thumbnails_*/*.png",
         config_kind="thumbnail", config_keys=("alignment.render",), requires=("thumbnail.optimization",), optional=True,
         supports_range=False),
    # meshing reads the coarse transform (thumbnail_align/tform) as the meshes' starting position,
    # so a new coarse solution makes the meshes stale and clearing it clears them
    Step("align.meshing", "align", "Generate meshes",
         "Finite-element meshes per section, honouring the material masks.",
         script="scripts/align_main.py", mode="meshing",
         out_subdir="align/mesh", config_kind="alignment", config_keys=("meshing",),
         requires=("thumbnail.matching", "thumbnail.optimization", "masks"), supports_range=False),
    Step("align.matching", "align", "Fine matching",
         "Refine the coarse matches at the working mip level by block matching.",
         script="scripts/align_main.py", mode="matching",
         out_subdir="align/matches", out_glob="*.h5", err_glob="*.h5_err",
         cardinality=Cardinality.PER_PAIR, config_kind="alignment", config_keys=("matching",),
         requires=("align.meshing", "thumbnail.optimization"),
         clear_paths=("align/matches/match_cover",), snapshot_paths=("align/matches",)),
    Step("align.optimization", "align", "Optimize stack",
         "Relax all meshes of the stack against the fine matches.",
         script="scripts/align_main.py", mode="optimization",
         out_subdir="align/tform", config_kind="alignment", config_keys=("optimization",),
         requires=("align.matching",), supports_range=False,
         clear_paths=("align/tform/canvas.json",), snapshot_paths=("align/tform",)),
    Step("align.rendering", "align", "Render aligned stack (PNG tiles)",
         "Write the aligned stack as non-overlapping PNG tiles (VAST-style).",
         script="scripts/align_main.py", mode="rendering",
         out_subdir="aligned_stack", out_glob="mip{mip}/*/metadata.txt", out_base="aligned",
         config_kind="alignment", config_keys=("rendering",), requires=("align.optimization",), optional=True),
    Step("align.downsample", "align", "Mipmaps for PNG stack",
         "Downsampled levels of the PNG stack for viewing.",
         script="scripts/align_main.py", mode="downsample",
         # every level of a section is written in turn; the coarsest one comes last
         out_subdir="aligned_stack", out_glob="mip{max_mip}/*/metadata.txt", out_base="aligned",
         config_kind="alignment", config_keys=("downsample",), requires=("align.rendering",), optional=True),
    # the volume steps are judged by align/ts_spec.json, which FEABAS writes only once the volume (or a
    # further mip level of it) is complete - the volume folder itself appears as soon as rendering starts
    Step("align.tsr", "align", "Render aligned volume (precomputed)",
         "Write the aligned stack as a Neuroglancer precomputed volume via TensorStore.",
         script="scripts/align_main.py", mode="tensorstore_rendering",
         out_subdir="align", out_glob="ts_spec.json", config_kind="alignment", config_keys=("tensorstore_rendering",),
         requires=("align.optimization",), optional=True, cardinality=Cardinality.SINGLE,
         clear_paths=("align/render_flags", "align/ts_spec.json", "align/mask.png")),
    Step("align.tsd", "align", "Mipmaps for volume",
         "Downsample the precomputed volume.",
         script="scripts/align_main.py", mode="tensorstore_downsample",
         out_subdir="align", out_glob="ts_spec.json", config_kind="alignment", config_keys=("tensorstore_downsample",),
         requires=("align.tsr",), optional=True, cardinality=Cardinality.PER_LEVEL,
         clear_paths=("align/mipmap_flags",)),
)

STEPS_BY_KEY = {s.key: s for s in STEPS}
STAGE_LABELS = {
    "stitch": "Stitching (xy montage)",
    "thumbnail": "Coarse alignment",
    "masks": "Masks",
    "align": "Fine alignment & rendering",
    "export": "Export",
}

# steps whose every output depends on all of their inputs (a solve of the whole stack): compared as a whole
GLOBAL_STEPS = frozenset({"thumbnail.optimization", "align.optimization"})


def downstream(key: str) -> list[Step]:
    """All steps that (transitively) require *key*, in pipeline order."""
    out: list[Step] = []
    closed = {key}
    changed = True
    while changed:
        changed = False
        for s in STEPS:
            if s.key in closed:
                continue
            if any(r in closed for r in s.requires):
                closed.add(s.key)
                changed = True
    for s in STEPS:
        if s.key in closed and s.key != key:
            out.append(s)
    return out


# ----------------------------------------------------------------------
# where FEABAS puts things
# ----------------------------------------------------------------------

def _configs_for(root: Path, configs: ConfigStore | None) -> ConfigStore | None:
    """The given configs, or the ones in <root>/configs (without the editor's hints), or None."""
    if configs is not None:
        return configs
    d = Path(root) / "configs"
    if not d.is_dir():
        return None
    try:
        return ConfigStore(d, hints=False)
    except ConfigError:
        return None


def _conf(configs: ConfigStore | None, kind: str, key: str, default=None):
    if configs is None:
        return default
    try:
        value = configs.get(kind, key, default)
    except KeyError:
        return default
    return default if value is None else value


def _local_folder(root: Path, value, default_name: str) -> Path | None:
    """A configured output folder as FEABAS resolves it; None for a remote store (gs://, s3://)."""
    root = Path(root)
    if not value:
        return root / default_name
    text = str(value).replace("\\", "/")
    if text.startswith("file://"):
        text = text[len("file://"):]
        if re.match(r"^/[A-Za-z]:/", text):          # file:///D:/... on Windows
            text = text[1:]
    if "://" in text:
        return None
    p = Path(os.path.expanduser(os.path.expandvars(text)))
    return p if p.is_absolute() else root / p


def stitched_dir(root, configs: ConfigStore | None = None) -> Path:
    """Where 'Render montages' writes: stitching rendering.out_dir, else <root>/stitched_sections."""
    cs = _configs_for(root, configs)
    return _local_folder(root, _conf(cs, "stitching", "rendering.out_dir"), "stitched_sections") or Path(root) / "stitched_sections"


def aligned_dir(root, configs: ConfigStore | None = None) -> Path:
    """Where the aligned PNG stack goes: alignment rendering.out_dir, else <root>/aligned_stack."""
    cs = _configs_for(root, configs)
    return _local_folder(root, _conf(cs, "alignment", "rendering.out_dir"), "aligned_stack") or Path(root) / "aligned_stack"


def tensorstore_dir(root, configs: ConfigStore | None = None) -> Path | None:
    """The precomputed volume: alignment tensorstore_rendering.out_dir, else <root>/aligned_tensorstore."""
    cs = _configs_for(root, configs)
    return _local_folder(root, _conf(cs, "alignment", "tensorstore_rendering.out_dir"), "aligned_tensorstore")


def aligned_render_mip(configs: ConfigStore | None) -> int:
    return int(_conf(configs, "alignment", "rendering.mip_level", 0) or 0)


def aligned_max_mip(configs: ConfigStore | None) -> int:
    return int(_conf(configs, "alignment", "downsample.max_mip", 7) or 0)


def tensorstore_render_mip(configs: ConfigStore | None) -> int:
    return int(_conf(configs, "alignment", "tensorstore_rendering.mip_level", 0) or 0)


def tensorstore_mip_levels(configs: ConfigStore | None) -> list[int]:
    """The mip levels 'Mipmaps for volume' adds (above the rendered one)."""
    levels = _conf(configs, "alignment", "tensorstore_downsample.mip_levels", [1, 3, 5, 7])
    if not isinstance(levels, (list, tuple)):
        levels = [levels]
    base = tensorstore_render_mip(configs)
    out = set()
    for lv in levels:
        try:
            if int(lv) > base:
                out.add(int(lv))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def _ts_spec(root: Path) -> tuple[Path, set[int]]:
    """align/ts_spec.json and the mip levels it records as finished."""
    p = Path(root) / "align" / "ts_spec.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return p, set()
    levels = set()
    for k in data if isinstance(data, dict) else ():
        try:
            levels.add(int(k))
        except (TypeError, ValueError):
            pass
    return p, levels


def step_output_dir(root, step: Step, configs: ConfigStore | None = None) -> Path:
    cs = _configs_for(root, configs)
    if step.out_base == "stitched":
        return stitched_dir(root, cs)
    if step.out_base == "aligned":
        return aligned_dir(root, cs)
    return Path(root) / step.out_subdir


def _patterns(step: Step, cs: ConfigStore | None) -> tuple[str, ...]:
    patterns = (step.out_glob,) if isinstance(step.out_glob, str) else tuple(step.out_glob)
    if any("{" in p for p in patterns):
        values = {"mip": aligned_render_mip(cs), "max_mip": aligned_max_mip(cs)}
        patterns = tuple(p.format(**values) for p in patterns)
    return patterns


_Z_PREFIX = re.compile(r"^\d+_(.+)$")


def _item_key(step: Step, path: Path, cs: ConfigStore | None, delim: str):
    """What one output file stands for: a section name, or a (section, section) pair."""
    if step.key == "stitch.rendering":
        return path.parent.name                       # mip0/<sec>/metadata.txt or <sec>/info
    if step.out_base == "aligned":
        name = path.parent.name                       # mipN/<zz>_<sec>/metadata.txt
        m = _Z_PREFIX.match(name) if _conf(cs, "alignment", "rendering.prefix_z_number", True) else None
        return m.group(1) if m else name
    if step.cardinality is Cardinality.PER_PAIR:
        stem = path.stem
        return tuple(stem.split(delim, 1)) if delim and delim in stem else (stem,)
    return path.stem


def _step_outputs(root: Path, step: Step, configs: ConfigStore | None = None) -> dict:
    """{item: file} for every finished output of *step*; the file's time is the item's time."""
    root = Path(root)
    cs = _configs_for(root, configs)
    if step.key == "align.tsr":
        spec, levels = _ts_spec(root)
        return {"volume": spec} if tensorstore_render_mip(cs) in levels else {}
    if step.key == "align.tsd":
        spec, levels = _ts_spec(root)
        return {lv: spec for lv in tensorstore_mip_levels(cs) if lv in levels}
    d = step_output_dir(root, step, cs)
    if not d.exists():
        return {}
    delim = str(_conf(cs, "thumbnail", "alignment.match_name_delimiter", "__to__")) if step.cardinality is Cardinality.PER_PAIR else ""
    out: dict = {}
    try:
        for pattern in _patterns(step, cs):         # several patterns: one output layout per render driver
            for p in d.glob(pattern):
                if p.name.startswith("."):
                    continue
                key = _item_key(step, p, cs, delim)
                if key in out and _mtime(out[key]) >= _mtime(p):
                    continue
                out[key] = p
    except OSError:
        return {}
    return out


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


# ----------------------------------------------------------------------
# scanning
# ----------------------------------------------------------------------

@dataclass
class StepStatus:
    step: Step
    state: State = State.EMPTY
    done: int = 0
    expected: int = 0
    errors: int = 0
    newest_output: float = 0.0
    oldest_output: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return (self.done / self.expected) if self.expected else 0.0

    def summary(self) -> str:
        if self.step.cardinality is Cardinality.SINGLE:
            base = "present" if self.done else "absent"
        elif self.step.cardinality is Cardinality.PER_LEVEL:
            base = f"{self.done}/{self.expected} mip levels"
        else:
            base = f"{self.done}/{self.expected}"
        if self.errors:
            base += f", {self.errors} error file(s)"
        return base


def expected_count(root: Path, n_sections: int, step: Step, configs: ConfigStore | None = None) -> int:
    """
    How many outputs a complete run of *step* leaves: one per section, one per matched pair
    (the coarse compare distance decides how many pairs there are; the fine steps mirror the
    coarse match list or align/match_name.txt), one per requested mip level, or a single output.
    """
    root = Path(root)
    n = int(n_sections)
    if step.cardinality is Cardinality.PER_SECTION:
        return n
    if step.cardinality is Cardinality.PER_PAIR:
        if step.key.startswith("thumbnail"):
            cd = configs.get("thumbnail", "alignment.compare_distance", 2) if configs is not None else 2
            if isinstance(cd, (list, tuple)):
                ks = [int(k) for k in cd]
            else:
                try:
                    ks = list(range(1, int(cd) + 1))
                except (TypeError, ValueError):
                    ks = [1]
            return max(0, sum(max(0, n - k) for k in ks))
        if step.key.startswith("align"):
            listed = read_fine_match_list(root)
            if listed is not None:
                return len(listed)
            return len(list((root / "thumbnail_align" / "matches").glob("*.h5"))) or max(0, n - 1)
        return max(0, n - 1)
    if step.cardinality is Cardinality.PER_LEVEL:
        return len(tensorstore_mip_levels(_configs_for(root, configs)))
    return 1


def count_outputs(root: Path, step: Step, configs: ConfigStore | None = None) -> int:
    n = len(_step_outputs(root, step, configs))
    if step.cardinality is Cardinality.SINGLE:
        return 1 if n else 0
    return n


def _item_kind(step: Step) -> str:
    if step.key in GLOBAL_STEPS or step.cardinality in (Cardinality.SINGLE, Cardinality.PER_LEVEL):
        return "global"
    return "pair" if step.cardinality is Cardinality.PER_PAIR else "section"


def _sections(key) -> tuple:
    return key if isinstance(key, tuple) else (key,)


def _newer_inputs(down: dict, up: dict, down_kind: str, up_kind: str) -> bool:
    """
    Whether an input that an output depends on is newer than that output ({item: mtime} each).

    Section-wise outputs are compared with the inputs of the same section (a pair with both of
    its sections, a section with every pair it takes part in), so running an earlier step for a
    few more sections leaves the finished ones alone. A stack solve depends on everything and is
    compared as a whole.
    """
    if not down or not up:
        return False
    if down_kind == "global" or up_kind == "global":
        return max(up.values()) > min(down.values()) + 1
    if up_kind == "pair":
        by_section: dict = {}
        for k, t in up.items():
            for s in _sections(k):
                by_section[s] = max(by_section.get(s, 0.0), t)
    else:
        by_section = up
    for k, t in down.items():
        if down_kind == "pair" and up_kind == "pair":
            u = up.get(k)
            if u is not None and u > t + 1:
                return True
            continue
        for s in _sections(k):
            u = by_section.get(s)
            if u is not None and u > t + 1:
                return True
    return False


class PipelineScan:
    """Snapshot of every step's state for a project root."""

    def __init__(self, root: Path, n_sections: int, configs: ConfigStore | None = None):
        self.root = Path(root)
        self.n_sections = n_sections
        self.configs = configs
        self.status: dict[str, StepStatus] = {}
        self.scan()

    def expected_for(self, step: Step) -> int:
        return expected_count(self.root, self.n_sections, step, self.configs)

    def scan(self) -> None:
        self.status = {}
        locations = _configs_for(self.root, self.configs)      # output folders, even without staleness configs
        items: dict[str, dict] = {}
        for step in STEPS:
            st = StepStatus(step)
            outs = _step_outputs(self.root, step, locations)
            times = {k: _mtime(p) for k, p in outs.items()}
            items[step.key] = times
            st.done = (1 if outs else 0) if step.cardinality is Cardinality.SINGLE else len(outs)
            st.expected = self.expected_for(step)
            if step.err_glob:
                d = self.root / step.out_subdir
                st.errors = len(list(d.glob(step.err_glob))) if d.exists() else 0
            if times:
                st.newest_output, st.oldest_output = max(times.values()), min(times.values())
            self.status[step.key] = st
        coords = {p.stem: _mtime(p) for p in (self.root / "stitch" / "stitch_coord").glob("*.txt")}
        # states, in order so upstream is known
        for step in STEPS:
            st = self.status[step.key]
            blocked = False
            for r in step.requires:
                up = self.status.get(r)
                if up is None:
                    continue
                if up.done == 0 and not up.step.optional:
                    blocked = True
                    st.reasons.append(f"waiting for '{up.step.label}'")
            if st.errors:
                st.state = State.ERROR
                continue
            if st.done == 0:
                st.state = State.BLOCKED if blocked else State.EMPTY
                continue
            if step.cardinality is Cardinality.SINGLE:
                st.state = State.COMPLETE
            else:
                st.state = State.COMPLETE if (st.expected and st.done >= st.expected) else State.PARTIAL
            mine = items[step.key]
            if self.configs is not None and step.config_kind and st.oldest_output:
                # only settings that decide this step's results: the thumbnail config also holds the
                # coarse alignment's settings, and worker counts change no result
                changed = self.configs.changed_keys(step.config_kind, st.oldest_output + 1,
                                                    step.config_keys, step.config_exclude)
                if changed == ["*"]:
                    # not saved by the settings pages (edited by hand, restored): which setting is unknown
                    self._stale(st, f"{step.config_kind} config file changed after these outputs")
                elif changed:
                    self._stale(st, f"{step.config_kind} config edited after these outputs: "
                                + ", ".join(changed[:3]) + (" …" if len(changed) > 3 else ""))
            if step.key == "stitch.matching" and _newer_inputs(mine, coords, "section", "section"):
                self._stale(st, "coordinate files changed after these outputs")
            for r in step.requires:
                up = self.status.get(r)
                if up is None:
                    continue
                if _newer_inputs(mine, items[r], _item_kind(step), _item_kind(up.step)):
                    self._stale(st, f"'{up.step.label}' produced newer outputs")
                elif up.state is State.STALE:
                    self._stale(st, f"'{up.step.label}' is stale")

    @staticmethod
    def _stale(st: StepStatus, reason: str) -> None:
        st.state = State.STALE
        st.reasons.append(reason)

    def __getitem__(self, key: str) -> StepStatus:
        return self.status[key]


# ----------------------------------------------------------------------
# running a step
# ----------------------------------------------------------------------

def step_argv(python: str, step: Step, start: int | None = None, stop: int | None = None,
              stride: int | None = None, filt: str | None = None, extra_args: list[str] | None = None) -> list[str]:
    """The command line of one FEABAS step: the vendored driver script with its --mode and range."""
    from .project import VENDOR_DIR
    argv = [python, str(VENDOR_DIR / step.script), "--mode", step.mode]
    if step.supports_range:
        if start:
            argv += ["--start", str(start)]
        if stop:
            argv += ["--stop", str(stop)]
        if stride and stride > 1:
            argv += ["--step", str(stride)]
    if filt and step.supports_filter:
        argv += ["--filter", filt]
    argv += list(extra_args or [])
    return argv


def expected_outputs(root: Path, step: Step, start: int | None = None, stop: int | None = None,
                     stride: int | None = None) -> int:
    """How many outputs a run of *step* should leave behind (sections, pairs, or one), for a range too."""
    root = Path(root)
    n = len([p for p in (root / "stitch" / "stitch_coord").glob("*.txt")])
    configs = _configs_for(root, None)
    expected = expected_count(root, n, step, configs)
    if start is not None or stop is not None:
        s0 = start or 0
        s1 = stop if stop else expected
        expected = max(0, len(range(s0, min(s1, expected), stride or 1)))
    return expected


# The steps of a plain run, in order: stitch, thumbnails (FEABAS writes default masks with them),
# coarse alignment, fine alignment. Rendering the aligned stack is optional and heavy, so it is
# a separate list.
STANDARD_PIPELINE: tuple[str, ...] = (
    "stitch.matching", "stitch.optimization", "stitch.rendering",
    "thumbnail.downsample",
    "thumbnail.matching", "thumbnail.optimization", "thumbnail.render",
    "align.meshing", "align.matching", "align.optimization",
)
RENDER_PIPELINE: tuple[str, ...] = ("align.rendering", "align.downsample")


# ----------------------------------------------------------------------
# step-specific progress
# ----------------------------------------------------------------------

def thumbnail_max_mip(configs: ConfigStore | None) -> int:
    """
    The highest stitched-section mip level 'Make thumbnails' builds, as thumbnail_main.py
    computes it for the PNG-tile render driver: downsample.max_mip if set, else one below the
    thumbnail mip, and never below the fine-alignment working mip.
    """
    if configs is None:
        return 0
    tm = int(configs.get("thumbnail", "thumbnail_mip_level", 2) or 0)
    mm = configs.get("thumbnail", "downsample.max_mip", None)
    max_mip = int(mm) if mm is not None else max(0, tm - 1)
    align_mip = int(configs.get("alignment", "matching.working_mip_level", 2) or 0)
    return max(align_mip, max_mip)


def thumbnail_progress(root: Path, configs: ConfigStore | None, n_sections: int) -> tuple[int, int, str]:
    """
    Progress of 'Make thumbnails' as (done, expected, message).

    FEABAS first builds the intermediate mip levels of every stitched section - by far the
    slowest part - and only then writes the thumbnails, so counting thumbnails alone shows
    nothing until the run is almost over. Every finished mip level of a section leaves a
    ``<stitched sections>/mipN/<sec>/metadata.txt`` behind, so those are counted too.
    """
    root = Path(root)
    n = max(0, int(n_sections))
    thumbs = count_outputs(root, STEPS_BY_KEY["thumbnail.downsample"], configs)
    driver = configs.get("stitching", "rendering.driver", "image") if configs is not None else "image"
    max_mip = thumbnail_max_mip(configs) if driver == "image" else 0
    if max_mip <= 0 or n == 0:
        return thumbs, n, f"thumbnails {thumbs}/{n}"
    ss = stitched_dir(root, configs)
    levels = 0
    for m in range(1, max_mip + 1):
        d = ss / f"mip{m}"
        if d.is_dir():
            levels += len([p for p in d.glob("*/metadata.txt")])
    levels = min(levels, n * max_mip)
    expected = n * max_mip + n
    return levels + thumbs, expected, f"mip levels {levels}/{n * max_mip}, thumbnails {thumbs}/{n}"


# ----------------------------------------------------------------------
# fine-alignment match list (align/match_name.txt)
# ----------------------------------------------------------------------

FINE_MATCH_LIST = "align/match_name.txt"


def read_fine_match_list(root: Path) -> list[str] | None:
    """The pair names listed in align/match_name.txt, or None when FEABAS uses every thumbnail match."""
    p = Path(root) / FINE_MATCH_LIST
    if not p.is_file():
        return None
    try:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return None
    return [ln[:-3] if ln.endswith(".h5") else ln for ln in lines if ln]


def fine_match_pairs(root: Path, section_names: list[str], compare_distance: int,
                     delimiter: str = "__to__") -> list[str]:
    """
    Thumbnail match pairs (file stems in thumbnail_align/matches) whose two sections are at most
    *compare_distance* apart in the section order. The coarse alignment often compares each
    section to two or more neighbours on either side for robustness; the fine alignment does not
    need that many pairs and every extra pair costs a full block-matching pass.
    """
    order = {name: i for i, name in enumerate(section_names)}
    out = []
    for p in sorted((Path(root) / "thumbnail_align" / "matches").glob("*.h5")):
        stem = p.stem
        if delimiter not in stem:
            continue
        a, b = stem.split(delimiter, 1)
        if a in order and b in order and abs(order[a] - order[b]) <= int(compare_distance):
            out.append(stem)
    return out


def write_fine_match_list(root: Path, section_names: list[str], compare_distance: int | None,
                          delimiter: str = "__to__") -> list[str] | None:
    """
    Write align/match_name.txt so that 'Generate meshes' and 'Fine matching' only use the pairs
    within *compare_distance*; None (or a distance that keeps every pair) removes the file, which
    is FEABAS's default of using every thumbnail match. Returns the pairs listed, or None.
    """
    p = Path(root) / FINE_MATCH_LIST
    if compare_distance is None or int(compare_distance) <= 0:
        if p.is_file():
            p.unlink()
        return None
    pairs = fine_match_pairs(root, section_names, int(compare_distance), delimiter)
    all_pairs = len(list((Path(root) / "thumbnail_align" / "matches").glob("*.h5")))
    if not pairs or len(pairs) == all_pairs:
        if p.is_file():
            p.unlink()
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(f"{s}.h5" for s in pairs) + "\n", encoding="utf-8")
    return pairs


# ----------------------------------------------------------------------
# clearing
# ----------------------------------------------------------------------

_MOUNT_POINT = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def is_link(p: Path) -> bool:
    """A symbolic link, or a directory junction on Windows (what test runs use to share folders)."""
    try:
        if Path(p).is_symlink():
            return True
        if os.name == "nt":
            return getattr(os.lstat(p), "st_reparse_tag", 0) == _MOUNT_POINT
    except OSError:
        return False
    return False


def _through_link(root: Path, p: Path) -> bool:
    """True when *p* is a link or is reached through one below *root*: its data belongs elsewhere."""
    if is_link(p):
        return True
    try:
        rel = p.relative_to(root)
    except ValueError:
        return False
    cur = root
    for part in rel.parts[:-1]:
        cur = cur / part
        if is_link(cur):
            return True
    return False


def _mip_dirs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("mip*") if p.is_dir() and p.name[3:].isdigit())


def _render_folder_targets(root: Path, folder: Path, default_name: str) -> list[Path]:
    """
    Everything a render step wrote into its folder. The workbench's own default folder goes as a
    whole; a folder chosen in the settings may hold other data, so only the mip levels and the
    per-section volumes (<section>/info) FEABAS put there are removed.
    """
    if folder == root / default_name:
        return [folder]
    out = _mip_dirs(folder)
    if folder.is_dir():
        out += sorted(p for p in folder.iterdir() if p.is_dir() and (p / "info").is_file())
    return out


def _volume_scales(volume: Path) -> list[dict]:
    try:
        info = json.loads((volume / "info").read_text(encoding="utf-8"))
        scales = info.get("scales") or []
        return sorted(scales, key=lambda s: float((s.get("resolution") or [0])[0]))
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def _scale_dir(volume: Path, scale: dict) -> Path | None:
    key = str(scale.get("key") or "")
    if not key or key.startswith(("/", "\\")) or ".." in key.replace("\\", "/").split("/") or ":" in key:
        return None
    return volume / key


def _step_clear_candidates(root: Path, step: Step, cs: ConfigStore | None) -> list[Path]:
    key = step.key
    if key == "stitch.rendering":
        return _render_folder_targets(root, stitched_dir(root, cs), "stitched_sections") + [root / p for p in step.clear_paths]
    if key == "thumbnail.downsample":
        out = [root / "thumbnail_align" / "thumbnails"] + [root / p for p in step.clear_paths]
        # the intermediate mip levels of the stitched sections are made by this step too
        out += [p for p in _mip_dirs(stitched_dir(root, cs)) if p.name != "mip0"]
        # and the workbench's masks are drawn on these thumbnails: made again, they no longer fit
        out += [root / "masks" / n for n in ("roi", "tissue", "folds", "structures", "manifest.json")]
        return out
    if key == "masks":
        # the hand-edited flags describe material masks that are gone
        return [root / step.out_subdir, root / "align" / "material_masks", root / "masks" / "manifest.json"]
    if key == "thumbnail.render":
        return sorted((root / "thumbnail_align").glob("aligned_thumbnails_*"))
    if key == "align.rendering":
        return _render_folder_targets(root, aligned_dir(root, cs), "aligned_stack")
    if key == "align.downsample":
        mip = aligned_render_mip(cs)
        return [p for p in _mip_dirs(aligned_dir(root, cs)) if int(p.name[3:]) > mip]
    if key == "align.tsr":
        out = []
        vol = tensorstore_dir(root, cs)
        if vol is not None:
            if vol == root / "aligned_tensorstore":
                out.append(vol)
            else:
                scales = [_scale_dir(vol, s) for s in _volume_scales(vol)]
                out += [s for s in scales if s is not None] + [vol / "info"]
        return out + [root / p for p in step.clear_paths] + sorted((root / "align").glob("ts_spec_*.json"))
    if key == "align.tsd":
        out = []
        vol = tensorstore_dir(root, cs)
        if vol is not None:
            out += [s for s in (_scale_dir(vol, sc) for sc in _volume_scales(vol)[1:]) if s is not None]
        return out + [root / p for p in step.clear_paths]
    return [root / step.out_subdir] + [root / p for p in step.clear_paths]


def clear_targets(root: Path, step: Step, cascade: bool = True, configs: ConfigStore | None = None,
                  skipped: list | None = None) -> list[Path]:
    """
    Paths that would be removed when clearing *step* (and downstream). Nothing is removed through
    a link: an alignment test run links the project's stitched sections, which are not its own.
    Such paths are left out (and listed in *skipped*, if given).
    """
    root = Path(root)
    cs = _configs_for(root, configs)
    steps = [step] + (downstream(step.key) if cascade else [])
    targets: list[Path] = []
    seen = set()
    for s in steps:
        for c in _step_clear_candidates(root, s, cs):
            if c in seen:
                continue
            seen.add(c)
            if not (c.exists() or c.is_symlink()):
                continue
            if _through_link(root, c):
                if skipped is not None:
                    skipped.append(c)
                continue
            targets.append(c)
    # something inside another target goes with it
    return [t for t in targets if not any(o in t.parents for o in targets if o != t)]


def cleared_keys(step: Step, cascade: bool = True) -> list[str]:
    return [step.key] + ([s.key for s in downstream(step.key)] if cascade else [])


def finish_clear(root: Path, keys, configs: ConfigStore | None = None) -> None:
    """
    Bookkeeping after the files of *keys* were removed. Clearing only the volume's mipmaps
    removes their scale folders; the volume's info file and align/ts_spec.json still list them,
    so both are cut back to the rendered level.
    """
    keys = set(keys)
    if "align.tsd" not in keys or "align.tsr" in keys:
        return
    root = Path(root)
    cs = _configs_for(root, configs)
    vol = tensorstore_dir(root, cs)
    if vol is not None and (vol / "info").is_file() and not is_link(vol):
        try:
            info = json.loads((vol / "info").read_text(encoding="utf-8"))
            scales = sorted(info.get("scales") or [], key=lambda s: float((s.get("resolution") or [0])[0]))
            if len(scales) > 1:
                info["scales"] = scales[:1]
                (vol / "info").write_text(json.dumps(info, indent=2), encoding="utf-8")
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    spec = root / "align" / "ts_spec.json"
    try:
        data = json.loads(spec.read_text(encoding="utf-8"))
        base = str(tensorstore_render_mip(cs))
        kept = {k: v for k, v in data.items() if str(k) == base}
        if kept != data:
            spec.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    except (OSError, ValueError, AttributeError):
        pass


def clear_step(root: Path, step: Step, cascade: bool = True, log=None, configs: ConfigStore | None = None) -> list[Path]:
    removed = []
    for p in clear_targets(root, step, cascade, configs):
        try:
            if p.is_dir() and not is_link(p):
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(p)
            if log:
                log(f"removed {p}")
        except OSError as e:
            if log:
                log(f"could not remove {p}: {e}")
    finish_clear(root, cleared_keys(step, cascade), configs)
    return removed


def drop_unreadable_outputs(root: Path, step: Step, since: float, configs: ConfigStore | None = None) -> list[Path]:
    """
    After a step was killed (Cancel): remove the .h5 outputs written since *since* that do not open.

    FEABAS writes its .h5 files in place and skips every output that exists, so a file cut short
    by the kill would count as finished and break the next step. Complete files written before
    the cancel are kept, as are all files from earlier runs.
    """
    root = Path(root)
    if not any(p.endswith(".h5") for p in ((step.out_glob,) if isinstance(step.out_glob, str) else step.out_glob)):
        return []
    try:
        import h5py
    except ImportError:
        return []
    removed = []
    for p in _step_outputs(root, step, configs).values():
        try:
            if p.suffix != ".h5" or p.stat().st_mtime < since - 1:
                continue
            with h5py.File(p, "r") as f:
                list(f.keys())
        except OSError:
            try:
                p.unlink()
                removed.append(p)
            except OSError:
                pass
    return removed


def clear_error_files(root: Path, step: Step) -> list[Path]:
    if not step.err_glob:
        return []
    d = root / step.out_subdir
    out = []
    for p in d.glob(step.err_glob) if d.exists() else []:
        try:
            p.unlink()
            out.append(p)
        except OSError:
            pass
    return out


# ----------------------------------------------------------------------
# snapshots
# ----------------------------------------------------------------------

SNAPSHOT_PATHS = (
    "configs",
    "section_order.txt",
    "stitch/stitch_coord",
    "stitch/match_h5",
    "stitch/tform",
    "thumbnail_align/matches",
    "thumbnail_align/manual_matches",
    "thumbnail_align/material_masks",
    "thumbnail_align/tform",
    "align/material_masks",
    "align/mesh",
    "align/matches",
    "align/tform",
)


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def estimate_snapshot_size(root: Path) -> int:
    total = 0
    for rel in SNAPSHOT_PATHS:
        p = root / rel
        if p.is_dir():
            total += _dir_size(p)
        elif p.is_file():
            total += p.stat().st_size
    return total


def create_snapshot(root: Path, label: str = "", log=None) -> Path:
    root = Path(root)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.strip())[:40]
    dest = root / "snapshots" / (f"{stamp}_{safe}" if safe else stamp)
    dest.mkdir(parents=True, exist_ok=False)
    included = []
    for rel in SNAPSHOT_PATHS:
        src = root / rel
        if src.is_dir():
            shutil.copytree(src, dest / rel, dirs_exist_ok=True)
            included.append(rel)
        elif src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest / rel)
            included.append(rel)
        if log:
            log(f"snapshot: {rel}")
    (dest / "snapshot.json").write_text(json.dumps({
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "label": label, "paths": included}, indent=2), encoding="utf-8")
    return dest


def list_snapshots(root: Path) -> list[dict]:
    out = []
    d = Path(root) / "snapshots"
    if not d.is_dir():
        return out
    for p in sorted(d.iterdir()):
        meta = p / "snapshot.json"
        if not meta.is_file():
            continue
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except ValueError:
            info = {}
        info["path"] = p
        info["name"] = p.name
        out.append(info)
    return out


def _sync_file(src: Path, dst: Path) -> bool:
    """Copy *src* over *dst* unless they are identical already. True if *dst* changed."""
    if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _sync_tree(src: Path, dst: Path, keep: frozenset = frozenset()) -> int:
    """
    Make *dst* hold exactly the files of *src*; identical files are left untouched. Files named in
    *keep* (relative paths) are neither copied nor removed. Returns the number of files changed.
    """
    changed = 0
    wanted = set()
    for f in sorted(src.rglob("*")):
        if f.is_file() and f.relative_to(src).as_posix() not in keep:
            rel = f.relative_to(src)
            wanted.add(rel)
            changed += _sync_file(f, dst / rel)
    if dst.is_dir():
        for f in sorted(dst.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            rel = f.relative_to(dst)
            if f.is_file() or f.is_symlink():
                if rel not in wanted and rel.as_posix() not in keep:
                    f.unlink()
                    changed += 1
            elif f.is_dir() and not any(f.iterdir()) and not (src / rel).is_dir():
                f.rmdir()
    return changed


def restore_snapshot(root: Path, snapshot_dir: Path, log=None) -> list[str]:
    """
    Put the snapshotted folders back as they were.

    Only files that differ are written, and they get the current time: whatever was computed
    from other inputs after the snapshot (later steps, rendered images) is then older than the
    restored inputs and shows as stale, while outputs whose inputs did not change stay done.
    """
    root = Path(root)
    snapshot_dir = Path(snapshot_dir)
    meta = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
    order = {rel: i for i, rel in enumerate(SNAPSHOT_PATHS)}
    restored = []
    for rel in sorted((r for r in meta.get("paths", []) if r in order), key=order.get):
        src, dst = snapshot_dir / rel, root / rel
        if _through_link(root, dst):
            if log:
                log(f"not restored (a link): {rel}")
            continue
        if src.is_dir():
            configs = rel == "configs"
            present = {name for name in CONFIG_FILES.values() if (dst / name).is_file()} if configs else set()
            # the record of which setting changed when stays: settings the restore rewrites get
            # new file times and count as changed, the ones it leaves alone stay as they are
            n = _sync_tree(src, dst, frozenset({CHANGES_FILE}) if configs else frozenset())
            for name in present:
                if not (dst / name).is_file():
                    # back to the defaults: kept as {} like the settings pages do, so the change shows
                    dump_yaml(dst / name, {}, header=f"Project overrides for default_{name}; "
                                                    "unspecified keys use the defaults.")
        elif src.is_file():
            n = int(_sync_file(src, dst))
        else:
            continue
        restored.append(rel)
        if log:
            log(f"restored {rel}" + (f" ({n} file(s) changed)" if n else " (unchanged)"))
    from .project import repair_working_directory
    repair_working_directory(root)          # the snapshot may come from before the project was moved
    return restored


def delete_snapshot(snapshot_dir: Path) -> None:
    shutil.rmtree(snapshot_dir)
