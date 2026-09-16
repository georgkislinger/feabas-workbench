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

Nothing here imports Qt.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

from .configs import ConfigStore


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


@dataclass(frozen=True)
class Step:
    key: str
    stage: str                  # stitch | thumbnail | masks | align | export
    label: str
    blurb: str
    script: str = ""            # vendored feabas script relative to vendor dir ('' for local steps)
    mode: str = ""              # --mode value
    out_subdir: str = ""        # main output folder relative to project root
    out_glob: str | tuple[str, ...] = "*.h5"
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


STEPS: tuple[Step, ...] = (
    Step("stitch.matching", "stitch", "Match tiles",
         "Find correspondences in the overlaps between neighbouring tiles of each section.",
         script="scripts/stitch_main.py", mode="matching",
         out_subdir="stitch/match_h5", out_glob="*.h5", err_glob="*.h5_err",
         config_kind="stitching", snapshot_paths=("stitch/match_h5",)),
    Step("stitch.optimization", "stitch", "Optimize montage",
         "Elastically relax the tiles of each section into a consistent montage.",
         script="scripts/stitch_main.py", mode="optimization",
         out_subdir="stitch/tform", config_kind="stitching", requires=("stitch.matching",),
         snapshot_paths=("stitch/tform",)),
    Step("stitch.rendering", "stitch", "Render montages",
         "Write each stitched section as PNG tiles or a TensorStore volume.",
         script="scripts/stitch_main.py", mode="rendering",
         # one output per finished section, whichever driver rendered it: PNG tiles write
         # mip0/<section>/metadata.txt last, the precomputed driver writes <section>/info
         out_subdir="stitched_sections", out_glob=("mip0/*/metadata.txt", "*/info"),
         config_kind="stitching", requires=("stitch.optimization",),
         clear_paths=("stitch/ts_specs", "stitch/hist_tf")),
    Step("thumbnail.downsample", "thumbnail", "Make thumbnails",
         "Downsample the stitched sections to the coarse-alignment mip level and create default masks.",
         script="scripts/thumbnail_main.py", mode="downsample",
         out_subdir="thumbnail_align/thumbnails", out_glob="*.png",
         config_kind="thumbnail", requires=("stitch.rendering",),
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
         clear_paths=("thumbnail_align/feature_matches",), snapshot_paths=("thumbnail_align/matches",)),
    Step("thumbnail.optimization", "thumbnail", "Optimize coarse stack",
         "Solve the coarse stack; its result seeds the fine alignment.",
         script="scripts/thumbnail_main.py", mode="optimization",
         out_subdir="thumbnail_align/tform", config_kind="thumbnail",
         requires=("thumbnail.matching",), supports_range=False,
         clear_paths=("thumbnail_align/mesh",), snapshot_paths=("thumbnail_align/tform",)),
    Step("thumbnail.render", "thumbnail", "Render coarse stack",
         "Aligned thumbnails to check the coarse result by eye (optional).",
         script="scripts/thumbnail_main.py", mode="render",
         out_subdir="thumbnail_align", out_glob="aligned_thumbnails_*/*.png",
         config_kind="thumbnail", requires=("thumbnail.optimization",), optional=True,
         supports_range=False),
    Step("align.meshing", "align", "Generate meshes",
         "Finite-element meshes per section, honouring the material masks.",
         script="scripts/align_main.py", mode="meshing",
         out_subdir="align/mesh", config_kind="alignment",
         requires=("thumbnail.matching", "masks"), supports_range=False),
    Step("align.matching", "align", "Fine matching",
         "Refine the coarse matches at the working mip level by block matching.",
         script="scripts/align_main.py", mode="matching",
         out_subdir="align/matches", out_glob="*.h5", err_glob="*.h5_err",
         cardinality=Cardinality.PER_PAIR, config_kind="alignment",
         requires=("align.meshing", "thumbnail.optimization"),
         clear_paths=("align/matches/match_cover",), snapshot_paths=("align/matches",)),
    Step("align.optimization", "align", "Optimize stack",
         "Relax all meshes of the stack against the fine matches.",
         script="scripts/align_main.py", mode="optimization",
         out_subdir="align/tform", config_kind="alignment",
         requires=("align.matching",), supports_range=False,
         clear_paths=("align/tform/canvas.json",), snapshot_paths=("align/tform",)),
    Step("align.rendering", "align", "Render aligned stack (PNG tiles)",
         "Write the aligned stack as non-overlapping PNG tiles (VAST-style).",
         script="scripts/align_main.py", mode="rendering",
         out_subdir="aligned_stack", out_glob="mip0/*", config_kind="alignment",
         requires=("align.optimization",), optional=True),
    Step("align.downsample", "align", "Mipmaps for PNG stack",
         "Downsampled levels of the PNG stack for viewing.",
         script="scripts/align_main.py", mode="downsample",
         out_subdir="aligned_stack", out_glob="mip1/*", config_kind="alignment",
         requires=("align.rendering",), optional=True, cardinality=Cardinality.SINGLE),
    Step("align.tsr", "align", "Render aligned volume (precomputed)",
         "Write the aligned stack as a Neuroglancer precomputed volume via TensorStore.",
         script="scripts/align_main.py", mode="tensorstore_rendering",
         out_subdir="aligned_tensorstore", out_glob="*", config_kind="alignment",
         requires=("align.optimization",), optional=True, cardinality=Cardinality.SINGLE,
         clear_paths=("align/render_flags", "align/ts_spec.json", "align/mask.png")),
    Step("align.tsd", "align", "Mipmaps for volume",
         "Downsample the precomputed volume.",
         script="scripts/align_main.py", mode="tensorstore_downsample",
         out_subdir="aligned_tensorstore", out_glob="*", config_kind="alignment",
         requires=("align.tsr",), optional=True, cardinality=Cardinality.SINGLE,
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
        else:
            base = f"{self.done}/{self.expected}"
        if self.errors:
            base += f", {self.errors} error file(s)"
        return base


def _iter_outputs(root: Path, step: Step) -> list[Path]:
    d = root / step.out_subdir
    if not d.exists():
        return []
    patterns = (step.out_glob,) if isinstance(step.out_glob, str) else tuple(step.out_glob)
    hits: list[Path] = []
    seen: set[Path] = set()
    try:
        for pattern in patterns:          # several patterns: one output layout per render driver
            for p in d.glob(pattern):
                if p.name.startswith(".") or p in seen:
                    continue
                seen.add(p)
                hits.append(p)
    except OSError:
        return []
    return hits


def count_outputs(root: Path, step: Step) -> int:
    hits = _iter_outputs(root, step)
    if step.cardinality is Cardinality.SINGLE:
        return 1 if hits else 0
    return len(hits)


def _mtimes(paths: Iterable[Path]) -> tuple[float, float]:
    newest = 0.0
    oldest = 0.0
    for p in paths:
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        newest = max(newest, m)
        oldest = m if oldest == 0.0 else min(oldest, m)
    return newest, oldest


class PipelineScan:
    """Snapshot of every step's state for a project root."""

    def __init__(self, root: Path, n_sections: int, configs: ConfigStore | None = None):
        self.root = Path(root)
        self.n_sections = n_sections
        self.configs = configs
        self.status: dict[str, StepStatus] = {}
        self.scan()

    def expected_for(self, step: Step) -> int:
        n = self.n_sections
        if step.cardinality is Cardinality.PER_SECTION:
            return n
        if step.cardinality is Cardinality.PER_PAIR:
            if step.key.startswith("thumbnail") and self.configs is not None:
                cd = self.configs.get("thumbnail", "alignment.compare_distance", 1)
                if isinstance(cd, (list, tuple)):
                    ks = [int(k) for k in cd]
                else:
                    try:
                        ks = list(range(1, int(cd) + 1))
                    except (TypeError, ValueError):
                        ks = [1]
                return max(0, sum(max(0, n - k) for k in ks))
            if step.key.startswith("align"):
                # fine matching mirrors the thumbnail match list, or align/match_name.txt when
                # the fine alignment uses its own (shorter) compare distance
                listed = read_fine_match_list(self.root)
                if listed is not None:
                    return len(listed)
                return len(list((self.root / "thumbnail_align" / "matches").glob("*.h5"))) or max(0, n - 1)
            return max(0, n - 1)
        return 1

    def scan(self) -> None:
        self.status = {}
        for step in STEPS:
            st = StepStatus(step)
            outs = _iter_outputs(self.root, step)
            st.done = (1 if outs else 0) if step.cardinality is Cardinality.SINGLE else len(outs)
            st.expected = self.expected_for(step)
            if step.err_glob:
                d = self.root / step.out_subdir
                st.errors = len(list(d.glob(step.err_glob))) if d.exists() else 0
            st.newest_output, st.oldest_output = _mtimes(outs)
            self.status[step.key] = st
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
            elif st.done == 0:
                st.state = State.BLOCKED if blocked else State.EMPTY
            else:
                st.state = State.COMPLETE if (st.expected and st.done >= st.expected) else State.PARTIAL
                if step.cardinality is Cardinality.SINGLE:
                    st.state = State.COMPLETE
                # staleness: config newer than oldest output, or upstream newer than our oldest
                if self.configs is not None and step.config_kind:
                    cm = self.configs.modified_time(step.config_kind)
                    if cm and st.oldest_output and cm > st.oldest_output + 1:
                        st.state = State.STALE
                        st.reasons.append(f"{step.config_kind} config edited after these outputs")
                for r in step.requires:
                    up = self.status.get(r)
                    if up and up.newest_output and st.oldest_output and up.newest_output > st.oldest_output + 1:
                        st.state = State.STALE
                        st.reasons.append(f"'{up.step.label}' produced newer outputs")

    def __getitem__(self, key: str) -> StepStatus:
        return self.status[key]


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
    ``stitched_sections/mipN/<sec>/metadata.txt`` behind, so those are counted too.
    """
    root = Path(root)
    n = max(0, int(n_sections))
    thumbs = len([p for p in (root / "thumbnail_align" / "thumbnails").glob("*.png")]) \
        if (root / "thumbnail_align" / "thumbnails").is_dir() else 0
    driver = configs.get("stitching", "rendering.driver", "image") if configs is not None else "image"
    max_mip = thumbnail_max_mip(configs) if driver == "image" else 0
    if max_mip <= 0 or n == 0:
        return thumbs, n, f"thumbnails {thumbs}/{n}"
    ss = root / "stitched_sections"
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

def clear_targets(root: Path, step: Step, cascade: bool = True) -> list[Path]:
    """Paths that would be removed when clearing *step* (and downstream)."""
    steps = [step] + (downstream(step.key) if cascade else [])
    targets: list[Path] = []
    seen = set()
    for s in steps:
        cand = [root / s.out_subdir] + [root / p for p in s.clear_paths]
        if s.key == "thumbnail.render":
            cand = list((root / "thumbnail_align").glob("aligned_thumbnails_*"))
        if s.key == "stitch.rendering":
            # mipmaps live next to mip0 inside stitched_sections; whole folder goes
            cand = [root / "stitched_sections", root / "stitch" / "ts_specs", root / "stitch" / "hist_tf"]
        if s.key == "thumbnail.downsample":
            cand = [root / "thumbnail_align" / "thumbnails", root / "thumbnail_align" / "material_masks",
                    root / "thumbnail_align" / "region_masks"]
            # mip levels generated inside stitched_sections beyond mip0 are also downsample outputs
            ss = root / "stitched_sections"
            if ss.is_dir():
                cand += [p for p in ss.glob("mip*") if p.name != "mip0"]
        for c in cand:
            if c in seen:
                continue
            seen.add(c)
            if c.exists():
                targets.append(c)
    return targets


def clear_step(root: Path, step: Step, cascade: bool = True, log=None) -> list[Path]:
    removed = []
    for p in clear_targets(root, step, cascade):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(p)
            if log:
                log(f"removed {p}")
        except OSError as e:
            if log:
                log(f"could not remove {p}: {e}")
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


def restore_snapshot(root: Path, snapshot_dir: Path, log=None) -> list[str]:
    """Replace the snapshotted folders with the snapshot's copies."""
    root = Path(root)
    meta = json.loads((Path(snapshot_dir) / "snapshot.json").read_text(encoding="utf-8"))
    restored = []
    for rel in meta.get("paths", []):
        src = Path(snapshot_dir) / rel
        dst = root / rel
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        restored.append(rel)
        if log:
            log(f"restored {rel}")
    return restored


def delete_snapshot(snapshot_dir: Path) -> None:
    shutil.rmtree(snapshot_dir)
