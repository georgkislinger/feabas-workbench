"""
FEABAS YAML configuration handling.

FEABAS merges ``default_<name>.yaml`` (from its default configuration folder)
with ``<workdir>/configs/<name>.yaml`` (user overrides, may be partial). This
module gives the GUI one merged view, tracks which keys are overridden, saves
back only the differences, and harvests the inline comments of the default
files as per-key hints so the settings editor can explain every field.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = {
    "stitching": "stitching_configs.yaml",
    "thumbnail": "thumbnail_configs.yaml",
    "alignment": "alignment_configs.yaml",
    "material": "material_table.yaml",
}


class ConfigError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# yaml helpers
# ----------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: {e}") from e
    return data if isinstance(data, dict) else {}


def dump_yaml(path: Path, data: dict, header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True)
    if header:
        text = "".join(f"# {ln}\n" for ln in header.splitlines()) + text
    path.write_text(text, encoding="utf-8")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def deep_diff(base: dict, current: dict) -> dict:
    """Keys in *current* that differ from *base* (nested)."""
    out: dict = {}
    for k, v in current.items():
        if k not in base:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(base[k], dict):
            sub = deep_diff(base[k], v)
            if sub:
                out[k] = sub
        elif v != base[k]:
            out[k] = copy.deepcopy(v)
    return out


def flatten(data: dict, prefix: str = "") -> list[tuple[str, Any]]:
    out = []
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(flatten(v, key))
        else:
            out.append((key, v))
    return out


def get_dotted(data: dict, dotted: str, default=None):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_dotted(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def del_dotted(data: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = data
    stack = []
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return
        stack.append((cur, p))
        cur = cur[p]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
    # prune empty parents
    for parent, key in reversed(stack):
        if isinstance(parent.get(key), dict) and not parent[key]:
            del parent[key]


# ----------------------------------------------------------------------
# hints from comments in the default yaml files
# ----------------------------------------------------------------------

def harvest_hints(path: Path) -> dict[str, str]:
    """Map dotted key -> inline/preceding comment from a YAML file."""
    hints: dict[str, str] = {}
    if not Path(path).is_file():
        return hints
    stack: list[tuple[int, str]] = []   # (indent, key)
    pending: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            pending = []
            continue
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("#").strip())
            continue
        m = re.match(r"([A-Za-z0-9_\-]+)\s*:\s*(.*)$", stripped)
        if not m:
            continue
        key, rest = m.group(1), m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        comment = ""
        if "#" in rest:
            comment = rest.split("#", 1)[1].strip()
        text = " ".join([c for c in pending if c] + ([comment] if comment else []))
        if text:
            hints[dotted] = text
        pending = []
        stack.append((indent, key))
    return hints


# curated hints for keys whose default comments are terse or missing
EXTRA_HINTS = {
    "stitching.section_thickness": "Section thickness in nm. Used to pick working mip levels so xy resolution is close to z.",
    "stitching.matching.num_workers": "Parallel processes for tile matching. Each caches images; lower if RAM is tight.",
    "stitching.rendering.driver": "'image' writes PNG tiles per section (simple, VAST-friendly); 'neuroglancer_precomputed' writes a TensorStore volume (faster downstream).",
    "stitching.rendering.loader_settings.apply_CLAHE": "Contrast-limited histogram equalisation during rendering. Usually leave on; turn off if you did your own histogram matching and want the raw look.",
    "stitching.rendering.loader_settings.inverse": "Invert grey values while rendering (BSE images are often acquired inverted).",
    "thumbnail.thumbnail_mip_level": "Mip level of thumbnails (each level halves resolution). Aim for roughly 500-2000 px thumbnails; larger ones make coarse matching very slow.",
    "thumbnail.downsample.thumbnail_highpass": "High-pass before downsampling to enhance somata. Rule of thumb: on for SE images, off for BSE images or very small mip levels.",
    "thumbnail.alignment.compare_distance": "How many neighbouring sections to match (1 = immediate neighbours only). 2 makes the stack more robust to a bad section.",
    "alignment.matching.working_mip_level": "Resolution for fine matching. Pick the mip whose xy pixel size is closest to the section thickness.",
    "alignment.meshing.mesh_size": "Mesh spacing at mip0 pixels. Finer meshes follow local distortions but can overfit real z change.",
    "alignment.meshing.mask_mip_level": "Mip level of the masks in align/material_masks (the fold/tissue masks written by the Masks step).",
    "alignment.optimization.chunk_settings.chunked_to_depth": "0 = sliding window over the whole stack; >0 = align chunks first, then chunks against each other.",
    "alignment.rendering.tile_size": "Output tile size (px) for PNG rendering. 4096 is fine for VAST.",
    "alignment.tensorstore_rendering.driver": "Neuroglancer precomputed is the recommended big-data output.",
}


# ----------------------------------------------------------------------
# config store
# ----------------------------------------------------------------------

@dataclass
class ConfigDoc:
    kind: str                      # stitching | thumbnail | alignment | material
    default_path: Path
    user_path: Path
    defaults: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)

    @property
    def merged(self) -> dict:
        return deep_merge(self.defaults, self.overrides)

    def is_overridden(self, dotted: str) -> bool:
        return get_dotted(self.overrides, dotted, _MISSING) is not _MISSING

    def set(self, dotted: str, value) -> None:
        if value == get_dotted(self.defaults, dotted, _MISSING):
            del_dotted(self.overrides, dotted)
        else:
            set_dotted(self.overrides, dotted, value)

    def reset(self, dotted: str) -> None:
        del_dotted(self.overrides, dotted)

    def hint(self, dotted: str) -> str:
        return EXTRA_HINTS.get(f"{self.kind}.{dotted}") or self.hints.get(dotted, "")


_MISSING = object()


class ConfigStore:
    """All four FEABAS config documents of a project."""

    def __init__(self, configs_dir: Path):
        self.configs_dir = Path(configs_dir)
        self.docs: dict[str, ConfigDoc] = {}
        self.reload()

    def reload(self) -> None:
        self.docs = {}
        for kind, name in CONFIG_FILES.items():
            dpath = self.configs_dir / f"default_{name}"
            upath = self.configs_dir / name
            doc = ConfigDoc(kind, dpath, upath)
            doc.defaults = load_yaml(dpath)
            doc.overrides = load_yaml(upath)
            doc.hints = harvest_hints(dpath)
            self.docs[kind] = doc

    def __getitem__(self, kind: str) -> ConfigDoc:
        return self.docs[kind]

    def save(self, kind: str | None = None) -> None:
        for k, doc in self.docs.items():
            if kind and k != kind:
                continue
            if doc.overrides:
                dump_yaml(doc.user_path, doc.overrides,
                          header=f"Project overrides for {doc.default_path.name}; unspecified keys use the defaults.")
            elif doc.user_path.is_file():
                doc.user_path.unlink()

    def get(self, kind: str, dotted: str, default=None):
        return get_dotted(self.docs[kind].merged, dotted, default)

    def set(self, kind: str, dotted: str, value) -> None:
        self.docs[kind].set(dotted, value)

    def modified_time(self, kind: str) -> float:
        p = self.docs[kind].user_path
        return p.stat().st_mtime if p.is_file() else 0.0


# ----------------------------------------------------------------------
# convenience: resolutions and mips
# ----------------------------------------------------------------------

def mip_resolution(pixel_size_nm: float, mip: int) -> float:
    return float(pixel_size_nm) * (2 ** int(mip))


def suggest_working_mip(pixel_size_nm: float, section_thickness_nm: float) -> int:
    """FEABAS rule: mip whose xy resolution is closest to (but not above) the section thickness."""
    import math
    if pixel_size_nm <= 0 or section_thickness_nm <= 0:
        return 2
    return max(0, int(math.floor(math.log2(section_thickness_nm / pixel_size_nm))))


def suggest_thumbnail_mip(section_w_px: float, section_h_px: float, target_px: float = 1000.0) -> int:
    """Mip level that brings the longer section side to about target_px."""
    import math
    longest = max(float(section_w_px), float(section_h_px), 1.0)
    return max(0, int(round(math.log2(longest / target_px))))


def parse_value(text: str, like):
    """Parse an edited string using the type of the current value."""
    t = text.strip()
    if isinstance(like, bool):
        return t.lower() in ("1", "true", "yes", "on")
    if like is None:
        if t.lower() in ("", "null", "none", "~"):
            return None
        return yaml.safe_load(t)
    if isinstance(like, int) and not isinstance(like, bool):
        try:
            return int(t)
        except ValueError:
            return yaml.safe_load(t)
    if isinstance(like, float):
        try:
            return float(t)
        except ValueError:
            return yaml.safe_load(t)
    if isinstance(like, (list, tuple, dict)):
        return yaml.safe_load(t)
    # a string-valued key keeps strings: 'NONE' is a real FEABAS value (e.g. the
    # rendering blend mode), so only the YAML null spellings clear such a key.
    if t.lower() in ("null", "~"):
        return None
    return t
