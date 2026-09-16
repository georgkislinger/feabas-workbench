"""
Test runs: sandboxed FEABAS working directories inside <project>/tests/<name>.

A montage test takes a subset of sections and/or a rectangular subset of tiles,
copies the project's configs, writes reduced stitch_coord files and runs the
stitching steps there, so parameters can be tried without touching the real
outputs. An alignment test links the real stitching results (read-only use)
and runs the thumbnail/alignment steps on a subset of sections.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .project import Project, DEFAULT_CONFIG_NAMES, USER_CONFIG_NAMES


def _link_dir(src: Path, dst: Path) -> None:
    """Directory junction (Windows) or symlink (POSIX); falls back to nothing if it fails."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, check=True)
        else:
            os.symlink(src, dst, target_is_directory=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not link {src} -> {dst}: {e}")


@dataclass
class TestRun:
    root: Path
    name: str
    kind: str          # montage | align
    sections: list[str]
    created: float

    @property
    def meta_file(self) -> Path:
        return self.root / "testrun.json"

    def save(self) -> None:
        self.meta_file.write_text(json.dumps({"name": self.name, "kind": self.kind, "sections": self.sections,
                                              "created": self.created}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, root: Path) -> "TestRun | None":
        mf = root / "testrun.json"
        if not mf.is_file():
            return None
        d = json.loads(mf.read_text(encoding="utf-8"))
        return cls(root, d["name"], d["kind"], d.get("sections", []), d.get("created", 0.0))


def list_test_runs(project: Project, kind: str | None = None) -> list[TestRun]:
    out = []
    if not project.tests_dir.is_dir():
        return out
    for d in sorted(project.tests_dir.iterdir()):
        tr = TestRun.load(d)
        if tr and (kind is None or tr.kind == kind):
            out.append(tr)
    return out


def _prepare_configs(project: Project, root: Path, overrides: dict[str, dict] | None = None) -> None:
    cfg = root / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_CONFIG_NAMES + USER_CONFIG_NAMES:
        src = project.configs_dir / name
        if src.is_file():
            shutil.copyfile(src, cfg / name)
    general = yaml.safe_load((project.configs_dir / "general_configs.yaml").read_text(encoding="utf-8")) or {}
    general["working_directory"] = root.as_posix()
    general["logging_directory"] = None
    (cfg / "general_configs.yaml").write_text(yaml.safe_dump(general, sort_keys=False), encoding="utf-8")
    for kind, over in (overrides or {}).items():
        p = cfg / f"{kind}_configs.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
        data = data or {}
        _deep_update(data, over)
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _deep_update(d: dict, u: dict) -> None:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def parse_stitch_coord(path: Path) -> dict:
    """Read one FEABAS coordinate file into {root, resolution, tile_size, tiles:[(rel, x, y)]}."""
    out = {"root": None, "resolution": None, "tile_size": None, "tiles": []}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.rstrip("\n").split("\t")
        if parts[0] == "{ROOT_DIR}":
            out["root"] = parts[1]
        elif parts[0] == "{RESOLUTION}":
            out["resolution"] = float(parts[1])
        elif parts[0] == "{TILE_SIZE}":
            out["tile_size"] = (int(parts[1]), int(parts[2]) if len(parts) > 2 else int(parts[1]))
        elif len(parts) >= 3:
            out["tiles"].append((parts[0], float(parts[1]), float(parts[2])))
    return out


def create_montage_test(project: Project, name: str, sections: list[str],
                        bbox: tuple[float, float, float, float] | None = None,
                        force_image_driver: bool = True) -> TestRun:
    """
    bbox: (x0, y0, x1, y1) in section pixel coordinates; tiles whose nominal
    rectangle intersects it are kept. None keeps all tiles.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()) or "test"
    root = project.tests_dir / safe
    if root.exists():
        shutil.rmtree(root)
    (root / "stitch" / "stitch_coord").mkdir(parents=True)
    overrides = {"stitching": {"rendering": {"driver": "image", "out_dir": None}}} if force_image_driver else None
    _prepare_configs(project, root, overrides)
    for sec in sections:
        src = project.stitch_coord_dir / f"{sec}.txt"
        if not src.is_file():
            continue
        info = parse_stitch_coord(src)
        th, tw = info["tile_size"] or (project.state.volume.tile_h, project.state.volume.tile_w)
        keep = []
        for rel, x, y in info["tiles"]:
            if bbox is None or (x < bbox[2] and x + tw > bbox[0] and y < bbox[3] and y + th > bbox[1]):
                keep.append((rel, x, y))
        if not keep:
            continue
        lines = []
        if info["root"]:
            lines.append(f"{{ROOT_DIR}}\t{info['root']}")
        if info["resolution"]:
            lines.append(f"{{RESOLUTION}}\t{info['resolution']:g}")
        lines.append(f"{{TILE_SIZE}}\t{th}\t{tw}")
        x0 = min(k[1] for k in keep)
        y0 = min(k[2] for k in keep)
        for rel, x, y in keep:
            lines.append(f"{rel}\t{x - x0:.1f}\t{y - y0:.1f}")
        (root / "stitch" / "stitch_coord" / f"{sec}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tr = TestRun(root, safe, "montage", sections, time.time())
    tr.save()
    return tr


def create_align_test(project: Project, name: str, sections: list[str]) -> TestRun:
    """Sandbox for coarse/fine alignment on a subset: links stitch results, fresh alignment folders."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip()) or "test"
    root = project.tests_dir / safe
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _prepare_configs(project, root)
    (root / "stitch").mkdir()
    (root / "stitch" / "stitch_coord").mkdir()
    (root / "stitch" / "tform").mkdir()
    for sec in sections:
        for sub in ("stitch_coord", "tform"):
            src = project.root / "stitch" / sub / (f"{sec}.txt" if sub == "stitch_coord" else f"{sec}.h5")
            if src.is_file():
                shutil.copyfile(src, root / "stitch" / sub / src.name)
        spec = project.root / "stitch" / "ts_specs" / f"{sec}.json"
        if spec.is_file():
            (root / "stitch" / "ts_specs").mkdir(exist_ok=True)
            shutil.copyfile(spec, root / "stitch" / "ts_specs" / spec.name)
    # stitched images are large: link, do not copy
    ss = project.root / "stitched_sections"
    if ss.is_dir():
        _link_dir(ss, root / "stitched_sections")
    # thumbnails and masks of the chosen sections are small: copy them so the test can start at matching
    for sub in ("thumbnail_align/thumbnails", "thumbnail_align/material_masks", "align/material_masks"):
        src_dir = project.root / sub
        if not src_dir.is_dir():
            continue
        for sec in sections:
            for f in src_dir.glob(f"{sec}.*"):
                (root / sub).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, root / sub / f.name)
    (root / "align").mkdir(exist_ok=True)
    (root / "thumbnail_align").mkdir(exist_ok=True)
    order = project.root / "section_order.txt"
    if order.is_file():
        keep = [ln for ln in order.read_text(encoding="utf-8").splitlines() if ln.strip() in set(sections)]
        (root / "section_order.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    tr = TestRun(root, safe, "align", sections, time.time())
    tr.save()
    return tr


def delete_test_run(tr: TestRun) -> None:
    # remove junctions/symlinks without following them
    for p in tr.root.rglob("*"):
        pass
    link = tr.root / "stitched_sections"
    if link.exists() and (link.is_symlink() or _is_junction(link)):
        if os.name == "nt":
            os.rmdir(link)
        else:
            link.unlink()
    shutil.rmtree(tr.root, ignore_errors=True)


def _is_junction(p: Path) -> bool:
    try:
        return bool(os.lstat(p).st_file_attributes & 0x400)   # FILE_ATTRIBUTE_REPARSE_POINT
    except (AttributeError, OSError):
        return False


def retarget_stitch_coords(project: Project, new_root: Path) -> int:
    """Rewrite the {ROOT_DIR} line of every coordinate file (raw <-> preprocessed tiles)."""
    n = 0
    for p in project.stitch_coord_dir.glob("*.txt"):
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        changed = False
        for ln in lines:
            if ln.startswith("{ROOT_DIR}"):
                out.append(f"{{ROOT_DIR}}\t{Path(new_root).as_posix()}")
                changed = True
            else:
                out.append(ln)
        if changed:
            p.write_text("\n".join(out) + "\n", encoding="utf-8")
            n += 1
    return n
