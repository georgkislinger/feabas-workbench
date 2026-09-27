"""
Project model.

A project folder *is* the FEABAS working directory. On top of FEABAS's own
layout (configs/, stitch/, thumbnail_align/, align/, logs/ ...) the workbench
keeps its own state in ``workbench_project.json`` and a few extra folders
(preprocessed tiles, models, masks, test runs, snapshots, exports).

FEABAS locates its *default* configuration folder by looking for
``configs/general_configs.yaml`` in the current working directory. The
workbench therefore writes that file into ``<project>/configs`` (pointing
``working_directory`` at the project itself) and copies the vendored
``default_*.yaml`` files next to it. Every FEABAS step is then run with the
project folder as the working directory, and the FEABAS installation itself is
never modified.

Nothing here imports Qt.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .tiles import NamingRule, LayoutParams

PROJECT_FILE = "workbench_project.json"
PROJECT_FORMAT = 1

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "feabas_3_0_5"
DEFAULT_CONFIG_NAMES = (
    "default_stitching_configs.yaml",
    "default_thumbnail_configs.yaml",
    "default_alignment_configs.yaml",
    "default_material_table.yaml",
)
USER_CONFIG_NAMES = (
    "stitching_configs.yaml",
    "thumbnail_configs.yaml",
    "alignment_configs.yaml",
    "material_table.yaml",
)

_KEEP = object()        # write_general_config: leave the setting as it is in the file
_GENERAL_HEADER = "# Written by FEABAS Workbench. working_directory points at this project.\n"


def _read_general(path: Path) -> dict:
    if not Path(path).is_file():
        return {}
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_general(path: Path, conf: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_name(Path(path).name + ".tmp")
    tmp.write_text(_GENERAL_HEADER + yaml.safe_dump(conf, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def _same_folder(value, root: Path) -> bool:
    """Whether a working_directory value names *root*, the way FEABAS expands it (relative to the root)."""
    if not value:
        return False
    text = os.path.expanduser(os.path.expandvars(str(value)))
    p = Path(text) if Path(text).is_absolute() else Path(root) / text
    try:
        return p.resolve() == Path(root).resolve()
    except OSError:
        return False


def repair_working_directory(root: os.PathLike | str) -> bool:
    """
    Point <root>/configs/general_configs.yaml at *root* again if it names another folder.

    FEABAS takes its working directory - where it reads the user configs and writes every output -
    from that file, not from the folder it is started in. A project folder that was copied, moved,
    renamed or opened through another path (a different drive letter or mount point) still names
    the old place, and a step run there would read and write the original project. Works for any
    FEABAS working directory (projects and test runs). Returns True when the file was changed.
    """
    root = Path(root)
    path = root / "configs" / "general_configs.yaml"
    conf = _read_general(path)
    if not conf or _same_folder(conf.get("working_directory"), root):
        return False
    old = str(conf.get("working_directory") or "")
    conf["working_directory"] = root.resolve().as_posix()
    logs = str(conf.get("logging_directory") or "")
    if old and logs and Path(logs).is_absolute():
        try:
            Path(logs).relative_to(Path(old))
            conf["logging_directory"] = None       # a log folder inside the old location: use <root>/logs
        except ValueError:
            pass
    _write_general(path, conf)
    return True


def forget_cached_resolution(root: os.PathLike | str, resolution_nm: float) -> bool:
    """
    FEABAS caches the tile resolution of the first run in <root>/configs/resolutions.yaml and
    never reads the coordinate files again. Remove that cache when the coordinate files now
    carry a different resolution, so a corrected pixel size is actually used. True if removed.
    """
    path = Path(root) / "configs" / "resolutions.yaml"
    if not path.is_file():
        return False
    try:
        cached = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("DATA_RESOLUTION")
        same = cached is not None and abs(float(cached) - float(resolution_nm)) <= 1e-6 * max(1.0, abs(float(resolution_nm)))
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
        same = False
    if same:
        return False
    path.unlink()
    return True


@dataclass
class SourceSettings:
    root_dir: str = ""
    rule: dict = field(default_factory=lambda: NamingRule().to_dict())
    layout: dict = field(default_factory=lambda: LayoutParams().to_dict())
    path_mode: str = "relative"

    def naming_rule(self) -> NamingRule:
        return NamingRule.from_dict(self.rule)

    def layout_params(self) -> LayoutParams:
        return LayoutParams.from_dict(self.layout)


@dataclass
class VolumeInfo:
    pixel_size_nm: float = 4.0
    section_thickness_nm: float = 50.0
    tile_w: int = 0
    tile_h: int = 0
    dtype: str = "uint8"
    n_sections: int = 0
    n_tiles: int = 0
    grid_rows: int = 0
    grid_cols: int = 0
    section_names: list[str] = field(default_factory=list)


@dataclass
class HistMatchSettings:
    enabled: bool = False
    template: str = ""
    ignore_black: bool = True
    ignore_white: bool = False
    workers: int = 8
    done: bool = False


@dataclass
class DenoiseSettings:
    enabled: bool = False
    method: str = "n2v"            # n2v | n2v2 | structn2v
    model_dir: str = ""            # careamics work dir of the trained run
    checkpoint: str = ""
    training_tiles: list[str] = field(default_factory=list)   # paths of tiles selected for training
    patch_size: int = 128
    batch_size: int = 12
    epochs: int = 100
    steps_per_epoch: int = 100
    roi_size: int = 11
    masked_pixel_percentage: float = 0.2
    struct_axes: str = "horizontal"
    struct_span: int = 5
    tile_size: int = 512
    tile_overlap: int = 64
    done: bool = False


@dataclass
class Preprocessing:
    histmatch: dict = field(default_factory=lambda: asdict(HistMatchSettings()))
    denoise: dict = field(default_factory=lambda: asdict(DenoiseSettings()))
    active_source: str = "raw"     # raw | histmatch | denoise


@dataclass
class ProjectState:
    format: int = PROJECT_FORMAT
    name: str = ""
    created: str = ""
    source: SourceSettings = field(default_factory=SourceSettings)
    volume: VolumeInfo = field(default_factory=VolumeInfo)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    envs: dict = field(default_factory=lambda: {"feabas_python": "", "dl_python": ""})
    masks: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)     # workbench-side alignment choices (fine compare distance ...)
    export: dict = field(default_factory=dict)
    local_execution: dict = field(default_factory=dict)  # opt-in per-stage workstation section scheduling
    suggested: dict = field(default_factory=dict)        # config values the workbench filled in itself (configs.apply_suggested_mips)
    notes: str = ""
    feabas_version: str = "3.0.5"


def _to_dataclass(cls, d):
    if d is None:
        return cls()
    if isinstance(d, cls):
        return d
    known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
    return cls(**known)


class Project:
    """A project folder and its persisted state."""

    def __init__(self, root: os.PathLike | str):
        self.root = Path(root).resolve()
        self.state = ProjectState(name=self.root.name)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    @property
    def file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def stitch_coord_dir(self) -> Path:
        return self.root / "stitch" / "stitch_coord"

    @property
    def preprocessed_dir(self) -> Path:
        return self.root / "preprocessed"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def workbench_log(self) -> Path:
        return self.root / "workbench.log"

    def ensure_layout(self) -> None:
        for d in (self.configs_dir, self.stitch_coord_dir, self.preprocessed_dir, self.models_dir,
                  self.masks_dir, self.tests_dir, self.snapshots_dir, self.exports_dir, self.logs_dir,
                  self.root / "thumbnail_align", self.root / "align"):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    @classmethod
    def exists(cls, root: os.PathLike | str) -> bool:
        return (Path(root) / PROJECT_FILE).is_file()

    @classmethod
    def create(cls, root: os.PathLike | str, name: str | None = None) -> "Project":
        p = cls(root)
        p.root.mkdir(parents=True, exist_ok=True)
        p.state.name = name or p.root.name
        p.state.created = _dt.datetime.now().isoformat(timespec="seconds")
        p.ensure_layout()
        p.install_default_configs()
        p.write_general_config()
        p.apply_workbench_defaults()
        p.save()
        return p

    def apply_workbench_defaults(self) -> None:
        """Settings the workbench prefers over FEABAS's defaults for a fresh project."""
        from .configs import ConfigStore
        cs = ConfigStore(self.configs_dir)
        # PNG tiles are the validated output on every platform and what VASTlite reads; the
        # TensorStore/precomputed driver stays available in the Stitching settings.
        if not cs["stitching"].is_overridden("rendering.driver"):
            cs.set("stitching", "rendering.driver", "image")
            cs.save("stitching")

    @classmethod
    def load(cls, root: os.PathLike | str) -> "Project":
        p = cls(root)
        raw = json.loads(p.file.read_text(encoding="utf-8"))
        st = ProjectState()
        st.format = raw.get("format", PROJECT_FORMAT)
        st.name = raw.get("name", p.root.name)
        st.created = raw.get("created", "")
        st.source = _to_dataclass(SourceSettings, raw.get("source"))
        st.volume = _to_dataclass(VolumeInfo, raw.get("volume"))
        pp = raw.get("preprocessing") or {}
        st.preprocessing = Preprocessing(
            histmatch={**asdict(HistMatchSettings()), **(pp.get("histmatch") or {})},
            denoise={**asdict(DenoiseSettings()), **(pp.get("denoise") or {})},
            active_source=pp.get("active_source", "raw"),
        )
        st.envs = {**{"feabas_python": "", "dl_python": ""}, **(raw.get("envs") or {})}
        st.masks = raw.get("masks") or {}
        st.structure = raw.get("structure") or {}
        st.alignment = raw.get("alignment") or {}
        st.export = raw.get("export") or {}
        st.local_execution = raw.get("local_execution") or {}
        st.suggested = raw.get("suggested") or {}
        st.notes = raw.get("notes", "")
        st.feabas_version = raw.get("feabas_version", "3.0.5")
        p.state = st
        p.ensure_layout()
        if not (p.configs_dir / "general_configs.yaml").is_file():
            p.install_default_configs()
            p.write_general_config()
        else:
            # a project that was copied, moved or opened through another path still names its old
            # folder: FEABAS would read and write there instead of here
            repair_working_directory(p.root)
        return p

    def save(self) -> None:
        data = {
            "format": PROJECT_FORMAT,
            "name": self.state.name,
            "created": self.state.created,
            "source": asdict(self.state.source),
            "volume": asdict(self.state.volume),
            "preprocessing": asdict(self.state.preprocessing),
            "envs": self.state.envs,
            "masks": self.state.masks,
            "structure": self.state.structure,
            "alignment": self.state.alignment,
            "export": self.state.export,
            "local_execution": self.state.local_execution,
            "suggested": self.state.suggested,
            "notes": self.state.notes,
            "feabas_version": self.state.feabas_version,
        }
        if not self.root.is_dir():
            return          # project folder was removed meanwhile; nothing sensible to save
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.file)

    # ------------------------------------------------------------------
    # FEABAS configuration files
    # ------------------------------------------------------------------
    def install_default_configs(self, overwrite: bool = False) -> list[Path]:
        """Copy vendored default_*.yaml into <project>/configs."""
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in DEFAULT_CONFIG_NAMES:
            src = VENDOR_DIR / "configs" / name
            dst = self.configs_dir / name
            if src.is_file() and (overwrite or not dst.is_file()):
                shutil.copyfile(src, dst)
                copied.append(dst)
        return copied

    def general_config_path(self) -> Path:
        return self.configs_dir / "general_configs.yaml"

    def write_general_config(self, cpu_budget=_KEEP, parallel_framework: str | None = None,
                             logfile_level: str | None = None) -> Path:
        """
        (Re)write configs/general_configs.yaml so FEABAS points at this project.

        The compute settings are kept as they are unless given: *cpu_budget* ``None`` means all
        physical cores (FEABAS's default), an int limits them. The file is only rewritten when
        something in it changes.
        """
        v = self.state.volume
        existing = _read_general(self.general_config_path())
        conf = {
            "working_directory": self.root.as_posix(),
            "cpu_budget": existing.get("cpu_budget") if cpu_budget is _KEEP else cpu_budget,
            "parallel_framework": parallel_framework or existing.get("parallel_framework") or "process",
            "full_resolution": float(v.pixel_size_nm),
            "section_thickness": float(v.section_thickness_nm),
            "logging_directory": None,
            "logfile_level": logfile_level or existing.get("logfile_level") or "INFO",
            "console_level": existing.get("console_level", "INFO"),
            "archive_level": existing.get("archive_level", "INFO"),
            "tensorstore_timeout": existing.get("tensorstore_timeout"),
        }
        if conf != existing or not self.general_config_path().is_file():
            _write_general(self.general_config_path(), conf)
        return self.general_config_path()

    # ------------------------------------------------------------------
    # sections
    # ------------------------------------------------------------------
    def section_names(self) -> list[str]:
        """Section names from stitch_coord files, honouring section_order.txt."""
        order_file = self.root / "section_order.txt"
        names = sorted(p.stem for p in self.stitch_coord_dir.glob("*.txt"))
        if order_file.is_file():
            ordered = [ln.strip() for ln in order_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            known = set(names)
            listed = set(ordered)
            names = list(dict.fromkeys(n for n in ordered if n in known)) + [n for n in names if n not in listed]
        return names

    # ------------------------------------------------------------------
    # tile source
    # ------------------------------------------------------------------
    def active_tile_root(self) -> Path | None:
        """Folder that stitch_coord files should point at (raw or preprocessed)."""
        pp = self.state.preprocessing
        if pp.active_source == "histmatch":
            return self.preprocessed_dir / "histmatch"
        if pp.active_source == "denoise":
            return self.preprocessed_dir / "denoised"
        return Path(self.state.source.root_dir) if self.state.source.root_dir else None

    def python_for(self, kind: str, fallback: str = "") -> str:
        """Project-level interpreter override or fallback (global setting)."""
        key = "feabas_python" if kind == "feabas" else "dl_python"
        return self.state.envs.get(key) or fallback

    def __repr__(self) -> str:
        return f"Project({self.root})"
