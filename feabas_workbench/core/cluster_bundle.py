"""Portable, immutable inputs for single-node FEABAS jobs. No Qt or SSH required."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from .configs import CONFIG_FILES, ConfigStore, dump_yaml, load_yaml
from .project import Project, VENDOR_DIR
from .steps import STEPS, STEPS_BY_KEY

MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_input(source, target, bundle, external):
    """Keep the SSH package small; immutable large inputs are staged by Globus."""
    used = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    if used + source.stat().st_size > MAX_BUNDLE_BYTES - 6 * 1024 * 1024:
        rel = target.relative_to(bundle / "inputs").as_posix()
        external[rel] = dict(source=str(source), size=source.stat().st_size, sha256=file_hash(source))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
PARTITIONS = {
    "cm4_tiny": ("cm4", 17, 112, 244, 24, "cm4_tiny"),
    "serial_std": ("serial", 1, 16, 100, 24, ""),
    "serial_long": ("serial", 1, 16, 100, 168, "cm4_serial_long"),
    "teramem_inter": ("inter", 1, 96, 2900, 240, ""),
}


def remote_path(value: str) -> str:
    p = PurePosixPath(value)
    if (not value.startswith("/") or value.startswith("//") or str(p) == "/"
            or ".." in p.parts or "\\" in value or any(ord(c) < 32 for c in value)
            or "%" in value):
        raise ValueError("Use an absolute Linux path without '..', backslashes, '%' or control characters.")
    return str(p)


def module_names(value: str) -> list[str]:
    names = value.split()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./+@-]*", n) for n in names):
        raise ValueError("Modules must be names separated by spaces, not shell commands.")
    return names


@dataclass
class ClusterResources:
    partition: str = "cm4_tiny"
    cpus: int = 64
    memory_gib: int = 128
    hours: int = 4
    workers: int = 16
    section_concurrency: int = 1

    @property
    def cluster(self) -> str:
        return PARTITIONS[self.partition][0]

    def validate(self) -> None:
        if self.partition not in PARTITIONS:
            raise ValueError("Choose a supported LRZ batch partition.")
        _, low, high, mem, hours, _ = PARTITIONS[self.partition]
        if any(type(x) is not int for x in (self.cpus, self.memory_gib, self.hours, self.workers, self.section_concurrency)):
            raise ValueError("Resources must be whole numbers.")
        if not low <= self.cpus <= high:
            raise ValueError(f"{self.partition}: choose {low}–{high} physical CPU cores.")
        if not 1 <= self.memory_gib <= mem or not 1 <= self.hours <= hours:
            raise ValueError(f"{self.partition}: maximum {mem} GiB and {hours} hours in this preset.")
        if not 1 <= self.workers <= self.cpus:
            raise ValueError("Workers must be between 1 and the allocated CPU count.")
        if not 1 <= self.section_concurrency <= self.cpus or self.section_concurrency * self.workers > self.cpus:
            raise ValueError("Simultaneous sections × workers per section must fit within the allocated CPU count.")


def _rewrite_coordinates(text: str, local_tiles: Path, remote_tiles: str, verify_files: bool = True) -> str:
    base = local_tiles.resolve()
    rows = [f"{{ROOT_DIR}}\t{remote_tiles}"]
    count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] == "{ROOT_DIR}":
            if len(parts) != 2:
                raise ValueError("Invalid coordinate root.")
            base = Path(parts[1]).resolve()
        elif parts[0].startswith("{"):
            rows.append(line)
        else:
            if len(parts) != 3:
                raise ValueError("Invalid stitch coordinate row.")
            tile = Path(parts[0])
            tile = (tile if tile.is_absolute() else base / tile).resolve()
            try:
                rel = tile.relative_to(local_tiles.resolve()).as_posix()
            except ValueError as e:
                raise ValueError(f"Tile outside the active tile folder: {tile}") from e
            if verify_files and not tile.is_file():
                raise ValueError(f"Tile is missing: {tile}")
            float(parts[1]); float(parts[2])
            rows.append("\t".join([rel, *parts[1:]]))
            count += 1
    if not count:
        raise ValueError("No tiles in a coordinate file.")
    return "\n".join(rows) + "\n"


def _portable_config(value, cpus: int, key: str = ""):
    if isinstance(value, dict):
        return {k: _portable_config(v, cpus, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable_config(v, cpus, key) for v in value]
    if key == "num_workers":
        # Exported workers use the remote budget. FEABAS divides the CPU budget
        # between them when setting BLAS/OpenMP threads.
        return cpus
    if key in {"out_dir", "mask_dir"} and value:
        raise ValueError(f"Custom {key} is not portable. Use the default project folder for cluster jobs.")
    if isinstance(value, str):
        if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
                or "://" in value or ".." in PurePosixPath(value.replace("\\", "/")).parts):
            raise ValueError(f"Config '{key}' contains an external path. Use project defaults: {value}")
    return value


def export_bundle(project: Project, destination: Path, remote_project: str, remote_tiles: str,
                  remote_python: str, resources: ClusterResources, step_keys: list[str],
                  modules: str = "", workspace: bool = False) -> Path:
    """Snapshot small inputs only. Existing local project and images are never changed."""
    resources.validate()
    rp, rt, python = map(remote_path, (remote_project, remote_tiles, remote_python))
    if not workspace and (PurePosixPath(rp) == PurePosixPath(rt) or PurePosixPath(rp) in PurePosixPath(rt).parents
            or PurePosixPath(rt) in PurePosixPath(rp).parents):
        raise ValueError("Keep the tile folder outside the dedicated remote work folder.")
    if not step_keys or any(k not in STEPS_BY_KEY or not STEPS_BY_KEY[k].script for k in step_keys):
        raise ValueError("Select at least one FEABAS step. Local preprocessing and neural networks are not batch steps.")
    ordered = [s.key for s in STEPS if s.key in step_keys]
    module_list = module_names(modules)
    coords = sorted(project.stitch_coord_dir.glob("*.txt"))
    if not coords and not workspace:
        raise ValueError("First write the stitch coordinates on the Project page.")
    local_tiles = project.active_tile_root()
    if local_tiles is None or (not workspace and not local_tiles.is_dir()):
        raise ValueError("The project's active tile folder is missing.")
    run_id = uuid.uuid4().hex
    out = Path(destination) / run_id
    out.mkdir(parents=True, exist_ok=False)
    remote_bundle = f"{rp}/.workbench-cluster/{run_id}"
    try:
        external = {}
        inputs = out / "inputs"
        (inputs / "stitch" / "stitch_coord").mkdir(parents=True)
        for src in coords:
            if src.is_symlink():
                raise ValueError("Coordinate files must not be symbolic links.")
            rewritten = _rewrite_coordinates(src.read_text(encoding="utf-8"), local_tiles, rt, verify_files=not workspace)
            (inputs / "stitch" / "stitch_coord" / src.name).write_text(rewritten, encoding="utf-8", newline="\n")
        config = ConfigStore(project.configs_dir)
        for kind, name in CONFIG_FILES.items():
            data = _portable_config(config[kind].merged, resources.workers)
            dump_yaml(inputs / "configs" / name, data)
            dump_yaml(inputs / "configs" / f"default_{name}", {})
        general = load_yaml(project.configs_dir / "general_configs.yaml")
        # General settings are a whitelist: never carry local directories to Linux.
        general = {k: general[k] for k in ("logfile_level", "console_level", "section_thickness",
                                         "full_resolution") if k in general}
        general.update(working_directory=rp, cpu_budget=resources.cpus, parallel_framework="process")
        dump_yaml(inputs / "configs" / "general_configs.yaml", general)
        for rel in ("section_order.txt", "thumbnail_align/material_masks", "align/material_masks"):
            src = project.root / rel
            paths = [src] if src.is_file() else sorted(src.rglob("*")) if src.is_dir() else []
            for f in paths:
                if f.is_symlink() or not f.resolve().is_relative_to(project.root.resolve()):
                    raise ValueError("Input links outside the project are not supported.")
                if f.is_file():
                    target = inputs / f.relative_to(project.root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if workspace:
                        stage_input(f, target, out, external)
                        continue
                    if sum(p.stat().st_size for p in inputs.rglob("*") if p.is_file()) + f.stat().st_size > MAX_BUNDLE_BYTES:
                        raise ValueError("Job inputs exceed 64 MiB. Stage large masks separately or use a smaller project.")
                    shutil.copyfile(f, target)
        # Ship only pure Python helpers and the pinned drivers, never GUI binaries.
        pkg = out / "runtime" / "feabas_workbench"
        (pkg / "core").mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        for src in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(src, pkg / "core" / src.name)
        shutil.copytree(VENDOR_DIR, pkg / "vendor" / "feabas_3_0_5", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(VENDOR_DIR.parent / "winfix", pkg / "vendor" / "winfix", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        hashes = {p.relative_to(inputs).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(inputs.rglob("*")) if p.is_file()}
        manifest = dict(format=1, run_id=run_id, remote_project=rp, remote_tiles=rt, remote_python=python,
                        resources=asdict(resources), steps=ordered, modules=module_list, input_hashes=hashes,
                        sections=project.section_names(), fine_compare_distance=project.state.alignment.get("fine_compare_distance", 0))
        if workspace:
            manifest["external_inputs"] = external
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        lines = ["#!/bin/bash", "# Generated by FEABAS Workbench; inspect before submission.",
                 "#SBATCH --job-name=feabas", f"#SBATCH --clusters={resources.cluster}",
                 f"#SBATCH --partition={resources.partition}"]
        qos = PARTITIONS[resources.partition][-1]
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        lines += ["#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={resources.cpus}",
                  "#SBATCH --hint=nomultithread", f"#SBATCH --mem={resources.memory_gib}G",
                  f"#SBATCH --time={resources.hours}:00:00", "#SBATCH --get-user-env", "#SBATCH --export=NONE",
                  f"#SBATCH --chdir={shlex.quote(remote_bundle)}",
                  f"#SBATCH --output={shlex.quote(remote_bundle + '/slurm-%j.out')}",
                  "set -euo pipefail", "module load slurm_setup"]
        if module_list:
            lines.append("module load " + " ".join(map(shlex.quote, module_list)))
        runtime = remote_bundle + "/runtime"
        lines += ["export PYTHONUNBUFFERED=1 PYTHONUTF8=1 MALLOC_ARENA_MAX=2",
                  "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
                  "export PYTHONPATH=" + shlex.quote(runtime + "/feabas_workbench/vendor/winfix:" + runtime),
                  "exec " + shlex.quote(python) + " -m feabas_workbench.core.cluster_runner " + shlex.quote(remote_bundle), ""]
        (out / "job.sh").write_text("\n".join(lines), encoding="utf-8", newline="\n")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > MAX_BUNDLE_BYTES:
            raise ValueError("Job bundle exceeds the 64 MiB small-file limit.")
        return out
    except Exception:
        shutil.rmtree(out)
        raise
