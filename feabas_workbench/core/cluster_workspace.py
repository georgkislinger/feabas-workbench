"""Project-scoped execution profiles, path mapping and reproducible cluster requests.

Passwords/tokens never belong in a project. The remote view is a separate cache so
remote previews cannot overwrite a workstation run's results.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path, PureWindowsPath

from .cluster_bundle import ClusterResources, MAX_BUNDLE_BYTES, export_bundle, remote_path, stage_input
from .project import Project
from .steps import PipelineScan, State, STEPS_BY_KEY

CONTROL = ".workbench-cluster"
INPUT_DIRS = ("configs", "stitch/stitch_coord", "masks", "structures", "thumbnail_align/material_masks",
              "thumbnail_align/manual_matches", "align/material_masks")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return copy.deepcopy(default)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def profile_for(project: Project) -> dict:
    path = project.root / CONTROL / "workspace.json"
    p = read_json(path, {})
    defaults = dict(format=2, mode="local", project_id=uuid.uuid4().hex,
                    host="cool.hpc.lrz.de", username="", port=22, key_filename="",
                    remote_project="", remote_tiles="", remote_python="", remote_dl_python="", modules="",
                    partition="serial_std", cpus=4, memory_gib=16, hours=1, workers=4, section_concurrency=1,
                    source_collection="", computer_collection="", destination_collection="",
                    source_path="", destination_tiles="", destination_project="",
                    include_existing=True, download_after_export=True, export_destination="", globus_cli="")
    # Migrate only connection/settings from the previous advanced batch dialog.
    legacy = read_json(project.root / CONTROL / "profile.json", {}) if not p else {}
    defaults.update({k: v for k, v in legacy.items() if k in defaults and k != "mode"})
    defaults.update({k: v for k, v in p.items() if k in defaults})
    return defaults


def resources_for(profile):
    return ClusterResources(**{k: profile[k] for k in ClusterResources.__dataclass_fields__})


def save_profile(project, profile):
    # Whitelist persisted fields. In particular, do not serialize authentication callbacks.
    allowed = profile_for(project)
    write_json(project.root / CONTROL / "workspace.json", {k: profile[k] for k in allowed if k in profile})


def view_project(local: Project) -> Project:
    root = local.root / CONTROL / "view"
    if not Project.exists(root):
        result = Project.create(root, local.state.name)
        result.state = copy.deepcopy(local.state)
        result.save()
        for rel in (*INPUT_DIRS, "section_order.txt"):
            src = local.root / rel
            files = [src] if src.is_file() else src.rglob("*") if src.is_dir() else []
            for f in files:
                if f.is_file() and not f.is_symlink():
                    dest = root / f.relative_to(local.root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
        result.write_general_config()
    return Project.load(root)


def local_globus_path(path: Path) -> str:
    """GCP's Windows drive syntax; NAS collections can use a different configured path."""
    text = str(path.resolve())
    p = PureWindowsPath(text)
    if p.drive and not p.drive.startswith("\\\\"):
        return "/" + p.drive.rstrip(":") + "/" + "/".join(p.parts[1:])
    if text.startswith("\\\\"):
        raise ValueError("Select the NAS folder in Globus and enter its collection path; a UNC path is not a Globus path.")
    return Path(text).as_posix()


def source_signature(project: Project) -> str:
    """Metadata fingerprint, not an expensive reread of terabytes of image contents."""
    root = Path(project.state.source.root_dir)
    if not root.is_dir():
        raise ValueError("The source image directory is not accessible on this computer.")
    h = hashlib.sha256(str(root.resolve()).encode())
    count = 0
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != CONTROL and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            p = Path(directory) / name
            if p.is_symlink():
                raise ValueError(f"Resolve source image links before synchronization: {p}")
            st = p.stat()
            h.update(f"{p.relative_to(root).as_posix()}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
            count += 1
    if not count:
        raise ValueError("The source image directory is empty.")
    return h.hexdigest()


class PathMapper:
    def __init__(self, project: Project, local: Project, profile: dict, bundle: Path, external=None):
        self.project, self.local, self.profile, self.bundle = project, local, profile, bundle
        self.external = external if external is not None else {}
        self.mappings = [(project.root, profile["remote_project"]), (local.root, profile["remote_project"]),
                         (Path(project.state.source.root_dir), profile["remote_tiles"])]
        self.mappings.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    def map(self, value):
        if isinstance(value, dict):
            return {k: self.map(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.map(v) for v in value]
        if not isinstance(value, str) or not (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            return value
        path = Path(value).resolve()
        for source, target in self.mappings:
            try:
                rel = path.relative_to(source.resolve()).as_posix()
                if target == self.profile["remote_project"] and path.is_file():
                    dest = self.bundle / "inputs" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    stage_input(path, dest, self.bundle, self.external)
                return remote_path(target) + ("/" + rel if rel != "." else "")
            except ValueError:
                continue
        # Standalone checkpoints/templates can travel with the small input bundle.
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"This input needs a cluster copy: {value}. Place it in the project and Sync project first.")
        name = hashlib.sha256(str(path).encode()).hexdigest()[:16] + "-" + path.name
        dest = self.bundle / "inputs" / "cluster_assets" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage_input(path, dest, self.bundle, self.external)
        return self.profile["remote_project"] + "/cluster_assets/" + name


def build_workspace_bundle(project, local, profile, commands, destination):
    res = resources_for(profile)
    res.validate()
    # Raw coordinates may refer to preprocessed files that only exist on the cluster.
    active = project.active_tile_root()
    raw = Path(project.state.source.root_dir).resolve()
    remote_tiles = profile["remote_tiles"]
    if active and active.resolve() != raw:
        remote_tiles = profile["remote_project"] + "/" + active.resolve().relative_to(project.root).as_posix()
    out = export_bundle(project, destination, profile["remote_project"], remote_tiles,
                        profile["remote_python"], res, ["stitch.matching"], profile["modules"], workspace=True)
    try:
        manifest = read_json(out / "manifest.json")
        mapper = PathMapper(project, local, profile, out, manifest["external_inputs"])
        prepared = []
        for original in commands:
            c = copy.deepcopy(original)
            if c["kind"] == "worker":
                if not re.fullmatch(r"[a-z][a-z0-9_]*", c["module"]):
                    raise ValueError("Invalid worker name.")
                if c.get("dl") and not profile.get("remote_dl_python"):
                    raise ValueError("This tool needs a cluster deep-learning environment. Set its Python in Advanced settings.")
                if c["module"] == "export_vast":
                    c["download_to"] = c["payload"]["out_dir"]
                    c["payload"]["out_dir"] = str(project.exports_dir / out.name)
                c["payload"] = mapper.map(c["payload"])
                if "workers" in c["payload"]:
                    c["payload"]["workers"] = res.cpus
                c["python"] = profile["remote_dl_python"] if c.get("dl") else profile["remote_python"]
            prepared.append(c)
        # User-authored edits, excluding model weights and large generated results.
        for rel in ("masks", "structures", "thumbnail_align/manual_matches"):
            src = project.root / rel
            for f in sorted(src.rglob("*")) if src.is_dir() else []:
                if f.is_symlink():
                    raise ValueError(f"Project input contains a link: {f}")
                if f.is_file():
                    dest = out / "inputs" / f.relative_to(project.root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    stage_input(f, dest, out, manifest["external_inputs"])
        pkg = out / "runtime" / "feabas_workbench"
        shutil.copytree(Path(__file__).parent.parent / "workers", pkg / "workers",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest.update(format=2, workspace=True, project_id=profile["project_id"], commands=prepared,
                        steps=[], raw_tiles=profile["remote_tiles"], remote_dl_python=profile.get("remote_dl_python", ""),
                        path_mappings=[(str(src), dst) for src, dst in mapper.mappings])
        manifest["input_hashes"] = {p.relative_to(out / "inputs").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((out / "inputs").rglob("*")) if p.is_file()}
        write_json(out / "manifest.json", manifest)
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > MAX_BUNDLE_BYTES:
            raise ValueError("Edited masks/assets exceed the small-file limit (64 MiB). Use Sync project to transfer large assets.")
        return out
    except Exception:
        shutil.rmtree(out)
        raise


def sync_scope(profile):
    keys = ("host", "username", "port", "remote_project", "remote_tiles", "source_collection", "source_path",
            "destination_collection", "destination_project", "destination_tiles")
    return hashlib.sha256(json.dumps({k: profile[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def remote_scan(project, configs, snapshot):
    from .cluster_remote import scientific_config
    from .steps import downstream
    scan = PipelineScan(project.root, len(project.section_names()), configs)
    for key, st in scan.status.items():
        row = snapshot.get("steps", {}).get(key)
        if row:
            for field in ("done", "expected", "errors", "newest_output", "oldest_output", "reasons"):
                if field in row:
                    setattr(st, field, row[field])
            st.state = State(row["state"])
        else:
            st.done = st.errors = 0
            st.state = State.EMPTY
            st.reasons = ["Remote state has not been checked yet."]
    for kind, key in (("stitching", "stitch.matching"), ("thumbnail", "thumbnail.downsample"),
                      ("alignment", "align.meshing"), ("material", "masks")):
        previous = snapshot.get("configurations", {}).get(kind)
        if previous is not None and scientific_config(previous) != scientific_config(configs[kind].merged):
            for step in [STEPS_BY_KEY[key], *downstream(key)]:
                if scan.status[step.key].done:
                    scan.status[step.key].state = State.STALE
                    scan.status[step.key].reasons.append("Local settings changed since the cluster run.")
    return scan


def section_tasks(command, sections, resources):
    """Only isolated section operations can fan out. Stack optimization stays single."""
    safe = {"stitch.matching", "stitch.optimization", "stitch.rendering", "thumbnail.downsample"}
    if command.get("step") not in safe or resources.section_concurrency == 1 or command.get("filter"):
        return [command]
    start, stop, stride = command.get("start") or 0, command.get("stop") or len(sections), command.get("stride") or 1
    if start < 0 or stop < 0 or stride < 1:
        raise ValueError("Cluster section ranges must use nonnegative indices and positive stride.")
    return [dict(command, start=i, stop=i + 1, stride=1) for i in range(start, min(stop, len(sections)), stride)]
