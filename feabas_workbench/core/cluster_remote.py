"""Headless workspace execution. All image computation occurs inside Slurm."""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from .cluster_bundle import ClusterResources, MAX_BUNDLE_BYTES, file_hash, _portable_config
from .cluster_workspace import CONTROL, read_json, write_json, section_tasks
from .configs import ConfigStore, CONFIG_FILES, dump_yaml, load_yaml
from .project import VENDOR_DIR
from .steps import (STEPS_BY_KEY, PipelineScan, cleared_keys, clear_targets, count_outputs, expected_outputs, finish_clear,
                    step_argv, write_fine_match_list, create_snapshot)


def archive_outputs(root, keys, run_id):
    paths = set()
    for key in keys:
        paths.update(clear_targets(root, STEPS_BY_KEY[key], cascade=True))
    moved = []
    for src in sorted(paths, key=lambda p: len(p.parts)):
        if not src.exists() or any(p in src.parents for p in moved):
            continue
        if not src.resolve().is_relative_to(root.resolve()) or src.is_symlink():
            raise RuntimeError("Output path escapes the managed workspace.")
        dest = root / CONTROL / "previous" / run_id / src.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved.append(src)
    finish_clear(root, {k for key in keys for k in cleared_keys(STEPS_BY_KEY[key])})
    return [str(p.relative_to(root)) for p in moved]


def changed_steps(old, new):
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    keys = set()
    for rel in changed:
        if rel.startswith("stitch/stitch_coord/") or rel == "configs/stitching_configs.yaml":
            keys.add("stitch.matching")
        elif rel in {"configs/thumbnail_configs.yaml", "section_order.txt"}:
            keys.add("thumbnail.downsample")
        elif rel == "configs/alignment_configs.yaml":
            keys.add("align.meshing")
        elif "material" in rel or rel.startswith("masks/"):
            keys.add("thumbnail.matching"); keys.add("align.meshing")
        elif rel.startswith("thumbnail_align/manual_matches/"):
            keys.add("thumbnail.matching")
    return keys


def scientific_config(value):
    if isinstance(value, dict):
        return {k: scientific_config(v) for k, v in value.items() if k not in {"num_workers", "cpu_budget", "logging_directory"}}
    if isinstance(value, list):
        return [scientific_config(v) for v in value]
    return value


def relocate_existing_results(root, manifest):
    """Translate explicit FEABAS path fields once, on a compute node."""
    marker = root / CONTROL / "paths-relocated.json"
    if marker.exists():
        return
    mappings = [(a.replace("\\", "/").rstrip("/"), b.rstrip("/")) for a, b in manifest.get("path_mappings", [])]
    mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
    def translated(text):
        norm = text.replace("\\", "/")
        for source, target in mappings:
            if norm.casefold() == source.casefold() or norm.casefold().startswith(source.casefold() + "/"):
                return target + norm[len(source):]
        return text
    import h5py
    import numpy as np
    for p in (root / "stitch/tform").glob("*.h5"):
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Mirrored result path escapes the workspace.")
        with h5py.File(p, "r+") as f:
            if "imgrootdir" in f:
                original = bytes(f["imgrootdir"][()]).decode("utf-8")
                value = translated(original)
                if value != original:
                    del f["imgrootdir"]
                    f.create_dataset("imgrootdir", data=np.frombuffer(value.encode("utf-8"), dtype=np.uint8))
    for folder in ("stitched_sections", "aligned_stack"):
        for p in (root / folder).rglob("metadata.txt"):
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("Mirrored metadata path escapes the workspace.")
            lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
            edited = []
            for line in lines:
                fields = line.rstrip("\r\n").split("\t")
                if fields and fields[0] == "{ROOT_DIR}" and len(fields) == 2:
                    fields[1] = translated(fields[1])
                elif fields and not fields[0].startswith("{"):
                    fields[0] = translated(fields[0])
                edited.append("\t".join(fields) + "\n")
            if "".join(edited) != "".join(lines):
                p.write_text("".join(edited), encoding="utf-8")
    def relocate_spec(value):
        if isinstance(value, dict):
            return {k: relocate_spec(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate_spec(v) for v in value]
        return translated(value) if isinstance(value, str) else value
    specs = list((root / "stitch/ts_specs").glob("*.json"))
    if (root / "align/ts_spec.json").is_file():
        specs.append(root / "align/ts_spec.json")
    for p in specs:
        original = read_json(p)
        value = relocate_spec(original)
        if value != original:
            write_json(p, value)
    write_json(marker, dict(project_id=manifest["project_id"]))


def install_workspace_inputs(bundle, root, manifest):
    control = root / CONTROL
    owner = read_json(control / "owner.json", {})
    if owner.get("project_id") != manifest["project_id"]:
        raise RuntimeError("Remote folder is not owned by this Workbench project. Run the setup check first.")
    relocate_existing_results(root, manifest)
    marker = control / "workspace-inputs.json"
    old = read_json(marker, {})
    hashes = dict(manifest["input_hashes"])
    hashes.update({k: v["sha256"] for k, v in manifest.get("external_inputs", {}).items()})
    def source_for(rel):
        return bundle / ("external_inputs" if rel in manifest.get("external_inputs", {}) else "inputs") / rel
    for rel, digest in hashes.items():
        src = source_for(rel)
        if not src.resolve().is_relative_to(bundle.resolve()) or src.is_symlink() or ".." in Path(rel).parts:
            raise RuntimeError("Unsafe bundle input path.")
        if file_hash(src) != digest:
            raise RuntimeError(f"Bundle input changed: {rel}")
    if old:
        # Runtime CPU budgets do not invalidate scientific outputs. Other general
        # settings (voxel size, section thickness) do.
        comparison = dict(old)
        for rel, digest in hashes.items():
            existing = root / rel
            if existing.is_file() and not existing.is_symlink():
                if file_hash(existing) == digest:
                    comparison[rel] = digest
                elif rel.startswith("configs/") and scientific_config(load_yaml(existing)) == scientific_config(load_yaml(bundle / "inputs" / rel)):
                    comparison[rel] = digest
        keys = changed_steps(comparison, hashes)
        a = load_yaml(root / "configs/general_configs.yaml")
        b = load_yaml(bundle / "inputs/configs/general_configs.yaml")
        if any(a.get(k) != b.get(k) for k in ("full_resolution", "section_thickness")):
            keys.add("stitch.matching")
        if "stitch.matching" in keys:
            # FEABAS caches the tile resolution of its first run here and never reads the
            # coordinate files again; new coordinates may carry a corrected pixel size
            cache = root / "configs" / "resolutions.yaml"
            if cache.is_file() and not cache.is_symlink():
                cache.unlink()
        archived = archive_outputs(root, keys, manifest["run_id"])
        if archived:
            print("Changed inputs: previous downstream outputs moved to " + CONTROL + "/previous/" + manifest["run_id"], flush=True)
    for rel in old.keys() - hashes.keys():
        dst = root / rel
        if dst.is_file() and not dst.is_symlink() and dst.resolve().is_relative_to(root.resolve()):
            archive = control / "previous" / manifest["run_id"] / rel
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(archive))
    for rel in hashes:
        dest = root / rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Input destination escapes the workspace.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.is_file() or file_hash(dest) != hashes[rel]:
            previous_time = dest.stat().st_mtime_ns if dest.is_file() else None
            semantic_same = bool(previous_time and rel.startswith("configs/") and
                                 scientific_config(load_yaml(dest)) == scientific_config(load_yaml(bundle / "inputs" / rel)))
            shutil.copyfile(source_for(rel), dest)
            if semantic_same:
                os.utime(dest, ns=(previous_time, previous_time))
    write_json(marker, hashes)


def write_snapshot(root, bundle, manifest, state, results, error="", peak_memory=0):
    configs = ConfigStore(root / "configs")
    scan = PipelineScan(root, len(manifest["sections"]), configs)
    snapshot = dict(state=state, timestamp=time.time(), error=error, results=results, peak_memory_bytes=peak_memory,
                    run_id=manifest["run_id"], project_id=manifest["project_id"], steps={},
                    configurations={k: configs[k].merged for k in ("stitching", "thumbnail", "alignment", "material")})
    for key, st in scan.status.items():
        snapshot["steps"][key] = {k: getattr(st, k) for k in ("state", "done", "expected", "errors",
                                                           "newest_output", "oldest_output", "reasons")}
    # A bounded preview package; bulk outputs always travel through Globus.
    included, skipped, size = {}, 0, 0
    folders = ["thumbnail_align/thumbnails", "thumbnail_align/material_masks", "masks", "structures",
               "thumbnail_align", "stitch/tform", "align", "models"]
    files = {}
    for rel in folders:
        d = root / rel
        for p in sorted(d.rglob("*")) if d.is_dir() else []:
            if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve()):
                r = p.relative_to(root).as_posix()
                if p.suffix.lower() in {".png", ".jpg", ".tif", ".tiff", ".h5", ".json", ".txt", ".yaml", ".pt", ".pth", ".ckpt"}:
                    files.setdefault(r, p)
    archive = bundle / "previews.zip"
    tmp = archive.with_suffix(".tmp")
    all_previews = {}
    # Larger UI caches travel via Globus. Full-resolution stacks are excluded.
    cache = bundle / "preview-cache"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel, p in files.items():
            n = p.stat().st_size
            digest = file_hash(p)
            all_previews[rel] = digest
            if state in {"COMPLETED", "FAILED"}:
                target = cache / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                try:
                    os.link(p, target)
                except OSError:
                    shutil.copyfile(p, target)
            if size + n > MAX_BUNDLE_BYTES - 1024 * 1024 or n > 16 * 1024 * 1024:
                skipped += 1
                continue
            data = p.read_bytes()
            z.writestr(rel, data)
            included[rel] = digest
            size += len(data)
    os.replace(tmp, archive)
    snapshot.update(previews=included, preview_manifest=all_previews, preview_bytes=sum(p.stat().st_size for p in files.values()),
                    previews_skipped=skipped,
                    export_bytes={d.name: sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                                  for d in (root / "exports").iterdir() if d.is_dir()} if (root / "exports").is_dir() else {},
                    preprocessing={name: sum(1 for p in (root / "preprocessed" / name).rglob("*") if p.is_file())
                                   for name in ("histmatch", "denoised")})
    snapshot["render_bytes"] = {name: sum(p.stat().st_size for p in (root / name).rglob("*") if p.is_file())
                                for name in ("aligned_stack", "aligned_tensorstore")}
    write_json(bundle / "status.json", snapshot)
    return snapshot


def run_commands(bundle, root, manifest):
    resources = ClusterResources(**manifest["resources"])
    resources.validate()
    results, peak = [], [0]
    active, lock = set(), threading.Lock()
    stopped = threading.Event()
    def sample_memory():
        try:
            import psutil
        except ImportError:
            return
        while not stopped.wait(1):
            total = 0
            with lock:
                pids = list(active)
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                    for p in [proc, *proc.children(recursive=True)]:
                        try:
                            total += p.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], total)
            write_json(bundle / "heartbeat.json", dict(timestamp=time.time(), peak_memory_bytes=peak[0], memory_bytes=total))
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    def execute(command, index, parallel=1):
        work = bundle / "commands" / str(index)
        work.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "configs", work / "configs", dirs_exist_ok=True)
        # Whole-stack operations may use the full allocation. Section fan-out
        # uses the explicit workers-per-section budget.
        budget = resources.workers if parallel > 1 else resources.cpus
        configs = ConfigStore(work / "configs")
        for kind, filename in CONFIG_FILES.items():
            dump_yaml(work / "configs" / filename, _portable_config(configs[kind].merged, budget))
        general = load_yaml(work / "configs/general_configs.yaml")
        general.update(cpu_budget=max(1, resources.cpus // parallel), logging_directory=str(work / "logs"))
        dump_yaml(work / "configs/general_configs.yaml", general)
        kind = command["kind"]
        if kind == "step":
            step = STEPS_BY_KEY[command["step"]]
            argv = step_argv(sys.executable, step, command.get("start"), command.get("stop"),
                             command.get("stride"), command.get("filter"), command.get("extra_args"))
        elif kind == "worker":
            payload = work / "payload.json"
            write_json(payload, command["payload"])
            argv = [command["python"], "-m", "feabas_workbench.workers." + command["module"], "--spec", str(payload)]
        elif kind == "tool":
            tool = (VENDOR_DIR / "tools" / command["tool"]).resolve()
            if not tool.is_relative_to((VENDOR_DIR / "tools").resolve()) or not tool.is_file():
                raise RuntimeError("Unknown FEABAS tool.")
            argv = [sys.executable, str(tool), *command["args"]]
        else:
            raise RuntimeError(f"Unsupported command type: {kind}")
        output = {}
        started = time.monotonic()
        with subprocess.Popen(argv, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
            with lock:
                active.add(proc.pid)
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    if line.startswith("##RESULT "):
                        output = json.loads(line[len("##RESULT "):])
                code = proc.wait()
            finally:
                with lock:
                    active.discard(proc.pid)
            if code:
                raise RuntimeError(f"{command.get('name', kind)} exited with code {code}.")
        return dict(command=command, result=output, seconds=time.monotonic() - started)
    try:
        write_snapshot(root, bundle, manifest, "RUNNING", results)
        for index, command in enumerate(manifest["commands"]):
            kind = command["kind"]
            if kind == "clear":
                archive_outputs(root, [command["step"]], manifest["run_id"] + "-clear")
                results.append(dict(command=command, result={}, seconds=0))
            elif kind == "snapshot":
                create_snapshot(root, command.get("label", "cluster"))
                results.append(dict(command=command, result={}, seconds=0))
            else:
                step = STEPS_BY_KEY[command["step"]] if kind == "step" else None
                if step:
                    if step.key.startswith("stitch.") and manifest.get("remote_tiles"):
                        for coord in (root / "stitch/stitch_coord").glob("*.txt"):
                            for line in coord.read_text(encoding="utf-8").splitlines():
                                if line and not line.startswith("{"):
                                    p = Path(manifest["remote_tiles"]) / line.split("\t")[0]
                                    if not p.is_file() or not os.access(p, os.R_OK):
                                        raise RuntimeError(f"Image missing or unreadable: {p}. Synchronize the selected source first.")
                    if step.key in {"align.meshing", "align.matching"}:
                        write_fine_match_list(root, manifest["sections"], manifest.get("fine_compare_distance") or None,
                                              ConfigStore(root / "configs").get("thumbnail", "alignment.match_name_delimiter", "__to__"))
                    for dep in step.requires:
                        if count_outputs(root, STEPS_BY_KEY[dep]) == 0:
                            raise RuntimeError(f"{step.label} needs {STEPS_BY_KEY[dep].label}. Run the earlier step first.")
                tasks = section_tasks(command, manifest["sections"], resources)
                if not tasks:
                    raise RuntimeError("The selected section range is empty.")
                concurrency = min(len(tasks), resources.section_concurrency)
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(execute, c, f"{index}-{i}", concurrency) for i, c in enumerate(tasks)]
                    command_results = [f.result() for f in futures]
                if step:
                    errors = list((root / step.out_subdir).glob(step.err_glob)) if step.err_glob else []
                    expected = expected_outputs(root, step, command.get("start"), command.get("stop"), command.get("stride"))
                    if errors or count_outputs(root, step) < expected:
                        raise RuntimeError(f"{step.label}: missing outputs or error markers; check the job log.")
                results.append(dict(command=command, result=command_results[-1]["result"],
                                    seconds=sum(r["seconds"] for r in command_results)))
                if step and step.key == "thumbnail.downsample":
                    for rel in manifest.get("input_hashes", {}).keys() | manifest.get("external_inputs", {}).keys():
                        if rel.startswith("thumbnail_align/material_masks/"):
                            src = bundle / ("external_inputs" if rel in manifest.get("external_inputs", {}) else "inputs") / rel
                            shutil.copyfile(src, root / rel)
            write_snapshot(root, bundle, manifest, "RUNNING", results, peak_memory=peak[0])
        return write_snapshot(root, bundle, manifest, "COMPLETED", results, peak_memory=peak[0])
    except Exception as e:
        write_snapshot(root, bundle, manifest, "FAILED", results, str(e), peak[0])
        raise
    finally:
        stopped.set()
        monitor.join(3)


def run(bundle):
    import fcntl
    from importlib.metadata import version
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit through Slurm. Image processing must not run on a login node.")
    if version("feabas") != "3.0.5":
        raise RuntimeError("The remote environment must contain FEABAS 3.0.5.")
    manifest = read_json(bundle / "manifest.json")
    root = Path(manifest["remote_project"])
    with (root / CONTROL / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            install_workspace_inputs(bundle, root, manifest)
            run_commands(bundle, root, manifest)
        except Exception as e:
            if not (bundle / "status.json").exists():
                write_json(bundle / "status.json", dict(state="FAILED", error=str(e), results=[],
                           project_id=manifest["project_id"], run_id=manifest["run_id"], steps={}))
            raise
