"""Headless Linux batch entry point; invoked only inside an allocation."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .configs import ConfigStore
from .project import VENDOR_DIR
from .steps import STEPS_BY_KEY, count_outputs, expected_count, write_fine_match_list


def install_inputs(bundle: Path, root: Path, manifest: dict) -> None:
    """Refuse different inputs or a nonempty foreign project; retain resumable outputs."""
    marker = root / ".workbench-cluster" / "inputs.json"
    hashes = manifest["input_hashes"]
    for rel, digest in hashes.items():
        p = bundle / "inputs" / rel
        if not p.resolve().is_relative_to((bundle / "inputs").resolve()) or p.is_symlink():
            raise RuntimeError("Invalid input path in bundle.")
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Bundle input changed: {rel}")
    if marker.exists():
        if json.loads(marker.read_text()) != hashes:
            raise RuntimeError("Remote project has different inputs/settings. Choose a NEW remote work folder.")
        for rel, digest in hashes.items():
            p = root / rel
            if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Remote input changed: {rel}. Use a new remote work folder.")
        return
    if any(p.name != ".workbench-cluster" for p in root.iterdir()):
        raise RuntimeError("Use an empty, dedicated remote work folder. Existing files are never overwritten.")
    # A partial failed install deliberately stays nonempty and requires a fresh work folder.
    for rel in hashes:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle / "inputs" / rel, target)
    marker.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")


def run(bundle: Path) -> None:
    import fcntl  # Linux only; the GUI remains portable.
    from importlib.metadata import version
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run job.sh with sbatch, never on a login node.")
    if version("feabas") != "3.0.5":
        raise RuntimeError("This bundle requires feabas==3.0.5 in the remote Python environment.")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("workspace"):
        from .cluster_remote import run as run_workspace
        return run_workspace(bundle)
    root = Path(manifest["remote_project"])
    root.mkdir(parents=True, exist_ok=True)
    control = root / ".workbench-cluster"
    control.mkdir(exist_ok=True)
    with (control / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError("Another Workbench job is using this remote project. Wait for it to finish.") from e
        install_inputs(bundle, root, manifest)
        # Check every referenced tile on the compute node, without reading image contents.
        tiles = Path(manifest["remote_tiles"])
        for coord in (root / "stitch" / "stitch_coord").glob("*.txt"):
            for line in coord.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("{"):
                    p = tiles / line.split("\t")[0]
                    if not p.is_file() or not os.access(p, os.R_OK):
                        raise RuntimeError(f"Stage the dataset before running: missing/unreadable {p}")
        config = ConfigStore(root / "configs")
        results = []
        for key in manifest["steps"]:
            step = STEPS_BY_KEY[key]
            if key in {"align.meshing", "align.matching"}:
                write_fine_match_list(root, manifest["sections"],
                                      int(manifest.get("fine_compare_distance", 0) or 0) or None,
                                      config.get("thumbnail", "alignment.match_name_delimiter", "__to__"))
            for dep in step.requires:
                if count_outputs(root, STEPS_BY_KEY[dep]) == 0:
                    raise RuntimeError(f"{key} needs {dep}; select the earlier steps for a fresh remote project.")
            print(f"\n=== {step.label} ({key}) ===", flush=True)
            subprocess.run([sys.executable, str(VENDOR_DIR / step.script), "--mode", step.mode], cwd=root, check=True)
            if key == "thumbnail.downsample":
                # FEABAS regenerates default masks when it creates thumbnails.
                # Restore the user's exported masks before coarse matching.
                for rel in manifest["input_hashes"]:
                    if rel.startswith("thumbnail_align/material_masks/"):
                        shutil.copyfile(bundle / "inputs" / rel, root / rel)
            expected = expected_count(root, len(manifest["sections"]), step, config)
            done = count_outputs(root, step)
            errors = list((root / step.out_subdir).glob(step.err_glob)) if step.err_glob else []
            if errors or (expected > 0 and done < expected):
                raise RuntimeError(f"{key}: {done}/{expected} outputs; {len(errors)} error markers. Inspect the log.")
            results.append(dict(step=key, done=done, expected=expected))
            (bundle / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("All selected FEABAS steps completed and outputs checked.", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
