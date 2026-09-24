"""Run independent FEABAS sections with a bounded worker budget on this PC."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from feabas_workbench.core.configs import ConfigStore
from feabas_workbench.core.jobs import kill_tree, feabas_env
from feabas_workbench.core.local_parallel import SECTION_STEPS, input_sections, selected_indices, plan
from feabas_workbench.core.project import Project
from feabas_workbench.core.steps import STEPS_BY_KEY, step_argv
from feabas_workbench.workers.common import load_spec, log, progress, result, run


def verify_section(root, key, source, configs):
    """FEABAS can catch a worker error and still return zero; require its finished output."""
    name = source.stem
    if key.startswith("stitch.") and key != "stitch.rendering":
        subdir = "match_h5" if key == "stitch.matching" else "tform"
        target = root / "stitch" / subdir / (name + ".h5")
        if target.with_suffix(".h5_err").exists() or not target.is_file():
            raise RuntimeError(f"{name}: {key} did not produce a completed result; check the section log.")
        return
    if key == "thumbnail.downsample":
        name = source.parent.name if source.name == "metadata.txt" else source.stem
        ext = configs.get("thumbnail", "downsample.thumbnail_format", "png")
        target = root / "thumbnail_align/thumbnails" / (name + "." + ext)
        mask = root / "thumbnail_align/material_masks" / (name + ".png")
        if not target.is_file() or not mask.is_file():
            raise RuntimeError(f"{name}: thumbnail or its material mask missing after processing.")
        return
    kind = "stitching" if key == "stitch.rendering" else "alignment"
    folder = "stitched_sections" if kind == "stitching" else "aligned_stack"
    base = Path(configs.get(kind, "rendering.out_dir") or root / folder)
    if not base.is_absolute():
        base = root / base
    mip = int(configs.get(kind, "rendering.mip_level", 0))
    if key == "align.downsample":
        if int(configs.get(kind, "downsample.max_mip", 8)) <= mip:
            return
        targets = [base / f"mip{level}" / source.parent.name / "metadata.txt"
                   for level in range(mip + 1, int(configs.get(kind, "downsample.max_mip", 8)) + 1)]
        if not all(p.is_file() for p in targets):
            raise RuntimeError(f"{name}: one or more requested mipmap levels are missing.")
        return
    elif key == "stitch.rendering" and configs.get(kind, "rendering.driver", "image") != "image":
        targets = [root / "stitch/ts_specs" / (name + ".json")]
    elif key == "stitch.rendering":
        targets = [base / f"mip{mip}" / name / "metadata.txt"]
    else:
        targets = [d / "metadata.txt" for d in (base / f"mip{mip}").glob("*")
                   if d.name == name or d.name.endswith("_" + name)]
    if not any(p.is_file() for p in targets):
        raise RuntimeError(f"{name}: finished render/mipmap metadata missing after processing.")


def execute(spec):
    root = Path(spec["root"]).resolve()
    key = spec["step"]
    step = STEPS_BY_KEY[key]
    configs = ConfigStore(root / "configs")
    paths, indices = selected_indices(input_sections(root, key, configs), key, spec.get("start"),
        spec.get("stop"), spec.get("stride"), spec.get("filter"), spec.get("reverse", False))
    if not indices:
        raise RuntimeError("No input sections match this range. Complete the preceding stage or change the range.")
    project = Project.load(spec["project"])
    setting = spec["settings"]
    allocation = plan(project, key, setting, count=len(indices))
    directory = root / "logs/local-parallel" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    log(f"Local parallelism: {allocation.sections} sections × {allocation.workers} workers; "
        f"CPU budget {allocation.cpu_budget}, RAM planning budget {allocation.ram_budget_gib:.1f} GiB.")
    if allocation.note:
        log(allocation.note)
    (directory / "plan.json").write_text(json.dumps(dict(step=key, inputs=[str(paths[i]) for i in indices],
        plan=allocation.to_dict()), indent=2), encoding="utf-8")
    stopped, active, mutex, output_lock = threading.Event(), set(), threading.Lock(), threading.Lock()
    first_failure = []
    peak = [0]
    monitor_stop = threading.Event()

    def memory_monitor():
        import psutil
        while not monitor_stop.wait(.5):
            with mutex:
                pids = list(active)
            usage = 0
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    for child in [p, *p.children(recursive=True)]:
                        try:
                            usage += child.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
            peak[0] = max(peak[0], usage)

    def stop_children():
        stopped.set()
        with mutex:
            pids = list(active)
        for pid in pids:
            kill_tree(pid)

    def launch(index, prepare=False):
        if stopped.is_set():
            raise RuntimeError("Section cancelled after another section failed.")
        name = "canvas" if prepare else f"{index:06d}"
        child_dir = directory / name
        child_dir.mkdir()
        kind, field, _ = SECTION_STEPS[key]
        resources = child_dir / "resources.json"
        resources.write_text(json.dumps(dict(kind=kind, field=field, workers=allocation.workers,
            logging_directory=str(child_dir), section_name=(paths[index].parent.name
                if paths[index].name == "metadata.txt" else paths[index].stem))), encoding="utf-8")
        env = dict(os.environ)
        env.update(feabas_env())
        env["FW_LOCAL_RESOURCES_FILE"] = str(resources)
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[variable] = "1"
        argv = step_argv(sys.executable, step, 0 if prepare else index, 0 if prepare else index + 1, 1,
                         spec.get("filter"))
        if prepare:
            # step_argv treats GUI stop=0 as 'to the end'; this internal pass must be empty.
            argv += ["--stop", "0"]
        elif key == "thumbnail.downsample":
            # Later phases discover newly created mipmaps, so their numeric positions
            # can change while other sections run. The storage hook selects by name.
            argv = step_argv(sys.executable, step, 0, 1)
        opts = dict(creationflags=subprocess.CREATE_NO_WINDOW) if os.name == "nt" else {}
        # The cwd stays the project, including relative custom paths. Runtime
        # resource overrides are process-local and inherited by FEABAS children.
        with output_lock:
            log("Preparing shared canvas" if prepare else f"Starting section {index}: {paths[index].parent.name if paths[index].name == 'metadata.txt' else paths[index].stem}")
        started = time.monotonic()
        try:
            with mutex:
                if stopped.is_set():
                    raise RuntimeError("Section cancelled.")
                proc = subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", **opts)
                active.add(proc.pid)
            try:
                with (child_dir / "console.log").open("w", encoding="utf-8") as transcript:
                    for line in proc.stdout:
                        transcript.write(line)
                        transcript.flush()
                        with output_lock:
                            print(f"[{name}] {line}", end="", flush=True)
                code = proc.wait()
                if code:
                    raise RuntimeError(f"Section {name} exited {code}. See {child_dir / 'console.log'}")
                if not prepare:
                    verify_section(root, key, paths[index], configs)
            finally:
                if proc.poll() is None:
                    kill_tree(proc.pid)
                    proc.wait()
                proc.stdout.close()
                with mutex:
                    active.discard(proc.pid)
        except BaseException as error:
            with mutex:
                if not first_failure:
                    first_failure.append(error)
            stop_children()
            raise
        return dict(index=index, seconds=time.monotonic() - started)

    monitor = threading.Thread(target=memory_monitor, daemon=True)
    monitor.start()
    completed = []
    try:
        progress(0, len(indices), f"{allocation.sections} sections at once × {allocation.workers} workers")
        # This driver writes shared canvas.json before rendering sections. Prepare
        # it exactly once with an empty range before allowing parallel readers.
        if key == "align.rendering" and configs.get("alignment", "rendering.offset_bbox", True):
            launch(0, prepare=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=allocation.sections) as pool:
            futures = [pool.submit(launch, i) for i in indices]
            try:
                for future in concurrent.futures.as_completed(futures):
                    completed.append(future.result())
                    progress(len(completed), len(indices), f"{len(completed)}/{len(indices)} sections complete")
            except BaseException:
                stop_children()
                for future in futures:
                    future.cancel()
                if first_failure:
                    raise first_failure[0]
                raise
    finally:
        monitor_stop.set()
        monitor.join(2)
    report = dict(plan=allocation.to_dict(), sections=completed, peak_rss_bytes=peak[0], logs=str(directory))
    (directory / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Observed peak summed process memory: {peak[0] / 1024**3:.2f} GiB (shared pages may be counted more than once).")
    result(report)
    return 0


def main():
    return execute(load_spec("Local section parallelism"))


if __name__ == "__main__":
    run(main)
