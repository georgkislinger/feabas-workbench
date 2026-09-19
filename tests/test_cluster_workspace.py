from __future__ import annotations

import copy
import hashlib
import os
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from feabas_workbench.core.cluster_bundle import ClusterResources
from feabas_workbench.core.cluster_workspace import (build_workspace_bundle, profile_for, save_profile, view_project,
    read_json, write_json, section_tasks, resources_for, source_signature, remote_scan, sync_scope)
from feabas_workbench.core.cluster_remote import install_workspace_inputs, run_commands
from feabas_workbench.core.configs import ConfigStore
from feabas_workbench.core.synthetic import make_demo_project
from feabas_workbench.core.steps import STEPS_BY_KEY, State
from feabas_workbench.core.globus_transfer import GlobusCLI


@pytest.fixture
def project(tmp_path):
    return make_demo_project(tmp_path / "project", n_sections=3, tile=128)


@pytest.fixture
def profile(project):
    p = profile_for(project)
    p.update(username="testuser", remote_project="/dss/example/work", remote_tiles="/dss/example/images",
             remote_python="/dss/env/bin/python", cpus=4, workers=2, section_concurrency=2)
    return p


def test_cpu_budget_and_teramem_are_validated():
    ClusterResources("teramem_inter", 32, 512, 8, 16, 2).validate()
    with pytest.raises(ValueError, match="Simultaneous"):
        ClusterResources("cm4_tiny", 32, 128, 4, 16, 3).validate()
    with pytest.raises(ValueError):
        ClusterResources("cm4_tiny", 64, 512, 4, 16, 1).validate()


def test_parallel_section_ranges_are_disjoint_and_stack_solve_is_single():
    resource = ClusterResources("cm4_tiny", 32, 128, 4, 16, 2)
    command = dict(kind="step", step="thumbnail.downsample", start=1, stop=7, stride=2)
    tasks = section_tasks(command, list(range(10)), resource)
    assert [(t["start"], t["stop"]) for t in tasks] == [(1, 2), (3, 4), (5, 6)]
    command["step"] = "align.optimization"
    assert section_tasks(command, list(range(10)), resource) == [command]


def test_cluster_view_keeps_workstation_outputs_separate(project):
    marker = project.root / "align/tform/local-only.txt"
    marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text("keep")
    view = view_project(project)
    assert view.root != project.root
    assert view.state.source.root_dir == project.state.source.root_dir
    assert list(view.stitch_coord_dir.glob("*.txt"))
    assert not (view.root / "align/tform/local-only.txt").exists()
    assert marker.read_text() == "keep"


def test_profile_does_not_persist_secrets(project, profile):
    profile.update(password="do not store", token="do not store")
    save_profile(project, profile)
    saved = read_json(project.root / ".workbench-cluster/workspace.json")
    assert "password" not in saved and "token" not in saved


def test_bundle_maps_workers_and_preserves_subset(project, profile, tmp_path):
    view = view_project(project)
    commands = [dict(kind="step", step="stitch.matching", start=1, stop=2, stride=1),
                dict(kind="worker", name="Export", module="export_vast", dl=False,
                     payload=dict(aligned_stack=str(view.root / "aligned_stack"), out_dir=str(tmp_path / "download")))]
    bundle = build_workspace_bundle(view, project, profile, commands, tmp_path / "bundles")
    m = read_json(bundle / "manifest.json")
    assert m["format"] == 2 and m["commands"][0]["stop"] == 2
    assert m["commands"][1]["payload"]["aligned_stack"] == "/dss/example/work/aligned_stack"
    assert m["commands"][1]["payload"]["out_dir"] == "/dss/example/work/exports/" + m["run_id"]
    assert m["commands"][1]["download_to"] == str(tmp_path / "download")
    assert (bundle / "runtime/feabas_workbench/workers/export_vast.py").is_file()
    assert m["resources"]["section_concurrency"] == 2


def test_new_source_file_invalidates_sync(project):
    old = source_signature(project)
    (Path(project.state.source.root_dir) / "new.txt").write_text("data")
    assert source_signature(project) != old


def test_scope_change_invalidates_sync(profile):
    other = dict(profile, remote_project="/dss/different/work")
    assert sync_scope(other) != sync_scope(profile)


def test_revised_settings_archive_downstream_but_keep_raw(project, profile, tmp_path):
    view = view_project(project)
    bundle = build_workspace_bundle(view, project, profile, [dict(kind="step", step="stitch.matching")], tmp_path / "bundles")
    m = read_json(bundle / "manifest.json")
    root = tmp_path / "remote"
    write_json(root / ".workbench-cluster/owner.json", dict(project_id=profile["project_id"]))
    install_workspace_inputs(bundle, root, m)
    output = root / "stitch/match_h5/old.h5"
    output.parent.mkdir(parents=True); output.write_bytes(b"previous result")
    raw = root / "raw-keep.tif"; raw.write_bytes(b"raw")
    config = ConfigStore(view.configs_dir)
    config.set("stitching", "matching.working_mip_level", 3); config.save()
    next_bundle = build_workspace_bundle(view, project, profile, [dict(kind="step", step="stitch.matching")], tmp_path / "bundles")
    m2 = read_json(next_bundle / "manifest.json")
    install_workspace_inputs(next_bundle, root, m2)
    assert not output.exists()
    assert (root / ".workbench-cluster/previous" / m2["run_id"] / "stitch/match_h5/old.h5").read_bytes() == b"previous result"
    assert raw.read_bytes() == b"raw"


def test_identical_inputs_keep_timestamps_and_outputs(project, profile, tmp_path):
    view = view_project(project)
    bundle = build_workspace_bundle(view, project, profile, [dict(kind="step", step="stitch.matching")], tmp_path / "bundles")
    m = read_json(bundle / "manifest.json")
    root = tmp_path / "remote"
    write_json(root / ".workbench-cluster/owner.json", dict(project_id=profile["project_id"]))
    install_workspace_inputs(bundle, root, m)
    before = (root / "configs/stitching_configs.yaml").stat().st_mtime_ns
    install_workspace_inputs(bundle, root, m)
    assert (root / "configs/stitching_configs.yaml").stat().st_mtime_ns == before


def test_remote_status_does_not_claim_local_results_are_remote(project):
    cfg = ConfigStore(project.configs_dir)
    scan = remote_scan(project, cfg, {})
    assert all(s.done == 0 for s in scan.status.values())
    snapshot = dict(steps={"stitch.matching": dict(state="done", done=3, expected=3)},
                    configurations={"stitching": copy.deepcopy(cfg["stitching"].merged)})
    assert remote_scan(project, cfg, snapshot)["stitch.matching"].state == State.COMPLETE
    cfg.set("stitching", "matching.working_mip_level", 5)
    assert remote_scan(project, cfg, snapshot)["stitch.matching"].state == State.STALE


def test_globus_transfer_encrypts_verifies_and_never_deletes(monkeypatch):
    calls = []
    cli = GlobusCLI()
    monkeypatch.setattr(cli, "run", lambda args: calls.append(args) or {"task_id": "test-id"})
    a, b = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    assert cli.transfer(a, "/source with spaces", b, "/dest", "test") == "test-id"
    assert "--verify-checksum" in calls[0] and "--encrypt" in calls[0]
    assert "checksum" in calls[0] and "--delete-destination-extra" not in calls[0]


def test_preview_extraction_preserves_manual_edits_and_rejects_escape(tmp_path):
    from feabas_workbench.ui.cluster_backend import apply_previews
    view = tmp_path / "view"; view.mkdir()
    file = view / "mask.png"; file.write_bytes(b"edited")
    archive = tmp_path / "preview.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mask.png", b"remote")
    snap = dict(previews={"mask.png": hashlib.sha256(b"remote").hexdigest()})
    assert apply_previews(archive, view, snap, {"mask.png": hashlib.sha256(b"old").hexdigest()}) == ["mask.png"]
    assert file.read_bytes() == b"edited"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape", b"remote")
    with pytest.raises(ValueError, match="path"):
        apply_previews(archive, view, dict(previews={"../escape": hashlib.sha256(b"remote").hexdigest()}), {})


@pytest.fixture
def window(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from feabas_workbench.core import envs
    from feabas_workbench.ui.main_window import MainWindow
    monkeypatch.setattr(envs, "SETTINGS_FILE", tmp_path / "settings.json")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    w = MainWindow(envs.Settings())
    yield w
    w.shutdown(); w.close(); app.processEvents()


def test_normal_run_buttons_dispatch_remote_without_local_environment(window, project, profile, monkeypatch):
    window.open_project(project.root)
    window.ctx.use_cluster(profile)
    captured = []
    monkeypatch.setattr(window.ctx.cluster, "submit", captured.append)
    monkeypatch.setattr(window.ctx.jobs.queue, "submit", lambda _: pytest.fail("Local queue must not run"))
    spec = window.ctx.feabas_step_spec(STEPS_BY_KEY["stitch.matching"], start=1, stop=2)
    window.ctx.jobs.submit(spec)
    assert captured[0].remote["stop"] == 2
    assert "LRZ cluster" in window.execution_label.text()
    assert window.page_count() == 7


def test_reopening_active_job_restores_cluster_mode_and_close_does_not_cancel(window, project, profile, monkeypatch):
    profile["mode"] = "cluster"
    save_profile(project, profile)
    write_json(project.root / ".workbench-cluster/workspace-jobs.json", [dict(state="PENDING", job_id="123")])
    window.open_project(project.root)
    assert window.ctx.cluster_enabled and window.ctx.jobs.running
    monkeypatch.setattr(window.ctx.cluster, "cancel_all", lambda: pytest.fail("Closing must not cancel a remote job"))
    window.shutdown()
    assert read_json(project.root / ".workbench-cluster/workspace-jobs.json")[0]["state"] == "PENDING"


def test_setup_uses_real_resource_values(window, project, profile):
    from feabas_workbench.ui.cluster_setup import ClusterSetupDialog
    window.open_project(project.root)
    dialog = ClusterSetupDialog(window.ctx, window)
    dialog.backend.profile.update(partition="cm4_tiny", cpus=64, memory_gib=128, workers=16)
    dialog._load()
    assert dialog.preset.currentText() == "Custom / saved settings"
    assert dialog.fields["cpus"].value() == 64
    dialog.preset.setCurrentIndex(2)
    p = dialog._values()
    assert p["partition"] == "teramem_inter" and p["memory_gib"] == 512
    resources_for(p).validate()
    dialog.shutdown(); dialog.close()


def test_stale_running_report_still_checks_slurm_for_oom(window, project, profile, monkeypatch):
    window.open_project(project.root); window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    entry = dict(state="PENDING", job_id="12", cluster="serial", remote_bundle="/dss/run", run_id="abc",
                 host=profile["host"], username=profile["username"], port=22, preview_stamp=1, last_scheduler_check=0,
                 started=time.time(), commands=[], notified=False)
    backend.journal = [entry]
    calls = []
    status = dict(state="RUNNING", timestamp=1, project_id=profile["project_id"], run_id="abc")
    backend.client = SimpleNamespace(connected=True, read_remote_json=lambda *a: status,
        status=lambda *a: calls.append(a) or "12|OUT_OF_MEMORY|0:125|00:01:00", close=lambda: None)
    monkeypatch.setattr(backend, "_work", lambda text, fn, done: done(fn()))
    backend.refresh()
    assert len(calls) == 1 and entry["state"] == "OUT_OF_MEMORY"


def test_real_headless_worker_creates_remote_export(tmp_path, monkeypatch):
    """Exercise process launch, runtime cwd/configs and worker result parsing together."""
    from PIL import Image
    from feabas_workbench.core.project import Project
    root = Project.create(tmp_path / "remote")
    tiles = root.root / "aligned_stack/mip0/0001_test"; tiles.mkdir(parents=True)
    Image.new("L", (32, 32), 77).save(tiles / "0001_test_tr0-tc0.png")
    (tiles / "metadata.txt").write_text("{ROOT_DIR}\t.\n0001_test_tr0-tc0.png\t0\t0\t32\t32\n")
    bundle = tmp_path / "bundle"; bundle.mkdir()
    command = dict(kind="worker", name="Export", module="export_vast", python=sys.executable,
                   payload=dict(aligned_stack=str(root.root / "aligned_stack"), out_dir=str(root.exports_dir),
                                name="sample", voxel_nm=[10, 10, 100], what="vast"))
    manifest = dict(resources=asdict(ClusterResources("serial_std", 2, 8, 1, 1, 1)), sections=["test"],
                    commands=[command], run_id="test-run", project_id="test-project")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    snap = run_commands(bundle, root.root, manifest)
    assert snap["state"] == "COMPLETED"
    assert (root.exports_dir / "vast/sample.vsvi").is_file() or list(root.exports_dir.rglob("*.vsvi"))
    assert (bundle / "previews.zip").is_file()


def test_large_assets_use_verified_bulk_staging(project, profile, tmp_path):
    from feabas_workbench.core.cluster_bundle import MAX_BUNDLE_BYTES
    view = view_project(project)
    asset = tmp_path / "checkpoint.pt"
    with asset.open("wb") as f:
        f.truncate(MAX_BUNDLE_BYTES)
    command = dict(kind="worker", module="fold_predict", payload=dict(checkpoint=str(asset)))
    bundle = build_workspace_bundle(view, project, profile, [command], tmp_path / "bundles")
    m = read_json(bundle / "manifest.json")
    assert len(m["external_inputs"]) == 1
    rel, planned = next(iter(m["external_inputs"].items()))
    assert planned["size"] == MAX_BUNDLE_BYTES
    assert not (bundle / "inputs" / rel).exists()
    assert sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()) < MAX_BUNDLE_BYTES
    root = tmp_path / "remote"
    write_json(root / ".workbench-cluster/owner.json", dict(project_id=profile["project_id"]))
    destination = bundle / "external_inputs" / rel
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"truncated transfer")
    with pytest.raises(RuntimeError, match="changed"):
        install_workspace_inputs(bundle, root, m)


def test_remote_preprocessed_source_can_be_inside_work_directory(project, profile, tmp_path):
    view = view_project(project)
    view.state.preprocessing.active_source = "histmatch"
    old = Path(view.state.source.root_dir).as_posix()
    for coord in view.stitch_coord_dir.glob("*.txt"):
        coord.write_text(coord.read_text().replace(old, (view.preprocessed_dir / "histmatch").as_posix()))
    bundle = build_workspace_bundle(view, project, profile, [dict(kind="step", step="stitch.matching")], tmp_path / "bundles")
    assert read_json(bundle / "manifest.json")["remote_tiles"] == "/dss/example/work/preprocessed/histmatch"


def test_mirrored_windows_result_paths_are_relocated_once(tmp_path):
    import h5py
    import numpy as np
    from feabas_workbench.core.cluster_remote import relocate_existing_results
    root = tmp_path / "remote"
    h5 = root / "stitch/tform/sec.h5"; h5.parent.mkdir(parents=True)
    with h5py.File(h5, "w") as f:
        f.create_dataset("imgrootdir", data=np.frombuffer(b"C:/raw tiles", dtype=np.uint8))
        f.create_dataset("preserved", data=[42])
    meta = root / "stitched_sections/mip0/sec/metadata.txt"; meta.parent.mkdir(parents=True)
    meta.write_text("{ROOT_DIR}\tC:/project/stitched_sections/mip0/sec\nimage.png\t0\t0\t32\t32\n")
    spec = root / "align/ts_spec.json"
    write_json(spec, dict(kvstore=dict(driver="file", path="C:/project/aligned_tensorstore/")))
    manifest = dict(project_id="test", path_mappings=[("C:/raw tiles", "/dss/images"), ("C:/project", "/dss/work")])
    relocate_existing_results(root, manifest)
    relocate_existing_results(root, manifest)
    with h5py.File(h5) as f:
        assert bytes(f["imgrootdir"][()]) == b"/dss/images" and f["preserved"][0] == 42
    assert "{ROOT_DIR}\t/dss/work/stitched_sections/mip0/sec" in meta.read_text()
    assert read_json(spec)["kvstore"]["path"] == "/dss/work/aligned_tensorstore/"


def test_bulk_previews_verify_all_files_before_merging_and_preserve_edits(tmp_path):
    from feabas_workbench.ui.cluster_backend import apply_preview_directory
    incoming, view = tmp_path / "incoming", tmp_path / "view"
    incoming.mkdir(); view.mkdir()
    (incoming / "a.png").write_bytes(b"new")
    (view / "a.png").write_bytes(b"manual edit")
    digest = hashlib.sha256(b"new").hexdigest()
    with pytest.raises(ValueError):
        apply_preview_directory(incoming, view, {"a.png": digest, "missing.png": digest}, {})
    assert (view / "a.png").read_bytes() == b"manual edit"
    assert apply_preview_directory(incoming, view, {"a.png": digest}, {}) == ["a.png"]
    previous = {"a.png": hashlib.sha256(b"manual edit").hexdigest()}
    assert apply_preview_directory(incoming, view, {"a.png": digest}, previous) == []
    assert (view / "a.png").read_bytes() == b"new"


def test_jobs_wait_for_all_large_inputs_and_never_submit_twice(window, project, profile, monkeypatch):
    window.open_project(project.root); window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    entry = dict(run_id="run1", state="STAGING", job_id="", remote_bundle="/dss/run1", cluster="serial",
                 input_plan={"model.pt": dict(source="ignored")})
    backend.journal = [entry]
    backend.transfers = [dict(job_input_key="run1/model.pt", state="ACTIVE")]
    calls = []
    from feabas_workbench.core.cluster_transport import JobRecord
    backend.client = SimpleNamespace(submit=lambda *a: calls.append(a) or JobRecord("1", "serial", "/dss/run1"), close=lambda: None)
    backend._stage_or_submit(entry)
    assert calls == []
    backend.transfers[0]["state"] = "SUCCEEDED"
    backend._stage_or_submit(entry)
    backend._stage_or_submit(entry)
    assert len(calls) == 1 and entry["state"] == "PENDING"


def test_partial_initial_sync_does_not_unlock_jobs(window, project, profile, monkeypatch):
    window.open_project(project.root); window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    backend.transfers = [dict(purpose="raw images", state="ACTIVE", task_id="1", batch="b", batch_size=2,
                             fingerprint="source", scope=sync_scope(profile))]
    monkeypatch.setattr(backend, "globus", lambda: SimpleNamespace(task=lambda _: dict(status="SUCCEEDED")))
    monkeypatch.setattr(backend, "_work", lambda text, fn, done: done(fn()))
    backend.refresh()
    assert not backend.synced


def test_globus_consent_scopes_are_remembered_and_granted_in_browser(monkeypatch):
    from feabas_workbench.core import globus_transfer
    scope = "https://auth.globus.org/scopes/11111111-1111-1111-1111-111111111111/data_access"
    class Process:
        returncode = 4
        def communicate(self, **kwargs):
            return "", f"Please run:\n  globus session consent '{scope}'\nto login with required scopes."
    cli = GlobusCLI()
    monkeypatch.setattr(cli, "command", lambda: ["globus"])
    monkeypatch.setattr(globus_transfer.subprocess, "Popen", lambda *a, **k: Process())
    with pytest.raises(RuntimeError, match="grant access"):
        cli.run(["stat", "example"])
    calls = []
    monkeypatch.setattr(cli, "run", lambda args, **kwargs: calls.append(args) or "Granted")
    assert cli.login() == "Granted"
    assert calls == [["session", "consent", scope]]
    assert not cli.required_scopes
