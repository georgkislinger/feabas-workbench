"""Offline contract tests for portable bundles, MFA, failures and queue behaviour."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time

import paramiko
import pytest
import yaml

from feabas_workbench.core.cluster_bundle import ClusterResources, export_bundle, remote_path
from feabas_workbench.core.cluster_runner import install_inputs
from feabas_workbench.core.cluster_transport import ClusterClient, ClusterError, ConnectionSettings, SubmissionUncertain
from feabas_workbench.core.jobs import JobQueue, JobSpec
from feabas_workbench.core.synthetic import make_demo_project


@pytest.fixture
def project(tmp_path):
    return make_demo_project(tmp_path / "local", n_sections=2, tile=128)


def bundle(project, tmp_path, **kw):
    return export_bundle(project, tmp_path / "bundles", kw.pop("remote_project", "/dss/work folder"),
                         "/dss/raw data", "/dss/env/bin/python", kw.pop("resources", ClusterResources()),
                         kw.pop("steps", ["stitch.matching", "stitch.optimization"]), **kw)


def test_bundle_maps_only_inputs_and_preserves_source(project, tmp_path):
    original = {p: p.read_bytes() for p in project.root.rglob("*") if p.is_file()}
    out = bundle(project, tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["sections"]) == 2
    coords = list((out / "inputs/stitch/stitch_coord").glob("*.txt"))
    assert len(coords) == 2
    assert all(p.read_text().startswith("{ROOT_DIR}\t/dss/raw data\n") for p in coords)
    assert not list(out.rglob("*.tif"))
    assert all(p.read_bytes() == data for p, data in original.items())
    config = yaml.safe_load((out / "inputs/configs/stitching_configs.yaml").read_text())
    assert config["matching"]["num_workers"] == 16
    assert yaml.safe_load((out / "inputs/configs/general_configs.yaml").read_text())["cpu_budget"] == 64
    script = (out / "job.sh").read_text()
    assert "#SBATCH --ntasks=1" in script and "#SBATCH --cpus-per-task=64" in script
    assert "#SBATCH --qos=cm4_tiny" in script and "#SBATCH --export=NONE" in script
    assert script.index("#SBATCH --output") < script.index("set -euo")
    assert b"\r" not in (out / "job.sh").read_bytes()
    assert (out / "runtime/feabas_workbench/vendor/winfix/sitecustomize.py").exists()


@pytest.mark.parametrize("bad", ["relative/path", "C:/foo", "/a/../b", "/", "/a\nx", "/a\\b", "/a/%j", "//server/path"])
def test_invalid_remote_paths(bad):
    with pytest.raises(ValueError): remote_path(bad)


@pytest.mark.parametrize("resources", [ClusterResources(cpus=16), ClusterResources(cpus=113),
    ClusterResources(memory_gib=245), ClusterResources(hours=25), ClusterResources(workers=65),
    ClusterResources("serial_std", 17, 50, 1, 1), ClusterResources("serial_long", 16, 101, 168, 8)])
def test_resource_limits(resources):
    with pytest.raises(ValueError): resources.validate()


def test_serial_script(project, tmp_path):
    out = bundle(project, tmp_path, resources=ClusterResources("serial_long", 8, 64, 168, 4))
    assert "#SBATCH --clusters=serial" in (out / "job.sh").read_text()
    assert "#SBATCH --qos=cm4_serial_long" in (out / "job.sh").read_text()


def test_external_coordinate_is_rejected(project, tmp_path):
    p = next(project.stitch_coord_dir.glob("*.txt"))
    p.write_text(p.read_text() + "../../outside.tif\t0\t0\n")
    with pytest.raises(ValueError, match="outside"): bundle(project, tmp_path)


def test_absolute_coordinates_remap(project, tmp_path):
    from feabas_workbench.core.cluster_bundle import _rewrite_coordinates
    tile = next(project.active_tile_root().rglob("*.tif"))
    text = f"{{RESOLUTION}}\t4\n{{TILE_SIZE}}\t128\t128\n{tile.as_posix()}\t0\t0\n"
    rewritten = _rewrite_coordinates(text, project.active_tile_root(), "/dss/tiles")
    assert str(tile.parent) not in rewritten
    assert tile.name in rewritten


def test_external_config_rejected(project, tmp_path):
    from feabas_workbench.core.configs import ConfigStore
    cs = ConfigStore(project.configs_dir); cs.set("stitching", "rendering.out_dir", "C:/outputs"); cs.save()
    with pytest.raises(ValueError, match="Custom out_dir"): bundle(project, tmp_path)


def test_modules_not_shell(project, tmp_path):
    with pytest.raises(ValueError, match="Modules"): bundle(project, tmp_path, modules="python; rm x")


def test_install_refuses_changed_inputs_and_retains_outputs(project, tmp_path):
    out = bundle(project, tmp_path)
    root = tmp_path / "remote"; (root / ".workbench-cluster").mkdir(parents=True)
    manifest = json.loads((out / "manifest.json").read_text())
    install_inputs(out, root, manifest)
    (root / "expensive-output.h5").write_bytes(b"checkpoint")
    install_inputs(out, root, manifest)
    assert (root / "expensive-output.h5").read_bytes() == b"checkpoint"
    next((root / "stitch/stitch_coord").glob("*.txt")).write_text("changed")
    with pytest.raises(RuntimeError, match="Remote input changed"): install_inputs(out, root, manifest)


def test_install_refuses_foreign_project(project, tmp_path):
    out = bundle(project, tmp_path)
    root = tmp_path / "remote"; (root / ".workbench-cluster").mkdir(parents=True)
    (root / "important.txt").write_text("keep")
    with pytest.raises(RuntimeError, match="empty, dedicated"):
        install_inputs(out, root, json.loads((out / "manifest.json").read_text()))
    assert (root / "important.txt").read_text() == "keep"


def client(tmp_path):
    return ClusterClient(ConnectionSettings(username="test"), lambda *_: "", lambda *_: False, tmp_path / "known_hosts")


def test_finished_job_falls_back_to_accounting(tmp_path, monkeypatch):
    c = client(tmp_path); commands = []
    def execute(cmd):
        commands.append(cmd)
        if "squeue" in cmd: raise ClusterError("slurm_load_jobs error: Invalid job id specified")
        return "123|COMPLETED|0:0|00:20:00|"
    monkeypatch.setattr(c, "execute", execute)
    assert "COMPLETED" in c.status("123", "cm4")
    assert len(commands) == 2 and all("-M cm4" in cmd for cmd in commands)
    monkeypatch.setattr(c, "execute", lambda _: "")
    assert c.status("123", "serial").startswith("UNKNOWN")


def test_submit_is_not_retried_after_disconnect(tmp_path, monkeypatch):
    c = client(tmp_path); commands = []
    def broken(cmd):
        commands.append(cmd); raise OSError("disconnected")
    monkeypatch.setattr(c, "execute", broken)
    with pytest.raises(SubmissionUncertain): c.submit("/dss/run", "cm4")
    assert len(commands) == 1 and "mkdir .submission-lock" in commands[0]
    monkeypatch.setattr(c, "execute", lambda _: "765;cm4")
    record = c.recover("/dss/run", "cm4")
    assert record.job_id == "765" and record.log_path.endswith("slurm-765.out")


@pytest.mark.parametrize("job,cluster", [("1; rm", "cm4"), ("", "cm4"), ("12", "bad")])
def test_invalid_jobs_never_reach_shell(tmp_path, job, cluster):
    with pytest.raises(ValueError): client(tmp_path).cancel(job, cluster)


def test_download_never_overwrites_and_removes_partial_file(tmp_path, monkeypatch):
    import io
    import stat
    from types import SimpleNamespace
    class SFTP:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def lstat(self, path): return SimpleNamespace(st_mode=stat.S_IFREG, st_size=4)
        def open(self, path, mode): return io.BytesIO(b"data")
    c = client(tmp_path); fake = SFTP(); monkeypatch.setattr(c, "_sftp", lambda: fake)
    destination = tmp_path / "result.h5"
    assert c.download_file("/dss/result.h5", destination) == 4
    with pytest.raises(FileExistsError): c.download_file("/dss/result.h5", destination)
    assert destination.read_bytes() == b"data"
    def broken(*args): raise OSError("disconnected")
    monkeypatch.setattr(fake, "open", broken)
    partial = tmp_path / "partial.h5"
    with pytest.raises(OSError): c.download_file("/dss/result.h5", partial)
    assert not partial.exists()


def test_changed_host_key_refused_before_password(tmp_path, monkeypatch):
    c = client(tmp_path)
    known = paramiko.HostKeys(); known.add("[cool.hpc.lrz.de]:22", "ssh-rsa", paramiko.RSAKey.generate(2048))
    known.save(str(c.known_hosts))
    other = paramiko.RSAKey.generate(2048)
    class Transport:
        def __init__(self, sock): pass
        def start_client(self, timeout): pass
        def get_remote_server_key(self): return other
        def close(self): pass
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: object())
    monkeypatch.setattr(paramiko, "Transport", Transport)
    c.prompt = lambda *_: pytest.fail("Password requested before verifying host")
    c.trust = lambda *_: pytest.fail("Changed host key was offered automatic acceptance")
    with pytest.raises(ClusterError, match="host key changed"): c.connect()


def test_runner_rejects_error_markers_despite_success_exit(project, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from feabas_workbench.core import cluster_runner as runner
    out = bundle(project, tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["remote_project"] = str(tmp_path / "run")
    manifest["remote_tiles"] = str(project.active_tile_root())
    (out / "manifest.json").write_text(json.dumps(manifest))
    # Linux's actual fcntl is unavailable on Windows; only its lock is mocked.
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *_: None))
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "3.0.5")
    calls = []
    def process(argv, cwd, check):
        calls.append(argv)
        outputs = cwd / "stitch/match_h5"; outputs.mkdir(parents=True)
        for name in manifest["sections"]: (outputs / f"{name}.h5").write_bytes(b"output")
        (outputs / "s0001.h5_err").write_text("error")
    monkeypatch.setattr(runner.subprocess, "run", process)
    with pytest.raises(RuntimeError, match="error markers"): runner.run(out)
    assert len(calls) == 1  # failed matching must not start optimization


@pytest.mark.parametrize("bad_log", [False, True])
def test_failed_local_start_does_not_deadlock(tmp_path, bad_log):
    finished = threading.Event(); results = []
    q = JobQueue(on_queue_finished=lambda rs: (results.extend(rs), finished.set()))
    q.stop_on_error = False
    blocker = tmp_path / "not-directory"; blocker.write_text("file")
    first = JobSpec("cannot start", [str(tmp_path / "missing.exe")], tmp_path,
                    log_file=blocker / "log" if bad_log else None)
    second = JobSpec("following job", [sys.executable, "-c", "print('ok')"], tmp_path)
    submitter = threading.Thread(target=q.submit, args=([first, second],), daemon=True)
    submitter.start(); submitter.join(2)
    assert not submitter.is_alive(), "submit deadlocked on synchronous start failure"
    assert finished.wait(10)
    assert [r.exit_code for r in results] == [127, 0]
    assert not q.running


@pytest.mark.parametrize("close_streams", [False, True])
def test_silent_pipeline_process_obeys_timeout(tmp_path, monkeypatch, close_streams):
    from feabas_workbench.core import pipeline
    code = "import os,time; " + ("os.close(1); os.close(2); " if close_streams else "") + "time.sleep(60)"
    monkeypatch.setattr(pipeline, "step_argv", lambda *_: [sys.executable, "-c", code])
    t0 = time.monotonic()
    result = pipeline.run_step(tmp_path, pipeline.STEPS_BY_KEY["stitch.matching"], log=lambda _: None, timeout=.3)
    assert time.monotonic() - t0 < 8
    assert result.exit_code != 0


@pytest.mark.parametrize("key_auth", [False, True])
def test_real_ssh_password_or_encrypted_key_then_mfa(tmp_path, key_auth):
    """Local SSH server verifies partial auth, MFA callbacks and channel output."""
    host_key = paramiko.RSAKey.generate(2048)
    user_key = paramiko.RSAKey.generate(2048)
    private = tmp_path / "private"; user_key.write_private_key_file(str(private), password="test-passphrase")
    listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(10)
    port = listener.getsockname()[1]
    stopped = threading.Event(); failures = []
    class Server(paramiko.ServerInterface):
        partial = False
        def get_allowed_auths(self, username):
            return "keyboard-interactive" if self.partial else ("publickey" if key_auth else "password")
        def check_auth_password(self, username, password):
            assert username == "tester" and password == "test-password"
            self.partial = True; return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        def check_auth_publickey(self, username, key):
            assert username == "tester" and key == user_key
            self.partial = True; return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        def check_auth_interactive(self, username, submethods):
            return paramiko.InteractiveQuery("MFA", "Enter token", ("Code:", False))
        def check_auth_interactive_response(self, responses):
            assert responses == ["123456"]
            return paramiko.AUTH_SUCCESSFUL
        def check_channel_request(self, kind, chanid): return paramiko.OPEN_SUCCEEDED
        def check_channel_exec_request(self, channel, command):
            def send():
                channel.send(b"connected\n"); channel.send_exit_status(0); channel.shutdown_write()
            threading.Thread(target=send, daemon=True).start()
            return True
    def serve():
        try:
            conn, _ = listener.accept()
            with paramiko.Transport(conn) as t:
                t.add_server_key(host_key); t.start_server(server=Server())
                channel = t.accept(10)
                stopped.wait(10)
                if channel: channel.close()
        except Exception as e: failures.append(e)
    server = threading.Thread(target=serve, daemon=True); server.start()
    prompts = []
    def prompt(text, secret):
        prompts.append(text); assert secret
        return "test-passphrase" if "passphrase" in text else "123456" if "Code:" in text else "test-password"
    c = ClusterClient(ConnectionSettings("127.0.0.1", "tester", port, str(private) if key_auth else ""),
                      prompt, lambda host, fp: fp.startswith("ssh-rsa SHA256:"), tmp_path / "hosts")
    try:
        c.connect(); assert c.connected
        assert c.execute("printf connected") == "connected"
        assert len(prompts) == 2
        assert not any(secret in (tmp_path / "hosts").read_text() for secret in ("test-password", "123456", "test-passphrase"))
    finally:
        c.close(); stopped.set(); listener.close(); server.join(3)
    assert not failures


def test_gui_cluster_keeps_seven_workstation_pages(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from feabas_workbench.core import envs
    from feabas_workbench.ui.main_window import MainWindow
    from feabas_workbench.ui.cluster_dialog import ClusterDialog
    monkeypatch.setattr(envs, "SETTINGS_FILE", tmp_path / "settings.json")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(envs.Settings())
    project = make_demo_project(tmp_path / "ui", n_sections=2, tile=128)
    window.open_project(project.root)
    dialog = ClusterDialog(window.ctx, window)
    try:
        assert window.page_count() == 7 and dialog.tabs.count() == 4
        dialog.remote_project.setText("/dss/project"); dialog.remote_tiles.setText("/dss/tiles")
        dialog.remote_python.setText("/dss/env/bin/python"); dialog._prepare()
        deadline = time.monotonic() + 15
        while dialog.runner.running and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        assert dialog.bundle and (dialog.bundle / "job.sh").exists(), dialog.output.toPlainText()
        assert not dialog.client
        dialog.partition.setCurrentText("serial_std")
        assert dialog.cpus.maximum() == 16 and dialog.workers.maximum() <= 16
        dialog.reject()
        saved = json.loads(dialog.profile_file.read_text())
        assert not any("password" in key or "token" in key for key in saved)
    finally:
        dialog.shutdown(); window.close(); app.processEvents()


def test_module_entrypoint_propagates_failed_selfcheck(monkeypatch):
    import runpy
    from feabas_workbench import app
    monkeypatch.setattr(app, "main", lambda: 1)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("feabas_workbench.__main__", run_name="__main__")
    assert stopped.value.code == 1


def test_cluster_menu_explains_missing_dependency(tmp_path, monkeypatch):
    import builtins
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from feabas_workbench.core import envs
    from feabas_workbench.ui.main_window import MainWindow
    from feabas_workbench.core.project import Project
    monkeypatch.setattr(envs, "SETTINGS_FILE", tmp_path / "settings.json")
    app = QApplication.instance() or QApplication([])
    window = MainWindow(envs.Settings())
    window.open_project(Project.create(tmp_path / "project").root)
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: messages.append((title, text)))
    original_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == "cluster_setup":
            raise ModuleNotFoundError("No module named 'paramiko'", name="paramiko")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    try:
        window.run_on_cluster()
        assert len(messages) == 1
        assert sys.executable in messages[0][1]
        assert 'pip install "paramiko>=3.4,<6"' in messages[0][1] if sys.platform == "win32" else 'pip install' in messages[0][1]
        assert "restart Workbench" in messages[0][1]
        assert window.page_count() == 7 and not getattr(window, "_cluster_dialog", None)
    finally:
        window.close(); app.processEvents()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch launcher regression")
def test_batch_launcher_warns_when_selected_environment_has_no_paramiko(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class NoParamiko:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'paramiko':\n"
        "            raise ModuleNotFoundError(\"No module named 'paramiko'\", name='paramiko')\n"
        "sys.meta_path.insert(0, NoParamiko())\n", encoding="utf-8")
    env = dict(os.environ, FW_PYTHON=sys.executable, PYTHONPATH=str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "start_gui.bat", "--help"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Cluster support is unavailable" in result.stdout
    assert sys.executable in result.stdout
    assert 'pip install "paramiko>=3.4,<6"' in result.stdout
    assert "--project" in result.stdout  # the launcher still reaches the app
