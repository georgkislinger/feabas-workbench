"""Home-folder projects: SSH transfers, the Miniforge environment, storage checks, leaving cluster mode."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import paramiko
import pytest

from feabas_workbench.core.cluster_storage import parse_dss_storage
from feabas_workbench.core.cluster_transport import ClusterClient, ClusterError, ConnectionSettings, install_script
from feabas_workbench.core.cluster_workspace import profile_for, read_json, save_profile, sync_scope, write_json
from feabas_workbench.core.synthetic import make_demo_project

FIXTURE = (Path(__file__).parent / "fixtures/dss-no-containers.txt").read_text()
HOME = "/dss/dsshome1/04/testuser"


# -- the environment script ---------------------------------------------------------------
def test_install_script_uses_miniforge_python_311_and_headless_opencv():
    script = install_script(HOME + "/feabas-workbench/feabas-env/bin/python")
    assert "module load miniforge3" in script and "python=3.11" in script
    assert f"conda create -y -q -p {HOME}/feabas-workbench/feabas-env" in script
    assert "--override-channels -c conda-forge" in script
    # headless cv2 replaces FEABAS's opencv-python after the main install
    assert script.index("uninstall -q -y opencv-python") > script.index("feabas==3.0.5")
    assert "--no-deps --force-reinstall opencv-python-headless" in script
    assert "module load python/3.10.12" in install_script("/dss/env/bin/python", "python/3.10.12")
    with pytest.raises(ValueError):
        install_script("/dss/env/python")
    with pytest.raises(ValueError):
        install_script("/dss/env/bin/python", "miniforge3; rm -rf ~")


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs a Linux bash")
def test_install_script_is_valid_bash():
    result = subprocess.run(["bash", "-n"], input=install_script("/dss/my env/bin/python", "a b"),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


# -- a real SSH server with an SFTP subsystem over a local folder ----------------------------
@pytest.fixture
def ssh_server(tmp_path):
    """Password 'pw', SFTP over a local folder, and canned replies to commands: replies maps a
    substring of the command to (exit code, output); every command is recorded."""
    root = tmp_path / "server"
    root.mkdir()
    replies, commands = {}, []
    host_key = paramiko.RSAKey.generate(2048)
    listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(10)
    stop, failures = threading.Event(), []

    def errno(e):
        return paramiko.SFTPServer.convert_errno(e.errno)

    class Handle(paramiko.SFTPHandle):
        def stat(self):
            return paramiko.SFTPAttributes.from_stat(os.fstat(self.readfile.fileno()))

        def chattr(self, attr):
            paramiko.SFTPServer.set_file_attr(self.filename, attr)
            return paramiko.SFTP_OK

    class Files(paramiko.SFTPServerInterface):
        def _real(self, path):
            return str(root) + self.canonicalize(path)

        def list_folder(self, path):
            try:
                out = []
                for name in os.listdir(self._real(path)):
                    a = paramiko.SFTPAttributes.from_stat(os.lstat(os.path.join(self._real(path), name)))
                    a.filename = name
                    out.append(a)
                return out
            except OSError as e:
                return errno(e)

        def stat(self, path):
            try:
                return paramiko.SFTPAttributes.from_stat(os.stat(self._real(path)))
            except OSError as e:
                return errno(e)

        lstat = stat

        def open(self, path, flags, attr):
            real = self._real(path)
            try:
                fd = os.open(real, flags | getattr(os, "O_BINARY", 0), 0o666)
            except OSError as e:
                return errno(e)
            mode = ("ab" if flags & os.O_APPEND else "r+b" if flags & os.O_RDWR
                    else "wb" if flags & os.O_WRONLY else "rb")
            handle = Handle(flags)
            handle.filename = real
            handle.readfile = handle.writefile = os.fdopen(fd, mode)
            return handle

        def remove(self, path):
            try:
                os.remove(self._real(path))
            except OSError as e:
                return errno(e)
            return paramiko.SFTP_OK

        def rename(self, old, new):
            if os.path.exists(self._real(new)):
                return paramiko.SFTP_FAILURE
            os.rename(self._real(old), self._real(new))
            return paramiko.SFTP_OK

        def posix_rename(self, old, new):
            os.replace(self._real(old), self._real(new))
            return paramiko.SFTP_OK

        def mkdir(self, path, attr):
            try:
                os.mkdir(self._real(path))
            except OSError as e:
                return errno(e)
            return paramiko.SFTP_OK

        def chattr(self, path, attr):
            paramiko.SFTPServer.set_file_attr(self._real(path), attr)
            return paramiko.SFTP_OK

    class Server(paramiko.ServerInterface):
        def get_allowed_auths(self, username):
            return "password"

        def check_auth_password(self, username, password):
            return paramiko.AUTH_SUCCESSFUL if password == "pw" else paramiko.AUTH_FAILED

        def check_channel_request(self, kind, chanid):
            return paramiko.OPEN_SUCCEEDED

        def check_channel_exec_request(self, channel, command):
            text = command.decode("utf-8")
            commands.append(text)
            code, out = next(((c, o) for key, (c, o) in replies.items() if key in text), (1, "unexpected command"))
            def send():
                channel.sendall(out.encode("utf-8")); channel.send_exit_status(code); channel.shutdown_write()
            threading.Thread(target=send, daemon=True).start()
            return True

    def serve():
        try:
            conn, _ = listener.accept()
            with paramiko.Transport(conn) as t:
                t.add_server_key(host_key)
                t.set_subsystem_handler("sftp", paramiko.SFTPServer, Files)
                t.start_server(server=Server())
                stop.wait(120)
        except Exception as e:  # noqa: BLE001 - reported by the fixture
            failures.append(e)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(port=listener.getsockname()[1], root=root, replies=replies, commands=commands)
    finally:
        stop.set(); listener.close(); thread.join(3)
    assert not failures


@pytest.fixture
def sftp(ssh_server, tmp_path):
    client = ClusterClient(ConnectionSettings("127.0.0.1", "tester", ssh_server.port),
                           lambda text, secret: "pw", lambda host, fp: True, tmp_path / "hosts")
    client.connect()
    try:
        yield client, ssh_server.root
    finally:
        client.close()


def test_ssh_upload_skips_unchanged_files_and_never_deletes(sftp, tmp_path):
    client, server = sftp
    src = tmp_path / "tiles"
    (src / "s1").mkdir(parents=True)
    (src / "s1" / "a.tif").write_bytes(b"a" * 70000)
    (src / "b.tif").write_bytes(b"b")
    (src / ".workbench-cluster").mkdir()
    (src / ".workbench-cluster" / "state.json").write_text("{}")
    reports = []
    skip = (".workbench-cluster",)
    result = client.upload_tree(src, HOME + "/images", exclude=skip, progress=lambda d, t: reports.append((d, t)))
    remote = server / HOME.lstrip("/") / "images"
    assert result == dict(files=2, copied=2, bytes=70001)
    assert (remote / "s1" / "a.tif").read_bytes() == b"a" * 70000
    assert int((remote / "b.tif").stat().st_mtime) == int((src / "b.tif").stat().st_mtime)
    assert not (remote / ".workbench-cluster").exists() and not list(server.rglob("*.fw-part"))
    assert reports[-1] == (70001, 70001)
    (remote / "extra.txt").write_text("keep")
    assert client.upload_tree(src, HOME + "/images", exclude=skip)["copied"] == 0
    (src / "b.tif").write_bytes(b"bb")
    assert client.upload_tree(src, HOME + "/images", exclude=skip)["copied"] == 1
    assert (remote / "b.tif").read_bytes() == b"bb" and (remote / "extra.txt").read_text() == "keep"


def test_ssh_download_copies_new_or_changed_files_only(sftp, tmp_path):
    client, server = sftp
    remote = server / "dss/work/exports/run1"
    (remote / "sub").mkdir(parents=True)
    (remote / "sub" / "x.png").write_bytes(b"x" * 100000)
    (remote / "y.vsvi").write_text("y")
    dest = tmp_path / "local"
    result = client.download_tree("/dss/work/exports/run1", dest)
    assert result["copied"] == 2 and (dest / "sub" / "x.png").read_bytes() == b"x" * 100000
    assert client.download_tree("/dss/work/exports/run1", dest)["copied"] == 0
    assert not list(dest.rglob("*.fw-part"))


def test_stopped_upload_never_leaves_a_complete_looking_file(sftp, tmp_path):
    client, server = sftp
    src = tmp_path / "tiles"
    src.mkdir()
    (src / "big.tif").write_bytes(os.urandom(300000))
    seen = []
    with pytest.raises(ClusterError, match="stopped"):
        client.upload_tree(src, "/dss/images", progress=lambda d, t: seen.append(d), cancelled=lambda: bool(seen))
    assert not (server / "dss/images/big.tif").exists()
    assert client.upload_tree(src, "/dss/images")["copied"] == 1   # the next sync completes it
    assert (server / "dss/images/big.tif").read_bytes() == (src / "big.tif").read_bytes()


# -- GUI: storage choice, SSH sync, leaving cluster mode -----------------------------------
@pytest.fixture
def project(tmp_path):
    return make_demo_project(tmp_path / "project", n_sections=3, tile=128)


@pytest.fixture
def profile(project):
    p = profile_for(project)
    p.update(username="testuser", remote_project=HOME + "/feabas-workbench/p/work", remote_tiles=HOME + "/feabas-workbench/p/images",
             remote_python=HOME + "/feabas-workbench/feabas-env/bin/python", transfer="ssh")
    return p


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


def run_inline(text, function, done=None, *, progress=False):
    result = function(lambda *a: None, lambda: False) if progress else function()
    if done:
        done(result)
    return True


def test_setup_offers_the_home_folder_and_warns_before_it_fills(window, project, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from feabas_workbench.ui.cluster_setup import ClusterSetupDialog
    window.open_project(project.root)
    dialog = ClusterSetupDialog(window.ctx, window)
    monkeypatch.setattr(dialog.backend, "connect", lambda done=None: None)
    try:
        # 30 GB of raw tiles need about 5 × 30 + 3 GB: more than an empty 100 GB home.
        dialog._discovered(dict(parse_dss_storage(FIXTURE), storage=FIXTURE, raw_bytes=30 * 10**9))
        assert [dialog.storage.itemText(i) for i in range(dialog.storage.count())] == [HOME]
        assert "Home folder · 0.0 of 100 GB used" in dialog.storage_note.text()
        assert "Warning: This project may need about 153 GB" in dialog.storage_note.text()
        asked = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or QMessageBox.No)
        dialog._choose_storage()
        assert "about 153 GB" in asked[0] and dialog.backend.profile["remote_project"] == ""
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        dialog._choose_storage()
        p = read_json(project.root / ".workbench-cluster/workspace.json")
        assert p["remote_project"].startswith(HOME + "/feabas-workbench/feabas-") and p["remote_project"].endswith("/work")
        assert p["remote_python"] == HOME + "/feabas-workbench/feabas-env/bin/python"
        assert p["transfer"] == "ssh"
        # the tutorial's 1.3 GB fit without a warning
        dialog._discovered(dict(parse_dss_storage(FIXTURE), storage=FIXTURE, raw_bytes=13 * 10**8))
        assert "Warning" not in dialog.storage_note.text()
    finally:
        dialog.shutdown(); dialog.close()


def test_ssh_sync_uploads_images_and_project_and_unlocks_runs(window, project, profile, monkeypatch):
    window.open_project(project.root)
    window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    calls = []
    def upload(src, dst, **kw):
        calls.append((Path(src), dst, kw.get("exclude", ())))
        return dict(files=3, copied=3, bytes=30)
    backend.client = SimpleNamespace(connected=True, prepare_workspace=lambda p: calls.append("prepared"),
                                     upload_tree=upload, close=lambda: None)
    monkeypatch.setattr(backend, "_work", run_inline)
    window.sync_cluster_btn.click()
    assert calls[0] == "prepared"
    assert calls[1][:2] == (Path(project.state.source.root_dir), profile["remote_tiles"])
    assert calls[2][:2] == (project.root, profile["remote_project"]) and ".workbench-cluster" in calls[2][2]
    assert backend.synced["scope"] == sync_scope(backend.profile)
    assert [t["purpose"] for t in backend.transfers] == ["raw images", "initial project"]
    assert not backend.transferring        # an SSH copy never stays open like a Globus task


def test_sign_in_and_ssh_sync_through_the_real_background_thread(window, project, profile, monkeypatch):
    """No _work stand-in: the ThreadRunner calls its function with progress=/cancelled= keywords."""
    import time
    from PySide6.QtWidgets import QApplication
    from feabas_workbench.ui import cluster_backend

    class FakeClient:
        connected = False

        def __init__(self, *args):
            pass

        def connect(self):
            FakeClient.connected = True
            return "Connected to test"

        def close(self):
            pass

        def prepare_workspace(self, p):
            return "ok"

        def upload_tree(self, src, dst, progress=None, cancelled=None, exclude=()):
            assert not cancelled()
            progress(5 * 10**9, 10**10)
            progress(10**10, 10**10)
            return dict(files=2, copied=2, bytes=10**10)

    monkeypatch.setattr(cluster_backend, "ClusterClient", FakeClient)
    window.open_project(project.root)
    window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    seen = []
    backend.changed.connect(lambda: seen.append(backend.message))
    backend.sync()
    app, deadline = QApplication.instance(), time.monotonic() + 15
    while (backend.busy or not backend.synced) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert backend.synced and backend.message == "Images synchronized over SSH · ready to run", seen
    assert "Connected to test" in seen
    assert any(s.endswith("5.00 of 10.00 GB") for s in seen)     # progress crossed the thread boundary
    assert not backend.copying and window.sync_cluster_btn.text() == "Sync project && images"


def test_ssh_export_download_reports_the_local_folder(window, project, profile, monkeypatch, tmp_path):
    profile["export_destination"] = str(tmp_path / "out")
    window.open_project(project.root)
    window.ctx.use_cluster(profile)
    backend = window.ctx.cluster
    fetched, emitted = [], []
    backend.client = SimpleNamespace(connected=True, close=lambda: None,
                                     download_tree=lambda src, dst, **kw: fetched.append((src, dst)) or dict(files=1, copied=1, bytes=5))
    monkeypatch.setattr(backend, "_work", run_inline)
    window.ctx.export_downloaded.connect(emitted.append)
    backend.download_exports(run_id="abc")
    assert fetched == [(profile["remote_project"] + "/exports/abc", tmp_path / "out" / "abc")]
    assert emitted == [str(tmp_path / "out" / "abc")]


def test_leaving_cluster_mode_keeps_remote_jobs_and_pauses_monitoring(window, project, profile, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    profile["mode"] = "cluster"
    save_profile(project, profile)
    write_json(project.root / ".workbench-cluster/workspace-jobs.json", [dict(state="PENDING", job_id="123")])
    window.open_project(project.root)
    assert window.ctx.cluster_enabled
    assert not window.local_btn.isHidden() and window.leave_cluster_action.isEnabled()
    assert window.sync_cluster_btn.text() == "Sync project && images"
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    window.leave_cluster_action.trigger()
    assert not window.ctx.cluster_enabled and window.ctx.project.root == project.root
    assert read_json(project.root / ".workbench-cluster/workspace.json")["mode"] == "local"
    assert read_json(project.root / ".workbench-cluster/workspace-jobs.json")[0]["state"] == "PENDING"
    assert window.local_btn.isHidden() and not window.leave_cluster_action.isEnabled()
    assert "LRZ jobs/transfers continue" in window.execution_label.text()
    monkeypatch.setattr(window.ctx.cluster, "_work", lambda *a, **k: pytest.fail("monitoring must pause on this PC"))
    window.ctx.cluster.refresh()
    window.ctx.use_cluster(profile)       # coming back resumes monitoring of the same job
    assert window.ctx.cluster_enabled and window.ctx.jobs.running


def test_leaving_waits_only_for_an_operation_in_progress(window, project, profile, monkeypatch):
    window.open_project(project.root)
    window.ctx.use_cluster(profile)
    monkeypatch.setattr(type(window.ctx.cluster), "busy", property(lambda self: True))
    with pytest.raises(RuntimeError, match="still busy"):
        window.ctx.use_local()
    assert window.ctx.cluster_enabled


def test_setup_walkthrough_against_a_local_ssh_server(window, project, ssh_server, monkeypatch, tmp_path):
    """The final-test clicks, end to end through the real dialog, backend, thread runner, SSH and SFTP:
    sign in, discover, home folder, prepare FEABAS, save, sync over SSH, submit one step."""
    import json
    import time
    from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
    from feabas_workbench.core.steps import STEPS_BY_KEY
    from feabas_workbench.ui import cluster_backend
    from feabas_workbench.ui.cluster_setup import ClusterSetupDialog
    home = "/dss/dsshome1/04/tester"
    ssh_server.replies.update({
        "expanduser": (0, json.dumps(dict(home=home, python="/usr/bin/python3", version=[3, 6, 15], user="tester"))),
        "dssusrinfo all": (0, FIXTURE.replace("testuser", "tester")),
        "conda create": (0, "FEABAS environment ready: Python 3.11.9"),
        "Tile folder missing": (0, "FEABAS 3.0.5 and dependencies available; paths accessible."),
        "sbatch --parsable": (0, "4242"),
    })
    monkeypatch.setattr(cluster_backend, "settings_dir", lambda: tmp_path / "settings")   # known_hosts stays here
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("pw", True))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    problems = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: problems.append(a[2]) or QMessageBox.Ok)
    app = QApplication.instance()

    def wait(condition, seconds=30):
        deadline = time.monotonic() + seconds
        while not condition() and not problems and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert condition(), problems or backend.message

    window.open_project(project.root)
    dialog = ClusterSetupDialog(window.ctx, window)
    backend = dialog.backend
    try:
        dialog.fields["host"].setText("127.0.0.1")
        dialog.port.setValue(ssh_server.port)
        dialog.fields["username"].setText("tester")
        dialog._discover()
        wait(lambda: dialog.storage.count() and not backend.busy)
        assert dialog.storage.currentText() == home and "Home folder" in dialog.storage_note.text()
        dialog._choose_storage()
        remote = backend.profile["remote_project"]
        wait(lambda: not backend.busy and backend.message.startswith("Remote workspace is writable"))
        assert remote.startswith(home + "/feabas-workbench/feabas-")
        dialog._install()
        wait(lambda: not backend.busy and "FEABAS environment ready" in backend.message)
        dialog._save()
        assert window.ctx.cluster_enabled
        window.sync_cluster_btn.click()
        wait(lambda: bool(backend.synced) and not backend.busy)
        images = ssh_server.root / backend.profile["remote_tiles"].lstrip("/")
        assert sorted(p.name for p in images.rglob("*.tif")) == sorted(
            p.name for p in Path(project.state.source.root_dir).rglob("*.tif"))
        window.ctx.jobs.submit(window.ctx.feabas_step_spec(STEPS_BY_KEY["stitch.matching"]))
        wait(lambda: backend.journal and backend.journal[-1].get("job_id") == "4242" and not backend.busy)
        run = ssh_server.root / (remote + "/.workbench-cluster/" + backend.journal[-1]["run_id"]).lstrip("/")
        assert (run / "READY").is_file() and backend.profile["remote_python"] in (run / "job.sh").read_text()
        assert backend.message.startswith("Queued at LRZ · job 4242") and not problems
    finally:
        dialog.shutdown(); dialog.close()