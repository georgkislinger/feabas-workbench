"""SSH/MFA and small-file transfers; never runs FEABAS on a login node."""
from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import paramiko

from .cluster_bundle import MAX_BUNDLE_BYTES, module_names, remote_path
from .cluster_storage import parse_dss_storage

# FEABAS 3.0.5 ships a pure-Python wheel; every dependency has manylinux wheels for 3.11.
FEABAS_PACKAGES = "'feabas==3.0.5' tifffile imagecodecs psutil"
PART_SUFFIX = ".fw-part"


class ClusterError(RuntimeError):
    pass


class Interrupted(ClusterError):
    """Stopped on purpose by the user: report it as a status, not as a failure."""


class SignInCancelled(Interrupted):
    pass


class TransferStopped(Interrupted):
    pass


def _stop_if(cancelled):
    if cancelled and cancelled():
        raise TransferStopped("Transfer stopped. Files copied so far are kept; the next sync continues.")


def _excluded(name, patterns):
    return name.endswith(PART_SUFFIX) or any(fnmatch.fnmatchcase(name, p) for p in patterns)


def install_script(python_path: str, modules: str = "") -> str:
    """Shell script that creates (once) and fills the private FEABAS environment on a login node.

    LRZ's python modules stop at 3.8 and LRZ recommends Miniforge for environments, so a
    conda-forge Python 3.11 is created by path; a python3 >= 3.10 from the given modules is
    the fallback. The environment's own python is called directly, so jobs need no activation.
    """
    dest = remote_path(python_path)
    if not dest.endswith("/bin/python"):
        raise ValueError("The environment Python should end in /bin/python.")
    env, py, q = dest[:-len("/bin/python")], shlex.quote(dest), shlex.quote
    lines = ["set -e"]
    mods = module_names(modules)
    if mods:
        lines.append("module load " + " ".join(map(q, mods)))
    lines += [
        f"if ! test -x {py}; then",
        "  command -v conda >/dev/null 2>&1 || module load miniforge3 >/dev/null 2>&1 || true",
        "  if command -v conda >/dev/null 2>&1; then",
        f"    conda create -y -q -p {q(env)} --override-channels -c conda-forge python=3.11 pip",
        "  elif python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then",
        f"    python3 -m venv {q(env)}",
        "  else",
        "    echo 'No Python 3.10 or newer: the miniforge3 module is missing and python3 is too old. Enter a module"
        " that provides conda or Python 3.10+ under Advanced > Environment modules.' >&2",
        "    exit 3",
        "  fi",
        "fi",
        f"{py} -m pip install -q --progress-bar off --only-binary=:all: {FEABAS_PACKAGES}",
        # FEABAS requires opencv-python, whose cv2 needs libGL; compute nodes may not have it.
        f"{py} -m pip uninstall -q -y opencv-python",
        f"{py} -m pip install -q --progress-bar off --only-binary=:all: --no-deps --force-reinstall opencv-python-headless",
        f"{py} -c " + q("import sys, feabas, cv2, numpy, h5py, psutil, tensorstore, tifffile; "
                        "print('FEABAS environment ready: Python ' + sys.version.split()[0])"),
    ]
    return "\n".join(lines)


class SubmissionUncertain(ClusterError):
    pass


@dataclass
class ConnectionSettings:
    host: str = "cool.hpc.lrz.de"
    username: str = ""
    port: int = 22
    key_filename: str = ""

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", self.host):
            raise ValueError("Enter an SSH hostname, without a URL or spaces.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.username):
            raise ValueError("Enter your cluster username.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Invalid SSH port.")


@dataclass
class JobRecord:
    job_id: str
    cluster: str
    remote_bundle: str
    log_path: str = ""


def _job(job_id: str, cluster: str) -> None:
    if not re.fullmatch(r"[1-9][0-9]*", job_id) or cluster not in {"cm4", "serial", "inter"}:
        raise ValueError("Invalid Slurm job or cluster.")


class ClusterClient:
    def __init__(self, settings: ConnectionSettings, prompt: Callable[[str, bool], str],
                 trust: Callable[[str, str], bool], known_hosts: Path):
        self.settings, self.prompt, self.trust, self.known_hosts = settings, prompt, trust, known_hosts
        self.transport = None

    @property
    def connected(self) -> bool:
        return self.transport is not None and self.transport.is_authenticated() and self.transport.is_active()

    def connect(self) -> str:
        self.settings.validate()
        self.close()
        s = self.settings
        sock = socket.create_connection((s.host, s.port), timeout=25)
        t = self.transport = paramiko.Transport(sock)
        t.banner_timeout = t.auth_timeout = 90
        try:
            t.start_client(timeout=25)
            key = t.get_remote_server_key()
            name = f"[{s.host}]:{s.port}"
            hosts = paramiko.HostKeys()
            if self.known_hosts.exists():
                hosts.load(str(self.known_hosts))
            stored = hosts.lookup(name)
            fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
            if stored is not None and not hosts.check(name, key):
                raise ClusterError("SSH host key changed. Connection refused. Verify with LRZ support; "
                                   "a round-robin alias may select another login node. Use its verified hostname.")
            if stored is None:
                if not self.trust(name, f"{key.get_name()} {fingerprint}"):
                    raise ClusterError("Host key was not accepted.")
                hosts.add(name, key.get_name(), key)
                self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
                hosts.save(str(self.known_hosts))
            if s.key_filename:
                try:
                    private_key = paramiko.PKey.from_path(s.key_filename)
                except (paramiko.PasswordRequiredException, TypeError):
                    # Positional password works with both Paramiko 3/4 and 5,
                    # which renamed its keyword. Cryptography expects bytes.
                    private_key = paramiko.PKey.from_path(s.key_filename, self.prompt("SSH key passphrase", True).encode("utf-8"))
                allowed = t.auth_publickey(s.username, private_key)
            else:
                try:
                    allowed = t.auth_none(s.username)
                except paramiko.BadAuthenticationType as e:
                    allowed = e.allowed_types
                # LRZ can offer password + MFA entirely through interactive
                # challenges. Do not ask for an unused password before those.
                if not t.is_authenticated() and "password" in allowed:
                    allowed = t.auth_password(s.username, self.prompt("LRZ cluster password", True), fallback=False)
            if not t.is_authenticated() and "keyboard-interactive" in allowed:
                def answer(title, instructions, prompts):
                    return [self.prompt("\n".join(x for x in (title, instructions, question) if x), not echo)
                            for question, echo in prompts]
                t.auth_interactive(s.username, answer)
            if not t.is_authenticated():
                raise ClusterError("Authentication incomplete. Check cluster entitlement, password and MFA.")
            t.set_keepalive(60)
            return f"Connected to {s.host} as {s.username}. No job submitted."
        except Exception:
            self.close()
            raise

    def close(self):
        t, self.transport = self.transport, None
        if t:
            t.close()

    def execute(self, command: str, timeout: int = 90) -> str:
        if not self.connected:
            raise ClusterError("Connect to the cluster first.")
        channel = self.transport.open_session(timeout=25)
        channel.settimeout(25)
        output, errors = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        try:
            channel.exec_command("bash -lc " + shlex.quote(command))
            while True:
                if channel.recv_ready():
                    output.extend(channel.recv(65536))
                if channel.recv_stderr_ready():
                    errors.extend(channel.recv_stderr(65536))
                if len(output) + len(errors) > 2 * 1024 * 1024:
                    raise ClusterError("Remote command output exceeded 2 MiB.")
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    break
                if time.monotonic() > deadline:
                    raise ClusterError("SSH command timed out; its remote outcome may be unknown.")
                time.sleep(.03)
            code = channel.recv_exit_status()
            text = output.decode("utf-8", errors="replace").strip()
            if code:
                raise ClusterError(f"Remote command failed ({code}):\n" + errors.decode("utf-8", errors="replace") + "\n" + text)
            return text
        finally:
            channel.close()

    def check_environment(self, remote_python: str, remote_project: str, remote_tiles: str, modules: str = "") -> str:
        python, rp, rt = map(remote_path, (remote_python, remote_project, remote_tiles))
        mods = module_names(modules)
        code = ("import os,sys; from pathlib import Path; from importlib.metadata import version; "
                "import feabas, numpy, cv2, h5py, tensorstore; "
                "assert version('feabas')=='3.0.5', 'Requires feabas==3.0.5'; "
                "p=Path(sys.argv[1]); t=Path(sys.argv[2]); "
                "assert t.is_dir() and os.access(t,os.R_OK|os.X_OK), 'Tile folder missing/unreadable'; "
                "parent=next(x for x in [p,*p.parents] if x.exists()); "
                "assert parent.is_dir() and os.access(parent,os.W_OK|os.X_OK), 'Work folder not writable'; "
                "print('FEABAS 3.0.5 and dependencies available; paths accessible. Check DSS quota before submission.')")
        command = "set -e; module load slurm_setup; "
        if mods:
            command += "module load " + " ".join(map(shlex.quote, mods)) + "; "
        command += "command -v sbatch >/dev/null; " + " ".join(map(shlex.quote, [python, "-c", code, rp, rt]))
        return self.execute(command)

    def discover(self) -> dict:
        """Small read-only account/storage checks, no workload on the login node."""
        info = self.execute("python3 -c " + shlex.quote(
            "import os,json,sys,shutil; print(json.dumps(dict(home=os.path.expanduser('~'), "
            "python=sys.executable, version=list(sys.version_info[:3]), user=os.environ.get('USER',''))))"))
        result = json.loads(info.splitlines()[-1])
        try:
            result["storage"] = self.execute("dssusrinfo all")
        except ClusterError as e:
            result["storage"] = str(e)
            result.update(directories=[], storage_state="error", home_used=None, home_limit=None)
        else:
            parsed = parse_dss_storage(result["storage"])
            parsed["home"] = parsed["home"] or result.get("home", "")
            result.update(parsed)
        return result

    def prepare_workspace(self, profile: dict) -> str:
        rp, rt = map(remote_path, (profile["remote_project"], profile["remote_tiles"]))
        if rp == rt or rp.startswith(rt + "/") or rt.startswith(rp + "/"):
            raise ValueError("Keep raw images and the remote work folder separate.")
        identifier = profile["project_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid project identity.")
        with self._sftp() as sftp:
            self._mkdirs(sftp, rp)
            control = rp + "/.workbench-cluster"
            try:
                with sftp.open(control + "/owner.json") as f:
                    existing = json.loads(f.read())
                if existing.get("project_id") != identifier:
                    raise ClusterError("This remote directory belongs to another project. Choose a new folder.")
            except FileNotFoundError:
                if sftp.listdir(rp):
                    raise ClusterError("Choose an empty remote work folder for the first setup.")
                self._mkdirs(sftp, control)
                with sftp.open(control + "/owner.json", "w") as f:
                    f.write(json.dumps({"project_id": identifier}))
            self._mkdirs(sftp, rt)
            with sftp.open(control + "/connection-" + identifier + ".txt", "w") as f:
                f.write(identifier)
            with sftp.open(rt + "/.workbench-connection-" + identifier + ".txt", "w") as f:
                f.write(identifier)
        return "Remote workspace is writable and belongs to this project."

    def install_environment(self, python_path: str, modules: str = "") -> str:
        return self.execute(install_script(python_path, modules), timeout=1800)

    def read_remote_json(self, path: str, missing=None):
        try:
            with self._sftp() as sftp, sftp.open(remote_path(path), "rb") as f:
                data = f.read(2 * 1024 * 1024 + 1)
                if len(data) > 2 * 1024 * 1024:
                    raise ClusterError("Remote status document exceeds 2 MiB.")
                return json.loads(data)
        except FileNotFoundError:
            return missing

    def _sftp(self):
        if not self.connected:
            raise ClusterError("Connect first.")
        sftp = paramiko.SFTPClient.from_transport(self.transport)
        sftp.get_channel().settimeout(30)
        return sftp

    @staticmethod
    def _mkdirs(sftp, path: str):
        cur = ""
        for part in PurePosixPath(remote_path(path)).parts[1:]:
            cur += "/" + part
            try:
                info = sftp.lstat(cur)
            except FileNotFoundError:
                sftp.mkdir(cur)
                info = sftp.lstat(cur)
            if not stat.S_ISDIR(info.st_mode):
                raise ClusterError(f"Upload path is not a real directory (links are refused): {cur}")

    def upload_bundle(self, bundle: Path) -> str:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if not re.fullmatch(r"[a-f0-9]{32}", manifest["run_id"]):
            raise ValueError("Invalid bundle ID.")
        rp = remote_path(manifest["remote_project"])
        remote = rp + "/.workbench-cluster/" + manifest["run_id"]
        files = sorted(p for p in bundle.rglob("*") if p.is_file())
        if any(p.is_symlink() or not p.resolve().is_relative_to(bundle.resolve()) for p in files):
            raise ValueError("Bundle must not contain links.")
        if sum(p.stat().st_size for p in files) > MAX_BUNDLE_BYTES:
            raise ValueError("Bundle exceeds 64 MiB.")
        with self._sftp() as sftp:
            self._mkdirs(sftp, rp + "/.workbench-cluster")
            sftp.mkdir(remote)  # Exclusive: a previous upload is never overwritten.
            for f in files:
                target = remote + "/" + f.relative_to(bundle).as_posix()
                self._mkdirs(sftp, str(PurePosixPath(target).parent))
                sftp.put(str(f), target, confirm=True)
            with sftp.open(remote + "/READY", "w") as f:
                f.write("complete\n")
        return remote

    def recover(self, remote_bundle: str, cluster: str) -> JobRecord:
        rb = remote_path(remote_bundle)
        receipt = self.execute("cat -- " + shlex.quote(rb + "/job-id"))
        return self._parse_receipt(receipt, rb, cluster)

    @staticmethod
    def _parse_receipt(receipt, rb, cluster):
        if not re.fullmatch(r"[1-9][0-9]*(;[a-zA-Z0-9_-]+)?", receipt):
            raise SubmissionUncertain("No valid submission receipt. Inspect the remote job queue before any new submission.")
        job_id, _, actual = receipt.partition(";")
        if actual and actual != cluster:
            raise SubmissionUncertain("Submission returned an unexpected cluster. Inspect the remote queue.")
        _job(job_id, cluster)
        return JobRecord(job_id, cluster, rb, rb + f"/slurm-{job_id}.out")

    def submit(self, remote_bundle: str, cluster: str) -> JobRecord:
        rb = remote_path(remote_bundle)
        if cluster not in {"cm4", "serial", "inter"}:
            raise ValueError("Invalid cluster.")
        # Lock remains after ambiguous failures; repeating submission cannot create a duplicate.
        cmd = ("set -e; module load slurm_setup; cd -- " + shlex.quote(rb) + "; test -f READY; "
               "if test -s job-id; then cat job-id; exit 0; fi; "
               "mkdir .submission-lock || { echo 'Submission already attempted; inspect queue and receipt.' >&2; exit 75; }; "
               "sbatch --parsable job.sh > job-id.tmp; mv job-id.tmp job-id; cat job-id")
        try:
            return self._parse_receipt(self.execute(cmd), rb, cluster)
        except Exception as e:
            raise SubmissionUncertain("Submission outcome not confirmed. Use 'Recover job ID'; do not submit again "
                                      "until the LRZ queue has been checked.\n" + str(e)) from e

    def status(self, job_id: str, cluster: str) -> str:
        _job(job_id, cluster)
        prefix = "module load slurm_setup && "
        try:
            queued = self.execute(prefix + f"squeue -M {cluster} -j {job_id} --noheader --format='%T|%M|%R'")
        except ClusterError as e:
            if "invalid job id" not in str(e).lower():
                raise
            queued = ""
        # Federation/cluster headings are sometimes printed despite --noheader.
        rows = [r for r in queued.splitlines() if "|" in r]
        if rows:
            return "\n".join(rows)
        history = self.execute(prefix + f"sacct -M {cluster} -j {job_id} -X -n -P --format=JobIDRaw,State,ExitCode,Elapsed")
        rows = [r for r in history.splitlines() if r.split("|")[0].strip() == job_id]
        return "\n".join(rows) if rows else "UNKNOWN — not in queue; accounting may be delayed. Refresh later."

    def cancel(self, job_id: str, cluster: str) -> str:
        _job(job_id, cluster)
        self.execute(f"module load slurm_setup && scancel -M {cluster} {job_id}")
        return "Cancellation requested. Refresh later to confirm."

    def tail_log(self, path: str) -> str:
        p = shlex.quote(remote_path(path))
        return self.execute("if test -f " + p + "; then tail -c 65536 -- " + p + "; fi")

    def download_file(self, remote: str, destination: Path) -> int:
        """Explicit single regular file, maximum 64 MiB. Bulk results use Globus."""
        remote = remote_path(remote)
        with self._sftp() as sftp:
            info = sftp.lstat(remote)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BUNDLE_BYTES:
                raise ClusterError("Choose a regular file up to 64 MiB. Use Globus for large results.")
            created = False
            try:
                with destination.open("xb") as out:
                    created = True
                    with sftp.open(remote, "rb") as src:
                        total = 0
                        while chunk := src.read(65536):
                            total += len(chunk)
                            if total > MAX_BUNDLE_BYTES:
                                raise ClusterError("Download exceeded 64 MiB.")
                            out.write(chunk)
            except Exception:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        return total

    # -- bulk transfers over SSH: the Globus-free path for projects that fit a home folder --
    @staticmethod
    def _replace(sftp, part: str, target: str):
        try:
            sftp.posix_rename(part, target)
        except OSError:
            try:
                sftp.remove(target)
            except FileNotFoundError:
                pass
            sftp.rename(part, target)

    def upload_tree(self, local: Path, remote: str, exclude=(), progress=None, cancelled=None) -> dict:
        """Copy new or changed files (size or modification time) into a remote folder.

        Remote files are never deleted. Each file is written under a temporary name and
        renamed once its size is confirmed, so an interrupted copy never looks complete;
        the next call skips everything that already arrived.
        """
        local, root = Path(local), remote_path(remote)
        if not local.is_dir():
            raise ClusterError(f"Folder not found on this computer: {local}")
        files = []
        for directory, dirs, names in os.walk(local):
            here = Path(directory)
            dirs[:] = sorted(d for d in dirs if not _excluded(d, exclude))
            for name in sorted(names):
                p = here / name
                if _excluded(name, exclude):
                    continue
                if p.is_symlink():
                    raise ClusterError(f"Resolve links before uploading: {p}")
                st = p.stat()
                files.append((p, p.relative_to(local).as_posix(), st.st_size, int(st.st_mtime)))
        total = sum(f[2] for f in files)
        done = copied = 0
        listings = {}
        with self._sftp() as sftp:
            def remote_files(folder):
                if folder not in listings:
                    try:
                        listings[folder] = {a.filename: a for a in sftp.listdir_attr(folder)}
                    except FileNotFoundError:
                        self._mkdirs(sftp, folder)
                        listings[folder] = {}
                return listings[folder]
            remote_files(root)
            for path, rel, size, mtime in files:
                _stop_if(cancelled)
                target = root + "/" + rel
                folder, name = target.rsplit("/", 1)
                have = remote_files(folder).get(name)
                if not (have and stat.S_ISREG(have.st_mode or 0) and have.st_size == size and have.st_mtime == mtime):
                    def moved(n, _size, base=done):
                        _stop_if(cancelled)
                        if progress:
                            progress(base + n, total)
                    sftp.put(str(path), target + PART_SUFFIX, callback=moved, confirm=True)
                    sftp.utime(target + PART_SUFFIX, (mtime, mtime))
                    self._replace(sftp, target + PART_SUFFIX, target)
                    copied += 1
                done += size
                if progress:
                    progress(done, total)
        return dict(files=len(files), copied=copied, bytes=total)

    def download_tree(self, remote: str, local: Path, progress=None, cancelled=None) -> dict:
        """Copy new or changed remote files into a local folder; local files are never deleted."""
        root, local = remote_path(remote), Path(local)
        files = []
        with self._sftp() as sftp:
            def walk(folder, rel):
                for a in sorted(sftp.listdir_attr(folder), key=lambda a: a.filename):
                    _stop_if(cancelled)
                    name = a.filename
                    if name in (".", "..") or "\\" in name or ":" in name or name.endswith(PART_SUFFIX):
                        continue
                    r = rel + "/" + name if rel else name
                    if stat.S_ISDIR(a.st_mode or 0):
                        walk(folder + "/" + name, r)
                    elif stat.S_ISREG(a.st_mode or 0):   # links are not followed
                        files.append((folder + "/" + name, r, a.st_size, a.st_mtime))
            walk(root, "")
            total = sum(f[2] for f in files)
            done = copied = 0
            for path, rel, size, mtime in files:
                _stop_if(cancelled)
                dest = local.joinpath(*rel.split("/"))
                if not dest.resolve().is_relative_to(local.resolve()):
                    raise ClusterError(f"Unsafe remote file name: {rel}")
                if not (dest.is_file() and dest.stat().st_size == size and int(dest.stat().st_mtime) == mtime):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    part = dest.with_name(dest.name + PART_SUFFIX)
                    def moved(n, _size, base=done):
                        _stop_if(cancelled)
                        if progress:
                            progress(base + n, total)
                    sftp.get(path, str(part), callback=moved)
                    if part.stat().st_size != size:
                        part.unlink(missing_ok=True)
                        raise ClusterError(f"Incomplete download: {rel}")
                    os.utime(part, (mtime, mtime))
                    os.replace(part, dest)
                    copied += 1
                done += size
                if progress:
                    progress(done, total)
        return dict(files=len(files), copied=copied, bytes=total)

    def upload_file(self, local: Path, remote: str, cancelled=None) -> dict:
        """One file to an exact remote path, renamed into place after its size is confirmed."""
        target = remote_path(remote)
        with self._sftp() as sftp:
            self._mkdirs(sftp, str(PurePosixPath(target).parent))
            sftp.put(str(local), target + PART_SUFFIX, callback=lambda *_: _stop_if(cancelled), confirm=True)
            self._replace(sftp, target + PART_SUFFIX, target)
        return dict(files=1, copied=1, bytes=Path(local).stat().st_size)
