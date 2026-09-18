"""SSH/MFA and small-file transfers; never runs FEABAS on a login node."""
from __future__ import annotations

import base64
import hashlib
import json
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


class ClusterError(RuntimeError):
    pass


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
    if not re.fullmatch(r"[1-9][0-9]*", job_id) or cluster not in {"cm4", "serial"}:
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
                    allowed = t.auth_password(s.username, self.prompt("LRZ cluster password", True), fallback=False)
                except paramiko.BadAuthenticationType as e:
                    allowed = e.allowed_types
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
        if cluster not in {"cm4", "serial"}:
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
        return self.execute("tail -c 65536 -- " + shlex.quote(remote_path(path)))

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
