"""Use the official Globus CLI's browser login and credential store.

No tokens/passwords are read into Workbench or saved in project settings. Transfers
are asynchronous and their IDs are persisted before the GUI begins monitoring.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


class GlobusAuthorizationRequired(RuntimeError):
    """The transfer API rejected the request before creating a task."""


def collection_id(value):
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError) as e:
        raise ValueError("Choose a Globus collection and copy its Collection UUID from the File Manager.") from e


def collection_path(value):
    if not value.startswith("/") or ".." in value.split("/") or any(ord(c) < 32 for c in value):
        raise ValueError("A Globus folder must be an absolute collection path without '..' or control characters.")
    return value.rstrip("/") or "/"


class GlobusCLI:
    def __init__(self, executable=""):
        self.executable = executable
        self.process = None
        self.required_scopes = []

    def command(self):
        if self.executable:
            p = Path(self.executable)
            if not p.is_file() or p.suffix.lower() in {".bat", ".cmd", ".ps1"}:
                raise ValueError("Choose the official globus executable, not a shell script.")
            return [str(p)]
        if getattr(sys, "frozen", False):
            return [sys.executable, "--globus-helper"]
        found = shutil.which("globus")
        if found and Path(found).suffix.lower() not in {".bat", ".cmd"}:
            return [found]
        if not getattr(sys, "frozen", False) and importlib.util.find_spec("globus_cli"):
            return [sys.executable, "-m", "globus_cli"]
        raise RuntimeError("Globus transfer support needs the official Globus CLI. Install globus-cli in the Workbench "
                           "Python environment, or choose globus.exe under Advanced. Globus Connect Personal is also "
                           "needed for transfers to/from this PC. Cluster SSH and local work remain available.")

    def run(self, args, *, json_output=True, timeout=120):
        argv = self.command() + list(args)
        if json_output:
            argv += ["--format", "json"]
        env = dict(os.environ, GLOBUS_PROFILE="feabas-workbench", PYTHONUTF8="1")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        folder = None
        if "--globus-helper" in argv:
            folder = Path(tempfile.gettempdir()) / ("feabas-globus-" + uuid.uuid4().hex)
            folder.mkdir()
            argv[2:2] = [str(folder / "stdout.txt"), str(folder / "stderr.txt")]
        try:
            self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, encoding="utf-8", errors="replace", env=env, creationflags=flags)
            try:
                stdout, stderr = self.process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate()
                raise RuntimeError("Globus timed out. Check Activity before retrying a transfer.") from None
            code = self.process.returncode
            if folder:
                stdout = (folder / "stdout.txt").read_text(encoding="utf-8") if (folder / "stdout.txt").exists() else stdout
                stderr = (folder / "stderr.txt").read_text(encoding="utf-8") if (folder / "stderr.txt").exists() else stderr
        finally:
            self.process = None
            if folder:
                for name in ("stdout.txt", "stderr.txt"):
                    (folder / name).unlink(missing_ok=True)
                folder.rmdir()
        if code:
            message = (stderr or stdout)[-3500:]
            match = re.search(r"globus session consent ([^\r\n]+)", message)
            if match:
                scopes = shlex.split(match.group(1))
                if scopes and all(s.startswith(("urn:globus:", "https://auth.globus.org/scopes/")) for s in scopes):
                    self.required_scopes = scopes
                    raise GlobusAuthorizationRequired("Globus needs collection access. Use Sign in / grant access in Cluster setup.\n" + message)
            raise RuntimeError("Globus needs attention. Use Sign in / grant access in Cluster setup.\n" + message)
        return json.loads(stdout) if json_output else stdout.strip()

    def login(self):
        # Default CLI login uses a loopback callback and opens the system browser.
        result = self.run(["session", "consent", *self.required_scopes] if self.required_scopes else ["login"],
                          json_output=False, timeout=600)
        self.required_scopes = []
        return result

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def local_collection(self):
        return collection_id(self.run(["endpoint", "local-id"], json_output=False).strip())

    def transfer(self, source, source_path, target, target_path, label, *, recursive=True, exclude=()):
        src, dst = collection_id(source), collection_id(target)
        sp, dp = collection_path(source_path), collection_path(target_path)
        args = ["transfer", f"{src}:{sp}", f"{dst}:{dp}", "--sync-level", "checksum",
                "--verify-checksum", "--encrypt", "--preserve-mtime", "--fail-on-quota-errors",
                "--notify", "off", "--label", label[:128], "--recursive" if recursive else "--no-recursive"]
        for pattern in exclude:
            args += ["--exclude", pattern]
        result = self.run(args)
        if not result.get("task_id"):
            raise RuntimeError("Globus did not return a transfer ID. Check Activity before retrying.")
        return result["task_id"]

    def task(self, task_id):
        return self.run(["task", "show", collection_id(task_id)])

    def stat(self, collection, path):
        return self.run(["stat", f"{collection_id(collection)}:{collection_path(path)}"])

    def cancel(self, task_id):
        return self.run(["task", "cancel", collection_id(task_id)])
