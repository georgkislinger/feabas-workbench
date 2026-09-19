"""Persistent cluster execution behind the normal workbench Run buttons."""
from __future__ import annotations

import hashlib
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from ..core.cluster_bundle import MAX_BUNDLE_BYTES, file_hash
from ..core.cluster_transport import ClusterClient, ConnectionSettings
from ..core.cluster_workspace import (CONTROL, TERMINAL, build_workspace_bundle, local_globus_path,
                                     read_json, resources_for, source_signature, sync_scope, write_json)
from ..core.envs import settings_dir
from ..core.globus_transfer import GlobusCLI, GlobusAuthorizationRequired
from ..core.jobs import JobResult, JobSpec
from .threads import ThreadRunner


def apply_previews(archive, view, snapshot, previous):
    """Verified bounded extraction. Preserve edits made locally while a job ran."""
    conflicts, total = [], 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            rel = info.filename
            target = (view / rel).resolve()
            if (not target.is_relative_to(view.resolve()) or "\\" in rel or ":" in rel
                    or rel not in snapshot.get("previews", {}) or info.is_dir()):
                raise ValueError("Invalid preview archive path.")
            total += info.file_size
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("Preview archive exceeds its size limit.")
            data = z.read(info)
            digest = hashlib.sha256(data).hexdigest()
            if digest != snapshot["previews"][rel]:
                raise ValueError("Preview changed during download. Refresh again.")
            if target.is_file():
                current = hashlib.sha256(target.read_bytes()).hexdigest()
                if current != digest and current != previous.get(rel):
                    conflicts.append(rel)
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".incoming")
            tmp.write_bytes(data)
            tmp.replace(target)
    # Remove only formerly cached files absent from the new snapshot and untouched
    # by the user. This keeps previews consistent after clearing remote outputs.
    for rel, digest in previous.items():
        if rel not in snapshot.get("preview_manifest", snapshot.get("previews", {})):
            p = (view / rel).resolve()
            if p.is_relative_to(view.resolve()) and p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest:
                p.unlink()
    return conflicts


def apply_preview_directory(incoming, view, manifest, previous):
    conflicts = []
    # Validate the complete transfer before modifying the visible cache.
    for rel, digest in manifest.items():
        src, dest = (incoming / rel).resolve(), (view / rel).resolve()
        if (not src.is_relative_to(incoming.resolve()) or not dest.is_relative_to(view.resolve())
                or ":" in rel or "\\" in rel or not src.is_file() or file_hash(src) != digest):
            raise ValueError("Downloaded preview is missing, changed, or has an unsafe path: " + rel)
    for rel, digest in manifest.items():
        src, dest = incoming / rel, view / rel
        if dest.is_file() and file_hash(dest) not in {digest, previous.get(rel)}:
            conflicts.append(rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".incoming")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)
    return conflicts


class ClusterBackend(QObject):
    changed = Signal()
    question = Signal(object)

    def __init__(self, ctx, local, profile):
        super().__init__(ctx)
        self.ctx, self.local, self.profile = ctx, local, profile
        self.directory = local.root / CONTROL
        self.journal = read_json(self.directory / "workspace-jobs.json", [])
        self.transfers = read_json(self.directory / "transfers.json", [])
        self.snapshot = read_json(self.directory / "remote-state.json", {})
        self.synced = read_json(self.directory / "synchronized.json", {})
        self.runner = ThreadRunner(self)
        self.client = None
        self._globus = None
        self.questions = []
        self._specs = {}
        self.message = "Connect to LRZ to check setup"
        self.question.connect(self._answer)
        self.timer = QTimer(self)
        self.timer.setInterval(30000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self._closed = False

    @property
    def running(self):
        return any(j.get("state") not in TERMINAL for j in self.journal)

    @property
    def busy(self):
        return self.runner.running

    @property
    def transferring(self):
        return any(t.get("state") in {"SUBMITTING", "ACTIVE"} for t in self.transfers)

    def status(self, text):
        self.message = text
        self.changed.emit()

    def persist(self):
        write_json(self.directory / "workspace-jobs.json", self.journal)
        write_json(self.directory / "transfers.json", self.transfers)

    def _answer(self, request):
        if request["kind"] == "trust":
            request["answer"] = QMessageBox.question(self.ctx.parent(), "Verify LRZ SSH host",
                "Confirm this fingerprint against LRZ's published SSH fingerprints before accepting:\n\n"
                + request["text"], QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
        else:
            value, ok = QInputDialog.getText(self.ctx.parent(), "LRZ sign in", request["text"],
                                             QLineEdit.Password if request["secret"] else QLineEdit.Normal)
            request["answer"] = value if ok else None
        request["event"].set()

    def _ask(self, text, secret=True, kind="password"):
        r = dict(text=text, secret=secret, kind=kind, event=threading.Event(), answer=None)
        self.questions.append(r)
        self.question.emit(r)
        if not r["event"].wait(300) or r["answer"] is None:
            raise RuntimeError("Sign in cancelled or timed out.")
        self.questions.remove(r)
        return r["answer"]

    def _work(self, text, function, done=None):
        if self.busy:
            return False
        self.status(text)
        def run(progress, cancelled):
            return function()
        def finished(result, error):
            if self._closed:
                return
            if error:
                self.timer.stop()  # do not repeatedly prompt for expired credentials every 30 seconds
                self.status("Needs attention: " + error.splitlines()[0])
                self.ctx.log(error, "error")
                QMessageBox.warning(self.ctx.parent(), "Cluster needs attention", error.split("Traceback")[0][:3000])
            elif done:
                try:
                    done(result)
                except Exception as e:
                    self.status("Needs attention: " + str(e))
                    self.ctx.log(str(e), "error")
                    QMessageBox.warning(self.ctx.parent(), "Cluster needs attention", str(e))
            else:
                self.status(str(result))
            if not error and not self._closed:
                self.timer.start()
            self.changed.emit()
        return self.runner.start(run, on_done=finished)

    def connect(self, done=None):
        if self.client and self.client.connected:
            if done:
                done(None)
            return
        settings = ConnectionSettings(**{k: self.profile[k] for k in ConnectionSettings.__dataclass_fields__})
        self.client = ClusterClient(settings, self._ask, lambda host, fp: self._ask(host + "\n" + fp, False, "trust"),
                                    settings_dir() / "cluster-known-hosts")
        def connected(result):
            self.status(result)
            if done:
                done(result)
        self._work("Waiting for LRZ sign in…", self.client.connect, connected)

    def globus(self):
        executable = self.profile.get("globus_cli", "")
        if self._globus is None or self._globus.executable != executable:
            self._globus = GlobusCLI(executable)
        return self._globus

    def _transfer(self, source, source_path, target, target_path, purpose, *, exclude=(), fingerprint="", recursive=True, metadata=None):
        from ..core.globus_transfer import collection_id, collection_path
        self.globus().command()
        collection_id(source); collection_id(target); collection_path(source_path); collection_path(target_path)
        entry = dict(state="SUBMITTING", task_id="", purpose=purpose, source=source, source_path=source_path,
                     target=target, target_path=target_path, fingerprint=fingerprint, created=time.time(),
                     recursive=recursive, exclude=list(exclude))
        entry.update(metadata or {})
        self.transfers.append(entry)
        self.persist()
        # An exception leaves SUBMITTING in the journal: never silently duplicate an
        # uncertain transfer. The setup screen provides its label for Globus Activity.
        label = "FEABAS " + purpose + " " + uuid.uuid4().hex[:12]
        entry["label"] = label
        self.persist()
        try:
            entry["task_id"] = self.globus().transfer(source, source_path, target, target_path, label, exclude=exclude, recursive=recursive)
        except GlobusAuthorizationRequired:
            entry.update(state="FAILED", details="Globus rejected authorization; grant collection access, then Retry a failed transfer")
            self.persist()
            raise
        entry["state"] = "ACTIVE"
        self.persist()
        return entry

    def sync(self):
        if self.running or self.transferring or self.busy:
            self.status("Finish the current job/transfer before synchronizing.")
            return
        def connected(_):
            p = dict(self.profile)
            def stage():
                self.globus().command()
                self.client.prepare_workspace(p)
                fp = source_signature(self.ctx.project)
                self.globus().stat(p["source_collection"], p["source_path"])
                # Verify that the selected Globus work path is the same SSH folder.
                probe = self.globus().stat(p["destination_collection"], p["destination_project"].rstrip("/")
                                           + "/.workbench-cluster/connection-" + p["project_id"] + ".txt")
                if probe.get("size") != 32:
                    raise RuntimeError("The Globus work folder does not match the SSH work folder.")
                probe = self.globus().stat(p["destination_collection"], p["destination_tiles"].rstrip("/")
                                           + "/.workbench-connection-" + p["project_id"] + ".txt")
                if probe.get("size") != 32:
                    raise RuntimeError("The Globus image folder does not match the SSH image folder.")
                batch = uuid.uuid4().hex
                initial = p.get("include_existing") and not self.synced
                meta = dict(batch=batch, scope=sync_scope(p), batch_size=2 if initial else 1)
                self._transfer(p["source_collection"], p["source_path"], p["destination_collection"],
                               p["destination_tiles"], "raw images", fingerprint=fp, metadata=meta)
                if initial:
                    self._transfer(p["computer_collection"], local_globus_path(self.local.root),
                                          p["destination_collection"], p["destination_project"], "initial project",
                                          exclude=(CONTROL, ".git", "snapshots", "workbench.log"), metadata=meta)
                self.persist()
                return "Uploading via Globus. Keep the source computer/NAS available; jobs unlock after verification."
            self._work("Checking folders and starting synchronization…", stage)
        self.connect(connected)

    def submit(self, specs):
        specs = specs if isinstance(specs, list) else [specs]
        if self.running or self.transferring or self.busy:
            self.status("Wait for the current cluster job or transfer to finish.")
            return
        if any(not spec.remote for spec in specs):
            QMessageBox.warning(self.ctx.parent(), "Cannot run on cluster", "This operation has no cluster implementation. "
                                "Switch explicitly to This PC to run it locally.")
            return
        if not self.synced:
            QMessageBox.information(self.ctx.parent(), "Synchronize first", "Use Sync project & images in the top bar first.")
            return
        def connected(_):
            p = dict(self.profile)
            def start():
                if source_signature(self.ctx.project) != self.synced.get("source_signature"):
                    raise RuntimeError("The source images have changed. Sync project & images before running.")
                if self.synced.get("scope") != sync_scope(p):
                    raise RuntimeError("Cluster folders or Globus settings have changed. Synchronize again before running.")
                self.client.check_environment(p["remote_python"], p["remote_project"], p["remote_tiles"], p["modules"])
                self.ctx.configs.save()
                shutil.copytree(self.local.configs_dir, self.ctx.project.configs_dir, dirs_exist_ok=True)
                self.ctx.project.write_general_config()
                bundle = build_workspace_bundle(self.ctx.project, self.local, p, [s.remote for s in specs], self.directory / "bundles")
                remote = p["remote_project"] + "/.workbench-cluster/" + bundle.name
                manifest = read_json(bundle / "manifest.json")
                entry = dict(run_id=bundle.name, remote_bundle=remote, job_id="", cluster=resources_for(p).cluster,
                             host=p["host"], username=p["username"], port=p["port"], state="UPLOADING",
                             started=time.time(), commands=[s.remote for s in specs], last_scheduler_check=0,
                             notified=False, preview_stamp=0, input_plan=manifest.get("external_inputs", {}))
                entry["remote_commands"] = manifest["commands"]
                self.journal.append(entry)
                self._specs[bundle.name] = specs
                self.persist()
                try:
                    self.client.upload_bundle(bundle)
                except Exception:
                    entry["state"] = "FAILED"
                    self.persist()
                    raise
                entry["state"] = "STAGING" if entry["input_plan"] else "READY"
                self.persist()
                self._stage_or_submit(entry)
                return entry
            def started(entry):
                self.ctx.jobs.job_started.emit(specs[0])
                self.ctx.jobs.running_changed.emit(True)
                self.status(f"Queued at LRZ · job {entry['job_id']} · {p['partition']}" if entry["job_id"] else
                            "Uploading large masks / model inputs before submission")
            self._work("Uploading settings and submitting job…", start, started)
        self.connect(connected)

    def _stage_or_submit(self, entry):
        if entry.get("job_id") or entry["state"] not in {"STAGING", "READY"}:
            return
        p = self.profile
        for rel, asset in entry.get("input_plan", {}).items():
            key = entry["run_id"] + "/" + rel
            existing = next((t for t in self.transfers if t.get("job_input_key") == key and not t.get("replaced")), None)
            if existing is None:
                self._transfer(p["computer_collection"], local_globus_path(Path(asset["source"])),
                               p["destination_collection"], p["destination_project"].rstrip("/") +
                               "/.workbench-cluster/" + entry["run_id"] + "/external_inputs/" + rel,
                               "job input", recursive=False, metadata=dict(job_input_key=key))
        assets = [t for t in self.transfers if t.get("job_input_key", "").startswith(entry["run_id"] + "/") and not t.get("replaced")]
        if any(t["state"] == "FAILED" for t in assets):
            entry["state"] = "FAILED"
            entry["result"] = dict(error="An input transfer failed. The computation was not submitted.", results=[])
        elif all(t["state"] == "SUCCEEDED" for t in assets):
            entry["state"] = "SUBMITTING"
            self.persist()
            entry.update(asdict(self.client.submit(entry["remote_bundle"], entry["cluster"])), state="PENDING")
        self.persist()

    def refresh(self):
        if self._closed or self.busy:
            return
        awaiting_notice = any(j.get("state") in TERMINAL and not j.get("notified") for j in self.journal)
        if not self.transferring and not self.running and not awaiting_notice:
            return
        if self.running and (not self.client or not self.client.connected):
            self.status("Disconnected · jobs continue at LRZ · open Cluster settings to reconnect")
            return
        def check():
            notices = []
            for t in self.transfers:
                if t["state"] == "SUBMITTING" and not t["task_id"]:
                    notices.append("Unconfirmed transfer. Check Globus Activity and recover its task ID in setup.")
                elif t["state"] == "ACTIVE":
                    result = self.globus().task(t["task_id"])
                    t.update(state=result["status"], bytes=result.get("bytes_transferred", 0),
                             details=result.get("nice_status_short_description", ""))
                if t["purpose"] == "download previews" and t["state"] == "SUCCEEDED" and not t.get("applied"):
                    conflicts = apply_preview_directory(Path(t["local_destination"]), self.ctx.project.root,
                                                        t["preview_manifest"], t.get("previous_previews", {}))
                    t["applied"] = True
                    notices.append("All previews downloaded" + (f"; preserved {len(conflicts)} local edits" if conflicts else ""))
            if self.transfers and not self.transferring:
                uploads = [t for t in self.transfers if t["purpose"] in {"raw images", "initial project"} and not t.get("replaced")]
                latest = uploads[-1].get("batch") if uploads else None
                uploads = [t for t in uploads if t.get("batch") == latest and t.get("scope") == sync_scope(self.profile)]
                if uploads and len(uploads) == uploads[0].get("batch_size", len(uploads)) and all(t["state"] == "SUCCEEDED" for t in uploads):
                    self.synced = dict(source_signature=uploads[-1].get("fingerprint") or
                                       next(t["fingerprint"] for t in reversed(uploads) if t.get("fingerprint")),
                                       timestamp=time.time(), scope=sync_scope(self.profile))
                    write_json(self.directory / "synchronized.json", self.synced)
                    notices.append("Images synchronized · ready to run")
            snapshots = []
            for entry in self.journal:
                if entry["state"] in TERMINAL:
                    continue
                if (entry["host"], entry["username"], entry["port"]) != (self.profile["host"], self.profile["username"], self.profile["port"]):
                    notices.append("Reconnect to the original account to monitor this job.")
                    continue
                if entry["state"] in {"READY", "STAGING"}:
                    self._stage_or_submit(entry)
                    if not entry["job_id"]:
                        notices.append("Waiting for verified input transfers")
                        continue
                if entry["state"] == "UPLOADING":
                    entry["state"] = "FAILED"
                    entry["result"] = dict(error="Workbench closed during package upload. No job was submitted. Run the step again.", results=[])
                    continue
                if not entry["job_id"]:
                    entry.update(asdict(self.client.recover(entry["remote_bundle"], entry["cluster"])), state="PENDING")
                state = self.client.read_remote_json(entry["remote_bundle"] + "/status.json", {})
                if state and (state.get("run_id") != entry["run_id"] or state.get("project_id") != self.profile["project_id"]):
                    raise RuntimeError("Remote status belongs to a different project or job.")
                if state and state.get("timestamp", 0) > entry["preview_stamp"]:
                    archive = self.directory / (entry["run_id"] + "-previews.zip")
                    archive.unlink(missing_ok=True)
                    if "previews" in state:
                        self.client.download_file(entry["remote_bundle"] + "/previews.zip", archive)
                        conflicts = apply_previews(archive, self.ctx.project.root, state, self.snapshot.get("previews", {}))
                        archive.unlink(missing_ok=True)
                        if conflicts:
                            notices.append(f"Preserved {len(conflicts)} local edits; remote copies are available via Globus.")
                    self.snapshot = state
                    write_json(self.directory / "remote-state.json", state)
                    entry["preview_stamp"] = state.get("timestamp", 0)
                    snapshots.append(state)
                if state.get("state") in TERMINAL:
                    entry["state"] = state["state"]
                    entry["result"] = state
                    notices.append(f"Job {entry['job_id']}: {entry['state']}")
                    if state.get("previews_skipped") and not any(t.get("preview_run") == entry["run_id"] for t in self.transfers):
                        incoming = self.directory / "incoming-previews" / entry["run_id"]
                        incoming.mkdir(parents=True, exist_ok=True)
                        if shutil.disk_usage(incoming).free < 2 * state.get("preview_bytes", 0):
                            raise RuntimeError("Not enough local disk space for the complete preview cache.")
                        self._transfer(self.profile["destination_collection"], self.profile["destination_project"].rstrip("/") +
                                       "/.workbench-cluster/" + entry["run_id"] + "/preview-cache",
                                       self.profile["computer_collection"], local_globus_path(incoming), "download previews",
                                       metadata=dict(preview_run=entry["run_id"], local_destination=str(incoming),
                                                     preview_manifest=state["preview_manifest"], previous_previews=self.snapshot.get("preview_manifest", {})))
                elif time.time() - entry["last_scheduler_check"] >= 600:
                    entry["last_scheduler_check"] = time.time()
                    self.persist()
                    status = self.client.status(entry["job_id"], entry["cluster"])
                    entry["scheduler_status"] = status
                    first = status.split("|")
                    word = first[1] if first[0].strip() == entry["job_id"] and len(first) > 1 else first[0]
                    word = word.strip().split()[0].rstrip("+") if word.strip() else "UNKNOWN"
                    if word in TERMINAL:
                        entry["state"] = "FAILED" if word == "COMPLETED" and state.get("state") != "COMPLETED" else word
                    notices.append(f"Job {entry['job_id']}: {word}")
                else:
                    if state.get("state") == "RUNNING":
                        entry["state"] = "RUNNING"
                    heartbeat = self.client.read_remote_json(entry["remote_bundle"] + "/heartbeat.json", {})
                    if heartbeat:
                        notices.append(f"Running · peak RAM {heartbeat.get('peak_memory_bytes', 0) / 2**30:.1f} GiB")
                if entry.get("log_path") and (state or entry["state"] in TERMINAL):
                    log = self.client.tail_log(entry["log_path"])
                    old = entry.get("log_tail", "")
                    if log != old:
                        # Tails are bounded at 64 KiB. Find their overlap after a
                        # long output burst so the visible log does not repeat.
                        overlap = min(len(old), len(log))
                        while overlap and old[-overlap:] != log[:overlap]:
                            overlap -= 1
                        self.ctx.jobs.output.emit("stdout", log[overlap:])
                        entry["log_tail"] = log
            self.persist()
            return notices
        def checked(notices):
            for entry in self.journal:
                if entry["state"] in TERMINAL and not entry.get("notified"):
                    self._finish(entry)
            for t in self.transfers:
                if t["purpose"] == "download export" and t["state"] == "SUCCEEDED" and not t.get("notified"):
                    t["notified"] = True
                    self.ctx.log("Export downloaded and checksum-verified: " + t["target_path"])
                    if t.get("local_destination"):
                        self.ctx.export_downloaded.emit(t["local_destination"])
                    notices.append("Export downloaded and verified")
            active = [t for t in self.transfers if t["state"] == "ACTIVE"]
            failed = [t for t in self.transfers if t["state"] == "FAILED"]
            if active:
                notices.append(f"Transferring · {sum(t.get('bytes', 0) for t in active) / 2**30:.2f} GiB copied")
            elif failed:
                notices.append("Transfer failed · see Cluster settings / Globus Activity")
            if notices:
                self.status(" · ".join(notices[-2:]))
            self.persist()
            self.ctx.state_changed.emit()
        self._work(self.message, check, checked)

    def _finish(self, entry):
        entry["notified"] = True
        self.persist()
        specs = self._specs.get(entry["run_id"], [JobSpec(c.get("name", c.get("step", "Cluster job")), [],
                                                       self.ctx.project.root, remote=c) for c in entry["commands"]])
        results = []
        remote_results = entry.get("result", {}).get("results", [])
        for i, spec in enumerate(specs):
            data = remote_results[i].get("result", {}) if i < len(remote_results) else {}
            succeeded = i < len(remote_results) or entry["state"] == "COMPLETED"
            cancelled = entry["state"] == "CANCELLED" or (not succeeded and i > len(remote_results))
            result = JobResult(spec, 0 if succeeded else 1, cancelled,
                               time.time() - entry["started"], result=data,
                               last_lines=[entry.get("result", {}).get("error") or entry.get("scheduler_status", entry["state"])])
            results.append(result)
            if succeeded or i == len(remote_results):
                self.ctx.jobs.job_finished.emit(result)
        self.ctx.jobs.queue_finished.emit(results)
        self.ctx.jobs.running_changed.emit(False)
        if entry["state"] == "COMPLETED" and self.profile.get("download_after_export"):
            exports = [c for c in entry.get("remote_commands", []) if c.get("module") == "export_vast"]
            if exports:
                QTimer.singleShot(0, lambda: self.download_exports(exports[-1].get("download_to"), entry["run_id"]))

    def download_exports(self, destination=None, run_id=None):
        if self.busy or self.running:
            self.status("Wait for the job to finish before downloading exports.")
            return
        if run_id is None:
            completed = [j for j in self.journal if j["state"] == "COMPLETED" and
                         any(c.get("module") == "export_vast" for c in j.get("remote_commands", []))]
            if not completed:
                self.status("No completed cluster export yet. Use Export in the usual Export page first.")
                return
            run_id = completed[-1]["run_id"]
        dest = Path(destination or self.profile.get("export_destination") or self.local.exports_dir / "cluster")
        if dest.is_relative_to(self.ctx.project.root):
            dest = self.local.exports_dir / "cluster"
        if dest.name != run_id:
            dest = dest / run_id
        p = self.profile
        def download():
            dest.mkdir(parents=True, exist_ok=True)
            required = self.snapshot.get("export_bytes", {}).get(run_id, 0)
            if shutil.disk_usage(dest).free < required:
                raise RuntimeError(f"Not enough local disk space for this export ({required / 2**30:.1f} GiB). Choose another export folder.")
            entry = self._transfer(p["destination_collection"], p["destination_project"].rstrip("/") + "/exports/" + run_id,
                                  p["computer_collection"], local_globus_path(dest), "download export",
                                  metadata=dict(local_destination=str(dest), export_run=run_id))
            return "Downloading exports · transfer " + entry["task_id"]
        self._work("Starting export download…", download)

    def download_rendered(self):
        if self.busy or self.running or self.transferring:
            self.status("Wait for the current job or transfer before downloading the rendered stack.")
            return
        folders = {name: size for name, size in self.snapshot.get("render_bytes", {}).items() if size > 0}
        if not folders:
            self.status("No rendered stack at LRZ yet. Run Render in the usual Export page first.")
            return
        total = sum(folders.values())
        if QMessageBox.question(self.ctx.parent(), "Download full-resolution stack",
                f"Download {total / 2**30:.1f} GiB to this project's cluster cache for local inspection?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        def download():
            if shutil.disk_usage(self.ctx.project.root).free < total:
                raise RuntimeError("Not enough local disk space for the rendered stack.")
            for name in folders:
                self._transfer(self.profile["destination_collection"], self.profile["destination_project"].rstrip("/") + "/" + name,
                               self.profile["computer_collection"], local_globus_path(self.ctx.project.root / name), "download rendered stack")
            return "Downloading full-resolution results for local viewers"
        self._work("Starting rendered-stack download…", download)

    def cancel_all(self):
        if self.busy:
            self.status("Wait for the current connection operation, then cancel the cluster job.")
            return
        def cancel():
            for entry in self.journal:
                if entry["state"] not in TERMINAL and entry["job_id"]:
                    self.client.cancel(entry["job_id"], entry["cluster"])
                elif entry["state"] in {"STAGING", "READY"}:
                    for t in self.transfers:
                        if t.get("job_input_key", "").startswith(entry["run_id"] + "/") and t["state"] == "ACTIVE":
                            self.globus().cancel(t["task_id"])
                    entry["state"] = "CANCELLED"
            self.persist()
            return "Cancellation requested; waiting for confirmation from LRZ."
        self.connect(lambda _: self._work("Requesting cancellation…", cancel))

    def shutdown(self):
        self._closed = True
        self.timer.stop()
        for r in self.questions:
            r["event"].set()
        if self.client:
            self.client.close()
        if self._globus:
            self._globus.close()
        self.runner.stop()
        self.persist()
