"""An opt-in remote workspace, kept out of all workstation pages."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
                              QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                              QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                              QScrollArea, QSpinBox, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..core.cluster_bundle import ClusterResources, PARTITIONS, export_bundle
from ..core.cluster_transport import ClusterClient, ConnectionSettings, JobRecord
from ..core.envs import settings_dir
from ..core.steps import STANDARD_PIPELINE, STEPS
from .threads import ThreadRunner


class ClusterDialog(QDialog):
    question = Signal(object)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run on cluster · LRZ / Slurm")
        self.resize(930, 800)
        self.ctx, self.project = ctx, ctx.project
        self.directory = self.project.root / ".workbench-cluster"
        self.profile_file = self.directory / "profile.json"
        self.history_file = self.directory / "jobs.json"
        self.client = None
        self.bundle = None
        self.runner = ThreadRunner(self)
        self.questions = []
        self.question.connect(self._answer)
        self.history = self._read(self.history_file, [])
        self._build()
        self._load_profile()
        self._show_history()

    @staticmethod
    def _read(path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def _write(self, path, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _build(self):
        layout = QVBoxLayout(self)
        title = QLabel("<h2>Send work to LRZ</h2>Keep using your PC while Slurm schedules the computation.")
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        connect = QWidget(); form = QFormLayout(connect)
        self.host = QLineEdit("cool.hpc.lrz.de")
        self.user = QLineEdit(); self.user.setPlaceholderText("Your cluster account")
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(22)
        self.key = QLineEdit(); self.key.setPlaceholderText("Optional private key file; port 2222 at LRZ")
        for label, widget in (("Host", self.host), ("Username", self.user), ("SSH port", self.port), ("Private key", self.key)):
            form.addRow(label, widget)
        browse = QPushButton("Choose private key…"); browse.clicked.connect(self._choose_key); form.addRow("", browse)
        note = QLabel("Password, key passphrase and MFA are requested when connecting and are never saved.\n"
                      "An Open OnDemand browser login does not authenticate this SSH connection.")
        note.setWordWrap(True); form.addRow(note)
        self.connect_btn = QPushButton("Connect…"); self.connect_btn.clicked.connect(self._connect)
        self.disconnect_btn = QPushButton("Disconnect"); self.disconnect_btn.clicked.connect(self._disconnect)
        form.addRow(self.connect_btn, self.disconnect_btn)
        self.connection_label = QLabel("Not connected"); form.addRow(self.connection_label)
        self.tabs.addTab(connect, "1  Connection")

        run = QWidget(); runlayout = QVBoxLayout(run); runform = QFormLayout()
        self.remote_project = QLineEdit(); self.remote_project.setPlaceholderText("/dss/…/my-project-run1 (new, dedicated folder)")
        self.remote_tiles = QLineEdit(); self.remote_tiles.setPlaceholderText("/dss/…/dataset (same relative tile layout)")
        self.remote_python = QLineEdit(); self.remote_python.setPlaceholderText("/dss/…/envs/feabas/bin/python")
        self.modules = QLineEdit(); self.modules.setPlaceholderText("Optional module names required by your Python environment")
        for label, w in (("Remote work folder", self.remote_project), ("Remote tile folder", self.remote_tiles),
                         ("Remote Python", self.remote_python), ("Environment modules", self.modules)):
            runform.addRow(label, w)
        active = self.project.active_tile_root()
        local = QLabel(f"Local tile folder → remote tile folder\n{active or 'Set a source on the Project page first'}")
        local.setWordWrap(True); runform.addRow(local)
        self.partition = QComboBox(); self.partition.addItems(list(PARTITIONS))
        self.cpus = QSpinBox(); self.cpus.setRange(17, 112); self.cpus.setValue(64)
        self.memory = QSpinBox(); self.memory.setRange(1, 244); self.memory.setValue(128); self.memory.setSuffix(" GiB")
        self.hours = QSpinBox(); self.hours.setRange(1, 24); self.hours.setValue(4); self.hours.setSuffix(" hours")
        self.workers = QSpinBox(); self.workers.setRange(1, 64); self.workers.setValue(16)
        resources = QHBoxLayout()
        for label, w in (("Partition", self.partition), ("CPUs", self.cpus), ("RAM", self.memory), ("Time", self.hours)):
            resources.addWidget(QLabel(label)); resources.addWidget(w)
        runform.addRow(resources)
        runform.addRow("FEABAS workers per step", self.workers)
        self.workers.setToolTip("Overrides worker counts in the exported configs only. More workers also need more RAM.")
        self.partition.currentTextChanged.connect(self._partition_changed)
        self.cpus.valueChanged.connect(self.workers.setMaximum)
        runlayout.addLayout(runform)
        note = QLabel("One compute node; no MPI. Presets are request limits, not an entitlement or availability guarantee.\n"
                      "Use a new work folder after changing inputs, settings or resources. Identical runs can resume cached outputs.")
        note.setWordWrap(True); runlayout.addWidget(note)
        self.steps = QListWidget(); self.steps.setMaximumHeight(160)
        for s in STEPS:
            if s.script:
                item = QListWidgetItem(s.label, self.steps); item.setData(Qt.UserRole, s.key)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if s.key in STANDARD_PIPELINE else Qt.Unchecked)
        runlayout.addWidget(self.steps)
        row = QHBoxLayout()
        self.prepare_btn = QPushButton("Prepare / preview job"); self.prepare_btn.clicked.connect(self._prepare)
        self.check_btn = QPushButton("Check remote setup"); self.check_btn.clicked.connect(self._check)
        self.submit_btn = QPushButton("Upload job files && submit"); self.submit_btn.clicked.connect(self._submit)
        for w in (self.prepare_btn, self.check_btn, self.submit_btn): row.addWidget(w)
        runlayout.addLayout(row)
        self.staged = QCheckBox("The tile dataset is already staged at the remote tile folder.")
        runlayout.addWidget(self.staged)
        runscroll = QScrollArea(); runscroll.setWidgetResizable(True); runscroll.setWidget(run)
        self.tabs.addTab(runscroll, "2  Prepare & run")

        jobs = QWidget(); jl = QVBoxLayout(jobs)
        self.jobs = QListWidget(); self.jobs.setMaximumHeight(170); jl.addWidget(self.jobs)
        row = QHBoxLayout()
        self.job_buttons = []
        for text, slot in (("Refresh status", self._refresh), ("Read log", self._log),
                           ("Recover job ID", self._recover), ("Cancel job…", self._cancel_job)):
            b = QPushButton(text); b.clicked.connect(slot); row.addWidget(b); self.job_buttons.append(b)
        jl.addLayout(row)
        note = QLabel("Refresh is manual and limited to once per job every 10 minutes. Closing this window leaves submitted jobs running.")
        note.setWordWrap(True); jl.addWidget(note)
        self.result_path = QLineEdit(); self.result_path.setPlaceholderText("Absolute path to a log, transform or preview file on LRZ")
        jl.addWidget(self.result_path)
        self.download_btn = QPushButton("Download one result file… (up to 64 MiB)")
        self.download_btn.clicked.connect(self._download); jl.addWidget(self.download_btn)
        jl.addStretch()
        self.tabs.addTab(jobs, "3  Jobs & results")
        helpview = QTextBrowser(); helpview.setOpenExternalLinks(True)
        helpview.setHtml(HELP)
        self.tabs.addTab(helpview, "Data & setup guide")
        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMaximumHeight(175)
        self.output.setPlaceholderText("Job preview, connection checks and logs appear here."); layout.addWidget(self.output)
        close = QPushButton("Close"); close.clicked.connect(self.reject); layout.addWidget(close, alignment=Qt.AlignRight)
        self.mutation_buttons = [self.connect_btn, self.disconnect_btn, self.prepare_btn, self.check_btn,
                                 self.submit_btn, self.download_btn, *self.job_buttons]
        screen = self.screen().availableGeometry()
        self.resize(min(930, screen.width()), min(800, screen.height()))

    def _partition_changed(self, name):
        _, low, high, mem, hours, _ = PARTITIONS[name]
        self.cpus.setRange(low, high); self.memory.setMaximum(mem); self.hours.setMaximum(hours)

    def _settings(self):
        return ConnectionSettings(self.host.text().strip(), self.user.text().strip(), self.port.value(), self.key.text().strip())

    def _profile(self):
        return {**asdict(self._settings()), "remote_project": self.remote_project.text().strip(),
                "remote_tiles": self.remote_tiles.text().strip(), "remote_python": self.remote_python.text().strip(),
                "modules": self.modules.text().strip(), "partition": self.partition.currentText(),
                "cpus": self.cpus.value(), "memory_gib": self.memory.value(), "hours": self.hours.value(),
                "workers": self.workers.value(), "steps": self._selected_steps()}

    def _selected_steps(self):
        return [self.steps.item(i).data(Qt.UserRole) for i in range(self.steps.count())
                if self.steps.item(i).checkState() == Qt.Checked]

    def _load_profile(self):
        d = self._read(self.profile_file, {})
        for key in ("host", "username", "key_filename", "remote_project", "remote_tiles", "remote_python", "modules"):
            widget = {"username": self.user, "key_filename": self.key}.get(key) or getattr(self, key, None)
            if key in d and widget: widget.setText(str(d[key]))
        if d.get("partition") in PARTITIONS: self.partition.setCurrentText(d["partition"])
        for key, w in (("port", self.port), ("cpus", self.cpus), ("memory_gib", self.memory), ("hours", self.hours), ("workers", self.workers)):
            if key in d: w.setValue(int(d[key]))
        if "steps" in d:
            for i in range(self.steps.count()):
                item = self.steps.item(i); item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in d["steps"] else Qt.Unchecked)

    def _choose_key(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose your SSH private key")
        if path: self.key.setText(path)

    def _ask(self, text, secret=True, trust=False):
        request = {"text": text, "secret": secret, "trust": trust, "event": threading.Event(), "value": None}
        self.questions.append(request); self.question.emit(request)
        if not request["event"].wait(180):
            raise RuntimeError("Authentication prompt timed out. Connect again.")
        self.questions.remove(request)
        if request["value"] is None: raise RuntimeError("Connection cancelled.")
        return request["value"]

    def _answer(self, request):
        if request["event"].is_set(): return
        if request["trust"]:
            value = QMessageBox.question(self, "Verify SSH host key", request["text"],
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
        else:
            value, ok = QInputDialog.getText(self, "Cluster authentication", request["text"],
                                             QLineEdit.Password if request["secret"] else QLineEdit.Normal)
            if not ok: value = None
        request["value"] = value; request["event"].set()

    def _work(self, fn, success=None):
        if self.runner.running: return
        for b in self.mutation_buttons: b.setEnabled(False)
        # Keep edits from changing the endpoint or payload during an operation.
        self.tabs.widget(0).setEnabled(False); self.tabs.widget(1).setEnabled(False)
        def done(result, error):
            for b in self.mutation_buttons: b.setEnabled(True)
            self.tabs.widget(0).setEnabled(True); self.tabs.widget(1).setEnabled(True)
            if error:
                self.output.setPlainText(error)
            elif success:
                success(result)
            else:
                self.output.setPlainText(str(result))
            self.connection_label.setText("Connected" if self.client and self.client.connected else "Not connected")
        self.output.setPlainText("Working…")
        self.runner.start(lambda **kwargs: fn(), on_done=done)

    def _connected(self):
        if not self.client or not self.client.connected:
            raise ValueError("Connect on the Connection tab first.")
        if self._settings() != self.client.settings:
            raise ValueError("Connection settings changed. Reconnect first.")

    def _connect(self):
        try:
            settings = self._settings(); settings.validate()
            self._write(self.profile_file, self._profile())
            if self.client: self.client.close()
            self.client = ClusterClient(settings, self._ask,
                lambda host, key: self._ask(f"{host}\n{key}\n\nCompare this fingerprint with LRZ's official SSH documentation "
                                            "before accepting. Accept this key?", trust=True),
                settings_dir() / "cluster_known_hosts")
            self._work(self.client.connect)
        except Exception as e: self.output.setPlainText(str(e))

    def _disconnect(self):
        if self.client: self.client.close()
        self.connection_label.setText("Not connected")
        self.output.setPlainText("Disconnected. Submitted jobs continue running.")

    def _check(self):
        try:
            self._connected(); p = self._profile()
            self._work(lambda: self.client.check_environment(p["remote_python"], p["remote_project"], p["remote_tiles"], p["modules"]))
        except Exception as e: self.output.setPlainText(str(e))

    def _prepare(self):
        try:
            self.ctx.save_project()
            p = self._profile(); self._write(self.profile_file, p)
            resources = ClusterResources(p["partition"], p["cpus"], p["memory_gib"], p["hours"], p["workers"])
            self.bundle = None
            def ready(path):
                self.bundle = path; self.prepared_profile = p
                self.output.setPlainText(f"Prepared locally: {path}\nRaw images are NOT included.\n\n" + (path / "job.sh").read_text())
            self._work(lambda: export_bundle(self.project, self.directory / "bundles", p["remote_project"],
                                             p["remote_tiles"], p["remote_python"], resources, p["steps"], p["modules"]), ready)
        except Exception as e: self.output.setPlainText(str(e))

    def _submit(self):
        try:
            self._connected()
            if not self.bundle or self._profile() != self.prepared_profile:
                raise ValueError("Prepare and review the job after your last settings change.")
            if not self.staged.isChecked(): raise ValueError("Stage the dataset first and check the data-ready box.")
            bundle, p = self.bundle, self.prepared_profile
            remote = p["remote_project"].rstrip("/") + "/.workbench-cluster/" + bundle.name
            if any(j["remote_bundle"] == remote for j in self.history):
                raise ValueError("Submission already attempted. Use Jobs & results to inspect or recover it.")
            entry = dict(job_id="", cluster=PARTITIONS[p["partition"]][0], remote_bundle=remote, log_path="",
                         host=p["host"], username=p["username"], port=p["port"], state="Submission pending / unconfirmed")
            self.history.append(entry); self._save_history()
            def submit():
                self.client.check_environment(p["remote_python"], p["remote_project"], p["remote_tiles"], p["modules"])
                self.client.upload_bundle(bundle)
                return self.client.submit(remote, entry["cluster"])
            def submitted(record):
                entry.update(asdict(record), state="Submitted"); self._save_history()
                self.tabs.setCurrentIndex(2); self.output.setPlainText(f"Submitted job {record.job_id} on {record.cluster}.")
            self._work(submit, submitted)
        except Exception as e: self.output.setPlainText(str(e))

    def _save_history(self):
        self._write(self.history_file, self.history); self._show_history()

    def _show_history(self):
        selected = self.jobs.currentRow()
        self.jobs.clear()
        for j in self.history:
            self.jobs.addItem(f"{j['job_id'] or '?'} · {j['cluster']} · {j['host']} · {j.get('state', 'Unknown')}")
        if self.history: self.jobs.setCurrentRow(selected if 0 <= selected < len(self.history) else len(self.history) - 1)

    def _selected_job(self, require_id=True):
        self._connected()
        index = self.jobs.currentRow()
        if index < 0: raise ValueError("Select a job first.")
        j = self.history[index]; s = self.client.settings
        if (j["host"], j["username"], j["port"]) != (s.host, s.username, s.port):
            raise ValueError("Reconnect to this job's original host, username and port.")
        if require_id and not j["job_id"]: raise ValueError("Recover the job ID first; do not resubmit an uncertain job.")
        return j

    def _refresh(self):
        try:
            j = self._selected_job()
            remaining = 600 - (time.time() - j.get("last_refresh", 0))
            if remaining > 0: raise ValueError(f"Please wait {int(remaining) + 1} seconds before refreshing this job again.")
            j["last_refresh"] = time.time(); self._save_history()
            def refreshed(state):
                j["state"] = state; self._save_history(); self.output.setPlainText(state)
            self._work(lambda: self.client.status(j["job_id"], j["cluster"]), refreshed)
        except Exception as e: self.output.setPlainText(str(e))

    def _log(self):
        try:
            j = self._selected_job(); self._work(lambda: self.client.tail_log(j["log_path"]))
        except Exception as e: self.output.setPlainText(str(e))

    def _recover(self):
        try:
            j = self._selected_job(False)
            def recovered(record: JobRecord):
                j.update(asdict(record), state="Recovered; refresh for status"); self._save_history()
                self.output.setPlainText(f"Recovered job {record.job_id}.")
            self._work(lambda: self.client.recover(j["remote_bundle"], j["cluster"]), recovered)
        except Exception as e: self.output.setPlainText(str(e))

    def _cancel_job(self):
        try:
            j = self._selected_job()
            if QMessageBox.question(self, "Cancel job", f"Cancel job {j['job_id']} on {j['cluster']}?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes: return
            self._work(lambda: self.client.cancel(j["job_id"], j["cluster"]))
        except Exception as e: self.output.setPlainText(str(e))

    def _download(self):
        try:
            self._connected(); remote = self.result_path.text().strip()
            folder = QFileDialog.getExistingDirectory(self, "Choose folder for this result file (no overwrites)")
            if not folder: return
            dest = Path(folder) / remote.rsplit("/", 1)[-1]
            self._work(lambda: self.client.download_file(remote, dest),
                       lambda size: self.output.setPlainText(f"Downloaded {size:,} bytes to {dest}"))
        except Exception as e: self.output.setPlainText(str(e))

    def reject(self):
        if self.runner.running:
            self.output.appendPlainText("Wait for the current operation, or cancel its authentication prompt, before closing.")
            return
        self._write(self.profile_file, self._profile())
        if self.client: self.client.close()
        super().reject()

    def shutdown(self):
        for request in self.questions: request["event"].set()
        if self.client: self.client.close()
        self.runner.stop()


HELP = """
<h2>Prepare once, send jobs from your PC</h2>
<p><b>Slurm is LRZ's job scheduler.</b> Your PC prepares a job; LRZ allocates a compute node,
runs FEABAS there, and retains the output on cluster storage. The PC can disconnect after submission.</p>
<ol>
<li>Obtain Linux Cluster access and a DSS project allocation from your institution/LRZ.
The desktop login alone does not establish your storage quota or batch entitlement.</li>
<li>Copy your NAS dataset to DSS using <a href="https://doku.lrz.de/file-systems-and-io-on-linux-cluster-10745972.html">LRZ's
transfer guidance (Globus for bulk data)</a>. Ask your institute to use a Globus endpoint/server that can read the NAS.
Retain the directory layout below your local tile folder. Do not put a 1 TB dataset in the 100 GB home directory.
Scratch is temporary and has a deletion policy. Keep the source copy.</li>
<li>Prepare a Linux Python environment with <b>feabas==3.0.5</b>, tensorstore and numpy&lt;2 on LRZ.
Use an LRZ-supported Python/Conda module and install dependencies before submitting. Compute nodes normally
have no Internet. Give the full path to its bin/python and any module names needed to load its libraries.</li>
<li>Write coordinates in the Workbench Project page. The remote tile folder must contain the same active
raw or preprocessed images. Select a new, dedicated remote work folder outside the tile folder.</li>
<li>Connect with SSH password + MFA (port 22), or private key + MFA (port 2222).
Verify the host fingerprint using <a href="https://doku.lrz.de/ssh-secure-shell-on-lrz-hpc-systems-10746160.html">LRZ's SSH documentation</a>.
Then check the remote setup, prepare the job, inspect its script, and submit.</li>
</ol>
<h3>Resources and results</h3>
<p>cm4_tiny: 17–112 physical cores on one node. The preset caps RAM at 244 GiB and time at 24 h.
serial_std: 1–16 cores / 24 h; serial_long: 1–16 cores / 168 h. Both serial presets cap RAM at 100 GiB.
These conservative limits do not guarantee admission: concurrent jobs and account policy also matter.
FEABAS worker counts are overridden only in the exported files. More workers need more RAM;
more CPUs do not guarantee faster matching. There is no automatic multi-node MPI scaling.</p>
<p>The job includes configs, coordinates, small material masks and pinned FEABAS drivers (up to 64 MiB).
It checks missing tiles and output/error files, stops on failure, and prevents simultaneous runs in the same work folder.
Identical inputs can reuse outputs; changed settings/resources require a new folder to avoid stale caches.</p>
<p>Local deep-learning training/prediction, preprocessing and workstation page buttons still run locally.
This window runs FEABAS stitching, alignment and rendering. For large output volumes use Globus again;
the result download button handles one small file and never overwrites an existing local file.</p>
<p>An unconfirmed submission must be checked before any new job is sent. Try <b>Recover job ID</b>.
If there is no receipt, inspect the LRZ queue and the bundle's job-id.tmp file, or ask LRZ support.</p>
<p>Official references: <a href="https://doku.lrz.de/running-parallel-jobs-on-the-linux-cluster-11484078.html">batch jobs</a> ·
<a href="https://doku.lrz.de/policies-on-the-linux-cluster-1307289651.html">current policies</a> ·
<a href="https://doku.lrz.de/job-processing-on-the-linux-cluster-10745970.html">partition limits</a>.</p>
"""
