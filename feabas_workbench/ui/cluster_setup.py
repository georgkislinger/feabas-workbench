"""Guided setup; workstation pages retain their familiar layout."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout,
                              QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                              QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..core.cluster_bundle import PARTITIONS, remote_path
from ..core.cluster_storage import (ENVIRONMENT_BYTES, STORAGE_FACTOR, estimate_project_bytes, fit_warning, is_home,
                                    preprocessing_copies, project_storage_path, raw_image_bytes, storage_message)
from ..core.cluster_workspace import local_globus_path, resources_for, save_profile, sync_scope, write_json

LRZ_COLLECTION = "c3f32bba-797e-11e6-8435-22000b97daec"
STORAGE_DOC = "https://doku.lrz.de/file-systems-and-io-on-linux-cluster-10745972.html"
# Above this much raw data, a transfer that needs Workbench open for hours is the wrong default.
GLOBUS_SUGGESTED_BYTES = 200 * 10**9


class ClusterSetupDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.backend = ctx.cluster_setup()
        self._discovery = {}
        self._closed = False
        self.setWindowTitle("Use LRZ from Workbench")
        self.resize(850, 790)
        outer = QVBoxLayout(self)
        title = QLabel("<h2>Use your normal workflow on LRZ</h2>Connect once, synchronize your images, then use the usual Run buttons.")
        title.setWordWrap(True)
        outer.addWidget(title)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.fields = {}
        setup = QWidget(); form = QFormLayout(setup)
        self._text(form, "username", "LRZ username", "Your cluster account")
        b = QPushButton("1. Sign in and find my storage")
        b.clicked.connect(self._discover); form.addRow(b)
        self.storage = QComboBox(); self.storage.setEditable(True)
        self.storage.setPlaceholderText("Choose your home folder or a DSS container after signing in")
        form.addRow("Storage folder", self.storage)
        self.storage_note = QLabel(); self.storage_note.setWordWrap(True); form.addRow(self.storage_note)
        self.storage.currentTextChanged.connect(self._storage_hint)
        b = QPushButton("2. Use this storage folder")
        b.clicked.connect(self._choose_storage); form.addRow(b)
        self.paths = QLabel("No cluster workspace selected yet.")
        self.paths.setWordWrap(True); form.addRow(self.paths)
        b = QPushButton("3. Prepare FEABAS at LRZ")
        b.clicked.connect(self._install); form.addRow(b)
        note = QLabel("This creates a private Python 3.11 environment (Miniforge, as LRZ recommends) next to the project "
                      "folders and installs FEABAS 3.0.5; allow 5-15 minutes. It does not change your workstation's Python. "
                      "Your home folder (100 GB) suits the tutorial and smaller projects; for large datasets request a DSS "
                      "container from your project administrator / LRZ.")
        note.setWordWrap(True); form.addRow(note)
        b = QPushButton("LRZ storage instructions")
        b.clicked.connect(lambda: self.open_url(STORAGE_DOC)); form.addRow(b)
        self.tabs.addTab(setup, "1  Cluster setup")

        transfer = QWidget(); form = self.transfer_form = QFormLayout(transfer)
        self.transfer = QComboBox()
        self.transfer.addItem("Directly over SSH: no extra software, for data that fits your storage", "ssh")
        self.transfer.addItem("Globus: large datasets or an institute NAS, runs without Workbench", "globus")
        self.transfer.currentIndexChanged.connect(self._transfer_changed)
        form.addRow("How files travel", self.transfer)
        self.ssh_note = QLabel("Files travel over your SSH login, encrypted and size-checked. Unchanged files are skipped "
                               "and nothing is deleted at either end, so an interrupted upload simply continues with the "
                               "next Sync. Keep Workbench open while it copies.")
        self.ssh_note.setWordWrap(True); form.addRow(self.ssh_note)
        self._globus_rows = []
        note = QLabel("For this PC, install Globus Connect Personal, select accessible folders, then sign in below. "
                      "For an institute NAS, its own Globus collection is preferable if one exists. LRZ serves DSS "
                      "containers through Globus, not home folders.")
        note.setWordWrap(True); form.addRow(note); self._globus_rows.append(note)
        b = QPushButton("Set up Globus Connect Personal")
        b.clicked.connect(lambda: self.open_url("https://docs.globus.org/globus-connect-personal/install/windows/"))
        form.addRow(b); self._globus_rows.append(b)
        b = QPushButton("Sign in / grant Globus access in browser")
        b.clicked.connect(self._globus_login); form.addRow(b); self._globus_rows.append(b)
        b = QPushButton("Find this computer's collection")
        b.clicked.connect(self._find_pc); form.addRow(b); self._globus_rows.append(b)
        for key, label in (("computer_collection", "This PC's collection UUID"), ("source_collection", "Images: source collection UUID"),
                           ("source_path", "Images: folder in Globus"), ("destination_collection", "LRZ collection UUID"),
                           ("destination_tiles", "Images: LRZ folder in Globus"), ("destination_project", "Project: LRZ folder in Globus")):
            self._text(form, key, label); self._globus_rows.append(self.fields[key])
        self.include_existing = QCheckBox("Include existing project results on the first synchronization")
        form.addRow(self.include_existing)
        self.download = QCheckBox("Automatically download after Export")
        form.addRow(self.download)
        b = QPushButton("Open these locations in Globus")
        b.clicked.connect(self._file_manager); form.addRow(b); self._globus_rows.append(b)
        self.tabs.addTab(transfer, "2  Data transfer")

        resource = QWidget(); form = QFormLayout(resource)
        self.preset = QComboBox()
        self.preset.addItems(["Tutorial / first check — 4 cores, 16 GiB", "CPU processing — 32 cores, 128 GiB",
                              "Large-memory processing — 32 cores, 512 GiB", "Custom / saved settings"])
        form.addRow("Starting point", self.preset)
        self.partition = QComboBox(); self.partition.addItems(PARTITIONS)
        form.addRow("LRZ partition", self.partition)
        for key, label, maximum in (("cpus", "Total CPU cores", 112), ("memory_gib", "RAM (GiB)", 2900),
                                    ("hours", "Maximum hours", 240), ("section_concurrency", "Simultaneous sections", 112),
                                    ("workers", "Workers per section", 112)):
            w = QSpinBox(); w.setRange(1, maximum); self.fields[key] = w; form.addRow(label, w)
        note = QLabel("Section concurrency applies to tile matching, montage optimization/rendering, and mipmapping/thumbnails. "
                      "Stack-wide alignment remains coordinated. Simultaneous sections × workers must fit in Total CPU cores. "
                      "Start with one section and use measured RAM before increasing concurrency. These are CPU-only partitions.")
        note.setWordWrap(True); form.addRow(note)
        self.size_hint = QLabel(); self.size_hint.setWordWrap(True); form.addRow(self.size_hint)
        self.tabs.addTab(resource, "3  Computing power")

        advanced = QWidget(); form = QFormLayout(advanced)
        for key, label in (("host", "SSH host"), ("key_filename", "Optional private key"),
                           ("remote_project", "Remote work directory"), ("remote_tiles", "Remote raw image directory"),
                           ("remote_python", "Remote FEABAS Python"), ("remote_dl_python", "Optional CPU deep-learning Python"),
                           ("modules", "Environment modules"), ("globus_cli", "Optional globus.exe path"),
                           ("export_destination", "Default local export folder")):
            self._text(form, key, label)
        self.port = QSpinBox(); self.port.setRange(1, 65535); form.addRow("SSH port", self.port)
        b = QPushButton("Check FEABAS and remote paths")
        b.clicked.connect(self._check); form.addRow(b)
        b = QPushButton("Recover an unconfirmed Globus transfer")
        b.clicked.connect(self._recover_transfer); form.addRow(b)
        b = QPushButton("Retry a failed transfer")
        b.clicked.connect(self._retry_transfer); form.addRow(b)
        b = QPushButton("Download completed exports")
        b.clicked.connect(lambda: self.backend.download_exports()); form.addRow(b)
        self.tabs.addTab(advanced, "Advanced")

        self.output = QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMaximumHeight(160)
        outer.addWidget(self.output)
        self.status = QLabel(); self.status.setWordWrap(True); outer.addWidget(self.status)
        bottom = QHBoxLayout()
        refresh = QPushButton("Refresh jobs / transfers"); refresh.clicked.connect(self.backend.refresh); bottom.addWidget(refresh)
        self.leave = QPushButton("Leave cluster mode")
        self.leave.setToolTip("Compute on this PC again. Submitted LRZ jobs keep running.")
        self.leave.clicked.connect(self._leave); bottom.addWidget(self.leave)
        bottom.addStretch(1)
        self.save = QPushButton("Save cluster settings && use cluster"); self.save.setObjectName("Primary")
        self.save.clicked.connect(self._save); bottom.addWidget(self.save)
        close = QPushButton("Close"); close.clicked.connect(self.reject); bottom.addWidget(close)
        outer.addLayout(bottom)
        self._load()
        self.preset.currentIndexChanged.connect(self._preset)
        self.preset.activated.connect(self._preset)
        self.backend.changed.connect(self._status)
        self._status()

    def _text(self, form, key, label, placeholder=""):
        field = QLineEdit(); field.setPlaceholderText(placeholder)
        self.fields[key] = field; form.addRow(label, field)

    @staticmethod
    def open_url(url):
        QDesktopServices.openUrl(QUrl(url))

    def _load(self):
        p = self.backend.profile
        for key, widget in self.fields.items():
            if isinstance(widget, QSpinBox):
                widget.setValue(p[key])
            else:
                widget.setText(str(p.get(key, "")))
        self.partition.setCurrentText(p["partition"])
        current = tuple(p[k] for k in ("partition", "cpus", "memory_gib", "hours", "workers", "section_concurrency"))
        presets = [("serial_std", 4, 16, 1, 4, 1), ("cm4_tiny", 32, 128, 4, 16, 2), ("teramem_inter", 32, 512, 8, 16, 2)]
        self.preset.setCurrentIndex(presets.index(current) if current in presets else 3)
        self.port.setValue(p["port"])
        self.include_existing.setChecked(p["include_existing"])
        self.download.setChecked(p["download_after_export"])
        self.transfer.setCurrentIndex(max(0, self.transfer.findData(p.get("transfer", "ssh"))))
        self._transfer_changed()
        if not self.fields["destination_collection"].text():
            self.fields["destination_collection"].setText(LRZ_COLLECTION)
        if not self.fields["source_path"].text() and self.ctx.project.state.source.root_dir:
            try:
                self.fields["source_path"].setText(local_globus_path(Path(self.ctx.project.state.source.root_dir)))
            except ValueError:
                pass
        self.paths.setText("Work: " + p["remote_project"] + "\nImages: " + p["remote_tiles"] if p["remote_project"]
                           else "Choose a storage folder after signing in.")
        v = self.ctx.project.state.volume
        if v.tile_w and v.tile_h:
            per = v.tile_w * v.tile_h * max(1, v.grid_rows * v.grid_cols) * (2 if "16" in v.dtype else 1) / 2**30
            self.size_hint.setText(f"This project's raw pixels: about {per:.2f} GiB per section. This excludes working arrays "
                                   "and does not predict peak RAM. Job reports record observed process memory.")
        self.leave.setVisible(self.ctx.cluster_enabled)

    def _values(self):
        p = dict(self.backend.profile)
        p.update({k: w.value() if isinstance(w, QSpinBox) else w.text().strip() for k, w in self.fields.items()})
        p.update(partition=self.partition.currentText(), port=self.port.value(), include_existing=self.include_existing.isChecked(),
                 download_after_export=self.download.isChecked(), transfer=self.transfer.currentData())
        return p

    def _store(self):
        if self.backend.busy:
            raise RuntimeError("Wait for the current connection operation to finish.")
        p = self._values()
        if (self.backend.running or self.backend.transferring) and p != self.backend.profile:
            raise RuntimeError("Wait for the running job before changing its connection or paths.")
        old = self.backend.profile
        connection = ("host", "username", "port", "key_filename")
        if self.backend.client and any(p[k] != old[k] for k in connection):
            self.backend.client.close()
        if sync_scope(p) != sync_scope(old):
            self.backend.synced = {}
            write_json(self.backend.directory / "synchronized.json", {})
            if any(p[k] != old[k] for k in (*connection, "remote_project")):
                self.backend.snapshot = {}
                write_json(self.backend.directory / "remote-state.json", {})
        self.backend.profile = p
        save_profile(self.ctx.local_project, p)
        return p

    def _safe(self, action):
        try:
            action()
        except Exception as e:
            QMessageBox.warning(self, "Setup needs attention", str(e))

    def _transfer_changed(self, *_):
        globus = self.transfer.currentData() == "globus"
        for widget in self._globus_rows:
            self.transfer_form.setRowVisible(widget, globus)
        self.transfer_form.setRowVisible(self.ssh_note, not globus)

    def _discover(self):
        def start():
            self._store()
            source = self.ctx.local_project.state.source.root_dir
            def connected(_):
                def check():
                    info = self.backend.client.discover()
                    # Metadata only; on a large NAS dataset this walk is better off the GUI thread.
                    info["raw_bytes"] = raw_image_bytes(source) if source and Path(source).is_dir() else 0
                    return info
                self.backend._work("Checking your LRZ account and storage…", check, self._discovered)
            self.backend.connect(connected)
        self._safe(start)

    def _discovered(self, info):
        if self._closed:
            return
        self._discovery = info
        self.storage.clear()
        self.storage.addItems(info["directories"] + ([info["home"]] if info.get("home") else []))
        # With a placeholder text QComboBox leaves new items unselected (currentText() == "").
        self.storage.setCurrentIndex(0 if self.storage.count() else -1)
        self.output.setPlainText(info["storage"])
        message = storage_message(info)
        self.paths.setText(message)
        self.backend.status(message)
        self._storage_hint()

    def _estimate(self):
        raw = self._discovery.get("raw_bytes") or 0
        return raw, (estimate_project_bytes(raw, preprocessing_copies(self.ctx.local_project.state)) + ENVIRONMENT_BYTES
                     if raw else 0)

    def _storage_hint(self, *_):
        text, info = self.storage.currentText().strip(), self._discovery
        lines = []
        if text and info:
            if is_home(text, info):
                used, limit = info.get("home_used"), info.get("home_limit")
                quota = f"{used / 1e9:.1f} of {limit / 1e9:.0f} GB used" if limit is not None and used is not None else "quota unknown"
                lines.append(f"Home folder · {quota} · backed up nightly. Good for the tutorial and smaller projects; "
                             "large datasets belong in a DSS container.")
            else:
                lines.append("DSS container · compare the estimate below with its free space in the report "
                             "(DSS Container usage and limits).")
        raw, estimate = self._estimate()
        if raw:
            lines.append(f"This project: {raw / 1e9:.1f} GB of raw images → roughly {estimate / 1e9:.0f} GB at LRZ once every "
                         f"step has run ({STORAGE_FACTOR}× the raw data plus the FEABAS environment; archived results of "
                         "changed settings come on top).")
            if text and is_home(text, info):
                warning = fit_warning(estimate, info.get("home_used"), info.get("home_limit"))
                if warning:
                    lines.append("Warning: " + warning)
        self.storage_note.setText("\n".join(lines))

    def _choose_storage(self):
        def choose():
            info = self._discovery
            if not info:
                raise ValueError("Click '1. Sign in and find my storage' first, so Workbench knows your folders and quota.")
            home = (info.get("home") or "").rstrip("/")
            base = project_storage_path(self.storage.currentText(), home)
            in_home = bool(home) and is_home(base, info)
            if in_home and base == home:
                base += "/feabas-workbench"      # keep the home folder itself tidy
            raw, estimate = self._estimate()
            warning = fit_warning(estimate, info.get("home_used"), info.get("home_limit")) if in_home else ""
            if warning and QMessageBox.question(self, "Storage may become too small", warning + "\n\nUse this folder anyway?",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            name = re.sub(r"[^A-Za-z0-9_-]", "-", self.ctx.local_project.state.name)[:40] or "project"
            root = base + "/feabas-" + name + "-" + self.backend.profile["project_id"][:8]
            for key, value in {"remote_project": root + "/work", "remote_tiles": root + "/images",
                               "remote_python": base + "/feabas-env/bin/python", "destination_project": root + "/work",
                               "destination_tiles": root + "/images"}.items():
                self.fields[key].setText(value)
            # LRZ's Globus does not serve home folders; very large data should not depend on Workbench staying open.
            method = "globus" if not in_home and raw > GLOBUS_SUGGESTED_BYTES else "ssh"
            self.transfer.setCurrentIndex(self.transfer.findData(method))
            p = self._store()
            self.paths.setText("Work: " + p["remote_project"] + "\nImages: " + p["remote_tiles"]
                               + "\nFiles travel " + ("through Globus (tab 2)." if method == "globus" else "over SSH (tab 2)."))
            self.backend.connect(lambda _: self.backend._work("Checking remote folders…", lambda: self.backend.client.prepare_workspace(p)))
        self._safe(choose)

    def _install(self):
        def start():
            p = self._store()
            if not p["remote_python"]:
                raise ValueError("Choose the storage folder first (step 2); the environment is created next to it.")
            remote_path(p["remote_python"])
            self.backend.connect(lambda _: self.backend._work("Preparing the private FEABAS environment; this takes 5-15 minutes…",
                                  lambda: self.backend.client.install_environment(p["remote_python"], p["modules"])))
        self._safe(start)

    def _check(self):
        def start():
            p = self._store()
            self.backend.connect(lambda _: self.backend._work("Checking FEABAS and remote paths…",
                lambda: self.backend.client.check_environment(p["remote_python"], p["remote_project"], p["remote_tiles"], p["modules"])))
        self._safe(start)

    def _globus_login(self):
        self._safe(lambda: (self._store(), self.backend._work("Complete Globus sign in in your browser…", self.backend.globus().login)))

    def _find_pc(self):
        def found(identifier):
            self.fields["computer_collection"].setText(identifier)
            if not self.fields["source_collection"].text():
                self.fields["source_collection"].setText(identifier)
            self.backend.status("Computer collection found")
        self._safe(lambda: (self._store(), self.backend._work("Finding this PC's Globus collection…", self.backend.globus().local_collection, found)))

    def _file_manager(self):
        p = self._values()
        self.open_url("https://app.globus.org/file-manager?" + urlencode(dict(origin_id=p["source_collection"],
                       origin_path=p["source_path"], destination_id=p["destination_collection"], destination_path=p["destination_tiles"])))

    def _preset(self, index):
        if index == 3:
            return
        values = [("serial_std", 4, 16, 1, 4, 1), ("cm4_tiny", 32, 128, 4, 16, 2),
                  ("teramem_inter", 32, 512, 8, 16, 2)][index]
        self.partition.setCurrentText(values[0])
        for key, value in zip(("cpus", "memory_gib", "hours", "workers", "section_concurrency"), values[1:]):
            self.fields[key].setValue(value)

    def _recover_transfer(self):
        pending = [t for t in self.backend.transfers if t["state"] == "SUBMITTING" and not t["task_id"]]
        if not pending:
            QMessageBox.information(self, "Transfers", "No unconfirmed transfer. Failed transfers remain visible in Globus Activity.")
            return
        t = pending[0]
        task, ok = QInputDialog.getText(self, "Recover transfer", "Find this label in Globus Activity, then paste its task UUID:\n" + t.get("label", t["purpose"]))
        if ok and task.strip():
            def verified(info):
                if info.get("label") != t.get("label") or info.get("source_endpoint") != t["source"] or info.get("destination_endpoint") != t["target"]:
                    raise ValueError("That task does not match this transfer's label and collections.")
                t.update(task_id=task.strip(), state=info["status"])
                self.backend.persist()
                self.backend.status("Transfer recovered")
            self.backend._work("Verifying transfer identity…", lambda: self.backend.globus().task(task.strip()), verified)
        elif ok:
            if QMessageBox.question(self, "No transfer in Globus Activity?",
                    "Only confirm if you checked the exact label in Globus Activity and no matching transfer exists. "
                    "Mark this attempt as not submitted so it can be retried?", QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) == QMessageBox.Yes:
                t.update(state="FAILED", details="User confirmed no matching task exists in Globus Activity")
                self.backend.persist()
                self.backend.status("Transfer can now be retried")

    def _retry_transfer(self):
        if self.backend.busy:
            return
        # SSH copies never stay open: a new Sync continues where an interrupted one stopped.
        failed = [t for t in self.backend.transfers if t["state"] == "FAILED" and not t.get("replaced") and t.get("method") != "ssh"]
        if not failed:
            QMessageBox.information(self, "Transfers", "No failed transfer. Recover unconfirmed transfers first.")
            return
        labels = [t.get("label", t["purpose"]) for t in failed]
        label, ok = QInputDialog.getItem(self, "Retry transfer", "Choose the failed transfer", labels, 0, False)
        if not ok:
            return
        t = failed[labels.index(label)]
        def retry():
            meta = {k: v for k, v in t.items() if k in {"batch", "scope", "batch_size", "job_input_key", "export_run",
                    "local_destination", "preview_manifest", "previous_previews", "preview_run"}}
            if t.get("scope") and t["scope"] != sync_scope(self.backend.profile):
                raise ValueError("These folders have changed. Use Sync project & images for the new settings.")
            replacement = self.backend._transfer(t["source"], t["source_path"], t["target"], t["target_path"], t["purpose"],
                exclude=t.get("exclude", ()), recursive=t.get("recursive", True), fingerprint=t.get("fingerprint", ""), metadata=meta)
            t["replaced"] = replacement["label"]
            self.backend.persist()
            return "Transfer retry started; existing identical files are skipped"
        self.backend._work("Retrying transfer…", retry)

    def _save(self):
        def save():
            p = self._store()
            resources_for(p).validate()
            self.ctx.use_cluster(p)
            self.backend.status("Cluster mode enabled · sync images before running" if not self.backend.synced else "Cluster mode enabled")
            self.accept()
        self._safe(save)

    def _leave(self):
        leave = getattr(self.parent(), "_use_local", None)
        if leave:
            leave()
        else:
            self._safe(self.ctx.use_local)
        if not self.ctx.cluster_enabled:
            self.reject()

    def _status(self):
        self.status.setText(self.backend.message)
        self.save.setEnabled(not self.backend.busy and not self.backend.running)
        self.leave.setVisible(self.ctx.cluster_enabled)
        rows = [f"Job {j['job_id'] or '?'}: {j['state']}" for j in self.backend.journal[-5:]]
        rows += [f"{t['purpose']}: {t['state']} {t.get('details', '')}" for t in self.backend.transfers[-5:]]
        if rows:
            self.output.setPlainText("\n".join(rows))

    def shutdown(self):
        # The backend belongs to AppContext; closing settings must keep jobs and
        # transfers alive and must not discard the authenticated SSH connection.
        self._closed = True
        try:
            self.backend.changed.disconnect(self._status)
        except RuntimeError:
            pass
