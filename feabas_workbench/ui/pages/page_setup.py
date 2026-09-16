"""Window 0: environments, GPU, external viewers, compute settings."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QPlainTextEdit)

from ...core import envs as E
from ...core.jobs import JobSpec
from ..widgets import PathPicker, card, hint, form_row, spin, combo
from .base import Page


class _Discover(QObject):
    done = Signal(object, object)   # list[ProbeResult], GpuInfo

    def run(self) -> None:
        try:
            res = E.discover_environments()
        except Exception:  # noqa: BLE001
            res = []
        self.done.emit(res, E.detect_gpu())


class SetupPage(Page):
    title = "Setup"
    subtitle = ("Which Python environments run FEABAS and the deep-learning models, GPU status, and where Fiji "
                "and VASTlite live. Everything here is remembered globally; projects can override the interpreters.")
    key = "setup"
    needs_project = False

    def build(self) -> None:
        s = self.ctx.settings
        f, lay = card("Environments")
        self.feabas_py = PathPicker("file", "python.exe of an environment with feabas installed", "Python (python*)")
        self.dl_py = PathPicker("file", "python.exe of an environment with torch + careamics + ultralytics + smp", "Python (python*)")
        self.feabas_py.setText(s.feabas_python)
        self.dl_py.setText(s.dl_python)
        lay.addWidget(form_row("FEABAS Python", self.feabas_py, "Runs the stitching/alignment steps."))
        lay.addWidget(form_row("Deep-learning Python", self.dl_py, "Runs fold detection, YOLO-seg and CAREamics denoising."))
        r = QHBoxLayout()
        self.detect_btn = QPushButton("Detect environments on this machine")
        self.probe_btn = QPushButton("Check selected")
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("Primary")
        r.addWidget(self.detect_btn); r.addWidget(self.probe_btn); r.addStretch(1); r.addWidget(self.save_btn)
        lay.addLayout(r)
        self.env_list = QPlainTextEdit()
        self.env_list.setReadOnly(True)
        self.env_list.setMaximumHeight(170)
        self.env_list.setPlaceholderText("detected environments appear here")
        lay.addWidget(self.env_list)
        self.env_pick = QComboBox()
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Use detected:")); r2.addWidget(self.env_pick, 1)
        use_f = QPushButton("as FEABAS"); use_d = QPushButton("as deep-learning")
        r2.addWidget(use_f); r2.addWidget(use_d)
        lay.addLayout(r2)
        self.detect_btn.clicked.connect(self._detect)
        self.probe_btn.clicked.connect(self._probe_selected)
        self.save_btn.clicked.connect(self._save)
        use_f.clicked.connect(lambda: self._use(self.feabas_py))
        use_d.clicked.connect(lambda: self._use(self.dl_py))
        self.body.addWidget(f)

        f, lay = card("Install environments")
        lay.addWidget(hint("Creates conda environments with conda-forge Python and pip-installs the packages: "
                           "'fw-feabas' (feabas 3.0.5, tensorstore) and 'fw-dl' (PyTorch for your driver, "
                           "segmentation-models-pytorch, ultralytics, careamics 0.3.2). Needs internet; several GB for fw-dl."))
        r = QHBoxLayout()
        self.conda = PathPicker("file", "conda / mamba / micromamba executable")
        c = E.find_conda()
        self.conda.setText(s.conda_exe or (str(c) if c else ""))
        r.addWidget(QLabel("conda")); r.addWidget(self.conda, 1)
        self.mm_btn = QPushButton("Download micromamba")
        r.addWidget(self.mm_btn)
        lay.addLayout(r)
        r = QHBoxLayout()
        self.torch_index = combo([("auto from driver", ""), ("CUDA 12.6", "https://download.pytorch.org/whl/cu126"),
                                  ("CUDA 11.8", "https://download.pytorch.org/whl/cu118"), ("CPU only", "https://download.pytorch.org/whl/cpu")], "")
        r.addWidget(QLabel("PyTorch build")); r.addWidget(self.torch_index)
        self.inst_feabas = QPushButton("Install fw-feabas")
        self.inst_dl = QPushButton("Install fw-dl")
        r.addWidget(self.inst_feabas); r.addWidget(self.inst_dl); r.addStretch(1)
        lay.addLayout(r)
        self.mm_btn.clicked.connect(self._micromamba)
        self.inst_feabas.clicked.connect(lambda: self._install("feabas"))
        self.inst_dl.clicked.connect(lambda: self._install("dl"))
        self.body.addWidget(f)

        f, lay = card("Hardware")
        import psutil
        cores = psutil.cpu_count(logical=False) or 0
        threads = psutil.cpu_count(logical=True) or 0
        ram = psutil.virtual_memory().total / 1024 ** 3
        cpu = QLabel(f"CPU: {cores} physical cores / {threads} threads, RAM {ram:.0f} GB. FEABAS uses one process per "
                     f"worker; set each step's 'workers' to about the number of physical cores (RAM permitting).")
        cpu.setObjectName("Hint"); cpu.setWordWrap(True)
        lay.addWidget(cpu)
        self.gpu_label = QLabel("GPU: not checked yet")
        self.gpu_label.setObjectName("Mono")
        lay.addWidget(self.gpu_label)
        lay.addWidget(hint("PyTorch wheels bring their own CUDA runtime; only the NVIDIA driver must be new enough "
                           "(≥ 528 for CUDA 12.6 builds). FEABAS itself does not use the GPU."))
        self.body.addWidget(f)

        f, lay = card("External viewers")
        self.fiji = PathPicker("file", "ImageJ-win64.exe / ImageJ-linux64")
        self.vast = PathPicker("file", "VAST_Lite.exe")
        self.fiji.setText(s.fiji_path or E.find_fiji())
        self.vast.setText(s.vast_path or E.find_vast())
        lay.addWidget(form_row("Fiji", self.fiji))
        lay.addWidget(form_row("VASTlite", self.vast))
        self.body.addWidget(f)

        f, lay = card("Interface")
        self.show_struct = QCheckBox("show the experimental 'Structure-guided' tab on the Alignment page")
        self.show_struct.setChecked(bool(getattr(s, "show_structure_tab", False)))
        self.show_struct.setToolTip("Alignment driven by biological structures found with a YOLO-seg model (nuclei, "
                                    "mitochondria, vessels ...) instead of anonymous texture features. Experimental "
                                    "and not needed for the normal FEABAS workflow, so it is hidden by default. "
                                    "Takes effect immediately and is remembered.")
        lay.addWidget(self.show_struct)
        self.show_struct.toggled.connect(self._toggle_structure_tab)
        self.body.addWidget(f)

        f, lay = card("Compute settings (this project)")
        self.cpu = spin(0, 512, 0); self.cpu.setSpecialValueText("all physical cores")
        self.framework = combo([("process (multiprocessing)", "process"), ("thread", "thread"), ("dask", "dask")], "process")
        self.loglevel = combo(["INFO", "DEBUG", "WARNING"], "INFO")
        lay.addWidget(form_row("CPU budget", self.cpu, "Estimated target for FEABAS; per-step num_workers are set in each step's settings."))
        lay.addWidget(form_row("Parallel framework", self.framework))
        lay.addWidget(form_row("Log file level", self.loglevel))
        self.apply_btn = QPushButton("Apply to project")
        lay.addWidget(self.apply_btn)
        self.apply_btn.clicked.connect(self._apply_general)
        self.body.addWidget(f)

        self._probes: list[E.ProbeResult] = []
        self._thread = None

    # ------------------------------------------------------------------
    def on_shown(self) -> None:
        if not self._probes and not self.ctx.settings.feabas_python:
            self._detect()

    def shutdown(self) -> None:
        # the probe thread may be blocked in `conda env list` or a probe interpreter; killing those
        # makes it return, so the QThread can end before Qt tears it down (see envs._run_probe)
        E.cancel_probes()
        super().shutdown()

    def _detect(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return          # a probe is already running (on_shown and the button both start one)
        E.reset_probe_cancel()
        self.detect_btn.setEnabled(False)
        self.env_list.setPlainText("probing environments (this takes a moment per environment)…")
        self._worker = _Discover()
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._detected)
        self._thread.start()

    def _detected(self, probes, gpu) -> None:
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(2000)
        self.detect_btn.setEnabled(True)
        self._probes = [p for p in probes if p.ok]
        lines = []
        self.env_pick.clear()
        for p in self._probes:
            tag = []
            if p.is_feabas_env:
                tag.append("FEABAS")
            if p.is_dl_env:
                tag.append("DL")
            lines.append(f"[{'/'.join(tag) or '-'}] {p.python}\n      {p.describe()}")
            self.env_pick.addItem(f"{Path(p.python).parent.name}: {p.describe()[:90]}", p.python)
        self.env_list.setPlainText("\n".join(lines) or "no usable environments found")
        self._show_gpu(gpu)
        # auto-fill empty fields
        if not self.feabas_py.text():
            for p in self._probes:
                if p.is_feabas_env:
                    self.feabas_py.setText(p.python)
                    break
        if not self.dl_py.text():
            best = None
            for p in self._probes:
                if p.is_dl_env and p.cuda and p.has("careamics", "ultralytics", "segmentation_models_pytorch"):
                    best = p; break
            if best is None:
                for p in self._probes:
                    if p.is_dl_env and p.cuda:
                        best = p; break
            if best:
                self.dl_py.setText(best.python)
        self.info(f"detected {len(self._probes)} Python environments")

    def _show_gpu(self, gpu: E.GpuInfo) -> None:
        if gpu.found:
            self.gpu_label.setText(f"{gpu.name}, driver {gpu.driver}, {gpu.memory_mb / 1024:.0f} GB → recommended PyTorch build: {gpu.torch_index().rsplit('/', 1)[-1]}")
        else:
            self.gpu_label.setText("no NVIDIA GPU / nvidia-smi found → CPU-only deep learning (slow)")

    def _use(self, target: PathPicker) -> None:
        p = self.env_pick.currentData()
        if p:
            target.setText(p)

    def _probe_selected(self) -> None:
        for label, picker in (("FEABAS", self.feabas_py), ("deep-learning", self.dl_py)):
            if picker.text():
                r = E.probe_python(picker.text())
                self.info(f"{label} env: {r.describe()}")
                if label == "FEABAS" and r.ok and not r.is_feabas_env:
                    self.warn("the FEABAS environment does not import 'feabas'")
                if label == "deep-learning" and r.ok and not r.is_dl_env:
                    self.warn("the deep-learning environment does not import 'torch'")
        self._show_gpu(E.detect_gpu())

    def _save(self) -> None:
        s = self.ctx.settings
        s.feabas_python = self.feabas_py.text()
        s.dl_python = self.dl_py.text()
        s.fiji_path = self.fiji.text()
        s.vast_path = self.vast.text()
        s.conda_exe = self.conda.text()
        s.save()
        self.info("settings saved")

    def _toggle_structure_tab(self, on: bool) -> None:
        self.ctx.settings.show_structure_tab = bool(on)
        self.ctx.settings.save()
        w = self.window()
        if hasattr(w, "apply_interface_settings"):
            w.apply_interface_settings()

    def _micromamba(self) -> None:
        try:
            exe = E.download_micromamba(log=self.info)
            self.conda.setText(str(exe))
            self.info(f"micromamba ready at {exe}")
        except Exception as e:  # noqa: BLE001
            self.error(f"micromamba download failed: {e}")

    def _install(self, kind: str) -> None:
        conda = self.conda.text()
        if not conda or not Path(conda).is_file():
            QMessageBox.information(self, "conda", "Choose a conda/mamba/micromamba executable or download micromamba first.")
            return
        gpu = E.detect_gpu()
        idx = self.torch_index.currentData() or gpu.torch_index()
        name = "fw-feabas" if kind == "feabas" else "fw-dl"
        pyver = "3.12" if kind == "feabas" else "3.11"
        plan = E.InstallPlan(kind, name, pyver, Path(conda), idx if kind == "dl" else "")
        if not self.confirm("Install", f"Create conda environment '{name}' (Python {pyver}) and install the "
                                       f"{'FEABAS' if kind == 'feabas' else 'deep-learning (' + idx.rsplit('/', 1)[-1] + ')'} packages?\n"
                                       f"This downloads a lot and can take 10–30 minutes."):
            return
        specs = []
        for cmd in plan.commands():
            specs.append(JobSpec(f"create env {name}", cmd, cwd=Path.home(), kind="shell"))
        # the pip installs run inside the new environment through `<manager> run -n <env>`, so
        # neither its location nor a helper interpreter is needed - the frozen exe has none to offer
        pips = plan.pip_commands_in_env()
        for i, cmd in enumerate(pips, 1):
            specs.append(JobSpec(f"pip install into {name} ({i}/{len(pips)})", cmd, cwd=Path.home(), kind="shell"))
        self._install_kind, self._install_plan, self._install_last = kind, plan, specs[-1].name
        self.ctx.jobs.job_finished.connect(self._install_done)
        self.submit(specs)

    def _install_done(self, res) -> None:
        if res.spec.name != getattr(self, "_install_last", None):
            return
        try:
            self.ctx.jobs.job_finished.disconnect(self._install_done)
        except (RuntimeError, TypeError):
            pass
        if not res.ok:
            return
        plan = self._install_plan
        d = E.env_dir_for(plan.conda, plan.env_name)
        py = E.env_python(d) if d else None
        if py is None:
            self.warn(f"environment {plan.env_name} was installed but its python.exe was not found; use 'Detect environments'")
            return
        (self.feabas_py if self._install_kind == "feabas" else self.dl_py).setText(str(py))
        self._save()
        self.info(f"environment ready: {py}")

    def _apply_general(self) -> None:
        if not self.require_project():
            return
        p = self.project
        p.write_general_config(cpu_budget=self.cpu.value() or None, parallel_framework=self.framework.currentData(),
                               logfile_level=self.loglevel.currentData())
        self.info("general_configs.yaml updated")

    def on_project_changed(self, project) -> None:
        if project is None:
            return
        import yaml
        try:
            g = yaml.safe_load(project.general_config_path().read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            g = {}
        self.cpu.setValue(int(g.get("cpu_budget") or 0))
        self.framework.setCurrentIndex(max(0, self.framework.findData(g.get("parallel_framework", "process"))))
        self.loglevel.setCurrentIndex(max(0, self.loglevel.findData(g.get("logfile_level", "INFO"))))
