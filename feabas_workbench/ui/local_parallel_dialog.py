"""Per-stage workstation resources, separate from scientific settings."""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QLabel, QMessageBox, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..core.configs import load_yaml
from ..core.local_parallel import MIPMAP_STEPS, SECTION_STEPS, hardware, hardware_threads, mipmap_plan, options, plan
from ..core.steps import STEPS_BY_KEY


class LocalParallelDialog(QDialog):
    def __init__(self, ctx, keys, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Local parallelism — this project")
        self.resize(720, 660)
        layout = QVBoxLayout(self)
        note = QLabel("Choose workers within each section, sections processed at once, or both; the mipmap "
                      "steps choose for themselves unless you pick a mode. Settings apply to this PC; cluster "
                      "resources are configured separately.")
        note.setWordWrap(True); layout.addWidget(note)
        physical, free = hardware()
        threads = hardware_threads()
        cpus = f"{physical} physical cores" + (f" ({threads} logical CPUs)" if threads > physical else "")
        layout.addWidget(QLabel(f"Available now: {cpus}; {free:.1f} GiB free RAM."))
        form = QFormLayout()
        self.cpu = QSpinBox(); self.cpu.setRange(0, max(1024, threads)); self.cpu.setSpecialValueText("All physical cores")
        self.cpu.setToolTip("Leave at 'All physical cores' (FEABAS's own default), or set a number: a set budget may "
                            "also use hyper-threads, up to the logical CPUs.")
        self.cpu.setValue(int(ctx.project.state.local_execution.get("cpu_budget",
            load_yaml(ctx.project.general_config_path()).get("cpu_budget")) or 0))
        self.ram = QDoubleSpinBox(); self.ram.setRange(0, 16384); self.ram.setSuffix(" GiB"); self.ram.setSpecialValueText("Auto: 80% of free RAM")
        self.ram.setValue(float(ctx.project.state.local_execution.get("ram_budget_gib", 0)))
        form.addRow("Total CPU budget", self.cpu); form.addRow("RAM planning budget", self.ram); layout.addLayout(form)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); rows = QVBoxLayout(content); self.rows = {}
        for key in keys:
            if key not in SECTION_STEPS:
                continue
            setting = options(ctx.project, key)
            box = QGroupBox(STEPS_BY_KEY[key].label); fields = QFormLayout(box)
            mode = QComboBox()
            if key in MIPMAP_STEPS:
                mode.addItem("Automatic (recommended)", "auto")
            mode.addItem("Existing FEABAS settings", "existing")
            if SECTION_STEPS[key][2]:
                mode.addItem("Within each section only", "within")
            mode.addItem("Across sections only", "across")
            if SECTION_STEPS[key][2]:
                mode.addItem("Both: within and across sections", "both")
            mode.setCurrentIndex(max(0, mode.findData(setting["mode"])))
            sections = QSpinBox(); sections.setRange(1, 1024); sections.setValue(setting["sections"])
            workers = QSpinBox(); workers.setRange(1, 1024); workers.setValue(setting["workers"])
            measured = QDoubleSpinBox(); measured.setRange(0, 16384); measured.setDecimals(2); measured.setSuffix(" GiB")
            measured.setSpecialValueText("Estimate from image dimensions"); measured.setValue(setting["measured_gib"])
            measured.setToolTip("Enter measured peak RAM for one section at this worker count, including a safety margin.")
            estimate = QLabel(); estimate.setWordWrap(True)
            fields.addRow("Parallelism", mode); fields.addRow("Sections at once (maximum)", sections)
            fields.addRow("Workers per section", workers); fields.addRow("RAM per section", measured); fields.addRow(estimate)
            self.rows[key] = mode, sections, workers, measured, estimate
            for widget in (mode, sections, workers, measured):
                (widget.currentIndexChanged if widget is mode else widget.valueChanged).connect(self.refresh)
            rows.addWidget(box)
        rows.addStretch(1); scroll.setWidget(content); layout.addWidget(scroll, 1)
        caveat = QLabel("CPU limit: sections × workers ≤ total CPU budget (the physical cores unless you set a "
            "budget; a set budget may use hyper-threads too). RAM is an estimate, not a guarantee; "
            "start with one section and use its logged peak to refine it. Whole-stack optimization and "
            "shared TensorStore volume writes keep their existing execution.")
        caveat.setWordWrap(True); layout.addWidget(caveat)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.cpu.valueChanged.connect(self.refresh); self.ram.valueChanged.connect(self.refresh)
        self.refresh()

    def refresh(self):
        import copy
        project = copy.copy(self.ctx.project)
        project.state = copy.deepcopy(project.state)
        project.state.local_execution["ram_budget_gib"] = self.ram.value()
        physical, free = hardware()
        threads = hardware_threads()
        for key, (mode, sections, workers, measured, label) in self.rows.items():
            value = mode.currentData()
            sections.setEnabled(value in {"across", "both"})
            workers.setEnabled(value in {"within", "both"} and SECTION_STEPS[key][2])
            measured.setEnabled(value not in {"existing", "auto"})
            if value == "existing":
                label.setText("Unchanged: uses the step's existing FEABAS worker settings.")
                continue
            if value == "auto":
                try:
                    auto = mipmap_plan(project, key, configs=self.ctx.configs, available=(physical, free),
                                       threads=threads, cpu_override=self.cpu.value())
                    if auto is None:
                        text = "Nothing to tune: the stitched sections are not PNG tiles, so FEABAS's settings apply."
                    elif auto.sections:
                        text = auto.note + " Decided again at each run, for the sections it covers."
                    else:
                        text = ("Sections in parallel when there are enough of them, else FEABAS's workers within "
                                "each section; a larger read cache either way.")
                except Exception as e:  # noqa: BLE001 - a description only
                    text = f"Automatic: {e}"
                label.setText(text)
                continue
            settings = dict(mode=value, sections=sections.value(), workers=workers.value(), measured_gib=measured.value())
            allocation = plan(project, key, settings, available=(physical, free), cpu_override=self.cpu.value(),
                              threads=threads)
            memory = f"~{allocation.per_section_gib:.2f} GiB/section" if allocation.per_section_gib else "RAM unknown"
            label.setText(f"Now: {allocation.sections} sections × {allocation.workers} workers = "
                          f"{allocation.sections * allocation.workers} CPUs; {memory}. {allocation.note}")

    def save(self):
        if self.ctx.jobs.running:
            QMessageBox.information(self, "Job running", "Wait for this job to finish before changing its resources.")
            return
        project = self.ctx.project
        project.state.local_execution["cpu_budget"] = self.cpu.value()
        project.state.local_execution["ram_budget_gib"] = self.ram.value()
        settings = project.state.local_execution.setdefault("steps", {})
        for key, (mode, sections, workers, measured, _) in self.rows.items():
            settings[key] = dict(mode=mode.currentData(), sections=sections.value(), workers=workers.value(),
                                 measured_gib=measured.value())
        project.save(); self.ctx.state_changed.emit(); self.accept()
