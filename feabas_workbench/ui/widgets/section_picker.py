from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)


class SectionPicker(QWidget):
    """Checkable list of section names with quick range selection."""

    selection_changed = Signal()
    current_changed = Signal(str)

    def __init__(self, parent=None, checkable: bool = True):
        super().__init__(parent)
        self.checkable = checkable
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.info = QLabel("0 sections")
        self.info.setObjectName("Hint")
        top.addWidget(self.info, 1)
        if checkable:
            self.all_btn = QPushButton("all"); self.all_btn.setObjectName("Flat")
            self.none_btn = QPushButton("none"); self.none_btn.setObjectName("Flat")
            self.sel_btn = QPushButton("✓ highlighted"); self.sel_btn.setObjectName("Flat")
            self.sel_btn.setToolTip("Check the rows highlighted in the list. Click, then shift-click for a range "
                                    "or ctrl-click to add single sections; the space bar toggles them too.")
            self.unsel_btn = QPushButton("✗ highlighted"); self.unsel_btn.setObjectName("Flat")
            self.every = QSpinBox(); self.every.setRange(1, 999); self.every.setPrefix("every "); self.every.setMaximumWidth(90)
            self.every_btn = QPushButton("apply"); self.every_btn.setObjectName("Flat")
            top.addWidget(self.all_btn); top.addWidget(self.none_btn)
            top.addWidget(self.sel_btn); top.addWidget(self.unsel_btn)
            top.addWidget(self.every); top.addWidget(self.every_btn)
            self.all_btn.clicked.connect(lambda: self.set_all(True))
            self.none_btn.clicked.connect(lambda: self.set_all(False))
            self.sel_btn.clicked.connect(lambda: self.check_selected(True))
            self.unsel_btn.clicked.connect(lambda: self.check_selected(False))
            self.every_btn.clicked.connect(self._apply_every)
        lay.addLayout(top)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.list, 1)
        self.list.itemChanged.connect(lambda _i: (self._update_info(), self.selection_changed.emit()))
        self.list.currentItemChanged.connect(lambda cur, _p: self.current_changed.emit(cur.text() if cur else ""))

    def set_sections(self, names: list[str], status: dict[str, str] | None = None) -> None:
        cur = self.current()
        # a refresh (after every computation) must not throw the user's selection away
        known = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        was_checked = set(self.checked()) if self.checkable and known else None
        self.list.blockSignals(True)
        self.list.clear()
        for n in names:
            it = QListWidgetItem(n)
            if self.checkable:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if was_checked is None or n in was_checked or n not in known
                                 else Qt.Unchecked)
            if status and n in status:
                it.setToolTip(status[n])
                it.setText(f"{n}   {status[n]}")
                it.setData(Qt.UserRole, n)
            else:
                it.setData(Qt.UserRole, n)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self._update_info()
        if cur:
            self.set_current(cur)
        elif names:
            self.list.setCurrentRow(0)

    def names(self) -> list[str]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())]

    def checked(self) -> list[str]:
        if not self.checkable:
            return self.names()
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def checked_indices(self) -> list[int]:
        return [i for i in range(self.list.count()) if self.list.item(i).checkState() == Qt.Checked]

    def set_all(self, on: bool) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_info()
        self.selection_changed.emit()

    def set_checked(self, names: set[str]) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in names else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_info()
        self.selection_changed.emit()

    def check_selected(self, on: bool) -> None:
        """Check (or uncheck) every highlighted row: click, shift-click, ctrl-click as usual."""
        rows = self.list.selectedItems()
        if not rows:
            return
        self.list.blockSignals(True)
        for it in rows:
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_info()
        self.selection_changed.emit()

    def keyPressEvent(self, event) -> None:
        if self.checkable and event.key() == Qt.Key_Space and self.list.selectedItems():
            on = any(it.checkState() != Qt.Checked for it in self.list.selectedItems())
            self.check_selected(on)
            return
        super().keyPressEvent(event)

    def _update_info(self) -> None:
        n = self.list.count()
        if not self.checkable:
            self.info.setText(f"{n} sections")
            return
        self.info.setText(f"{len(self.checked())} of {n} sections")

    def _apply_every(self) -> None:
        k = self.every.value()
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if i % k == 0 else Qt.Unchecked)
        self.list.blockSignals(False)
        self._update_info()
        self.selection_changed.emit()

    def current(self) -> str:
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else ""

    def set_current(self, name: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == name:
                self.list.setCurrentRow(i)
                return

    def step(self, delta: int) -> None:
        row = self.list.currentRow()
        n = self.list.count()
        if n:
            self.list.setCurrentRow(max(0, min(n - 1, row + delta)))
