"""
Tree editor for one FEABAS config document: merged view, overridden keys in
blue, hints from the default file's comments, reset per key.
"""

from __future__ import annotations

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.configs import ConfigDoc, parse_value, flatten, get_dotted
from .. import theme


class ConfigEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: ConfigDoc | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search keys")
        self.search.setMaximumWidth(240)
        self.status = QLabel("")
        self.status.setObjectName("Hint")
        reset_all = QPushButton("Reset all to defaults")
        reset_all.setObjectName("Flat")
        top.addWidget(self.search)
        top.addWidget(self.status, 1)
        top.addWidget(reset_all)
        lay.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["setting", "value", "default"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.header().setStretchLastSection(True)
        lay.addWidget(self.tree, 1)
        self.hint = QLabel("")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(36)
        lay.addWidget(self.hint)
        self.tree.currentItemChanged.connect(self._show_hint)
        self.search.textChanged.connect(self._filter)
        reset_all.clicked.connect(self._reset_all)
        self._building = False

    def set_doc(self, doc: ConfigDoc | None) -> None:
        self.doc = doc
        self.rebuild()

    def rebuild(self) -> None:
        self._building = True
        self.tree.clear()
        if self.doc is None:
            self._building = False
            return
        merged = self.doc.merged
        self._add_children(self.tree.invisibleRootItem(), merged, "")
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.setColumnWidth(0, max(220, self.tree.columnWidth(0)))
        self.tree.setColumnWidth(1, 220)
        n_over = len(flatten(self.doc.overrides))
        self.status.setText(f"{n_over} overridden setting(s)" if n_over else "all defaults")
        self._building = False

    def _add_children(self, parent: QTreeWidgetItem, data: dict, prefix: str) -> None:
        for k, v in data.items():
            dotted = f"{prefix}.{k}" if prefix else str(k)
            it = QTreeWidgetItem(parent)
            it.setText(0, str(k))
            it.setData(0, Qt.UserRole, dotted)
            if isinstance(v, dict):
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._add_children(it, v, dotted)
            else:
                it.setText(1, self._fmt(v))
                it.setText(2, self._fmt(get_dotted(self.doc.defaults, dotted)))
                it.setFlags(it.flags() | Qt.ItemIsEditable)
                it.setForeground(2, QBrush(QColor(theme.MUTED)))
                self._style(it, dotted)
            h = self.doc.hint(dotted)
            if h:
                it.setToolTip(0, h)
                it.setToolTip(1, h)

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, dict)):
            return yaml.safe_dump(v, default_flow_style=True).strip()
        return str(v)

    def _style(self, it: QTreeWidgetItem, dotted: str) -> None:
        over = self.doc.is_overridden(dotted)
        col = QColor(theme.OVERRIDE) if over else QColor(theme.TEXT)
        it.setForeground(0, QBrush(col))
        it.setForeground(1, QBrush(col))
        f = it.font(1)
        f.setBold(over)
        it.setFont(1, f)
        it.setFont(0, f)

    def _item_changed(self, it: QTreeWidgetItem, col: int) -> None:
        if self._building or col != 1 or self.doc is None:
            return
        dotted = it.data(0, Qt.UserRole)
        current = get_dotted(self.doc.merged, dotted)
        try:
            val = parse_value(it.text(1), current)
        except Exception as e:  # noqa: BLE001
            self.hint.setText(f"could not parse value: {e}")
            self._building = True
            it.setText(1, self._fmt(current))
            self._building = False
            return
        self.doc.set(dotted, val)
        self._building = True
        it.setText(1, self._fmt(get_dotted(self.doc.merged, dotted)))
        self._style(it, dotted)
        self._building = False
        n_over = len(flatten(self.doc.overrides))
        self.status.setText(f"{n_over} overridden setting(s)" if n_over else "all defaults")
        self.changed.emit()

    def _menu(self, pos) -> None:
        it = self.tree.itemAt(pos)
        if it is None or self.doc is None:
            return
        dotted = it.data(0, Qt.UserRole)
        m = QMenu(self)
        a = m.addAction("Reset to default")
        a.triggered.connect(lambda: self._reset(it, dotted))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _reset(self, it: QTreeWidgetItem, dotted: str) -> None:
        self.doc.reset(dotted)
        self.rebuild()
        self.changed.emit()

    def _reset_all(self) -> None:
        if self.doc is None:
            return
        self.doc.overrides.clear()
        self.rebuild()
        self.changed.emit()

    def _show_hint(self, cur: QTreeWidgetItem | None, _prev) -> None:
        if cur is None or self.doc is None:
            self.hint.setText("")
            return
        dotted = cur.data(0, Qt.UserRole)
        h = self.doc.hint(dotted)
        self.hint.setText(f"{dotted}: {h}" if h else dotted)

    def _filter(self, text: str) -> None:
        text = text.strip().lower()

        def visit(it: QTreeWidgetItem) -> bool:
            dotted = (it.data(0, Qt.UserRole) or "").lower()
            match = (not text) or (text in dotted)
            child_match = False
            for i in range(it.childCount()):
                if visit(it.child(i)):
                    child_match = True
            show = match or child_match
            it.setHidden(not show)
            return show

        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            visit(root.child(i))

    def set_value(self, dotted: str, value) -> None:
        """Programmatic edit (used by the quick-settings forms)."""
        if self.doc is None:
            return
        self.doc.set(dotted, value)
        self.rebuild()
        self.changed.emit()
