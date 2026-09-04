"""Global search (Ctrl+K) - jump to a Task, Work, or run by typing.

Searches Task/Work names and, for runs, server/model/dataset/paths/notes/
tags/failure_reason/exec_command (via Database.search_runs). Selecting a
result calls back into whoever opened the dialog with enough information
to navigate there; this widget doesn't know about the rest of the app.
"""
from __future__ import annotations

from typing import Any, Callable

from .. import theme
from ..db import Database
from ..qt import Qt, QtCore, QtWidgets

MAX_RESULTS = 60


class GlobalSearchDialog(QtWidgets.QDialog):
    def __init__(
        self,
        db: Database,
        on_select: Callable[[dict[str, Any]], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self._on_select = on_select

        self.setWindowTitle("Search")
        self.setModal(True)
        self.setMinimumSize(560, 420)

        self.query_edit = QtWidgets.QLineEdit(self)
        self.query_edit.setPlaceholderText(
            "Search Tasks, Works, and runs (model, path, notes, tags...)"
        )
        self.query_edit.textChanged.connect(self._refresh_results)

        self.list = QtWidgets.QListWidget(self)
        self.list.setAlternatingRowColors(True)
        self.list.itemActivated.connect(self._activate)

        hint = QtWidgets.QLabel(
            f"<span style='color:{theme.color('text.muted')}'>"
            "↑↓ to move · Enter to open · Esc to close</span>",
            self,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.query_edit)
        layout.addWidget(self.list, 1)
        layout.addWidget(hint)

        self.query_edit.installEventFilter(self)
        self._refresh_results("")

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.query_edit and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self.list.setFocus()
                self.list.keyPressEvent(event)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                item = self.list.currentItem() or self.list.item(0)
                if item is not None:
                    self._activate(item)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    # -- results ---------------------------------------------------------------
    def _refresh_results(self, text: str) -> None:
        self.list.clear()
        needle = text.strip().lower()

        for task in self.db.list_tasks():
            if needle and needle not in task["name"].lower():
                continue
            self._add_item(f"📁 {task['name']}", "Task", {"kind": "task", "task_id": task["id"]})

        for task in self.db.list_tasks():
            for work in self.db.list_works(task["id"]):
                haystack = f"{task['name']} {work['name']}".lower()
                if needle and needle not in haystack:
                    continue
                self._add_item(
                    f"📂 {task['name']} ▸ {work['name']}",
                    "Work",
                    {"kind": "work", "task_id": task["id"], "work_id": work["id"]},
                )

        if needle:
            for row in self.db.search_runs(needle, limit=MAX_RESULTS):
                mode = "Train" if row["kind"] == "train" else "Evaluation"
                label = (
                    f"#{row['id']}  {row.get('model') or '-'}  ·  "
                    f"{row['task_name']} ▸ {row['work_name']}"
                )
                self._add_item(label, mode, {
                    "kind": "run",
                    "run_kind": row["kind"],
                    "task_id": row["task_id"],
                    "work_id": row["work_id"],
                    "run_id": row["id"],
                })

        if self.list.count():
            self.list.setCurrentRow(0)

    def _add_item(self, text: str, tag: str, payload: dict[str, Any]) -> None:
        item = QtWidgets.QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, payload)
        item.setToolTip(tag)
        self.list.addItem(item)

    def _activate(self, item: QtWidgets.QListWidgetItem) -> None:
        payload = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(payload, dict):
            self._on_select(payload)
        self.accept()
