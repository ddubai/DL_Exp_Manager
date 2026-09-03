"""좌측 네비게이션 - Level 1(DL Task) / Level 2(Work ID) 드릴다운 트리."""
from __future__ import annotations

from typing import Any

from .. import editing
from ..config_store import OptionsConfig
from ..db import Database
from ..qt import Qt, QtWidgets, Signal

ROLE_KIND = int(Qt.ItemDataRole.UserRole)
ROLE_ID = int(Qt.ItemDataRole.UserRole) + 1


class NavigationPanel(QtWidgets.QWidget):
    """Task ▸ Work 계층 트리.

    선택이 바뀌면 `selectionChanged(task_id, work_id)` 를 보낸다.
    work 를 고르지 않고 task 만 고르면 work_id 는 -1 (해당 Task 전체 보기).
    """

    selectionChanged = Signal(int, int)  # task_id, work_id (없으면 -1)

    def __init__(
        self,
        db: Database,
        config: OptionsConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.config = config

        title = QtWidgets.QLabel("<b>DL Task ▸ Work ID</b>", self)

        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setPlaceholderText("Search Task / Work…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setHeaderLabels(["Name", "Train", "Infer"])
        self.tree.setColumnWidth(0, 190)
        self.tree.setColumnWidth(1, 52)
        self.tree.setColumnWidth(2, 52)
        self.tree.setAlternatingRowColors(True)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.currentItemChanged.connect(lambda *_: self._emit_selection())
        self.tree.itemDoubleClicked.connect(lambda *_: self.edit_current())

        self.add_task_btn = QtWidgets.QToolButton(self)
        self.add_task_btn.setText("＋Task")
        self.add_task_btn.setToolTip("Add a new DL Task (Level 1)")
        self.add_task_btn.clicked.connect(self.add_task)

        self.add_work_btn = QtWidgets.QToolButton(self)
        self.add_work_btn.setText("＋Work")
        self.add_work_btn.setToolTip("Add a Work ID under the selected Task (Level 2)")
        self.add_work_btn.clicked.connect(self.add_work)

        self.edit_btn = QtWidgets.QToolButton(self)
        self.edit_btn.setText("✎ Rename")
        self.edit_btn.setToolTip(f"Rename/describe the selected item  ({editing.RENAME_KEY})")
        self.edit_btn.clicked.connect(self.edit_current)

        self.del_btn = QtWidgets.QToolButton(self)
        self.del_btn.setText("🗑")
        self.del_btn.setToolTip(f"Delete the selected item  ({editing.DELETE_KEY})")
        self.del_btn.clicked.connect(self.delete_current)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(4)
        for widget in (self.add_task_btn, self.add_work_btn, self.edit_btn, self.del_btn):
            buttons.addWidget(widget)
        buttons.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)

        editing.install_shortcuts(
            self.tree,
            on_rename=self.edit_current,
            on_delete=self.delete_current,
            on_add=self.add_work,
        )
        self.setMinimumWidth(300)

    # -- 상태 조회 -----------------------------------------------------------
    def current_kind_and_id(self) -> tuple[str | None, int | None]:
        item = self.tree.currentItem()
        if item is None:
            return None, None
        return item.data(0, ROLE_KIND), item.data(0, ROLE_ID)

    def current_task_id(self) -> int | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        if item.data(0, ROLE_KIND) == "task":
            return int(item.data(0, ROLE_ID))
        parent = item.parent()
        return int(parent.data(0, ROLE_ID)) if parent else None

    def current_work_id(self) -> int | None:
        kind, ident = self.current_kind_and_id()
        return int(ident) if kind == "work" and ident is not None else None

    # -- 새로고침 ------------------------------------------------------------
    def refresh(self, select_work_id: int | None = None, select_task_id: int | None = None) -> None:
        keep_kind, keep_id = self.current_kind_and_id()
        if select_work_id is not None:
            keep_kind, keep_id = "work", select_work_id
        elif select_task_id is not None:
            keep_kind, keep_id = "task", select_task_id

        expanded = {
            item.data(0, ROLE_ID)
            for item in self._iter_items()
            if item.data(0, ROLE_KIND) == "task" and item.isExpanded()
        }

        self.tree.blockSignals(True)
        self.tree.clear()
        target: QtWidgets.QTreeWidgetItem | None = None

        for task in self.db.list_tasks():
            task_item = QtWidgets.QTreeWidgetItem([task["name"], "", ""])
            task_item.setData(0, ROLE_KIND, "task")
            task_item.setData(0, ROLE_ID, task["id"])
            task_item.setToolTip(0, task.get("description") or task["name"])
            font = task_item.font(0)
            font.setBold(True)
            task_item.setFont(0, font)
            self.tree.addTopLevelItem(task_item)

            train_total = infer_total = 0
            for work in self.db.list_works(task["id"]):
                n_train, n_infer = self.db.counts_for_work(work["id"])
                train_total += n_train
                infer_total += n_infer
                work_item = QtWidgets.QTreeWidgetItem(
                    [work["name"], str(n_train), str(n_infer)]
                )
                work_item.setData(0, ROLE_KIND, "work")
                work_item.setData(0, ROLE_ID, work["id"])
                work_item.setToolTip(0, work.get("description") or work["name"])
                task_item.addChild(work_item)
                if keep_kind == "work" and keep_id is not None and int(keep_id) == work["id"]:
                    target = work_item

            task_item.setText(1, str(train_total))
            task_item.setText(2, str(infer_total))
            if keep_kind == "task" and keep_id is not None and int(keep_id) == task["id"]:
                target = task_item
            if not expanded or task["id"] in expanded:
                task_item.setExpanded(True)

        self.tree.blockSignals(False)

        if target is None:
            target = self._first_selectable()
        if target is not None:
            self.tree.setCurrentItem(target)
        else:
            self._emit_selection()
        self._apply_filter(self.filter_edit.text())

    def _first_selectable(self) -> QtWidgets.QTreeWidgetItem | None:
        for i in range(self.tree.topLevelItemCount()):
            task_item = self.tree.topLevelItem(i)
            if task_item.childCount():
                return task_item.child(0)
        return self.tree.topLevelItem(0) if self.tree.topLevelItemCount() else None

    def _iter_items(self):
        for i in range(self.tree.topLevelItemCount()):
            task_item = self.tree.topLevelItem(i)
            yield task_item
            for j in range(task_item.childCount()):
                yield task_item.child(j)

    def _emit_selection(self) -> None:
        task_id = self.current_task_id()
        work_id = self.current_work_id()
        self.selectionChanged.emit(int(task_id or -1), int(work_id or -1))

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            task_item = self.tree.topLevelItem(i)
            task_match = needle in task_item.text(0).lower()
            child_visible = False
            for j in range(task_item.childCount()):
                child = task_item.child(j)
                visible = not needle or task_match or needle in child.text(0).lower()
                child.setHidden(not visible)
                child_visible = child_visible or visible
            task_item.setHidden(bool(needle) and not task_match and not child_visible)
            if needle and not task_item.isHidden():
                task_item.setExpanded(True)

    # -- CRUD ----------------------------------------------------------------
    def add_task(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Add DL Task", "Task name (e.g. SR, DN, Clustering, Classification):"
        )
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(self, "Add DL Task", "Description (optional):")
        task_name = name.strip()
        task_id = self.db.add_task(task_name, desc.strip())
        if self.config is not None:
            # 새 Task 도 바로 옵션/지표/컬럼을 가질 수 있도록 설정에 자리를 만든다.
            self.config.ensure_task(task_name)
        self.refresh(select_task_id=task_id)

    def add_work(self) -> None:
        task_id = self.current_task_id()
        if task_id is None:
            QtWidgets.QMessageBox.information(self, "Add Work", "Select a Task first.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Add Work ID", "Work ID (e.g. SSL2SL):"
        )
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(self, "Add Work ID", "Description (optional):")
        work_id = self.db.add_work(task_id, name.strip(), desc.strip())
        self.refresh(select_work_id=work_id)

    def edit_current(self) -> None:
        kind, ident = self.current_kind_and_id()
        if kind is None or ident is None:
            return
        ident = int(ident)
        if kind == "task":
            row: dict[str, Any] | None = next(
                (t for t in self.db.list_tasks() if t["id"] == ident), None
            )
            label = "Task"
        else:
            row = self.db.get_work(ident)
            label = "Work ID"
        if row is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, f"Edit {label}", "Name:", text=row["name"]
        )
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(
            self, f"Edit {label}", "Description:", text=row.get("description") or ""
        )
        if kind == "task":
            self.db.update_task(ident, name.strip(), desc.strip())
            self.refresh(select_task_id=ident)
        else:
            self.db.update_work(ident, name.strip(), desc.strip())
            self.refresh(select_work_id=ident)

    def delete_current(self) -> None:
        kind, ident = self.current_kind_and_id()
        if kind is None or ident is None:
            return
        ident = int(ident)
        if kind == "task":
            works = self.db.list_works(ident)
            message = (
                f"This Task, its {len(works)} Work(s), and all their run records will be deleted.\n"
                "Continue?"
            )
        else:
            n_train, n_infer = self.db.counts_for_work(ident)
            message = (
                f"This Work and its {n_train} Train / {n_infer} Inference record(s) will be deleted.\nContinue?"
            )
        answer = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if kind == "task":
            self.db.delete_task(ident)
        else:
            self.db.delete_work(ident)
        self.refresh()

    def manage_datasets(self) -> None:
        work_id = self.current_work_id()
        if work_id is None:
            QtWidgets.QMessageBox.information(self, "Manage Datasets", "Select a Work ID first.")
            return
        work = self.db.get_work(work_id)
        from .dataset_dialog import DatasetManagerDialog

        dialog = DatasetManagerDialog(self.db, work_id, work["name"] if work else "", self)
        dialog.exec()

    def _context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is not None:
            self.tree.setCurrentItem(item)
        kind, _ = self.current_kind_and_id()
        label = "Task" if kind == "task" else "Work ID"

        if item is None:
            menu = editing.build_item_menu(self, add_label="Add DL Task", on_add=self.add_task)
        else:
            extra_top = [("+ Add DL Task", self.add_task)]
            if kind == "work":
                extra_top.append(("📦 Manage Datasets…", self.manage_datasets))
            menu = editing.build_item_menu(
                self,
                add_label="Add Work ID",
                on_add=self.add_work,
                rename_label=f"Edit {label} Name/Description",
                on_rename=self.edit_current,
                delete_label=f"Delete {label}",
                on_delete=self.delete_current,
                extra_top=extra_top,
            )
        menu.exec(self.tree.viewport().mapToGlobal(pos))
