"""Work 별 데이터셋 레지스트리 관리 - 이름 + (선택) 변형 + 위치를 등록해 두고,
Train/Inference 등록 폼에서 그중 하나를 골라 경로를 바로 불러온다.

같은 데이터셋이라도 "전체 페어"와 "특정 서브셋"처럼 여러 판이 있을 수 있어,
이름은 같게 두고 Variant 만 다르게 등록할 수 있게 한다
(예: DIV2K · Full Pair / DIV2K · Subset A).
"""
from __future__ import annotations

from typing import Any

from .. import editing, theme
from ..db import Database
from ..qt import Qt, QtWidgets, Signal
from .common import PathEdit, toast


class DatasetEditDialog(QtWidgets.QDialog):
    """데이터셋 한 건 추가/수정 - 이름 + variant + 경로 + 메모."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, dataset: dict[str, Any] | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Dataset" if dataset else "Add Dataset")
        self.setMinimumWidth(440)

        self.name_edit = QtWidgets.QLineEdit(dataset.get("name", "") if dataset else "", self)
        self.name_edit.setPlaceholderText("e.g. DIV2K")

        self.variant_edit = QtWidgets.QLineEdit(dataset.get("variant", "") if dataset else "", self)
        self.variant_edit.setPlaceholderText("optional, e.g. Full Pair / Subset A (blank = default)")

        self.path_edit = PathEdit(self, "/mnt/data/DIV2K/train")
        self.path_edit.set_path(dataset.get("path") if dataset else "")

        self.notes_edit = QtWidgets.QLineEdit(dataset.get("notes", "") if dataset else "", self)
        self.notes_edit.setPlaceholderText("optional notes")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Variant:", self.variant_edit)
        form.addRow("Path:", self.path_edit)
        form.addRow("Notes:", self.notes_edit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(
            QtWidgets.QLabel(
                f"<span style='color:{theme.color('text.muted')}'>"
                "Same dataset, different pick: use Variant to register the full paired set "
                "and a specific subset separately (e.g. \"Full Pair\" vs \"Subset A\").</span>",
                self,
            )
        )
        layout.addWidget(buttons)

    def result_values(self) -> tuple[str, str, str, str]:
        return (
            self.name_edit.text().strip(),
            self.variant_edit.text().strip(),
            self.path_edit.path(),
            self.notes_edit.text().strip(),
        )


class DatasetManagerDialog(QtWidgets.QDialog):
    """한 Work 의 데이터셋 목록 - 추가/수정/삭제."""

    datasetsChanged = Signal()

    def __init__(
        self,
        db: Database,
        work_id: int,
        work_name: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.work_id = work_id
        self._rows: list[dict[str, Any]] = []
        self.setWindowTitle(f"Manage Datasets · {work_name}")
        self.resize(760, 420)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Variant", "Path", "Notes"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(220)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(lambda *_: self._edit_selected())
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)

        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("+ Add Dataset")
        add_btn.clicked.connect(self._add)
        edit_btn = QtWidgets.QToolButton(self)
        edit_btn.setText("✎ Edit")
        edit_btn.clicked.connect(self._edit_selected)
        del_btn = QtWidgets.QToolButton(self)
        del_btn.setText("- Remove")
        del_btn.clicked.connect(self._remove_selected)
        editing.install_shortcuts(
            self.table, on_add=self._add, on_rename=self._edit_selected, on_delete=self._remove_selected
        )

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(add_btn)
        controls.addWidget(edit_btn)
        controls.addWidget(del_btn)
        controls.addStretch(1)

        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Datasets registered here appear in the Train/Inference form's "
                "\"Registered Dataset\" picker for this Work.",
                self,
            )
        )
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)
        layout.addLayout(footer)

        self._reload()

    # -- data ---------------------------------------------------------------
    def _reload(self) -> None:
        self._rows = self.db.list_datasets(self.work_id)
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(row["name"]))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(row.get("variant") or ""))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(row.get("path") or ""))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(row.get("notes") or ""))

    def _selected_row(self) -> dict[str, Any] | None:
        index = self.table.currentRow()
        if 0 <= index < len(self._rows):
            return self._rows[index]
        return None

    @staticmethod
    def _label(row: dict[str, Any]) -> str:
        return f"{row['name']} · {row['variant']}" if row.get("variant") else row["name"]

    def _context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if index.isValid():
            self.table.setCurrentCell(index.row(), index.column())
        row = self._selected_row()
        menu = editing.build_item_menu(
            self,
            add_label="Add Dataset",
            on_add=self._add,
            rename_label=f"Edit '{self._label(row)}'" if row else "Edit",
            on_rename=self._edit_selected if row else None,
            delete_label=f"Remove '{self._label(row)}'" if row else "Remove",
            on_delete=self._remove_selected if row else None,
        )
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # -- actions --------------------------------------------------------------
    def _add(self) -> None:
        dialog = DatasetEditDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes = dialog.result_values()
        if not name:
            toast(self, False, "Enter a dataset name.", "Add Dataset")
            return
        self.db.add_dataset(self.work_id, name, variant, path, notes)
        self._reload()
        self.datasetsChanged.emit()

    def _edit_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            toast(self, False, "Select a dataset first.", "Edit Dataset")
            return
        dialog = DatasetEditDialog(self, row)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes = dialog.result_values()
        if not name:
            toast(self, False, "Enter a dataset name.", "Edit Dataset")
            return
        self.db.update_dataset(row["id"], name, variant, path, notes)
        self._reload()
        self.datasetsChanged.emit()

    def _remove_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            toast(self, False, "Select a dataset first.", "Remove Dataset")
            return
        used = self.db.count_runs_using_dataset(self.work_id, row["name"], row.get("variant") or "")
        if not editing.confirm_delete(self, self._label(row), used):
            return
        self.db.delete_dataset(row["id"])
        self._reload()
        self.datasetsChanged.emit()
