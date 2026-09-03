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
from ..qt import Qt, QtCore, QtGui, QtWidgets, Signal
from .common import PathEdit, toast


def _format_count(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


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

        self.sample_count_spin = QtWidgets.QSpinBox(self)
        self.sample_count_spin.setRange(0, 1_000_000_000)
        self.sample_count_spin.setSpecialValueText("(unspecified)")
        self.sample_count_spin.setGroupSeparatorShown(True)
        if dataset and dataset.get("sample_count"):
            self.sample_count_spin.setValue(int(dataset["sample_count"]))

        self.image_size_edit = QtWidgets.QLineEdit(dataset.get("image_size", "") if dataset else "", self)
        self.image_size_edit.setPlaceholderText("optional, e.g. 256x256")

        self.extension_edit = QtWidgets.QLineEdit(dataset.get("extension", "") if dataset else "", self)
        self.extension_edit.setPlaceholderText("optional, e.g. png, tiff, jpg")

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
        form.addRow("Total samples:", self.sample_count_spin)
        form.addRow("Image size:", self.image_size_edit)
        form.addRow("Extension:", self.extension_edit)
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

    def result_values(self) -> tuple[str, str, str, str, int | None, str, str]:
        return (
            self.name_edit.text().strip(),
            self.variant_edit.text().strip(),
            self.path_edit.path(),
            self.notes_edit.text().strip(),
            self.sample_count_spin.value() or None,
            self.image_size_edit.text().strip(),
            self.extension_edit.text().strip(),
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
        self.resize(900, 420)

        self.table = QtWidgets.QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Variant", "Path", "Samples", "Image Size", "Ext", "Notes"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.Stretch)
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
                "Dataset picker for this Work.",
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
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(_format_count(row.get("sample_count"))))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(row.get("image_size") or ""))
            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(row.get("extension") or ""))
            self.table.setItem(r, 6, QtWidgets.QTableWidgetItem(row.get("notes") or ""))

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
        name, variant, path, notes, sample_count, image_size, extension = dialog.result_values()
        if not name:
            toast(self, False, "Enter a dataset name.", "Add Dataset")
            return
        self.db.add_dataset(self.work_id, name, variant, path, notes, sample_count, image_size, extension)
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
        name, variant, path, notes, sample_count, image_size, extension = dialog.result_values()
        if not name:
            toast(self, False, "Enter a dataset name.", "Edit Dataset")
            return
        self.db.update_dataset(row["id"], name, variant, path, notes, sample_count, image_size, extension)
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


_SENTINEL_ROLE = int(Qt.ItemDataRole.UserRole) + 31
_SENTINEL_TEXT = "+  Add new dataset…"


class _DatasetSentinelDelegate(QtWidgets.QStyledItemDelegate):
    """드롭다운의 '＋ 새 데이터셋 추가…' 행 위에 구분선을 그린다."""

    def paint(self, painter, option, index) -> None:  # noqa: D401
        super().paint(painter, option, index)
        if not index.data(_SENTINEL_ROLE):
            return
        painter.save()
        pen = QtGui.QPen(QtGui.QColor(theme.color("border.subtle")))
        pen.setWidth(1)
        painter.setPen(pen)
        y = option.rect.top()
        painter.drawLine(option.rect.left() + 4, y, option.rect.right() - 4, y)
        painter.restore()


class DatasetCombo(QtWidgets.QComboBox):
    """New Run 폼의 Dataset 선택 - 현재 Work 에 등록된 데이터셋 레지스트리와 바로 연동된다.

    고르면 연결된 dataset row 를 `datasetSelected` 로 알려 호출부가 Dataset Path 를
    채울 수 있게 한다. 드롭다운 맨 아래 `＋ 새 데이터셋 추가…` 로 새로 등록하고,
    항목 우클릭(또는 F2/Del)으로 수정·삭제한다 - `ManagedCombo` 와 같은 조작감이다.
    """

    datasetSelected = Signal(dict)

    def __init__(self, db: Database, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db
        self._work_id: int | None = None
        self._last_valid_text = ""

        self.setEditable(False)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

        view = QtWidgets.QListView(self)
        view.setUniformItemSizes(True)
        view.setItemDelegate(_DatasetSentinelDelegate(view))
        self.setView(view)
        view.installEventFilter(self)
        view.viewport().installEventFilter(self)

        self.activated.connect(self._on_activated)
        self.reload()

    # -- 목록 구성 -----------------------------------------------------------
    def set_work(self, work_id: int | None, keep_text: bool = True) -> None:
        self._work_id = work_id
        self.reload(keep_text=keep_text)

    def reload(self, keep_text: bool = True) -> None:
        current = self.current_text() if keep_text else ""
        self.blockSignals(True)
        self.clear()
        if self._work_id:
            for row in self.db.list_datasets(self._work_id):
                self.addItem(self._label(row), row["id"])
        self._append_sentinel()
        self.setCurrentIndex(-1)
        self.set_text(current)
        self.blockSignals(False)
        self._last_valid_text = self.current_text()

    def _append_sentinel(self) -> None:
        self.addItem(_SENTINEL_TEXT)
        index = self.count() - 1
        model = self.model()
        item_index = model.index(index, 0)
        model.setData(item_index, True, _SENTINEL_ROLE)
        model.setData(
            item_index, QtGui.QBrush(QtGui.QColor(theme.color("accent"))), Qt.ItemDataRole.ForegroundRole
        )
        model.setData(
            item_index, "Register a new dataset for this Work.", Qt.ItemDataRole.ToolTipRole
        )

    @staticmethod
    def _label(row: dict[str, Any]) -> str:
        return f"{row['name']} · {row['variant']}" if row.get("variant") else row["name"]

    def _is_sentinel(self, index: int) -> bool:
        if index < 0 or index >= self.count():
            return False
        return bool(self.model().index(index, 0).data(_SENTINEL_ROLE))

    # -- 값 ------------------------------------------------------------------
    def current_text(self) -> str:
        text = self.currentText().strip()
        return "" if text == _SENTINEL_TEXT else text

    def set_text(self, value: str | None) -> None:
        value = (value or "").strip()
        index = self.findText(value) if value else -1
        if value and index < 0:
            # 레지스트리에 없는 값(예전 기록 등)도 잃지 않고 임시로 보여준다.
            insert_at = max(0, self.count() - 1)
            self.insertItem(insert_at, value)
            index = insert_at
        self.setCurrentIndex(index)
        self._last_valid_text = value if index >= 0 else ""

    # -- 상호작용 --------------------------------------------------------------
    def _on_activated(self, index: int) -> None:
        if not self._is_sentinel(index):
            self._last_valid_text = self.itemText(index)
            dataset_id = self.itemData(index)
            if dataset_id is not None:
                row = self.db.get_dataset(int(dataset_id))
                if row:
                    self.datasetSelected.emit(row)
            return
        # 센티넬은 값이 아니므로 직전 값을 되돌리고 추가 흐름으로 넘어간다.
        self.setCurrentIndex(self.findText(self._last_valid_text) if self._last_valid_text else -1)
        QtCore.QTimer.singleShot(0, self.add_item)

    def eventFilter(self, obj, event):  # noqa: N802
        view = self.view()
        if view is not None and obj in (view, view.viewport()):
            kind = event.type()
            if kind == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    index = view.indexAt(event.pos())
                    self._popup_context_menu(index, event.globalPosition().toPoint())
                    return True
            elif kind == QtCore.QEvent.Type.KeyPress:
                key = event.key()
                index = view.currentIndex()
                if key == Qt.Key.Key_F2:
                    self._edit_index(index)
                    return True
                if key == Qt.Key.Key_Delete:
                    self._delete_index(index)
                    return True
                if key == Qt.Key.Key_Insert:
                    self.hidePopup()
                    QtCore.QTimer.singleShot(0, self.add_item)
                    return True
        return super().eventFilter(obj, event)

    def _popup_context_menu(self, index, global_pos) -> None:
        row_idx = index.row() if index.isValid() else -1
        is_sentinel = self._is_sentinel(row_idx)
        self.hidePopup()  # 팝업을 띄운 채 메뉴를 열면 포커스가 엉킨다

        if row_idx < 0 or is_sentinel:
            menu = editing.build_item_menu(self, add_label="Add Dataset", on_add=self.add_item)
        else:
            value = self.itemText(row_idx)
            menu = editing.build_item_menu(
                self,
                add_label="Add Dataset",
                on_add=self.add_item,
                rename_label=f"Edit '{value}'",
                on_rename=lambda: self._edit_row(row_idx),
                delete_label=f"Delete '{value}'",
                on_delete=lambda: self._delete_row(row_idx),
            )
        menu.exec(global_pos)

    def _edit_index(self, index) -> None:
        if not index.isValid() or self._is_sentinel(index.row()):
            return
        row_idx = index.row()
        self.hidePopup()
        QtCore.QTimer.singleShot(0, lambda: self._edit_row(row_idx))

    def _delete_index(self, index) -> None:
        if not index.isValid() or self._is_sentinel(index.row()):
            return
        row_idx = index.row()
        self.hidePopup()
        QtCore.QTimer.singleShot(0, lambda: self._delete_row(row_idx))

    # -- 등록 / 수정 / 삭제 --------------------------------------------------------
    def add_item(self) -> None:
        if not self._work_id:
            toast(self, False, "Select a Work on the left first.", "Add Dataset")
            return
        dialog = DatasetEditDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes, sample_count, image_size, extension = dialog.result_values()
        if not name:
            return
        self.db.add_dataset(self._work_id, name, variant, path, notes, sample_count, image_size, extension)
        row = next(
            (r for r in self.db.list_datasets(self._work_id)
             if r["name"] == name and (r.get("variant") or "") == variant),
            None,
        )
        self.reload(keep_text=False)
        self.set_text(self._label(row) if row else name)
        if row:
            self.datasetSelected.emit(row)

    def _edit_row(self, row_idx: int) -> None:
        dataset_id = self.itemData(row_idx)
        if dataset_id is None:
            return
        row = self.db.get_dataset(int(dataset_id))
        if row is None:
            return
        dialog = DatasetEditDialog(self, row)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes, sample_count, image_size, extension = dialog.result_values()
        if not name:
            return
        self.db.update_dataset(row["id"], name, variant, path, notes, sample_count, image_size, extension)
        updated = self.db.get_dataset(row["id"])
        self.reload(keep_text=False)
        if updated:
            self.set_text(self._label(updated))
            self.datasetSelected.emit(updated)

    def _delete_row(self, row_idx: int) -> None:
        dataset_id = self.itemData(row_idx)
        if dataset_id is None:
            return
        row = self.db.get_dataset(int(dataset_id))
        if row is None:
            return
        used = self.db.count_runs_using_dataset(
            self._work_id, row["name"], row.get("variant") or ""
        )
        if not editing.confirm_delete(self, self._label(row), used):
            return
        self.db.delete_dataset(row["id"])
        if self.current_text() == self._label(row):
            self.set_text("")
        self.reload()
