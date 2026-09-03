"""재사용 위젯 모음."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..qt import Qt, QtCore, QtGui, QtWidgets, Signal
from ..utils import (
    coerce_number,
    format_number,
    open_in_file_manager,
    rows_to_tsv,
)

FOLDER_ICON = "📁"


def monospace_font(point_delta: int = 0) -> QtGui.QFont:
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    if point_delta:
        font.setPointSize(max(7, font.pointSize() + point_delta))
    return font


def toast(parent: QtWidgets.QWidget | None, ok: bool, message: str, title: str = "알림") -> None:
    """상태바가 있으면 상태바에, 없으면 메시지 박스로 알린다."""
    window = parent.window() if parent is not None else None
    bar = getattr(window, "statusBar", None)
    if ok and callable(bar):
        try:
            bar().showMessage(message.replace("\n", " "), 4000)
            return
        except (RuntimeError, TypeError):
            pass
    if ok:
        QtWidgets.QMessageBox.information(parent, title, message)
    else:
        QtWidgets.QMessageBox.warning(parent, title, message)


def copy_to_clipboard(text: str, parent: QtWidgets.QWidget | None = None, label: str = "내용") -> None:
    QtWidgets.QApplication.clipboard().setText(text or "")
    toast(parent, True, f"{label}을(를) 클립보드에 복사했습니다. ({len(text or '')} 자)")


class OpenFolderButton(QtWidgets.QToolButton):
    """OS 탐색기(Finder/탐색기)에서 경로를 여는 버튼."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, text: str = f"{FOLDER_ICON} 폴더 열기"):
        super().__init__(parent)
        self.setText(text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setToolTip("이 경로를 OS 파일 탐색기(macOS Finder / Windows 탐색기)에서 엽니다.")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._provider = lambda: ""
        self.clicked.connect(self._on_click)

    def set_path_provider(self, provider) -> None:
        self._provider = provider

    def _on_click(self) -> None:
        path = self._provider() or ""
        ok, message = open_in_file_manager(path)
        toast(self, ok, message, "폴더 열기")


class PathEdit(QtWidgets.QWidget):
    """경로 입력란 + [찾아보기] + [📁 폴더 열기]."""

    pathChanged = Signal(str)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "/mnt/data/...",
        directory: bool = True,
    ) -> None:
        super().__init__(parent)
        self._directory = directory

        self.edit = QtWidgets.QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.setFont(monospace_font())
        self.edit.textChanged.connect(self.pathChanged.emit)

        self.browse_btn = QtWidgets.QToolButton(self)
        self.browse_btn.setText("찾아보기…")
        self.browse_btn.setToolTip("파일 선택 대화상자로 경로를 고릅니다.")
        self.browse_btn.clicked.connect(self._browse)

        self.open_btn = OpenFolderButton(self)
        self.open_btn.set_path_provider(self.path)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_btn)
        layout.addWidget(self.open_btn)

    # -- API ----------------------------------------------------------------
    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str | None) -> None:
        self.edit.setText(value or "")

    def clear(self) -> None:
        self.edit.clear()

    def _browse(self) -> None:
        start = self.path() or QtCore.QDir.homePath()
        if self._directory:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 선택", start)
        else:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self, "파일 선택", start)
        if chosen:
            self.set_path(chosen)


class EditableCombo(QtWidgets.QComboBox):
    """목록에서 고르거나 직접 타이핑할 수 있는 콤보박스."""

    def __init__(
        self,
        items: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "선택하거나 직접 입력",
    ) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.lineEdit().setPlaceholderText(placeholder)
        self.lineEdit().setClearButtonEnabled(True)
        self.set_items(items)

        completer = self.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)

    def set_items(self, items: Iterable[str], keep_text: bool = True) -> None:
        current = self.current_text() if keep_text else ""
        self.blockSignals(True)
        self.clear()
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                self.addItem(text)
        self.setCurrentIndex(-1)
        self.setCurrentText(current)
        self.blockSignals(False)

    def merge_items(self, items: Iterable[str]) -> None:
        """기존 프리셋에 DB 에 이미 존재하는 값들을 합친다."""
        existing = [self.itemText(i) for i in range(self.count())]
        self.set_items(list(existing) + [str(i) for i in items])

    def current_text(self) -> str:
        return self.currentText().strip()

    def set_text(self, value: str | None) -> None:
        self.setCurrentText(value or "")


class LabeledText(QtWidgets.QWidget):
    """제목 + [복사] 버튼 + 멀티라인 텍스트 박스.

    read_only=True 이면 상세 뷰어, False 이면 입력 폼으로 사용한다.
    """

    def __init__(
        self,
        title: str,
        parent: QtWidgets.QWidget | None = None,
        read_only: bool = False,
        placeholder: str = "",
        mono: bool = True,
        min_height: int = 120,
    ) -> None:
        super().__init__(parent)
        self._title = title

        self.label = QtWidgets.QLabel(f"<b>{title}</b>", self)
        self.copy_btn = QtWidgets.QToolButton(self)
        self.copy_btn.setText("복사")
        self.copy_btn.setToolTip(f"{title} 내용을 클립보드로 복사")
        self.copy_btn.clicked.connect(self._copy)

        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setReadOnly(read_only)
        self.editor.setPlaceholderText(placeholder)
        self.editor.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setMinimumHeight(min_height)
        if mono:
            self.editor.setFont(monospace_font())
        self.editor.setTabChangesFocus(True)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.label)
        header.addStretch(1)
        header.addWidget(self.copy_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.editor, 1)

    def _copy(self) -> None:
        copy_to_clipboard(self.text(), self, self._title)

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, value: str | None) -> None:
        self.editor.setPlainText(value or "")

    def clear(self) -> None:
        self.editor.clear()

    def add_header_widget(self, widget: QtWidgets.QWidget) -> None:
        layout = self.layout().itemAt(0).layout()
        layout.insertWidget(layout.count() - 1, widget)


class MetricsEditor(QtWidgets.QWidget):
    """key/value 형태의 평가 지표 편집기 (PSNR, SSIM, ... 자유 확장)."""

    def __init__(
        self,
        presets: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presets = list(presets)

        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

        self.preset_combo = EditableCombo(self._presets, self, "지표명 (예: PSNR)")
        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("+ 추가")
        add_btn.clicked.connect(self._add_from_combo)
        remove_btn = QtWidgets.QToolButton(self)
        remove_btn.setText("− 삭제")
        remove_btn.clicked.connect(self._remove_selected)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.preset_combo, 1)
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)

    # -- API ----------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            key = (key_item.text().strip() if key_item else "")
            if not key:
                continue
            out[key] = coerce_number(val_item.text() if val_item else "")
        return out

    def set_metrics(self, metrics: dict[str, Any] | None) -> None:
        self.table.setRowCount(0)
        for key, value in sorted((metrics or {}).items()):
            self._append_row(str(key), format_number(value))

    def clear(self) -> None:
        self.table.setRowCount(0)

    # -- 내부 ----------------------------------------------------------------
    def _append_row(self, key: str, value: str = "") -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
        return row

    def _add_from_combo(self) -> None:
        key = self.preset_combo.current_text()
        if not key:
            QtWidgets.QMessageBox.information(self, "지표 추가", "지표 이름을 입력하세요.")
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text().strip().lower() == key.lower():
                self.table.setCurrentCell(row, 1)
                self.table.editItem(self.table.item(row, 1))
                return
        row = self._append_row(key)
        self.preset_combo.set_text("")
        self.table.setCurrentCell(row, 1)
        self.table.editItem(self.table.item(row, 1))

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.rowCount():
            rows = [self.table.rowCount() - 1]
        for row in rows:
            self.table.removeRow(row)


class FormRow(QtWidgets.QWidget):
    """QFormLayout 안에서 위젯 + 보조 버튼을 한 줄로 묶기 위한 컨테이너."""

    def __init__(self, *widgets: QtWidgets.QWidget, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for widget in widgets:
            layout.addWidget(widget)


def table_selection_to_tsv(
    view: QtWidgets.QTableView, headers: Sequence[str], selected_only: bool
) -> str:
    """QTableView 의 (선택 영역 또는 전체) 내용을 TSV 로 직렬화."""
    model = view.model()
    if model is None:
        return ""
    if selected_only:
        rows = sorted({idx.row() for idx in view.selectionModel().selectedRows()})
        if not rows:
            rows = sorted({idx.row() for idx in view.selectionModel().selectedIndexes()})
    else:
        rows = list(range(model.rowCount()))
    if not rows:
        return ""
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                str(model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole) or "")
                for col in range(model.columnCount())
            ]
        )
    return rows_to_tsv(headers, body)
