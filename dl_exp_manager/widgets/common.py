"""재사용 위젯 모음."""
from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

from .. import editing, theme
from ..qt import Qt, QtCore, QtGui, QtWidgets, Signal
from ..theme import icons
from ..utils import (
    coerce_number,
    format_number,
    open_in_file_manager,
    parse_gpu_count,
    rows_to_tsv,
)


def monospace_font(point_delta: int = 0) -> QtGui.QFont:
    from ..theme.fonts import mono_font

    return mono_font(theme.FONT_SIZES["mono"] + point_delta)


def toast(parent: QtWidgets.QWidget | None, ok: bool, message: str, title: str = "Notice") -> None:
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


def copy_to_clipboard(text: str, parent: QtWidgets.QWidget | None = None, label: str = "content") -> None:
    QtWidgets.QApplication.clipboard().setText(text or "")
    toast(parent, True, f"Copied {label} to the clipboard. ({len(text or '')} chars)")


def colorize_status_items(
    combo: QtWidgets.QComboBox, statuses: Iterable[str], offset: int = 0
) -> None:
    """콤보 항목의 글자색을 상태색으로 칠한다 (드롭다운에서 상태를 한눈에 구분하도록)."""
    for i, status in enumerate(statuses, start=offset):
        combo.setItemData(
            i,
            QtGui.QBrush(QtGui.QColor(theme.status_color(status))),
            Qt.ItemDataRole.ForegroundRole,
        )


class OpenFolderButton(QtWidgets.QToolButton):
    """OS 탐색기(Finder/탐색기)에서 경로를 여는 버튼."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, compact: bool = False):
        super().__init__(parent)
        self.setIcon(icons.icon("folder", theme.color("text.secondary")))
        if compact:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        else:
            self.setText("Open Folder")
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setToolTip("Open this path in the OS file manager (Finder on macOS / Explorer on Windows).")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._provider = lambda: ""
        self.clicked.connect(self._on_click)

    def set_path_provider(self, provider) -> None:
        self._provider = provider

    def _on_click(self) -> None:
        path = self._provider() or ""
        ok, message = open_in_file_manager(path)
        toast(self, ok, message, "Open Folder")


class PathEdit(QtWidgets.QWidget):
    """경로 입력란 + [찾아보기] + [📁 폴더 열기].

    탐색기에서 폴더(또는 파일)를 끌어다 놓으면 경로가 자동으로 채워진다.
    """

    pathChanged = Signal(str)
    folderDropped = Signal(str)  # 드롭으로 채워졌을 때만 (타이핑/찾아보기 구분용)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "/mnt/data/...",
        directory: bool = True,
        compact: bool = False,
    ) -> None:
        """compact=True 면 좁은 폼에서 입력란이 눌리지 않도록 버튼을 아이콘만 남긴다."""
        super().__init__(parent)
        self._directory = directory
        self.setAcceptDrops(True)

        self.edit = QtWidgets.QLineEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.setFont(monospace_font())
        self.edit.setMinimumWidth(80)
        self.edit.textChanged.connect(self.pathChanged.emit)
        self.edit.setToolTip("You can also drag a folder here from Finder/Explorer.")

        self.browse_btn = QtWidgets.QToolButton(self)
        self.browse_btn.setText("…" if compact else "Browse…")
        self.browse_btn.setToolTip("Pick a path with a file dialog.")
        self.browse_btn.clicked.connect(self._browse)

        self.open_btn = OpenFolderButton(self, compact=compact)
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
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start)
        else:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", start)
        if chosen:
            self.set_path(chosen)

    # -- drag & drop ----------------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        local_path = urls[0].toLocalFile()
        if not local_path:
            return
        # 파일이 이 필드에 떨어졌는데 폴더를 기대하는 필드라면, 그 파일이 있는 폴더를 쓴다
        # (결과 폴더 안의 아무 로그/이미지 파일을 끌어다 놔도 자연스럽게 동작하도록).
        if self._directory and os.path.isfile(local_path):
            local_path = os.path.dirname(local_path)
        self.set_path(local_path)
        self.folderDropped.emit(local_path)
        event.acceptProposedAction()


class EditableCombo(QtWidgets.QComboBox):
    """목록에서 고르거나 직접 타이핑할 수 있는 콤보박스."""

    def __init__(
        self,
        items: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "Select or type",
    ) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
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


SENTINEL_ROLE = int(Qt.ItemDataRole.UserRole) + 11
SENTINEL_TEXT = "+  Add new item…"


class _SentinelDelegate(QtWidgets.QStyledItemDelegate):
    """드롭다운의 '＋ 새 항목 추가…' 행 위에 구분선을 그린다."""

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        if not index.data(SENTINEL_ROLE):
            return
        painter.save()
        pen = QtGui.QPen(QtGui.QColor(theme.color("border.subtle")))
        pen.setWidth(1)
        painter.setPen(pen)
        y = option.rect.top()
        painter.drawLine(option.rect.left() + 4, y, option.rect.right() - 4, y)
        painter.restore()

class ManagedCombo(QtWidgets.QComboBox):
    """options.yaml 을 백엔드로 쓰는 편집 가능 콤보박스.

    - 목록에서 고르거나 직접 타이핑 (`setEditable(True)`)
    - 드롭다운 **맨 아래 `＋ 새 항목 추가…`** 행으로 새 값 등록
    - 드롭다운 항목 **우클릭 → 이름 변경 / 삭제**, **F2 / Del** 단축키
    - 선택지는 현재 Task 기준으로 읽고, 추가 시 Task 전용/전체 공통을 고를 수 있다
    """

    optionsChanged = Signal()
    renameRequested = Signal(str, str)   # (old, new) - 기존 기록 일괄 변경 요청

    def __init__(
        self,
        field: str,
        field_label: str = "",
        parent: QtWidgets.QWidget | None = None,
        placeholder: str = "",
        config=None,
        task_getter=None,
        usage_counter=None,
    ) -> None:
        super().__init__(parent)
        self.field = field
        self.field_label = field_label or field
        self._config = config
        self._task_getter = task_getter or (lambda: None)
        self._usage_counter = usage_counter or (lambda value: 0)
        self._last_valid = ""
        self._items: list[str] = []

        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        # 항목이 길어도 폼 너비를 밀어내지 않게 한다(툴팁/드롭다운에서 전체를 볼 수 있다).
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.lineEdit().setPlaceholderText(placeholder or f"Select {self.field_label} or type")
        self.lineEdit().setClearButtonEnabled(True)

        view = QtWidgets.QListView(self)
        view.setUniformItemSizes(True)
        view.setItemDelegate(_SentinelDelegate(view))
        self.setView(view)
        view.installEventFilter(self)
        view.viewport().installEventFilter(self)

        completer = self.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)

        self.activated.connect(self._on_activated)
        self.reload()

    # -- 설정 연결 -----------------------------------------------------------
    def set_config(self, config, task_getter=None) -> None:
        self._config = config
        if task_getter is not None:
            self._task_getter = task_getter
        self.reload()

    def current_task(self) -> str | None:
        return self._task_getter()

    def _config_items(self) -> list[str]:
        if self._config is None:
            return list(self._items)
        return self._config.options_for(self.current_task(), self.field)

    # -- 목록 구성 -----------------------------------------------------------
    def reload(self, keep_text: bool = True) -> None:
        current = self.current_text() if keep_text else ""
        items = self._config_items()
        self.blockSignals(True)
        self.clear()
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                self.addItem(text)
        self._items = list(seen)
        self._append_sentinel()
        self.setCurrentIndex(-1)
        self.setCurrentText(current)
        self.blockSignals(False)
        self._last_valid = current

    def _append_sentinel(self) -> None:
        self.addItem(SENTINEL_TEXT)
        index = self.count() - 1
        model = self.model()
        item_index = model.index(index, 0)
        model.setData(item_index, True, SENTINEL_ROLE)
        model.setData(item_index, QtGui.QBrush(QtGui.QColor(theme.color("accent"))),
                      Qt.ItemDataRole.ForegroundRole)
        model.setData(item_index, "Creates a new item and saves it to this Task's config.",
                      Qt.ItemDataRole.ToolTipRole)

    def _is_sentinel(self, index: int) -> bool:
        if index < 0 or index >= self.count():
            return False
        return bool(self.model().index(index, 0).data(SENTINEL_ROLE))

    def merge_items(self, items) -> None:
        """설정에 없지만 DB 에 남아 있는 값들을 목록에 합쳐 둔다."""
        extra = [str(i).strip() for i in items if str(i).strip()]
        missing = [i for i in extra if i not in self._items]
        if not missing:
            return
        current = self.current_text()
        self.blockSignals(True)
        self.removeItem(self.count() - 1)  # 센티넬 제거
        for item in missing:
            self.addItem(item)
            self._items.append(item)
        self._append_sentinel()
        self.setCurrentIndex(-1)
        self.setCurrentText(current)
        self.blockSignals(False)

    # -- 값 ------------------------------------------------------------------
    def current_text(self) -> str:
        text = self.currentText().strip()
        return "" if text == SENTINEL_TEXT else text

    def set_text(self, value: str | None) -> None:
        self.setCurrentText(value or "")
        self._last_valid = value or ""

    # -- 상호작용 -------------------------------------------------------------
    def _on_activated(self, index: int) -> None:
        if not self._is_sentinel(index):
            self._last_valid = self.itemText(index).strip()
            return
        # 센티넬은 값이 아니므로 직전 값을 되돌리고 추가 흐름으로 넘어간다.
        self.setCurrentText(self._last_valid)
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
                    self._rename_index(index)
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
        row = index.row() if index.isValid() else -1
        value = self.itemText(row).strip() if row >= 0 else ""
        is_sentinel = self._is_sentinel(row)
        self.hidePopup()  # 팝업을 띄운 채 메뉴를 열면 포커스가 엉킨다

        if row < 0 or is_sentinel:
            menu = editing.build_item_menu(
                self, add_label=f"Add {self.field_label}", on_add=self.add_item
            )
        else:
            menu = editing.build_item_menu(
                self,
                add_label=f"Add {self.field_label}",
                on_add=self.add_item,
                rename_label=f"Rename '{value}'",
                on_rename=lambda: self._rename_value(value),
                delete_label=f"Delete '{value}'",
                on_delete=lambda: self._delete_value(value),
            )
        menu.exec(global_pos)

    def _rename_index(self, index) -> None:
        if not index.isValid() or self._is_sentinel(index.row()):
            return
        value = self.itemText(index.row()).strip()
        self.hidePopup()
        QtCore.QTimer.singleShot(0, lambda: self._rename_value(value))

    def _delete_index(self, index) -> None:
        if not index.isValid() or self._is_sentinel(index.row()):
            return
        value = self.itemText(index.row()).strip()
        self.hidePopup()
        QtCore.QTimer.singleShot(0, lambda: self._delete_value(value))

    # -- 설정 변경 -------------------------------------------------------------
    def add_item(self) -> None:
        if self._config is None:
            return
        task = self.current_task()
        result = editing.AddOptionDialog.run(self, self.field_label, task)
        if result is None:
            return
        value, task_scope = result
        scope = task if task_scope else None
        if not self._config.add_option(scope, self.field, value):
            QtWidgets.QMessageBox.information(
                self, "Add Item", f"'{value}' is already in the list."
            )
        self.reload(keep_text=False)
        self.set_text(value)
        self.optionsChanged.emit()

    def _rename_value(self, old: str) -> None:
        if self._config is None or not old:
            return
        new = editing.prompt_text(self, "Rename", f"New {self.field_label} name:", old)
        if new is None or new == old:
            return
        task = self.current_task()
        scope = task if self._is_task_scoped(task) else None
        if not self._config.rename_option(scope, self.field, old, new):
            QtWidgets.QMessageBox.warning(self, "Rename", "Could not find that item in the config.")
            return
        used = self._usage_counter(old)
        if used and editing.confirm(
            self,
            "Update Existing Records",
            f"{used} existing run(s) use '{old}'.\n"
            f"Update those records to '{new}' as well?",
        ):
            self.renameRequested.emit(old, new)
        self.reload(keep_text=False)
        self.set_text(new)
        self.optionsChanged.emit()

    def _delete_value(self, value: str) -> None:
        if self._config is None or not value:
            return
        if not editing.confirm_delete(self, value, self._usage_counter(value)):
            return
        task = self.current_task()
        scope = task if self._is_task_scoped(task) else None
        if not self._config.remove_option(scope, self.field, value):
            QtWidgets.QMessageBox.warning(self, "Delete", "Could not find that item in the config.")
            return
        if self.current_text() == value:
            self.set_text("")
        self.reload()
        self.optionsChanged.emit()

    def _is_task_scoped(self, task: str | None) -> bool:
        """이 필드가 Task 전용 목록으로 정의돼 있는지."""
        if self._config is None or not task:
            return False
        task_def = self._config.task(task)
        return bool(task_def and self.field in task_def.options)


class ServerCombo(QtWidgets.QComboBox):
    """서버 선택 콤보 - 상단 Servers 목록(config.servers)에서만 고를 수 있다.

    자유 입력을 막아 두는 것 자체가 요구사항이다: 서버는 상단 서버 바에서만
    추가/이름변경/삭제하고, 각 실행 폼에서는 그 목록 중 하나를 고르기만 한다.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(False)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def set_items(self, names: Iterable[str], keep_text: bool = True) -> None:
        current = self.current_text() if keep_text else ""
        self.blockSignals(True)
        self.clear()
        self.addItem("", "")
        seen: set[str] = set()
        for name in names:
            text = str(name).strip()
            if text and text not in seen:
                seen.add(text)
                self.addItem(text, text)
        # 값이 목록에서 지워졌어도(서버 삭제) 현재 실행 기록의 값은 잃지 않는다.
        if current and current not in seen:
            self.addItem(f"{current}  (not in Servers list)", current)
        self.set_text(current)
        self.blockSignals(False)

    def current_text(self) -> str:
        data = self.currentData()
        return str(data) if data else ""

    def set_text(self, value: str | None) -> None:
        value = (value or "").strip()
        index = self.findData(value) if value else 0
        self.setCurrentIndex(index if index >= 0 else 0)


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
        wrap: bool = False,
    ) -> None:
        super().__init__(parent)
        self._title = title

        self.label = QtWidgets.QLabel(f"<b>{title}</b>", self)
        self.copy_btn = QtWidgets.QToolButton(self)
        self.copy_btn.setText("Copy")
        self.copy_btn.setToolTip(f"Copy {title} to the clipboard")
        self.copy_btn.clicked.connect(self._copy)

        self.editor = QtWidgets.QPlainTextEdit(self)
        self.editor.setReadOnly(read_only)
        self.editor.setPlaceholderText(placeholder)
        self.editor.setLineWrapMode(
            QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
            if wrap
            else QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
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
    """key/value 형태의 평가 지표 편집기.

    행 우클릭 / F2 / Del 로 추가·이름변경·삭제할 수 있고,
    Task 지표 정의(options.yaml)에 등록하는 메뉴도 제공한다.
    """

    metricsDefined = Signal()

    def __init__(
        self,
        presets: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
        config=None,
        task_getter=None,
    ) -> None:
        super().__init__(parent)
        self._presets = list(presets)
        self._config = config
        self._task_getter = task_getter or (lambda: None)

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

        self.preset_combo = EditableCombo(self._presets, self, "Metric name (e.g. PSNR)")
        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("+ Add")
        add_btn.clicked.connect(self._add_from_combo)
        remove_btn = QtWidgets.QToolButton(self)
        remove_btn.setText("- Remove")
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

        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        editing.install_shortcuts(
            self.table,
            on_rename=self._rename_selected,
            on_delete=self._remove_selected,
            on_add=self._add_from_combo,
        )

    # -- 설정 연결 -----------------------------------------------------------
    def set_config(self, config, task_getter=None) -> None:
        self._config = config
        if task_getter is not None:
            self._task_getter = task_getter
        self.refresh_presets()

    def refresh_presets(self) -> None:
        if self._config is None:
            return
        keys = self._config.metric_keys(self._task_getter())
        self.preset_combo.set_items(keys or list(self._presets), keep_text=True)

    def _selected_key(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text().strip() if item else ""

    def _rename_selected(self) -> None:
        old = self._selected_key()
        if not old:
            return
        new = editing.prompt_text(self, "Rename Metric", "New name:", old)
        if new and new != old:
            self.table.item(self.table.currentRow(), 0).setText(new)

    def _register_in_config(self, key: str) -> None:
        task = self._task_getter()
        if self._config is None or not task or not key:
            return
        from ..config_store import MetricDef

        if self._config.add_metric(task, MetricDef(key=key)):
            self.refresh_presets()
            self.metricsDefined.emit()
        else:
            QtWidgets.QMessageBox.information(
                self, "Metric Definition", f"'{key}' is already defined as a metric for {task}."
            )

    def _context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if index.isValid():
            self.table.setCurrentCell(index.row(), index.column())
        key = self._selected_key()
        task = self._task_getter()
        extra: list = []
        if key and task and self._config is not None and key not in self._config.metric_keys(task):
            extra.append((f"Add '{key}' to {task} metric definitions", lambda: self._register_in_config(key)))
        menu = editing.build_item_menu(
            self,
            add_label="Add Metric",
            on_add=self._add_from_combo,
            rename_label=f"Rename '{key}'" if key else "Rename",
            on_rename=self._rename_selected if key else None,
            delete_label=f"Delete '{key}'" if key else "Delete",
            on_delete=self._remove_selected if key else None,
            extra_top=extra,
        )
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # -- API ----------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            val_item = self.table.item(row, 1)
            key = (key_item.text().strip() if key_item else "")
            value_text = val_item.text().strip() if val_item else ""
            if not key or not value_text:
                # 값을 아직 안 채운 프리필 행(Task 지표 기본값)은 저장하지 않는다.
                continue
            out[key] = coerce_number(value_text)
        return out

    def set_metrics(self, metrics: dict[str, Any] | None) -> None:
        self.table.setRowCount(0)
        for key, value in sorted((metrics or {}).items()):
            self._append_row(str(key), format_number(value))

    def set_value(self, key: str, value: Any) -> None:
        """하나의 지표 값만 채우거나 갱신한다 (로그 자동 파싱 결과 반영용)."""
        text = value if isinstance(value, str) else format_number(value)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text().strip() == key:
                self.table.item(row, 1).setText(text)
                return
        self._append_row(key, text)

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
            QtWidgets.QMessageBox.information(self, "Add Metric", "Enter a metric name.")
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


class GpuSelector(QtWidgets.QWidget):
    """GPU 개수 선택 - 특정 인덱스가 아니라 몇 장을 쓸지만 정한다.

    실제로 어떤 GPU 슬롯에 배정할지는 서버 운영진/스케줄러가 정하는 경우가
    많아, 폼에서는 "몇 장"만 물어보는 편이 실제 사용 흐름에 더 가깝다.
    서버를 고르면 그 서버가 가진 GPU 수를 넘지 않게 상한을 건다.
    """

    changed = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._max = 0

        self.spin = QtWidgets.QSpinBox(self)
        self.spin.setRange(0, 64)
        self.spin.setSuffix(" GPU(s)")
        self.spin.setToolTip("Select a server first to see its GPU capacity.")
        self.spin.valueChanged.connect(self._on_changed)

        self.hint = QtWidgets.QLabel("", self)
        self.hint.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.hint.setFont(monospace_font(-1))
        self.hint.setWordWrap(True)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.spin)
        row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(row)
        layout.addWidget(self.hint)
        self.set_server(None)

    def set_server(self, server) -> None:
        """server 는 config_store.ServerDef 또는 None."""
        current = self.spin.value()
        gpus = list(getattr(server, "gpus", []) or [])
        self._max = len(gpus)

        self.spin.blockSignals(True)
        if self._max:
            self.spin.setRange(0, self._max)
            self.spin.setToolTip(f"This server has {self._max} GPU(s) available.")
        else:
            self.spin.setRange(0, 64)
            self.spin.setToolTip(
                "This server has no GPU info." if server is not None
                else "Select a server first to see its GPU capacity."
            )
        self.spin.setValue(min(current, self._max) if self._max else current)
        self.spin.blockSignals(False)
        self._update_hint()

    def _on_changed(self, *_args) -> None:
        self._update_hint()
        self.changed.emit()

    def _update_hint(self) -> None:
        n = self.spin.value()
        if n and self._max:
            self.hint.setText(f"{n} of {self._max} GPU(s) on this server")
        elif n:
            self.hint.setText(f"{n} GPU(s) requested")
        else:
            self.hint.setText("")

    def value(self) -> str:
        n = self.spin.value()
        return str(n) if n else ""

    def set_value(self, text: str | None) -> None:
        n = parse_gpu_count(text)
        self.spin.blockSignals(True)
        self.spin.setValue(min(n, self.spin.maximum()))
        self.spin.blockSignals(False)
        self._update_hint()

    def clear(self) -> None:
        self.set_value("")

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
