"""좌측 네비게이션 - Task ▸ Work 드릴다운 (Option A).

트리 대신 한 번에 한 단계만 보여준다: Tasks 목록 -> (고르면) 그 Task 의 Works 목록
-> (고르면) 그 Work 의 Dataset 목록. 브레드크럼으로 언제든 위 단계로 돌아간다.
Work 까지 들어가면 그 Work 에 등록된 Dataset(이름 + 위치)을 바로 보여주고 손댈 수 있다.
"""
from __future__ import annotations

from typing import Any, Callable

from .. import editing, theme
from ..config_store import OptionsConfig
from ..db import Database
from ..qt import Qt, QtCore, QtGui, QtWidgets, Signal
from ..theme import icons
from ..utils import open_in_file_manager
from .common import monospace_font, toast
from .dataset_dialog import DatasetEditDialog


def _registered_bit(created_at: Any) -> str:
    """등록 시각(created_at)을 날짜만 잘라 메타 줄에 붙일 짧은 표기로."""
    return str(created_at or "").strip()[:10]


def _icon_button(
    parent: QtWidgets.QWidget, icon_name: str, tooltip: str
) -> QtWidgets.QToolButton:
    """아이콘만 있는 작은 툴버튼 (폴더/수정/삭제처럼 행마다 반복되는 것들)."""
    btn = QtWidgets.QToolButton(parent)
    btn.setIcon(icons.icon(icon_name, theme.color("text.secondary")))
    btn.setToolTip(tooltip)
    btn.setStyleSheet(
        "QToolButton { border: none; background: transparent; }"
    )
    return btn


class _ClickableRow(QtWidgets.QFrame):
    """왼쪽 클릭 = 드릴다운, 오른쪽 클릭 = 컨텍스트 메뉴(이름변경/삭제)."""

    clicked = Signal()
    rightClicked = Signal(QtCore.QPoint)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(event.globalPosition().toPoint())
        super().mousePressEvent(event)


class NavigationPanel(QtWidgets.QWidget):
    """Task ▸ Work 드릴다운 + 선택된 Work 의 Dataset 인라인 표시.

    선택이 바뀌면 `selectionChanged(task_id, work_id)` 를 보낸다.
    work 를 고르지 않고 task 만 고르면 work_id 는 -1 (해당 Task 전체 보기).
    """

    selectionChanged = Signal(int, int)

    def __init__(
        self,
        db: Database,
        config: OptionsConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.config = config
        self._task_id: int | None = None
        self._work_id: int | None = None
        self._initialized = False
        self._row_widgets: list[tuple[QtWidgets.QWidget, str]] = []

        self.breadcrumb_layout = QtWidgets.QHBoxLayout()
        self.breadcrumb_layout.setContentsMargins(0, 0, 0, 0)
        self.breadcrumb_layout.setSpacing(4)
        breadcrumb_host = QtWidgets.QWidget(self)
        breadcrumb_host.setLayout(self.breadcrumb_layout)

        self.meta_label = QtWidgets.QLabel("", self)
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet(f"color: {theme.color('text.muted')}; font-size: 11px;")

        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)

        self.list_host = QtWidgets.QWidget(self)
        self.list_layout = QtWidgets.QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(2)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.list_host)

        self.toolbar_layout = QtWidgets.QHBoxLayout()
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(4)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(breadcrumb_host)
        layout.addWidget(self.meta_label)
        layout.addWidget(self.filter_edit)
        layout.addWidget(scroll, 1)
        layout.addLayout(self.toolbar_layout)

        editing.install_shortcuts(
            self, on_rename=self._shortcut_rename, on_delete=self._shortcut_delete, on_add=self._shortcut_add
        )
        self.setMinimumWidth(300)
        self.refresh()

    # ==================================================================
    # 상태 조회 (main_window 가 쓴다)
    # ==================================================================
    def current_task_id(self) -> int | None:
        return self._task_id

    def current_work_id(self) -> int | None:
        return self._work_id

    # ==================================================================
    # 새로고침 / 이동
    # ==================================================================
    def refresh(self, select_work_id: int | None = None, select_task_id: int | None = None) -> None:
        if select_work_id:
            work = self.db.get_work(select_work_id)
            if work is not None:
                self._task_id = int(work["task_id"])
                self._work_id = int(select_work_id)
                self._initialized = True
        elif select_task_id:
            self._task_id = int(select_task_id)
            self._work_id = None
            self._initialized = True
        else:
            self._ensure_valid_selection()
        self._render()

    def _ensure_valid_selection(self) -> None:
        """기존 선택이 여전히 유효한지 확인하고, 처음 실행일 때만 첫 Task/Work 로 들어간다."""
        tasks = self.db.list_tasks()
        task_ids = {t["id"] for t in tasks}
        if self._task_id is not None and self._task_id not in task_ids:
            self._task_id = None
            self._work_id = None
        if self._work_id is not None:
            work = self.db.get_work(self._work_id)
            if work is None or (self._task_id is not None and int(work["task_id"]) != self._task_id):
                self._work_id = None
        if not self._initialized:
            self._initialized = True
            if self._task_id is None and tasks:
                # Prefer a Task that actually has a Work to land on, so a fresh
                # install doesn't open on an empty Task while another has runs.
                preferred = next((t for t in tasks if self.db.list_works(t["id"])), None)
                self._task_id = (preferred or tasks[0])["id"]
            if self._task_id is not None and self._work_id is None:
                works = self.db.list_works(self._task_id)
                if works:
                    self._work_id = works[0]["id"]

    def _go_root(self) -> None:
        self._task_id = None
        self._work_id = None
        self._render()

    def _go_task(self) -> None:
        self._work_id = None
        self._render()

    def _enter_task(self, task_id: int) -> None:
        self._task_id = task_id
        self._work_id = None
        self._render()

    def _enter_work(self, work_id: int) -> None:
        self._work_id = work_id
        self._render()

    # ==================================================================
    # 렌더링
    # ==================================================================
    def _render(self) -> None:
        self._build_breadcrumb()
        self.filter_edit.blockSignals(True)
        self.filter_edit.clear()
        self.filter_edit.blockSignals(False)

        if self._task_id is None:
            self.filter_edit.setVisible(True)
            self.filter_edit.setPlaceholderText("Search Tasks…")
            self.meta_label.setVisible(False)
            self._build_task_list()
            self._build_toolbar("root")
        elif self._work_id is None:
            self.filter_edit.setVisible(True)
            self.filter_edit.setPlaceholderText("Search Works…")
            self.meta_label.setVisible(False)
            self._build_work_list()
            self._build_toolbar("task")
        else:
            self.filter_edit.setVisible(False)
            self.meta_label.setVisible(True)
            self._build_work_detail()
            self._build_toolbar("work")

        self._emit_selection()

    def _emit_selection(self) -> None:
        self.selectionChanged.emit(int(self._task_id or -1), int(self._work_id or -1))

    def _build_breadcrumb(self) -> None:
        self._clear_layout(self.breadcrumb_layout)
        crumbs: list[tuple[str, Callable[[], None]]] = [("All Tasks", self._go_root)]
        if self._task_id is not None:
            task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
            crumbs.append((task["name"] if task else "?", self._go_task))
        if self._work_id is not None:
            work = self.db.get_work(self._work_id)
            crumbs.append((work["name"] if work else "?", self._go_task))

        for i, (label, handler) in enumerate(crumbs):
            if i > 0:
                sep = QtWidgets.QLabel("▸", self)
                sep.setStyleSheet(f"color: {theme.color('text.muted')}; font-size: 12px;")
                self.breadcrumb_layout.addWidget(sep)
            pill = QtWidgets.QToolButton(self)
            pill.setText(label)
            pill.clicked.connect(handler)
            radius = theme.METRICS["radius.medium"]
            if i == len(crumbs) - 1:
                pill.setStyleSheet(
                    f"QToolButton {{ background: {theme.color('accent.bg')}; color: {theme.color('text.primary')};"
                    f" border: 1px solid {theme.color('accent.border')}; border-radius: {radius}px;"
                    f" padding: 4px 10px; font-weight: 600; font-size: 12px; }}"
                )
            else:
                pill.setCursor(Qt.CursorShape.PointingHandCursor)
                pill.setStyleSheet(
                    f"QToolButton {{ background: {theme.color('bg.elevated')}; color: {theme.color('text.secondary')};"
                    f" border: 1px solid {theme.color('border.default')}; border-radius: {radius}px;"
                    f" padding: 4px 10px; font-size: 12px; }}"
                    f" QToolButton:hover {{ background: {theme.color('bg.hover')}; color: {theme.color('text.primary')}; }}"
                )
            self.breadcrumb_layout.addWidget(pill)
        self.breadcrumb_layout.addStretch(1)

    # -- Level 1: Tasks -------------------------------------------------------
    def _build_task_list(self) -> None:
        self._clear_layout(self.list_layout)
        self._row_widgets = []
        tasks = self.db.list_tasks()
        if not tasks:
            self.list_layout.addWidget(self._empty_hint("No Tasks yet — use + Task below."))
            self.list_layout.addStretch(1)
            return
        for task in tasks:
            works = self.db.list_works(task["id"])
            n_train = sum(self.db.counts_for_work(w["id"])[0] for w in works)
            n_eval = sum(self.db.counts_for_work(w["id"])[1] for w in works)
            row = self._make_row(task["name"], task.get("description") or "", f"{n_train} / {n_eval}", bold=True)
            row.clicked.connect(lambda t=task: self._enter_task(t["id"]))
            row.rightClicked.connect(lambda pos, t=task: self._task_context_menu(pos, t))
            self.list_layout.addWidget(row)
            self._row_widgets.append((row, task["name"].lower()))
        self.list_layout.addStretch(1)

    # -- Level 2: Works ---------------------------------------------------------
    def _build_work_list(self) -> None:
        self._clear_layout(self.list_layout)
        self._row_widgets = []
        works = self.db.list_works(self._task_id)
        if not works:
            self.list_layout.addWidget(self._empty_hint("No Works yet — use + Work below."))
            self.list_layout.addStretch(1)
            return
        for work in works:
            n_train, n_eval = self.db.counts_for_work(work["id"])
            row = self._make_row(work["name"], work.get("description") or "", f"{n_train} / {n_eval}")
            row.clicked.connect(lambda w=work: self._enter_work(w["id"]))
            row.rightClicked.connect(lambda pos, w=work: self._work_context_menu(pos, w))
            self.list_layout.addWidget(row)
            self._row_widgets.append((row, work["name"].lower()))
        self.list_layout.addStretch(1)

    def _make_row(self, title: str, subtitle: str, counts: str, bold: bool = False) -> _ClickableRow:
        row = _ClickableRow(self.list_host)
        radius = theme.METRICS["radius.medium"]
        row.setStyleSheet(
            f"_ClickableRow {{ border-radius: {radius}px; background: transparent; }}"
            f" _ClickableRow:hover {{ background: {theme.color('bg.hover')}; }}"
        )
        outer = QtWidgets.QVBoxLayout(row)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(1)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        name_label = QtWidgets.QLabel(title, row)
        name_label.setStyleSheet(
            f"color: {theme.color('text.primary')}; font-size: 12.5px; font-weight: {600 if bold else 500};"
        )
        counts_label = QtWidgets.QLabel(counts, row)
        counts_label.setFont(monospace_font(-1))
        counts_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        top.addWidget(name_label, 1)
        top.addWidget(counts_label)
        outer.addLayout(top)

        if subtitle:
            sub_label = QtWidgets.QLabel(subtitle, row)
            sub_label.setStyleSheet(f"color: {theme.color('text.muted')}; font-size: 10.5px;")
            outer.addWidget(sub_label)
        return row

    def _empty_hint(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text, self.list_host)
        label.setStyleSheet(f"color: {theme.color('text.muted')}; font-size: 11.5px; padding: 12px 4px;")
        label.setWordWrap(True)
        return label

    # -- Level 3: 선택된 Work 의 Dataset ------------------------------------------
    def _build_work_detail(self) -> None:
        self._clear_layout(self.list_layout)
        self._row_widgets = []
        work = self.db.get_work(self._work_id) if self._work_id else None
        if work is None:
            self._work_id = None
            return

        n_train, n_eval = self.db.counts_for_work(work["id"])
        desc = f"{work['description']}  ·  " if work.get("description") else ""
        self.meta_label.setText(f"{desc}{n_train} Train · {n_eval} Evaluation")

        header = QtWidgets.QLabel("DATASET", self.list_host)
        header.setStyleSheet(
            f"color: {theme.color('text.muted')}; font-size: 10px; font-weight: 600;"
            f" letter-spacing: 0.06em; padding: 6px 4px 2px;"
        )
        self.list_layout.addWidget(header)

        datasets = self.db.list_datasets(work["id"])
        if not datasets:
            self.list_layout.addWidget(self._empty_hint("No datasets registered for this Work yet."))
        else:
            box = QtWidgets.QFrame(self.list_host)
            radius = theme.METRICS["radius.large"]
            box.setStyleSheet(
                f"QFrame {{ background: {theme.color('bg.elevated')}; border: 1px solid {theme.color('border.subtle')};"
                f" border-radius: {radius}px; }}"
            )
            box_layout = QtWidgets.QVBoxLayout(box)
            box_layout.setContentsMargins(0, 0, 0, 0)
            box_layout.setSpacing(0)
            for i, dataset in enumerate(datasets):
                if i > 0:
                    line = QtWidgets.QFrame(box)
                    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                    line.setStyleSheet(f"color: {theme.color('border.subtle')}; max-height: 1px;")
                    box_layout.addWidget(line)
                box_layout.addWidget(self._make_dataset_row(dataset))
            self.list_layout.addWidget(box)

        add_btn = QtWidgets.QToolButton(self.list_host)
        add_btn.setText("＋ Add dataset…")
        add_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.setStyleSheet(
            f"QToolButton {{ color: {theme.color('accent')}; font-size: 11.5px; border: none;"
            f" background: transparent; padding: 6px 4px; }}"
        )
        add_btn.clicked.connect(self._add_dataset)
        self.list_layout.addWidget(add_btn)
        self.list_layout.addStretch(1)

    def _make_dataset_row(self, dataset: dict[str, Any]) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget(self.list_host)
        outer = QtWidgets.QVBoxLayout(row)
        outer.setContentsMargins(10, 7, 6, 7)
        outer.setSpacing(2)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        name_label = QtWidgets.QLabel(dataset["name"], row)
        name_label.setStyleSheet(f"color: {theme.color('text.primary')}; font-size: 12px; font-weight: 500;")
        top.addWidget(name_label)
        if dataset.get("variant"):
            variant_label = QtWidgets.QLabel(dataset["variant"], row)
            variant_label.setFont(monospace_font(-1))
            variant_label.setStyleSheet(
                f"color: {theme.color('text.muted')}; background: {theme.color('bg.input')};"
                f" border-radius: {theme.METRICS['radius.small']}px; padding: 0px 5px;"
            )
            top.addWidget(variant_label)
        if dataset.get("sample_count"):
            count_label = QtWidgets.QLabel(f"{dataset['sample_count']:,}", row)
            count_label.setFont(monospace_font(-1))
            count_label.setStyleSheet(f"color: {theme.color('text.muted')};")
            top.addWidget(count_label)
        top.addStretch(1)

        folder_btn = _icon_button(row, "folder", "Open dataset path in file manager")
        folder_btn.clicked.connect(lambda _=False, d=dataset: self._open_dataset_folder(d))
        edit_btn = _icon_button(row, "edit", "Edit dataset")
        edit_btn.clicked.connect(lambda _=False, d=dataset: self._edit_dataset(d))
        del_btn = _icon_button(row, "delete", "Remove dataset")
        del_btn.clicked.connect(lambda _=False, d=dataset: self._delete_dataset(d))
        top.addWidget(folder_btn)
        top.addWidget(edit_btn)
        top.addWidget(del_btn)
        outer.addLayout(top)

        meta_bits = [
            b for b in (
                dataset.get("image_size"),
                dataset.get("extension"),
                _registered_bit(dataset.get("created_at")),
            ) if b
        ]
        if meta_bits:
            meta_label = QtWidgets.QLabel(" · ".join(meta_bits), row)
            meta_label.setFont(monospace_font(-1))
            meta_label.setStyleSheet(f"color: {theme.color('text.muted')}; font-size: 10.5px;")
            outer.addWidget(meta_label)
        return row

    def _add_dataset(self) -> None:
        if self._work_id is None:
            return
        dialog = DatasetEditDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes, sample_count, image_size, extension, registered_at = dialog.result_values()
        if not name:
            return
        self.db.add_dataset(
            self._work_id, name, variant, path, notes, sample_count, image_size, extension, registered_at
        )
        self._render()

    def _open_dataset_folder(self, dataset: dict[str, Any]) -> None:
        ok, message = open_in_file_manager(dataset.get("path") or "")
        toast(self, ok, message, "Open Dataset Folder")

    def _edit_dataset(self, dataset: dict[str, Any]) -> None:
        dialog = DatasetEditDialog(self, dataset)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name, variant, path, notes, sample_count, image_size, extension, registered_at = dialog.result_values()
        if not name:
            return
        self.db.update_dataset(
            dataset["id"], name, variant, path, notes, sample_count, image_size, extension, registered_at
        )
        self._render()

    def _delete_dataset(self, dataset: dict[str, Any]) -> None:
        label = f"{dataset['name']} · {dataset['variant']}" if dataset.get("variant") else dataset["name"]
        used = self.db.count_runs_using_dataset(self._work_id, dataset["name"], dataset.get("variant") or "")
        if not editing.confirm_delete(self, label, used):
            return
        self.db.delete_dataset(dataset["id"])
        self._render()

    # -- Search -----------------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for widget, haystack in self._row_widgets:
            widget.setVisible(not needle or needle in haystack)

    # -- Toolbar ----------------------------------------------------------------
    def _build_toolbar(self, level: str) -> None:
        self._clear_layout(self.toolbar_layout)
        if level == "root":
            self.toolbar_layout.addWidget(self._toolbar_btn("Task", self.add_task, "add"))
        elif level == "task":
            self.toolbar_layout.addWidget(self._toolbar_btn("Work", self.add_work, "add"))
            self.toolbar_layout.addWidget(
                self._toolbar_btn("Edit Task", self._edit_current_task, "edit")
            )
            self.toolbar_layout.addWidget(
                self._toolbar_btn("Delete Task", self._delete_current_task, "delete")
            )
        else:
            self.toolbar_layout.addWidget(self._toolbar_btn("Work", self.add_work, "add"))
            self.toolbar_layout.addWidget(
                self._toolbar_btn("Edit Work", self._edit_current_work, "edit")
            )
            self.toolbar_layout.addWidget(
                self._toolbar_btn("Delete Work", self._delete_current_work, "delete")
            )
        self.toolbar_layout.addStretch(1)

    @staticmethod
    def _toolbar_btn(
        text: str, handler: Callable[[], None], icon_name: str
    ) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setIcon(icons.icon(icon_name, theme.color("text.secondary")))
        btn.setText(text)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.clicked.connect(handler)
        return btn

    # ==================================================================
    # CRUD - Task / Work
    # ==================================================================
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
        self._task_id = task_id
        self._work_id = None
        self._initialized = True
        self._render()

    def add_work(self) -> None:
        if self._task_id is None:
            QtWidgets.QMessageBox.information(self, "Add Work", "Select a Task first.")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Work ID", "Work ID (e.g. SSL2SL):")
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(self, "Add Work ID", "Description (optional):")
        work_id = self.db.add_work(self._task_id, name.strip(), desc.strip())
        self._work_id = work_id
        self._render()

    def _edit_current_task(self) -> None:
        if self._task_id is None:
            return
        task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
        if task is not None:
            self._edit_task_or_work("Task", task, is_task=True)

    def _edit_current_work(self) -> None:
        if self._work_id is None:
            return
        work = self.db.get_work(self._work_id)
        if work is not None:
            self._edit_task_or_work("Work ID", work, is_task=False)

    def _edit_task_or_work(self, label: str, row: dict[str, Any], is_task: bool) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, f"Edit {label}", "Name:", text=row["name"])
        if not ok or not name.strip():
            return
        desc, _ = QtWidgets.QInputDialog.getText(
            self, f"Edit {label}", "Description:", text=row.get("description") or ""
        )
        if is_task:
            self.db.update_task(row["id"], name.strip(), desc.strip())
        else:
            self.db.update_work(row["id"], name.strip(), desc.strip())
        self._render()

    def _delete_current_task(self) -> None:
        if self._task_id is None:
            return
        task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
        if task is not None:
            self._delete_task(task)

    def _delete_current_work(self) -> None:
        if self._work_id is None:
            return
        work = self.db.get_work(self._work_id)
        if work is not None:
            self._delete_work(work)

    def _delete_task(self, task: dict[str, Any]) -> None:
        works = self.db.list_works(task["id"])
        message = f"This Task, its {len(works)} Work(s), and all their run records will be deleted.\nContinue?"
        if not editing.confirm(self, "Confirm Delete", message):
            return
        self.db.delete_task(task["id"])
        if self._task_id == task["id"]:
            self._task_id = None
            self._work_id = None
        self._render()

    def _delete_work(self, work: dict[str, Any]) -> None:
        n_train, n_eval = self.db.counts_for_work(work["id"])
        message = f"This Work and its {n_train} Train / {n_eval} Evaluation record(s) will be deleted.\nContinue?"
        if not editing.confirm(self, "Confirm Delete", message):
            return
        self.db.delete_work(work["id"])
        if self._work_id == work["id"]:
            self._work_id = None
        self._render()

    def _task_context_menu(self, pos: QtCore.QPoint, task: dict[str, Any]) -> None:
        menu = editing.build_item_menu(
            self,
            add_label="Add DL Task",
            on_add=self.add_task,
            rename_label=f"Edit '{task['name']}'",
            on_rename=lambda: self._edit_task_or_work("Task", task, is_task=True),
            delete_label=f"Delete '{task['name']}'",
            on_delete=lambda: self._delete_task(task),
        )
        menu.exec(pos)

    def _work_context_menu(self, pos: QtCore.QPoint, work: dict[str, Any]) -> None:
        menu = editing.build_item_menu(
            self,
            add_label="Add Work ID",
            on_add=self.add_work,
            rename_label=f"Edit '{work['name']}'",
            on_rename=lambda: self._edit_task_or_work("Work ID", work, is_task=False),
            delete_label=f"Delete '{work['name']}'",
            on_delete=lambda: self._delete_work(work),
        )
        menu.exec(pos)

    # -- 단축키 (F2 / Del / Ins) - 현재 단계의 "활성 대상"에 적용 -------------------
    def _shortcut_rename(self) -> None:
        if self._work_id is not None:
            self._edit_current_work()
        elif self._task_id is not None:
            self._edit_current_task()

    def _shortcut_delete(self) -> None:
        if self._work_id is not None:
            self._delete_current_work()
        elif self._task_id is not None:
            self._delete_current_task()

    def _shortcut_add(self) -> None:
        if self._task_id is None:
            self.add_task()
        else:
            self.add_work()

    # -- 내부 유틸 -----------------------------------------------------------------
    @staticmethod
    def _clear_layout(layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            sub_layout = item.layout()
            if sub_layout is not None:
                NavigationPanel._clear_layout(sub_layout)
