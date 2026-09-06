"""Train / Evaluation dashboard panel.

Layout::

    ┌ Filter · sort · CSV/clipboard toolbar ─────────────────────┐
    │ QTableView (click header to sort)                          │
    ├──────────────────────────────────────────────────────────── ┤
    │ Selected run detail (paths + open folder / command / config)│
    └───────────────────────────────────────────────────────────── ┘

The "register / edit run" form used to sit permanently in a third pane.
It is now a popup dialog (`self.form_dialog`) shown only when adding or
editing a run, so the table gets the full width the rest of the time.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Sequence

from .. import APP_NAME, ORG_NAME
from .. import constants as C
from .. import editing, theme
from ..command_builder import RenderedCommand, render_command
from ..config_store import MetricDef, OptionsConfig
from ..db import Database
from ..log_parser import canonical_metric_name, parse_loss_log, parse_train_config
from ..theme import icons
from ..models import (
    FIELD_SPECS,
    PATH_KEYS,
    RAW_ROLE,
    ColumnSpec,
    RunFilterProxy,
    RunTableModel,
    build_columns,
)
from ..qt import Qt, QtCore, QtWidgets, Signal
from ..utils import (
    format_duration,
    format_number,
    loads_metrics,
    metrics_to_text,
    now_iso,
    open_in_file_manager,
    parse_duration,
    parse_gpu_count,
    render_html_report,
    render_markdown_report,
    scan_result_folder,
    write_csv,
)
from .common import (
    EditableCombo,
    GpuSelector,
    LabeledText,
    ManagedCombo,
    MetricsEditor,
    OpenFolderButton,
    PathEdit,
    ServerCombo,
    colorize_status_items,
    copy_to_clipboard,
    monospace_font,
    table_selection_to_tsv,
    toast,
)
from .compare_dialog import CompareRunsDialog
from .curve_chart import CurveDialog
from .dataset_dialog import DatasetCombo, DatasetManagerDialog
from .image_viewer import ImageViewerDialog
from .log_viewer import LogViewerDialog


class AddColumnDialog(QtWidgets.QDialog):
    """Add a table column - pick an existing field or type a new name."""

    def __init__(self, parent: QtWidgets.QWidget | None, candidates: Sequence[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Column")
        self.setMinimumWidth(340)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.combo.addItems(list(candidates))
        self.combo.setCurrentIndex(-1)
        self.combo.lineEdit().setPlaceholderText("Pick an existing field or type a new name")

        self.metric_check = QtWidgets.QCheckBox("If new, register it as a metric for this Task", self)
        self.metric_check.setChecked(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Column to add to the table", self))
        layout.addWidget(self.combo)
        layout.addWidget(self.metric_check)
        layout.addWidget(
            QtWidgets.QLabel(
                f"<span style='color:{theme.color('text.muted')}'>"
                "Saved to this Task's columns file.</span>",
                self,
            )
        )
        layout.addWidget(buttons)

    def value(self) -> str:
        return self.combo.currentText().strip()

    def as_metric(self) -> bool:
        return self.metric_check.isChecked()


class MetricSettingsDialog(QtWidgets.QDialog):
    """Metric display settings - unit / decimal digits / direction."""

    def __init__(self, parent: QtWidgets.QWidget | None, metric: MetricDef) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Metric Settings · {metric.key}")
        self._key = metric.key

        self.unit_edit = QtWidgets.QLineEdit(metric.unit, self)
        self.unit_edit.setPlaceholderText("e.g. dB, % (optional)")
        self.digits_spin = QtWidgets.QSpinBox(self)
        self.digits_spin.setRange(0, 8)
        self.digits_spin.setValue(metric.digits)
        self.direction = QtWidgets.QComboBox(self)
        self.direction.addItem("Higher is better", True)
        self.direction.addItem("Lower is better", False)
        self.direction.setCurrentIndex(0 if metric.higher_is_better else 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("Unit:", self.unit_edit)
        form.addRow("Decimal digits:", self.digits_spin)
        form.addRow("Direction:", self.direction)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_metric(self) -> MetricDef:
        return MetricDef(
            key=self._key,
            unit=self.unit_edit.text().strip(),
            digits=self.digits_spin.value(),
            higher_is_better=bool(self.direction.currentData()),
        )


class BaseRunPanel(QtWidgets.QWidget):
    """Shared Train / Evaluation dashboard."""

    KIND: str = "train"
    METRIC_PRESETS: Sequence[str] = ()
    DETAIL_PATHS: Sequence[tuple[str, str]] = ()
    SAMPLE_COMMAND: str = ""
    SHOW_SERVER: bool = True
    SHOW_GPU: bool = True
    SHOW_TRAINING_CURVE: bool = True
    # Train 패널에서만 "이 학습으로 평가 만들기" 를 띄운다.
    OFFERS_EVALUATION_HANDOFF: bool = False

    runsChanged = Signal()
    configChanged = Signal()
    # Train 표에서 "이 학습으로 평가 만들기" 를 고르면 run id 를 실어 보낸다
    # (받는 쪽은 main_window - Evaluation 탭으로 옮기고 폼을 채운다).
    evaluationRequested = Signal(int)

    def __init__(
        self,
        db: Database,
        config: OptionsConfig,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.config = config
        self._task_id: int | None = None
        self._task_name: str | None = None
        self._work_id: int | None = None
        self._editing_id: int | None = None
        # 즐겨찾기는 표/상세에서만 토글하고 폼에는 입력란이 없으므로, 편집 중인 값을
        # 여기 들고 있다가 저장할 때 그대로 되돌려 준다 (안 그러면 매 저장마다 꺼진다).
        self._editing_favorite: bool = False
        self._hidden_headers: set[str] = set()
        self._custom_widgets: dict[str, ManagedCombo] = {}
        self._column_settings = QtCore.QSettings(ORG_NAME, APP_NAME)
        self._restoring_columns = False

        self.model = RunTableModel(self)
        self.proxy = RunFilterProxy(self)
        self.proxy.setSourceModel(self.model)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._build_table_area())
        splitter.addWidget(self._build_detail_area())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([560, 340])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        self._build_form_dialog()
        self._clear_detail()
        self._update_form_buttons()

    # ==================================================================
    # 1) Table area
    # ==================================================================
    def _build_table_area(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)

        new_run_btn = QtWidgets.QPushButton("+ New Run", container)
        new_run_btn.setProperty("variant", "cta")
        new_run_btn.setToolTip("Open a form to register a new run.")
        new_run_btn.clicked.connect(self.open_new_run_dialog)

        self.filter_edit = QtWidgets.QLineEdit(container)
        self.filter_edit.setPlaceholderText("Search all columns (model, path, metric value...)")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.proxy.set_text_filter)

        self.status_filter = QtWidgets.QComboBox(container)
        self.status_filter.addItem("All statuses", "")
        for status in C.STATUS_LIST:
            self.status_filter.addItem(f"●  {status}", status)
        colorize_status_items(self.status_filter, C.STATUS_LIST, offset=1)
        self.status_filter.currentIndexChanged.connect(
            lambda: self.proxy.set_status_filter(self.status_filter.currentData())
        )

        self.favorites_btn = QtWidgets.QToolButton(container)
        self.favorites_btn.setText("★ Favorites")
        self.favorites_btn.setCheckable(True)
        self.favorites_btn.setToolTip("Show only favorited runs.")
        self.favorites_btn.toggled.connect(self.proxy.set_favorites_only)

        refresh_btn = QtWidgets.QToolButton(container)
        refresh_btn.setText("↻ Refresh")
        refresh_btn.clicked.connect(self.reload)

        export_btn = QtWidgets.QToolButton(container)
        export_btn.setText("⤓ Export")
        export_btn.setToolTip("Export the currently filtered/sorted table.")
        export_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QtWidgets.QMenu(export_btn)
        export_menu.addAction("Export to CSV…", self.export_csv)
        export_menu.addAction("Export Report (Markdown/HTML)…", self.export_report)
        export_btn.setMenu(export_menu)

        copy_btn = QtWidgets.QToolButton(container)
        copy_btn.setText("⧉ Copy")
        copy_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        copy_menu = QtWidgets.QMenu(copy_btn)
        copy_menu.addAction("Copy Selected Rows", lambda: self.copy_table(selected_only=True))
        copy_menu.addAction("Copy Entire Table", lambda: self.copy_table(selected_only=False))
        copy_btn.setMenu(copy_menu)

        dup_btn = QtWidgets.QToolButton(container)
        dup_btn.setText("⎘ Duplicate")
        dup_btn.setToolTip("Duplicate the selected run with the same settings (status becomes queued).")
        dup_btn.clicked.connect(self.duplicate_selected)

        compare_btn = QtWidgets.QToolButton(container)
        compare_btn.setText("⇄ Compare")
        compare_btn.setToolTip("Compare 2-3 selected runs side by side (Ctrl/Cmd-click rows).")
        compare_btn.clicked.connect(self.compare_selected)

        del_btn = QtWidgets.QToolButton(container)
        del_btn.setIcon(icons.icon("delete", theme.color("text.secondary")))
        del_btn.setText("Delete")
        del_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        del_btn.clicked.connect(self.delete_selected)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addWidget(new_run_btn)
        toolbar.addWidget(self.filter_edit, 1)
        toolbar.addWidget(self.status_filter)
        toolbar.addWidget(self.favorites_btn)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(export_btn)
        toolbar.addWidget(copy_btn)
        toolbar.addWidget(dup_btn)
        toolbar.addWidget(compare_btn)
        toolbar.addWidget(del_btn)

        self.view = QtWidgets.QTableView(container)
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)  # click header to sort
        self.view.sortByColumn(0, Qt.SortOrder.DescendingOrder)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setWordWrap(False)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._table_context_menu)
        self.view.doubleClicked.connect(self._on_double_click)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(24)

        header = self.view.horizontalHeader()
        header.setSectionsMovable(True)
        header.setStretchLastSection(True)
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._header_context_menu)
        header.sectionMoved.connect(self._on_column_geometry_changed)
        header.sectionResized.connect(self._on_column_geometry_changed)
        header.setToolTip("Click a header to sort, right-click to manage columns")

        self.view.selectionModel().selectionChanged.connect(lambda *_: self._on_selection_changed())
        self._context_column: ColumnSpec | None = None
        editing.install_shortcuts(
            header,
            on_rename=self._rename_current_column,
            on_add=self.add_column,
        )

        self.count_label = QtWidgets.QLabel("", container)
        self.count_label.setStyleSheet("color: #666;")

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(toolbar)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.count_label)
        return container

    # ==================================================================
    # 2) Detail area (command / config.yml / paths)
    # ==================================================================
    def _build_detail_area(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)

        self.detail_title = QtWidgets.QLabel("Select a row to see details.", container)
        self.detail_title.setStyleSheet("font-weight: bold;")
        self.detail_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.detail_meta = QtWidgets.QLabel("", container)
        self.detail_meta.setStyleSheet("color: #555;")
        self.detail_meta.setWordWrap(True)
        self.detail_meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.paths_box = QtWidgets.QGroupBox("Paths", container)
        paths_box = self.paths_box
        paths_layout = QtWidgets.QFormLayout(paths_box)
        paths_layout.setContentsMargins(8, 8, 8, 8)
        paths_layout.setSpacing(4)
        self._detail_path_edits: dict[str, QtWidgets.QLineEdit] = {}
        for key, label in self.DETAIL_PATHS:
            edit = QtWidgets.QLineEdit(paths_box)
            edit.setReadOnly(True)
            edit.setFont(monospace_font())
            copy_btn = QtWidgets.QToolButton(paths_box)
            copy_btn.setText("Copy")
            copy_btn.clicked.connect(
                lambda _=False, e=edit, l=label: copy_to_clipboard(e.text(), self, l)
            )
            open_btn = OpenFolderButton(paths_box)
            open_btn.set_path_provider(lambda e=edit: e.text())
            row = QtWidgets.QWidget(paths_box)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(edit, 1)
            row_layout.addWidget(copy_btn)
            row_layout.addWidget(open_btn)
            paths_layout.addRow(f"{label}:", row)
            self._detail_path_edits[key] = edit

        self.detail_command = LabeledText(
            "Execution Command", container, read_only=True, min_height=90
        )
        self.detail_config = LabeledText("config.yml", container, read_only=True, min_height=90)
        self.detail_notes = LabeledText("Metrics & Notes", container, read_only=True, min_height=90)
        self.detail_history = LabeledText("History", container, read_only=True, mono=False, min_height=90)

        self.detail_tabs = QtWidgets.QTabWidget(container)
        self.detail_tabs.addTab(self.detail_command, "Command")
        self.detail_tabs.addTab(self.detail_config, "config.yml")
        self.detail_tabs.addTab(self.detail_notes, "Metrics / Notes")
        self.detail_tabs.addTab(self.detail_history, "History")

        self.detail_favorite_btn = QtWidgets.QPushButton("☆ Favorite", container)
        self.detail_favorite_btn.clicked.connect(lambda: self.toggle_favorite())

        log_btn = QtWidgets.QPushButton("📄 View Log", container)
        log_btn.setToolTip("Show the last lines of this run's log file (from its result folder).")
        log_btn.clicked.connect(self.view_log)

        image_btn = QtWidgets.QPushButton("🖼 View Image", container)
        image_btn.setToolTip("Show one representative image from this run's result folder.")
        image_btn.clicked.connect(self.view_image)

        edit_btn = QtWidgets.QPushButton("Edit This Run", container)
        edit_btn.setIcon(icons.icon("edit", theme.color("text.secondary")))
        edit_btn.clicked.connect(self.open_edit_dialog)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addWidget(self.detail_favorite_btn)
        action_row.addWidget(log_btn)
        action_row.addWidget(image_btn)
        if self.SHOW_TRAINING_CURVE:
            curve_btn = QtWidgets.QPushButton("📈 Training Curve", container)
            curve_btn.setToolTip("Parse the result folder's log and plot a metric over iterations.")
            curve_btn.clicked.connect(self.view_curve)
            action_row.addWidget(curve_btn)
        action_row.addWidget(edit_btn, 1)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.detail_title)
        layout.addWidget(self.detail_meta)
        layout.addWidget(paths_box)
        layout.addWidget(self.detail_tabs, 1)
        layout.addLayout(action_row)
        return container

    # ==================================================================
    # 3) Register / edit form (popup dialog)
    # ==================================================================
    def _build_form_dialog(self) -> None:
        self.form_dialog = QtWidgets.QDialog(self)
        self.form_dialog.setModal(True)
        self.form_dialog.setMinimumSize(900, 640)
        self.form_dialog.resize(1000, 700)
        dialog_layout = QtWidgets.QVBoxLayout(self.form_dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(self._build_form_area())

    def open_new_run_dialog(self) -> None:
        self.reset_form()
        self._show_form_dialog()

    def open_edit_dialog(self) -> None:
        if self.load_selected_into_form():
            self._show_form_dialog()

    def view_log(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run first.", "View Log")
            return
        dialog = LogViewerDialog(
            str(row.get("result_path") or ""),
            self,
            title=f"Log · Run #{row.get('id')}",
        )
        dialog.exec()

    def view_image(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run first.", "View Image")
            return
        dialog = ImageViewerDialog(
            str(row.get("result_path") or ""),
            self,
            title=f"Image · Run #{row.get('id')}",
        )
        dialog.exec()

    def view_curve(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run first.", "Training Curve")
            return
        dialog = CurveDialog(
            str(row.get("result_path") or ""),
            self,
            title=f"Training Curve · Run #{row.get('id')}",
        )
        dialog.exec()

    def compare_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) < 2:
            toast(self, False, "Select 2-3 runs to compare (Ctrl/Cmd-click rows).", "Compare")
            return
        if len(rows) > 3:
            toast(self, False, "Pick at most 3 runs to compare.", "Compare")
            return
        dialog = CompareRunsDialog(rows, self.config, self._task_name, self)
        dialog.exec()

    def _show_form_dialog(self) -> None:
        self.form_dialog.show()
        self.form_dialog.raise_()
        self.form_dialog.activateWindow()

    def _build_form_area(self) -> QtWidgets.QWidget:
        """팝업을 좌(실행 설정)/우(경로 + 실행 코드)로 나눈다."""
        container = QtWidgets.QWidget(self.form_dialog)
        outer_layout = QtWidgets.QVBoxLayout(container)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, container)
        splitter.setChildrenCollapsible(False)

        # -- LEFT: Work / Server / GPU / Model / Dataset / Status / ... --------
        left_scroll, left_inner = self._make_form_pane(splitter)
        self.form_layout = QtWidgets.QFormLayout(left_inner)
        self._style_form_layout(self.form_layout)

        self._build_left_fields(left_inner)

        self._custom_section = self._section("Task-Specific Fields", left_inner)
        self.form_layout.addRow(self._custom_section)
        self._custom_host = QtWidgets.QWidget(left_inner)
        self._custom_form = QtWidgets.QFormLayout(self._custom_host)
        self._custom_form.setContentsMargins(0, 0, 0, 0)
        self._custom_form.setSpacing(6)
        self._custom_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.form_layout.addRow(self._custom_host)

        # Train 의 Training Hyperparameters 는 여기(Task-Specific Fields 다음, Metrics 전)
        # 에 온다 - Evaluation 은 _build_left_fields 에서 이미 다 그려서 여기선 no-op.
        self._build_extra_form_rows(left_inner)

        self.metrics_editor = MetricsEditor(
            self.METRIC_PRESETS, left_inner, config=self.config, task_getter=self.current_task_name
        )
        self.metrics_editor.metricsDefined.connect(self._on_metrics_defined)
        self.form_layout.addRow(self._section("Evaluation Metrics", left_inner))
        self.form_layout.addRow(self.metrics_editor)

        left_scroll.setWidget(left_inner)

        # -- RIGHT: Paths + Execution Code / Config -----------------------------
        right_scroll, right_inner = self._make_form_pane(splitter)
        self.right_form_layout = QtWidgets.QFormLayout(right_inner)
        self._style_form_layout(self.right_form_layout)
        self._build_right_fields(right_inner)
        right_scroll.setWidget(right_inner)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 500])

        # -- Buttons - 스크롤 영역 밖에 둬서 폼을 내려도 항상 보이게 -------------------
        self.save_btn = QtWidgets.QPushButton("+ Register Run", container)
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_form)

        self.new_btn = QtWidgets.QPushButton("Clear", container)
        self.new_btn.setToolTip("Clear the form for a new entry (keeps this dialog open).")
        self.new_btn.clicked.connect(self.reset_form)

        self.cancel_btn = QtWidgets.QPushButton("Cancel", container)
        self.cancel_btn.clicked.connect(self.form_dialog.hide)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(12, 8, 12, 10)
        buttons.setSpacing(6)
        buttons.addWidget(self.save_btn, 2)
        buttons.addWidget(self.new_btn, 1)
        buttons.addWidget(self.cancel_btn, 1)

        outer_layout.addWidget(splitter, 1)
        outer_layout.addLayout(buttons)

        # 폼이 스크롤 영역 안에 있어서, 휠로 스크롤하다 커서가 콤보/스핀박스 위를
        # 지나면 그 값이 실수로 바뀌기 쉽다 - 이 폼 안의 모든 콤보/스핀박스에서 막는다.
        editing.disable_wheel_scrolling(container)

        # 서버·데이터셋·모델 등을 고르면 실행 명령어가 따라 만들어지게 한다.
        self._watch_command_sources()

        return container

    @staticmethod
    def _make_form_pane(parent: QtWidgets.QWidget) -> tuple[QtWidgets.QScrollArea, QtWidgets.QWidget]:
        scroll = QtWidgets.QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget(scroll)
        return scroll, inner

    @staticmethod
    def _style_form_layout(layout: QtWidgets.QFormLayout) -> None:
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    # ==================================================================
    # LEFT column - Train 의 기본 순서. Evaluation 은 이 메서드를 통째로 오버라이드한다.
    # ==================================================================
    def _build_left_fields(self, parent: QtWidgets.QWidget) -> None:
        self.work_combo = self._make_work_combo(parent)
        self.form_layout.addRow("Work ID:", self.work_combo)

        self.server_combo = self._make_server_combo(parent)
        if self.SHOW_SERVER:
            self.form_layout.addRow("Server:", self.server_combo)
        else:
            self.server_combo.setVisible(False)

        self.gpu_selector = GpuSelector(parent)
        if self.SHOW_GPU:
            self.form_layout.addRow("GPU:", self.gpu_selector)
        else:
            # 레이아웃에 올리지 않은 위젯은 부모 위 (0,0) 에 떠 있는 채로 계속
            # 보이므로, 아예 숨겨야 한다 (그냥 자식으로만 두면 안 됨).
            self.gpu_selector.setVisible(False)

        self.model_combo = self._make_option_combo("model", "Model", parent)
        self.form_layout.addRow("Model:", self.model_combo)

        dataset_row = self._make_dataset_row(parent)
        self.form_layout.addRow("Dataset:", dataset_row)

        self.status_combo = self._make_status_combo(parent)
        self.form_layout.addRow("Status:", self.status_combo)

        started_row = self._make_started_row(parent)
        self.form_layout.addRow("Started At:", started_row)

        self.duration_edit = self._make_duration_edit(parent)
        self.form_layout.addRow("Duration:", self.duration_edit)

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        """Subclasses add their own input fields here (Train: Training Hyperparameters).

        Called by `_build_form_area` right after Task-Specific Fields, before
        Evaluation Metrics - not from `_build_left_fields` itself, so it lands in
        the same spot regardless of how a subclass orders its other fields.
        """

    # -- 공용 위젯 팩토리 - LEFT 컬럼 순서를 재구성하는 서브클래스(Evaluation)도 함께 쓴다 ---
    @staticmethod
    def _make_work_combo(parent: QtWidgets.QWidget) -> EditableCombo:
        combo = EditableCombo([], parent, "Work ID (new if not found)")
        combo.setToolTip("The Work ID this run belongs to. Typing a new value creates it.")
        return combo

    def _make_server_combo(self, parent: QtWidgets.QWidget) -> ServerCombo:
        combo = ServerCombo(parent)
        combo.currentTextChanged.connect(self._on_server_changed)
        return combo

    def _make_dataset_row(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.dataset_combo = DatasetCombo(self.db, parent)
        self.dataset_combo.setToolTip(
            "Datasets registered for this Work. Picking one also fills in "
            f"{self._dataset_path_label()}."
        )
        self.dataset_combo.datasetSelected.connect(self._on_dataset_selected)
        manage_datasets_btn = QtWidgets.QToolButton(parent)
        manage_datasets_btn.setText("📦")
        manage_datasets_btn.setToolTip(
            "Manage all datasets registered for this Work "
            "(name, variant, path, sample count, image size, extension)."
        )
        manage_datasets_btn.clicked.connect(self._open_dataset_manager)
        row = QtWidgets.QWidget(parent)
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(self.dataset_combo, 1)
        row_layout.addWidget(manage_datasets_btn)
        return row

    @staticmethod
    def _make_status_combo(parent: QtWidgets.QWidget) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent)
        for status in C.STATUS_LIST:
            combo.addItem(f"●  {status}", status)
        combo.setCurrentIndex(C.STATUS_LIST.index(C.STATUS_QUEUED))
        colorize_status_items(combo, C.STATUS_LIST)
        return combo

    def _make_started_row(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self.started_edit = QtWidgets.QLineEdit(parent)
        self.started_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        now_btn = QtWidgets.QToolButton(parent)
        now_btn.setText("Now")
        now_btn.clicked.connect(lambda: self.started_edit.setText(now_iso()))
        row = QtWidgets.QWidget(parent)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.started_edit, 1)
        layout.addWidget(now_btn)
        return row

    @staticmethod
    def _make_duration_edit(parent: QtWidgets.QWidget) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(parent)
        edit.setPlaceholderText("e.g. 3h 20m / 01:30:00 / 5400 (seconds)")
        return edit

    # ==================================================================
    # RIGHT column - Paths + Execution Code / Config (양쪽 폼 공통)
    # ==================================================================
    def _build_right_fields(self, parent: QtWidgets.QWidget) -> None:
        self.dataset_path_edit = PathEdit(parent, "/mnt/data/DIV2K/train", compact=True)
        self.result_path_edit = PathEdit(parent, "/mnt/exp/SSL2SL/restormer_x4", compact=True)
        self.result_path_edit.folderDropped.connect(self._on_result_folder_dropped)
        parse_btn = QtWidgets.QToolButton(parent)
        parse_btn.setText("⇪ Parse")
        parse_btn.setToolTip(
            "Read config.yaml + the training log in Result Folder Path and fill in "
            "Model/Dataset/hyperparameters/Metrics/Duration automatically."
        )
        parse_btn.clicked.connect(self._parse_result_folder)
        result_path_row = QtWidgets.QWidget(parent)
        result_path_layout = QtWidgets.QHBoxLayout(result_path_row)
        result_path_layout.setContentsMargins(0, 0, 0, 0)
        result_path_layout.setSpacing(4)
        result_path_layout.addWidget(self.result_path_edit, 1)
        result_path_layout.addWidget(parse_btn)
        self.right_form_layout.addRow(self._section("Paths", parent))
        self.right_form_layout.addRow(self._dataset_path_label() + ":", self.dataset_path_edit)
        self.right_form_layout.addRow("Result Folder Path:", result_path_row)

        self.command_input = LabeledText(
            "Execution Command",
            parent,
            placeholder=self.SAMPLE_COMMAND,
            min_height=90,
            wrap=True,  # 생성된 Hydra 명령어는 길어서 가로 스크롤보다 줄바꿈이 낫다
        )
        generate_cmd_btn = QtWidgets.QToolButton(parent)
        generate_cmd_btn.setText("⚙ Generate")
        generate_cmd_btn.setToolTip(
            "Build the command from this form using the Task's template\n"
            f"(config/tasks/<Task>.yaml → commands.{self.KIND}).\n"
            "Values left blank drop their whole argument."
        )
        generate_cmd_btn.clicked.connect(self.generate_command)
        self.command_input.add_header_widget(generate_cmd_btn)

        sample_cmd_btn = QtWidgets.QToolButton(parent)
        sample_cmd_btn.setText("Sample")
        sample_cmd_btn.setToolTip("Fill in an example command.")
        sample_cmd_btn.clicked.connect(lambda: self.command_input.set_text(self.SAMPLE_COMMAND))
        self.command_input.add_header_widget(sample_cmd_btn)

        # 사용자가 직접 손댄 명령어는 절대 덮어쓰지 않는다. 손대기 전까지만
        # 폼 값이 바뀔 때마다 다시 생성해 준다(_sync_generated_command).
        self._command_dirty = False
        self.command_input.editor.textChanged.connect(self._on_command_edited)

        self.config_input = LabeledText(
            "config.yml", parent, placeholder="# Paste YAML config content here", min_height=140
        )
        load_cfg_btn = QtWidgets.QToolButton(parent)
        load_cfg_btn.setText("Load From File")
        load_cfg_btn.clicked.connect(self._load_config_from_file)
        self.config_input.add_header_widget(load_cfg_btn)

        self.notes_input = LabeledText(
            "Notes", parent, placeholder="Free-form notes", mono=False, min_height=70
        )

        self.tags_edit = QtWidgets.QLineEdit(parent)
        self.tags_edit.setPlaceholderText("comma-separated, e.g. paper-final, ablation")
        self.tags_edit.setToolTip("Free-form labels. Also searchable from the toolbar search box.")

        self.failure_reason_edit = QtWidgets.QLineEdit(parent)
        self.failure_reason_edit.setPlaceholderText("e.g. CUDA OOM at batch 32")
        self.status_combo.currentIndexChanged.connect(self._sync_failure_reason_visibility)

        self.right_form_layout.addRow(self._section("Execution Code / Config", parent))
        self.right_form_layout.addRow(self.command_input)
        self.right_form_layout.addRow(self.config_input)
        self.right_form_layout.addRow(self.notes_input)
        self.right_form_layout.addRow("Tags:", self.tags_edit)
        self.right_form_layout.addRow("Failure Reason:", self.failure_reason_edit)
        self._sync_failure_reason_visibility()

    # -- option combos ------------------------------------------------------------
    def current_task_name(self) -> str | None:
        return self._task_name

    def _make_option_combo(
        self, field: str, label: str, parent: QtWidgets.QWidget
    ) -> ManagedCombo:
        combo = ManagedCombo(
            field,
            label,
            parent,
            config=self.config,
            task_getter=self.current_task_name,
            usage_counter=lambda value, f=field: self._count_usage(f, value),
        )
        combo.optionsChanged.connect(self._on_options_changed)
        combo.renameRequested.connect(
            lambda old, new, f=field: self._bulk_rename(f, old, new)
        )
        return combo

    def _count_usage(self, field: str, value: str) -> int:
        if field in self.db._COUNTABLE:
            return self.db.count_runs_using(field, value)
        return self.db.count_extra_value(field, value)

    def _bulk_rename(self, field: str, old: str, new: str) -> None:
        if field in self.db._COUNTABLE:
            changed = self.db.rename_value_in_runs(field, old, new)
            if changed:
                self.reload()
                self.runsChanged.emit()
                toast(self, True, f"Updated {field} to '{new}' in {changed} existing record(s).")

    def _on_options_changed(self) -> None:
        self._rebuild_custom_fields()
        self.reload_columns()
        self.configChanged.emit()

    def _on_metrics_defined(self) -> None:
        self.reload_columns()
        self.configChanged.emit()

    def _on_server_changed(self, name: str) -> None:
        self.gpu_selector.set_server(self.config.server(name.strip()))

    def _sync_failure_reason_visibility(self) -> None:
        """Failure reason only matters once a run is marked failed - keep it out of the
        way otherwise so the form doesn't ask about failures that haven't happened."""
        is_failed = self.status_combo.currentData() == C.STATUS_FAILED
        self.right_form_layout.setRowVisible(self.failure_reason_edit, is_failed)

    def _dataset_path_label(self) -> str:
        return "Dataset Path"

    # ==================================================================
    # 실행 명령어 생성 (Task 의 commands 템플릿 + 지금 폼 값)
    # ==================================================================
    def _command_values(self) -> dict[str, str]:
        """템플릿 자리표시자에 넣을 값들. 서브클래스가 확장한다.

        사용자 정의 옵션 필드(`algo` 같은 것)도 그대로 이름을 쓴다 - Task 파일의
        `options:` 에 이름을 추가하면 폼에 콤보가 생기고 템플릿에서 바로 쓸 수 있다.
        """
        task = self._task_name or ""
        server_name = self.server_combo.current_text()
        server = self.config.server(server_name) if server_name else None
        gpu_count = parse_gpu_count(self.gpu_selector.value())
        work = self.db.get_work(self._work_id) if self._work_id else None

        values: dict[str, str] = {
            "task": task,
            "task_lower": task.lower(),
            "work": str(work["name"]) if work else "",
            "model": self.model_combo.current_text(),
            "dataset": self.dataset_combo.current_text(),
            "dataset_path": self.dataset_path_edit.path(),
            "result_path": self.result_path_edit.path(),
            "server": server_name,
            "host": server.host if server else "",
            "gpus": str(gpu_count) if gpu_count else "",
            # 0,1,2... - 개수만 알고 있으므로 앞에서부터 채운 예시 값이다.
            "cuda_devices": ",".join(str(i) for i in range(gpu_count)) if gpu_count else "",
            "status": self.status_combo.currentData() or "",
        }
        values.update(
            {name: combo.current_text() for name, combo in self._custom_widgets.items()}
        )
        return values

    def generate_command(self) -> None:
        """템플릿으로 명령어를 만들어 Execution Command 칸을 채운다."""
        rendered = self._render_command()
        self.command_input.set_text(rendered.text)
        self._command_dirty = False
        if rendered.unknown:
            toast(
                self,
                False,
                "The template uses placeholders this form does not know: "
                + ", ".join(f"{{{n}}}" for n in rendered.unknown)
                + "\n\nAdd them under this Task's `options:` (they become combo boxes), "
                "or fix the name in `commands:`.",
                "Generate Command",
            )

    def _render_command(self) -> RenderedCommand:
        template = self.config.command_template(self._task_name, self.KIND)
        return render_command(template, self._command_values())

    def _on_command_edited(self) -> None:
        # 프로그램이 채워 넣는 동안에는(_syncing_command) 손댄 것으로 치지 않는다.
        if not getattr(self, "_syncing_command", False):
            self._command_dirty = True

    def _sync_generated_command(self) -> None:
        """폼 값이 바뀌면 명령어를 다시 만든다 - 단, 사용자가 손대기 전까지만."""
        if self._command_dirty:
            return
        rendered = self._render_command()
        self._syncing_command = True
        try:
            self.command_input.set_text(rendered.text)
        finally:
            self._syncing_command = False

    def _watch_command_sources(self) -> None:
        """명령어에 들어가는 입력들이 바뀌면 다시 생성하도록 연결한다."""
        for widget in self._command_source_widgets():
            signal = getattr(widget, "currentTextChanged", None) or getattr(
                widget, "textChanged", None
            )
            if signal is not None:
                signal.connect(lambda *_: self._sync_generated_command())

    def _command_source_widgets(self) -> list[QtWidgets.QWidget]:
        widgets: list[QtWidgets.QWidget] = [
            self.model_combo, self.dataset_combo, self.server_combo,
            self.status_combo, self.dataset_path_edit.edit, self.result_path_edit.edit,
        ]
        widgets.extend(self._custom_widgets.values())
        return widgets

    @staticmethod
    def _section(title: str, parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(f"<span style='color:#1a73e8'><b>— {title} —</b></span>", parent)
        label.setContentsMargins(0, 8, 0, 0)
        return label

    def _load_config_from_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Config File", QtCore.QDir.homePath(), "YAML/JSON (*.yml *.yaml *.json);;All Files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                self.config_input.set_text(fp.read())
        except OSError as exc:
            toast(self, False, f"Could not read file:\n{exc}", "Load Config")
            return
        toast(self, True, f"Config loaded: {os.path.basename(path)}")

    def _on_result_folder_dropped(self, path: str) -> None:
        """A folder was dragged onto the Result Folder Path field (#9).

        Fills the path (already done by PathEdit itself) and parses it the
        same way the explicit "⇪ Parse" button does - drop the whole
        experiment folder and the boring part fills itself in.
        """
        self._parse_result_folder(path)

    def _parse_result_folder(self, path: str | None = None) -> None:
        """train.py's config.yaml + training log -> auto-fill the form.

        This is the "자동 로깅" entry point: read config.yaml (model/dataset/
        hyperparameters) and the training log (latest validation metrics +
        elapsed time) from the Result Folder Path and fill in the matching
        fields. Never overwrites text the user already typed into raw areas
        (config.yml paste box, Dataset Path, Duration) - safe to re-run.
        """
        path = path if path is not None else self.result_path_edit.path()
        if not path:
            toast(self, False, "Set Result Folder Path first.", "Parse Config + Log")
            return
        found = scan_result_folder(path)
        filled: list[str] = []

        config_path = found.get("config")
        if config_path:
            fields = parse_train_config(config_path)
            if fields.get("model"):
                self.model_combo.set_text(fields["model"])
                filled.append("Model")
            if fields.get("dataset"):
                self.dataset_combo.set_text(fields["dataset"])
                filled.append("Dataset")
            if fields.get("dataset_path") and not self.dataset_path_edit.path():
                self.dataset_path_edit.set_path(fields["dataset_path"])
                filled.append("Dataset Path")
            self._apply_parsed_hyperparams(fields, filled)
            for key, combo in self._custom_widgets.items():
                if fields.get(key):
                    combo.set_text(fields[key])
                    filled.append(key)
            if not self.config_input.text().strip():
                try:
                    with open(config_path, "r", encoding="utf-8") as fp:
                        self.config_input.set_text(fp.read())
                    filled.append("config.yml text")
                except OSError:
                    pass

        log_path = found.get("log")
        if log_path:
            log_result = parse_loss_log(log_path)
            if log_result.latest_metrics:
                self._apply_parsed_metrics(log_result.latest_metrics)
                filled.append("Metrics")
            if log_result.duration_sec and not self.duration_edit.text().strip():
                self.duration_edit.setText(format_duration(log_result.duration_sec))
                filled.append("Duration")

        if filled:
            toast(self, True, "Parsed from result folder: " + ", ".join(filled), "Parse Config + Log")
        else:
            # 정보성일 뿐 사용자 실수가 아니므로(예: 아직 config/log 가 없는 폴더) - 경고 팝업 대신
            # 상태바 메시지로 조용히 알린다.
            toast(
                self,
                True,
                "No config.yaml/log recognized in that folder (or nothing new to fill in).",
                "Parse Config + Log",
            )

    def _apply_parsed_hyperparams(self, fields: dict[str, str], filled: list[str]) -> None:
        """Subclasses fill their own hyperparameter fields here (Train: epochs/batch/lr/optimizer)."""

    def _apply_parsed_metrics(self, metrics: dict[str, float]) -> None:
        known = (
            {key.lower(): key for key in self.config.metric_keys(self._task_name)}
            if self._task_name
            else {}
        )
        for raw_key, value in metrics.items():
            display_key = known.get(raw_key.lower()) or canonical_metric_name(raw_key)
            self.metrics_editor.set_value(display_key, value)

    # ==================================================================
    # Scope / loading
    # ==================================================================
    def set_scope(self, task_id: int | None, work_id: int | None) -> None:
        self._task_id = task_id if task_id and task_id > 0 else None
        self._work_id = work_id if work_id and work_id > 0 else None
        self._task_name = self._resolve_task_name()
        self._rebuild_custom_fields()
        self.reload()

    def _resolve_task_name(self) -> str | None:
        if self._work_id:
            work = self.db.get_work(self._work_id)
            if work:
                return str(work["task_name"])
        if self._task_id:
            task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
            if task:
                return str(task["name"])
        return None

    def reload(self) -> None:
        rows = self._fetch_rows()
        columns = build_columns(
            self.config,
            self._task_name,
            self.KIND,
            RunTableModel.metric_keys_in(rows),
        )
        self.model.set_content(rows, columns)
        self._apply_column_sizing()
        self._refresh_combo_sources()
        self._refresh_work_combo()
        self._update_count_label()
        self._clear_detail()
        self._update_form_buttons()

    def reload_columns(self) -> None:
        """Keep rows as-is and just rebuild the column layout (config changed)."""
        selected = self._current_row()
        self.reload()
        if selected:
            self._select_run(int(selected["id"]))

    def _fetch_rows(self) -> list[dict[str, Any]]:
        if self.KIND == "train":
            return self.db.list_train_runs(work_id=self._work_id, task_id=self._task_id)
        return self.db.list_evaluation_runs(work_id=self._work_id, task_id=self._task_id)

    def _apply_column_sizing(self) -> None:
        # 프로그램이 너비/순서/숨김을 다시 맞추는 동안은 사용자가 직접 조정한 것으로
        # 착각해 저장하지 않도록 막는다 - reload() 는 스코프를 바꿀 때마다 불린다.
        self._restoring_columns = True
        try:
            header = self.view.horizontalHeader()
            restored = self._restore_column_state()
            for index, spec in enumerate(self.model.columns()):
                if not restored:
                    self.view.setColumnWidth(index, spec.width)
                header.setSectionHidden(index, spec.header in self._hidden_headers)
        finally:
            self._restoring_columns = False

    # -- Column order/width persistence (per Task, per Train/Evaluation) -------
    def _column_settings_key(self) -> str:
        return f"columns/{self.KIND}/{self._task_name or '_default'}"

    def _restore_column_state(self) -> bool:
        data = self._column_settings.value(self._column_settings_key())
        if not isinstance(data, QtCore.QByteArray):
            return False
        return bool(self.view.horizontalHeader().restoreState(data))

    def _on_column_geometry_changed(self, *_args: Any) -> None:
        if self._restoring_columns:
            return
        self._column_settings.setValue(
            self._column_settings_key(), self.view.horizontalHeader().saveState()
        )

    def _refresh_combo_sources(self) -> None:
        """Base the list on config, but merge in legacy values only found in the DB."""
        scope = self._task_id  # keep values scoped so other Tasks' data doesn't leak in
        # Server 는 상단 Servers 목록(config.servers)에서만 고른다 - 여기서 관리하지 않는다.
        self.server_combo.set_items(self.config.server_names())
        self.model_combo.reload()
        self.model_combo.merge_items(self.db.distinct_values(self.KIND, "model", scope))
        # Dataset 선택은 Task 옵션이 아니라 이 Work 에 등록된 데이터셋 레지스트리와 연동된다.
        self.dataset_combo.set_work(self._work_id)
        self.metrics_editor.refresh_presets()
        for combo in self._custom_widgets.values():
            combo.reload()
        self._refresh_extra_combo_sources()

    # -- Task-specific custom fields --------------------------------------------
    def _rebuild_custom_fields(self) -> None:
        """Build combo rows for this Task's custom fields (scale, noise_sigma, ...)."""
        fields = self.config.custom_fields(self._task_name) if self.config else []
        current = {name: combo.current_text() for name, combo in self._custom_widgets.items()}

        while self._custom_form.count():
            item = self._custom_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)   # deleteLater alone leaves it on screen until next tick
                widget.deleteLater()
        self._custom_widgets = {}

        for name in fields:
            combo = self._make_option_combo(name, name, self._custom_host)
            combo.set_text(current.get(name, ""))
            self._custom_form.addRow(f"{name}:", combo)
            self._custom_widgets[name] = combo

        has_fields = bool(fields)
        self._custom_section.setVisible(has_fields)
        self._custom_host.setVisible(has_fields)
        for combo in self._custom_widgets.values():
            combo.currentTextChanged.connect(lambda *_: self._sync_generated_command())

    def _refresh_extra_combo_sources(self) -> None:
        """Subclasses refresh their own combo sources here."""

    def _refresh_work_combo(self) -> None:
        works = self.db.list_works(self._task_id) if self._task_id else []
        current = self.work_combo.current_text()
        self.work_combo.set_items([w["name"] for w in works], keep_text=False)
        if current and any(w["name"] == current for w in works):
            self.work_combo.set_text(current)
        elif self._work_id:
            work = self.db.get_work(self._work_id)
            self.work_combo.set_text(work["name"] if work else "")
        elif works:
            self.work_combo.set_text(works[0]["name"])
        else:
            self.work_combo.set_text("")

    # -- Work 별 데이터셋 레지스트리 -----------------------------------------------
    def _on_dataset_selected(self, row: dict[str, Any]) -> None:
        """Dataset 콤보에서 등록된 항목을 고르면(직접 추가한 경우도 포함) 경로를 함께 채운다."""
        if row.get("path"):
            self.dataset_path_edit.set_path(row["path"])

    def _open_dataset_manager(self) -> None:
        if not self._work_id:
            toast(self, False, "Select a Work on the left first.", "Manage Datasets")
            return
        work = self.db.get_work(self._work_id)
        dialog = DatasetManagerDialog(self.db, self._work_id, work["name"] if work else "", self)
        dialog.datasetsChanged.connect(lambda: self.dataset_combo.reload(keep_text=True))
        dialog.exec()
        self.dataset_combo.reload(keep_text=True)

    def _update_count_label(self) -> None:
        visible = self.proxy.rowCount()
        total = self.model.rowCount()
        scope = "All"
        if self._work_id:
            work = self.db.get_work(self._work_id)
            if work:
                scope = f"{work['task_name']} ▸ {work['name']}"
        elif self._task_id:
            task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
            if task:
                scope = f"{task['name']} (All)"
        self.count_label.setText(f"Scope: {scope}   ·   Showing {visible} / {total}")

    # ==================================================================
    # Selection / detail
    # ==================================================================
    def _selected_rows(self) -> list[dict[str, Any]]:
        selection = self.view.selectionModel()
        if selection is None:
            return []
        rows: list[dict[str, Any]] = []
        for index in selection.selectedRows():
            data = self.proxy.data(index, RAW_ROLE)
            if isinstance(data, dict):
                rows.append(data)
        return rows

    def _current_row(self) -> dict[str, Any] | None:
        rows = self._selected_rows()
        if rows:
            return rows[0]
        index = self.view.currentIndex()
        if index.isValid():
            data = self.proxy.data(index, RAW_ROLE)
            if isinstance(data, dict):
                return data
        return None

    def _on_selection_changed(self) -> None:
        row = self._current_row()
        if row is None:
            self._clear_detail()
        else:
            self._show_detail(row)

    def _clear_detail(self) -> None:
        self.detail_title.setText("Select a row to see details.")
        self.detail_meta.setText("")
        self.detail_favorite_btn.setText("☆ Favorite")
        self.detail_favorite_btn.setProperty("variant", None)
        self.detail_favorite_btn.style().unpolish(self.detail_favorite_btn)
        self.detail_favorite_btn.style().polish(self.detail_favorite_btn)
        for edit in self._detail_path_edits.values():
            edit.clear()
        self.detail_command.clear()
        self.detail_config.clear()
        self.detail_notes.clear()
        self.detail_history.clear()
        # Paths / Execution Command 등은 실행을 하나 고르기 전엔 보여 줄 내용이 없다.
        self.paths_box.setVisible(False)
        self.detail_tabs.setVisible(False)

    def _show_detail(self, row: dict[str, Any]) -> None:
        self.paths_box.setVisible(True)
        self.detail_tabs.setVisible(True)
        star = "★ " if row.get("favorite") else ""
        title = (
            f"{star}#{row.get('id')}  ·  {row.get('task_name', '')} ▸ {row.get('work_name', '')}  ·  "
            f"{row.get('model') or '-'}  @  {row.get('server') or '-'}"
        )
        self.detail_title.setText(title)
        self.detail_meta.setText(self._detail_meta_text(row))
        self.detail_favorite_btn.setText("★ Favorited" if row.get("favorite") else "☆ Favorite")
        self.detail_favorite_btn.setProperty("variant", "primary" if row.get("favorite") else None)
        self.detail_favorite_btn.style().unpolish(self.detail_favorite_btn)
        self.detail_favorite_btn.style().polish(self.detail_favorite_btn)
        for key, edit in self._detail_path_edits.items():
            edit.setText(str(row.get(key) or ""))
        self.detail_command.set_text(row.get("exec_command") or "")
        self.detail_config.set_text(row.get("config_yaml") or "")

        metrics = loads_metrics(row.get("metrics_json"))
        notes_text = metrics_to_text(metrics, sep="\n") or "(no metrics recorded)"
        notes = (row.get("notes") or "").strip()
        tags = (row.get("tags") or "").strip()
        failure_reason = (row.get("failure_reason") or "").strip()
        sections = [f"[Metrics]\n{notes_text}", f"[Notes]\n{notes or '(none)'}"]
        if tags:
            sections.append(f"[Tags]\n{tags}")
        if failure_reason:
            sections.append(f"[Failure Reason]\n{failure_reason}")
        self.detail_notes.set_text("\n\n".join(sections))
        self.detail_history.set_text(self._format_history(int(row["id"])))

    def _format_history(self, run_id: int) -> str:
        entries = self.db.list_history(self.KIND, run_id)
        if not entries:
            return "(no history recorded)"
        lines = []
        for entry in entries:
            when = entry.get("created_at") or ""
            action = str(entry.get("action") or "").capitalize()
            detail = entry.get("detail") or ""
            lines.append(f"{when}  ·  {action}\n{detail}" if detail else f"{when}  ·  {action}")
        return "\n\n".join(lines)

    def _detail_meta_text(self, row: dict[str, Any]) -> str:
        status = str(row.get("status") or "")
        duration = format_duration(row.get("duration_sec"))
        gpu_count = parse_gpu_count(row.get("gpu_indices"))
        gpu_text = f"{gpu_count} GPU(s)" if gpu_count else "-"
        parts = [
            f"● {status}",
            f"{row.get('server') or '-'} · {gpu_text}",
            f"Started {row.get('started_at') or '-'}",
            f"Duration {duration or '-'}",
            f"Logged {row.get('created_at') or '-'}",
        ]
        return "   |   ".join(p for p in parts if p.strip())

    # ==================================================================
    # Context menus
    # ==================================================================
    def _table_context_menu(self, pos) -> None:
        index = self.view.indexAt(pos)
        if index.isValid():
            self.view.selectRow(index.row())
        row = self._current_row()
        menu = QtWidgets.QMenu(self)

        if row is not None:
            star_label = "☆ Remove from Favorites" if row.get("favorite") else "★ Mark as Favorite"
            menu.addAction(star_label, lambda: self.toggle_favorite(row))
            menu.addSeparator()
            folder_icon = icons.icon("folder", theme.color("text.secondary"))
            for key, label in self.DETAIL_PATHS:
                path = str(row.get(key) or "")
                action = menu.addAction(folder_icon, f"Open {label}")
                action.setEnabled(bool(path))
                action.triggered.connect(lambda _=False, p=path: self._open_path(p))
            menu.addSeparator()
            menu.addAction("Copy Command", lambda: copy_to_clipboard(row.get("exec_command") or "", self, "Execution Command"))
            menu.addAction("Copy config.yml", lambda: copy_to_clipboard(row.get("config_yaml") or "", self, "config.yml"))
            menu.addSeparator()
            menu.addAction("📄 View Log", self.view_log)
            menu.addAction("🖼 View Image", self.view_image)
            if self.SHOW_TRAINING_CURVE:
                menu.addAction("📈 Training Curve", self.view_curve)
            menu.addAction(icons.icon("edit", theme.color("text.secondary")), "Edit This Run", self.open_edit_dialog)
            menu.addAction("⎘ Duplicate", self.duplicate_selected)
            if self.OFFERS_EVALUATION_HANDOFF:
                action = menu.addAction("▷ Create Evaluation Run from This")
                action.setToolTip("Open the Evaluation form prefilled from this training run.")
                action.triggered.connect(
                    lambda _=False, rid=int(row["id"]): self.evaluationRequested.emit(rid)
                )
            menu.addSeparator()

        menu.addAction("⇄ Compare Selected (2-3)", self.compare_selected)
        menu.addAction("Copy Selected Rows (TSV)", lambda: self.copy_table(selected_only=True))
        menu.addAction("Export Entire Table to CSV", self.export_csv)
        menu.addAction("Export Report (Markdown/HTML)", self.export_report)
        if row is not None:
            menu.addSeparator()
            menu.addAction(icons.icon("delete", theme.color("text.secondary")), "Delete Selected", self.delete_selected)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _header_context_menu(self, pos) -> None:
        header = self.view.horizontalHeader()
        section = header.logicalIndexAt(pos)
        spec = self.model.column_spec(section)
        self._context_column = spec

        menu = QtWidgets.QMenu(self)
        if spec is not None:
            menu.addAction(f"+ Add Column...\t{editing.ADD_KEY}", self.add_column)
            if spec.editable_label:
                menu.addAction(
                    f"Rename '{self.model.header_text(spec)}'\t{editing.RENAME_KEY}",
                    lambda: self.rename_column(spec),
                )
            if self._can_remove_column(spec):
                menu.addAction(
                    f"Remove Column '{self.model.header_text(spec)}'", lambda: self.remove_column(spec)
                )
            if spec.is_metric:
                menu.addAction(
                    f"Metric Settings for '{spec.source_name}'...", lambda: self.edit_metric(spec)
                )
            menu.addSeparator()
        menu.addAction("Hide This Column", lambda: self._hide_column(section))
        menu.addAction("Restore Default Columns", self.reset_columns)
        menu.addSeparator()

        presets = menu.addMenu("Column Presets")
        presets.addAction("Simple", self.apply_preset_simple)
        presets.addAction("Paper-ready (metrics only)", self.apply_preset_paper)
        presets.addAction("Full (show all)", self.apply_preset_full)
        menu.addSeparator()

        visibility = menu.addMenu("Visible Columns")
        for index, column in enumerate(self.model.columns()):
            label = self.model.header_text(column)
            action = visibility.addAction(label)
            action.setCheckable(True)
            action.setChecked(label not in self._hidden_headers)
            action.triggered.connect(
                lambda checked, h=label, i=index: self._toggle_column(h, i, checked)
            )
        menu.exec(header.mapToGlobal(pos))

    def _rename_current_column(self) -> None:
        spec = self._context_column or self.model.column_spec(
            self.proxy.mapToSource(self.view.currentIndex()).column()
        )
        if spec is not None and spec.editable_label:
            self.rename_column(spec)

    def _hide_column(self, section: int) -> None:
        spec = self.model.column_spec(section)
        if spec is not None:
            self._toggle_column(self.model.header_text(spec), section, False)

    # -- Column management (saved to config) ---------------------------------
    def _require_task(self) -> str | None:
        if not self._task_name:
            toast(self, False, "Select a Task on the left first.", "Manage Columns")
            return None
        return self._task_name

    def _current_column_ids(self) -> list[str]:
        configured = self.config.columns_for(self._task_name, self.KIND)
        if configured:
            return configured
        return [
            c.source_name
            for c in self.model.columns()
            if c.key not in ("id", "notes")
        ]

    def _can_remove_column(self, spec: ColumnSpec) -> bool:
        # id/favorite/notes come from LEADING_COLUMNS/TRAILING_COLUMNS, not from the
        # Task's configured column list, so "removing" them from config would be a no-op.
        return spec.key not in ("id", "favorite", "status", "notes")

    def add_column(self) -> None:
        task = self._require_task()
        if task is None:
            return
        used = set(self._current_column_ids())
        candidates = [key for key in FIELD_SPECS if key not in used and key not in ("id", "notes")]
        candidates += [m for m in self.config.metric_keys(task) if m not in used]
        candidates += [f for f in self.config.custom_fields(task) if f not in used]

        dialog = AddColumnDialog(self, candidates)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name = dialog.value()
        if not name:
            return
        columns = list(self._current_column_ids())
        if name in columns:
            toast(self, False, f"'{name}' column already exists.", "Add Column")
            return
        columns.append(name)
        self.config.set_columns(task, self.KIND, columns)
        if dialog.as_metric() and name not in self.config.metric_keys(task):
            self.config.add_metric(task, MetricDef(key=name))
        self.reload_columns()
        self.configChanged.emit()

    def rename_column(self, spec: ColumnSpec) -> None:
        task = self._require_task()
        if task is None:
            return
        old_label = self.model.header_text(spec)
        new = editing.prompt_text(self, "Rename Column", "New display name:", old_label)
        if not new or new == old_label:
            return
        if spec.is_metric:
            # For a metric, rename the config key itself (columns update along with it).
            self.config.rename_metric(task, spec.source_name, new)
        else:
            # Built-in/custom fields only get a display-name change; the data key stays.
            self.model.set_header_label(spec.key, new)
        self.reload_columns()
        self.configChanged.emit()

    def remove_column(self, spec: ColumnSpec) -> None:
        task = self._require_task()
        if task is None:
            return
        label = self.model.header_text(spec)
        if not editing.confirm(
            self,
            "Remove Column",
            f"Remove '{label}' from the {task} {self.KIND} table?\n"
            "Data is kept; it just won't be shown.",
        ):
            return
        columns = [c for c in self._current_column_ids() if c != spec.source_name]
        self.config.set_columns(task, self.KIND, columns)
        self.reload_columns()
        self.configChanged.emit()

    def edit_metric(self, spec: ColumnSpec) -> None:
        task = self._require_task()
        if task is None or not spec.is_metric:
            return
        metric = self.config.metric_def(task, spec.source_name) or MetricDef(key=spec.source_name)
        dialog = MetricSettingsDialog(self, metric)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        updated = dialog.result_metric()
        self.config.update_metric(
            task,
            spec.source_name,
            unit=updated.unit,
            digits=updated.digits,
            higher_is_better=updated.higher_is_better,
        )
        self.reload_columns()
        self.configChanged.emit()

    def reset_columns(self) -> None:
        task = self._require_task()
        if task is None:
            return
        if not editing.confirm(self, "Restore Columns", f"Reset the {task} {self.KIND} column layout to defaults?"):
            return
        from ..config_store import BUILTIN

        default = (
            BUILTIN.get("tasks", {}).get(task, {}).get("columns", {}).get(self.KIND)
        )
        self.config.set_columns(task, self.KIND, default or [])
        self._hidden_headers.clear()
        self.reload_columns()
        self.configChanged.emit()

    def _toggle_column(self, header: str, index: int, visible: bool) -> None:
        if visible:
            self._hidden_headers.discard(header)
        else:
            self._hidden_headers.add(header)
        self.view.horizontalHeader().setSectionHidden(index, not visible)

    def _show_all_columns(self) -> None:
        self._hidden_headers.clear()
        for index in range(self.model.columnCount()):
            self.view.horizontalHeader().setSectionHidden(index, False)

    # -- Column presets (session-only visibility, not saved to config) ---------
    # These don't touch the Task's configured column *list* - just which of its
    # columns are currently hidden. Switching Tasks or reloading keeps the choice
    # for the session; it isn't written to options.yaml since it's a quick view
    # toggle, not a Task-level decision.
    _SIMPLE_HIDE_KINDS = {"path"}
    _SIMPLE_HIDE_KEYS = {
        "notes", "tags", "failure_reason", "epochs", "batch_size", "lr",
        "optimizer", "device", "input_size",
    }
    _PAPER_KEEP_KEYS = {"model"}

    def _apply_preset(self, keep: Callable[[ColumnSpec], bool]) -> None:
        self._hidden_headers = {
            self.model.header_text(spec) for spec in self.model.columns() if not keep(spec)
        }
        self._apply_column_sizing()

    def apply_preset_simple(self) -> None:
        """Hide paths and rarely-scanned hyperparameters; keep identity + metrics."""
        self._apply_preset(
            lambda spec: spec.kind not in self._SIMPLE_HIDE_KINDS
            and spec.key not in self._SIMPLE_HIDE_KEYS
        )

    def apply_preset_paper(self) -> None:
        """Model + metrics + Task-specific settings (scale, noise_sigma, ...) only -
        the columns you'd actually copy into a paper's results table."""
        self._apply_preset(
            lambda spec: spec.is_metric or spec.is_extra or spec.key in self._PAPER_KEEP_KEYS
        )

    def apply_preset_full(self) -> None:
        self._show_all_columns()

    def _open_path(self, path: str) -> None:
        ok, message = open_in_file_manager(path)
        toast(self, ok, message, "Open Folder")

    def _on_double_click(self, index) -> None:
        spec = self.model.column_spec(self.proxy.mapToSource(index).column())
        row = self._current_row()
        if spec is not None and spec.key == "favorite" and row is not None:
            self.toggle_favorite(row)
            return
        if spec is not None and spec.key in PATH_KEYS and row is not None:
            self._open_path(str(row.get(spec.key) or ""))
            return
        self.open_edit_dialog()

    def toggle_favorite(self, row: dict[str, Any] | None = None) -> None:
        row = row or self._current_row()
        if row is None:
            toast(self, False, "Select a run first.", "Favorite")
            return
        new_state = self.db.toggle_favorite(self.KIND, int(row["id"]))
        self.reload()
        self._select_run(int(row["id"]))
        toast(self, True, f"{'Marked' if new_state else 'Unmarked'} run #{row['id']} as favorite.")

    # ==================================================================
    # Export / clipboard
    # ==================================================================
    def export_csv(self) -> None:
        if self.proxy.rowCount() == 0:
            toast(self, False, "No rows to export.", "Export CSV")
            return
        default_name = f"{self.KIND}_runs_{QtCore.QDate.currentDate().toString('yyyyMMdd')}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            os.path.join(QtCore.QDir.homePath(), default_name),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        headers = self.model.headers()
        rows = [
            [
                str(self.proxy.data(self.proxy.index(r, c), Qt.ItemDataRole.DisplayRole) or "")
                for c in range(self.proxy.columnCount())
            ]
            for r in range(self.proxy.rowCount())
        ]
        try:
            count = write_csv(path, headers, rows)
        except OSError as exc:
            toast(self, False, f"Failed to save CSV:\n{exc}", "Export CSV")
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Export Complete",
            f"Saved {count} row(s).\n{path}\n\nOpen the folder?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            open_in_file_manager(path, reveal=True)

    def export_report(self) -> None:
        if self.proxy.rowCount() == 0:
            toast(self, False, "No rows to export.", "Export Report")
            return
        default_name = f"{self.KIND}_report_{QtCore.QDate.currentDate().toString('yyyyMMdd')}.md"
        path, chosen_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Report",
            os.path.join(QtCore.QDir.homePath(), default_name),
            "Markdown (*.md);;HTML (*.html)",
        )
        if not path:
            return
        is_html = path.lower().endswith(".html") or "html" in chosen_filter.lower()
        if is_html and not path.lower().endswith(".html"):
            path += ".html"
        elif not is_html and not path.lower().endswith(".md"):
            path += ".md"

        headers = self.model.headers()
        rows = [
            [
                str(self.proxy.data(self.proxy.index(r, c), Qt.ItemDataRole.DisplayRole) or "")
                for c in range(self.proxy.columnCount())
            ]
            for r in range(self.proxy.rowCount())
        ]
        title = f"{self.KIND.title()} Runs Report"
        content = (
            render_html_report(title, headers, rows)
            if is_html
            else render_markdown_report(title, headers, rows)
        )
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(content)
        except OSError as exc:
            toast(self, False, f"Failed to save report:\n{exc}", "Export Report")
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Export Complete",
            f"Saved report.\n{path}\n\nOpen the folder?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            open_in_file_manager(path, reveal=True)

    def copy_table(self, selected_only: bool) -> None:
        text = table_selection_to_tsv(self.view, self.model.headers(), selected_only)
        if not text:
            toast(self, False, "No rows to copy.", "Copy to Clipboard")
            return
        line_count = max(0, len(text.splitlines()) - 1)
        QtWidgets.QApplication.clipboard().setText(text)
        toast(self, True, f"Copied {line_count} row(s) to the clipboard. (Paste directly into Excel.)")

    # ==================================================================
    # Form <-> DB
    # ==================================================================
    def reset_form(self) -> None:
        self._editing_id = None
        self._editing_favorite = False
        self.server_combo.set_text("")
        self.model_combo.set_text("")
        self.dataset_combo.set_work(self._work_id, keep_text=False)
        self.status_combo.setCurrentIndex(C.STATUS_LIST.index(C.STATUS_QUEUED))
        self.started_edit.setText(now_iso())
        self.duration_edit.clear()
        self.gpu_selector.clear()
        self.dataset_path_edit.clear()
        self.result_path_edit.clear()
        # 같은 Task 안에서는 지표를 공유한다 - 이전에 이 Task 의 어느 Run 에서든
        # 등록한 지표는 새 Run 을 만들 때 값 없는 행으로 미리 채워 둔다.
        self.metrics_editor.set_metrics(
            {key: "" for key in self.config.metric_keys(self._task_name)}
        )
        for combo in self._custom_widgets.values():
            combo.set_text("")
        self.command_input.clear()
        self.config_input.clear()
        self.notes_input.clear()
        self.tags_edit.clear()
        self.failure_reason_edit.clear()
        self._reset_extra_fields()
        self._refresh_work_combo()
        self._update_form_buttons()
        # 빈 폼에서 시작하므로 명령어도 다시 만들어 준다 (사용자가 손대면 그때부터 멈춘다).
        self._command_dirty = False
        self._sync_generated_command()

    def _reset_extra_fields(self) -> None:
        """Subclasses reset their own fields here."""

    def load_selected_into_form(self) -> bool:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run in the table first.", "Load Run")
            return False
        self._editing_id = int(row["id"])
        self._editing_favorite = bool(row.get("favorite"))
        work = self.db.get_work(int(row["work_id"]))
        self.work_combo.set_text(work["name"] if work else "")
        self.server_combo.set_text(row.get("server"))
        self.gpu_selector.set_server(self.config.server(str(row.get("server") or "")))
        self.gpu_selector.set_value(row.get("gpu_indices"))
        self.model_combo.set_text(row.get("model"))
        self.dataset_combo.set_text(row.get("dataset"))
        status = str(row.get("status") or C.STATUS_DONE)
        if status in C.STATUS_LIST:
            self.status_combo.setCurrentIndex(C.STATUS_LIST.index(status))
        self._sync_failure_reason_visibility()  # setCurrentIndex above may be a no-op
        self.started_edit.setText(row.get("started_at") or "")
        self.duration_edit.setText(format_duration(row.get("duration_sec")))
        self.dataset_path_edit.set_path(row.get("dataset_path"))
        self.result_path_edit.set_path(row.get("result_path"))
        self.metrics_editor.set_metrics(loads_metrics(row.get("metrics_json")))
        # 저장된 명령어는 "실제로 돌린 것"의 기록이다 - 폼 값이 바뀌어도 덮어쓰지 않는다.
        self.command_input.set_text(row.get("exec_command"))
        self._command_dirty = True
        self.config_input.set_text(row.get("config_yaml"))
        self.notes_input.set_text(row.get("notes"))
        self.tags_edit.setText(str(row.get("tags") or ""))
        self.failure_reason_edit.setText(str(row.get("failure_reason") or ""))
        extra = self._row_extra(row)
        for name, combo in self._custom_widgets.items():
            combo.set_text(str(extra.get(name, "")))
        self._load_extra_fields(row)
        self._update_form_buttons()
        return True

    @staticmethod
    def _row_extra(row: dict[str, Any]) -> dict[str, Any]:
        cached = row.get("_extra")
        if isinstance(cached, dict):
            return cached
        try:
            data = json.loads(row.get("extra_json") or "{}")
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _load_extra_fields(self, row: dict[str, Any]) -> None:
        """Subclasses load their own fields here."""

    def _collect_extra_fields(self) -> dict[str, Any]:
        """Subclasses collect their own fields here."""
        return {}

    def _update_form_buttons(self) -> None:
        if self._editing_id is None:
            self.form_dialog.setWindowTitle("New Run")
            self.save_btn.setText("+ Register Run")
        else:
            self.form_dialog.setWindowTitle(f"Editing Run #{self._editing_id}")
            self.save_btn.setText(f"💾 Save #{self._editing_id}")

    def _resolve_work_id(self) -> int | None:
        """Resolve the form's Work ID text to a real work row (create if missing)."""
        name = self.work_combo.current_text()
        if self._task_id is None:
            QtWidgets.QMessageBox.warning(
                self, "Cannot Save", "Select a DL Task on the left first."
            )
            return None
        if not name:
            QtWidgets.QMessageBox.warning(self, "Cannot Save", "Enter or select a Work ID.")
            return None
        for work in self.db.list_works(self._task_id):
            if work["name"].lower() == name.lower():
                return int(work["id"])
        answer = QtWidgets.QMessageBox.question(
            self,
            "New Work ID",
            f"Work '{name}' does not exist. Create it?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return None
        return self.db.add_work(self._task_id, name)

    def _collect_form(self) -> dict[str, Any] | None:
        work_id = self._resolve_work_id()
        if work_id is None:
            return None
        if not self.model_combo.current_text():
            QtWidgets.QMessageBox.warning(self, "Cannot Save", "Enter a Model.")
            self.model_combo.setFocus()
            return None

        duration_text = self.duration_edit.text().strip()
        duration = parse_duration(duration_text)
        if duration_text and duration is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot Save",
                "Could not parse the duration.\nExamples: 3h 20m · 01:30:00 · 5400",
            )
            self.duration_edit.setFocus()
            return None

        metrics = self.metrics_editor.metrics()
        self._register_new_metrics(metrics)

        data: dict[str, Any] = {
            "work_id": work_id,
            "server": self.server_combo.current_text(),
            "model": self.model_combo.current_text(),
            "dataset": self.dataset_combo.current_text(),
            "dataset_path": self.dataset_path_edit.path(),
            "result_path": self.result_path_edit.path(),
            "status": self.status_combo.currentData(),
            "started_at": self.started_edit.text().strip(),
            "duration_sec": duration,
            "gpu_indices": self.gpu_selector.value(),
            "extra_json": {
                name: combo.current_text()
                for name, combo in self._custom_widgets.items()
                if combo.current_text()
            },
            "metrics_json": metrics,
            "exec_command": self.command_input.text(),
            "config_yaml": self.config_input.text(),
            "notes": self.notes_input.text(),
            "favorite": self._editing_favorite,
            "tags": self.tags_edit.text().strip(),
            "failure_reason": self.failure_reason_edit.text().strip(),
        }
        data.update(self._collect_extra_fields())
        return data

    def _register_new_metrics(self, metrics: dict[str, Any]) -> None:
        """Metrics are shared within a Task: a value typed for a new metric name
        on any Run becomes that Task's default, pre-filled on the next New Run."""
        if not self._task_name or not metrics:
            return
        known = set(self.config.metric_keys(self._task_name))
        added = False
        for key in metrics:
            if key not in known:
                self.config.add_metric(self._task_name, MetricDef(key=key))
                known.add(key)
                added = True
        if added:
            self.configChanged.emit()

    def save_form(self) -> None:
        data = self._collect_form()
        if data is None:
            return
        if self._editing_id is None:
            run_id = self.db.insert_run(self.KIND, data)
            message = f"Registered new run #{run_id}."
        else:
            run_id = self._editing_id
            self.db.update_run(self.KIND, run_id, data)
            message = f"Saved run #{run_id}."

        self._work_id = int(data["work_id"]) if self._work_id is not None else self._work_id
        # Refresh the tree/server indicator first, then the table, so the selection sticks.
        self.runsChanged.emit()
        self.reload()
        self._select_run(run_id)
        self.form_dialog.hide()
        toast(self, True, message, "Saved")

    def duplicate_selected(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run to duplicate.", "Duplicate")
            return
        new_id = self.db.duplicate_run(self.KIND, int(row["id"]))
        self.runsChanged.emit()
        self.reload()
        if new_id:
            self._select_run(new_id)
        toast(self, True, f"Duplicated run #{row['id']} → #{new_id}.")

    def delete_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            toast(self, False, "Select runs to delete.", "Delete")
            return
        ids = [int(r["id"]) for r in rows]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(ids)} selected run(s)?\n(#{', #'.join(map(str, ids))})",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        deleted = self.db.delete_runs(self.KIND, ids)
        if self._editing_id in ids:
            self._editing_id = None
        self.runsChanged.emit()
        self.reload()
        toast(self, True, f"Deleted {deleted} run(s).")

    def _select_run(self, run_id: int) -> None:
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            data = self.proxy.data(index, RAW_ROLE)
            if isinstance(data, dict) and int(data.get("id", -1)) == int(run_id):
                self.view.selectRow(row)
                self.view.scrollTo(index)
                return


class TrainPanel(BaseRunPanel):
    """Train dashboard."""

    KIND = "train"
    OFFERS_EVALUATION_HANDOFF = True
    SAMPLE_COMMAND = C.SAMPLE_TRAIN_CMD
    METRIC_PRESETS = C.TRAIN_METRIC_PRESETS
    DETAIL_PATHS = (
        ("dataset_path", "Dataset Path"),
        ("result_path", "Result Folder Path"),
    )

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        self.epochs_edit = QtWidgets.QLineEdit(parent)
        self.epochs_edit.setPlaceholderText("e.g. 300000 iter / 200 epoch")
        self.batch_edit = QtWidgets.QLineEdit(parent)
        self.batch_edit.setPlaceholderText("e.g. 8")
        self.crop_size_edit = QtWidgets.QLineEdit(parent)
        self.crop_size_edit.setPlaceholderText("e.g. 256x256 / 192")
        self.lr_edit = QtWidgets.QLineEdit(parent)
        self.lr_edit.setPlaceholderText("e.g. 3e-4")
        self.optimizer_combo = self._make_option_combo("optimizer", "Optimizer", parent)

        self.form_layout.addRow(self._section("Training Hyperparameters", parent))
        self.form_layout.addRow("Epochs / Iter:", self.epochs_edit)
        self.form_layout.addRow("Batch size:", self.batch_edit)
        self.form_layout.addRow("Crop size:", self.crop_size_edit)
        self.form_layout.addRow("Learning rate:", self.lr_edit)
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

    def _command_values(self) -> dict[str, str]:
        values = super()._command_values()
        values.update(
            {
                "epochs": self.epochs_edit.text().strip(),
                "batch_size": self.batch_edit.text().strip(),
                "crop_size": self.crop_size_edit.text().strip(),
                "lr": self.lr_edit.text().strip(),
                "optimizer": self.optimizer_combo.current_text(),
            }
        )
        return values

    def _command_source_widgets(self) -> list[QtWidgets.QWidget]:
        return super()._command_source_widgets() + [
            self.epochs_edit, self.batch_edit, self.crop_size_edit,
            self.lr_edit, self.optimizer_combo,
        ]

    def _refresh_extra_combo_sources(self) -> None:
        self.optimizer_combo.reload()
        self.optimizer_combo.merge_items(self.db.distinct_values("train", "optimizer", self._task_id))

    def _apply_parsed_hyperparams(self, fields: dict[str, str], filled: list[str]) -> None:
        if fields.get("epochs"):
            self.epochs_edit.setText(fields["epochs"])
            filled.append("Epochs/Iter")
        if fields.get("batch_size"):
            self.batch_edit.setText(fields["batch_size"])
            filled.append("Batch size")
        if fields.get("crop_size"):
            self.crop_size_edit.setText(fields["crop_size"])
            filled.append("Crop size")
        if fields.get("lr"):
            self.lr_edit.setText(fields["lr"])
            filled.append("LR")
        if fields.get("optimizer"):
            self.optimizer_combo.set_text(fields["optimizer"])
            filled.append("Optimizer")

    def _reset_extra_fields(self) -> None:
        self.epochs_edit.clear()
        self.batch_edit.clear()
        self.crop_size_edit.clear()
        self.lr_edit.clear()
        self.optimizer_combo.set_text("")

    def _load_extra_fields(self, row: dict[str, Any]) -> None:
        self.epochs_edit.setText(str(row.get("epochs") or ""))
        self.batch_edit.setText(str(row.get("batch_size") or ""))
        self.crop_size_edit.setText(str(row.get("crop_size") or ""))
        self.lr_edit.setText(str(row.get("lr") or ""))
        self.optimizer_combo.set_text(row.get("optimizer"))

    def _collect_extra_fields(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs_edit.text().strip(),
            "batch_size": self.batch_edit.text().strip(),
            "crop_size": self.crop_size_edit.text().strip(),
            "lr": self.lr_edit.text().strip(),
            "optimizer": self.optimizer_combo.current_text(),
        }


class EvaluationPanel(BaseRunPanel):
    """Evaluation dashboard."""

    KIND = "evaluation"
    SAMPLE_COMMAND = C.SAMPLE_EVAL_CMD
    METRIC_PRESETS = C.EVAL_METRIC_PRESETS
    SHOW_SERVER = True  # Server 는 보여준다 - 어느 서버에서 돌렸는지는 여전히 유용
    SHOW_GPU = False  # GPU 는 계속 뺀다 - Train Run + Epoch/Iter 가 핵심 식별자
    SHOW_TRAINING_CURVE = False  # 학습 곡선은 loss.log 기반 - 추론에는 해당 없음
    DETAIL_PATHS = (
        ("checkpoint_path", "Checkpoint Path"),
        ("dataset_path", "Test Dataset Path"),
        ("result_path", "Result Folder Path"),
    )

    NO_SOURCE_RUN = "(none - fill in manually)"

    def _dataset_path_label(self) -> str:
        return "Test Dataset Path"

    def _build_left_fields(self, parent: QtWidgets.QWidget) -> None:
        """Train 과 순서가 크게 달라(같은 Work 의 Train Run 을 먼저 고르는 흐름) 통째로 재구성한다."""
        self.work_combo = self._make_work_combo(parent)
        self.form_layout.addRow("Work ID:", self.work_combo)

        self.source_run_combo = QtWidgets.QComboBox(parent)
        self.source_run_combo.setToolTip(
            "Pick a Train run from this Work to base the evaluation on. "
            "Sets Model automatically; you still pick which checkpoint/epoch."
        )
        self.source_run_combo.currentIndexChanged.connect(self._on_source_run_changed)
        self.form_layout.addRow("Train Run:", self.source_run_combo)

        self.checkpoint_epoch_edit = QtWidgets.QLineEdit(parent)
        self.checkpoint_epoch_edit.setPlaceholderText("e.g. 300000 (iter) or 200 (epoch)")
        self.form_layout.addRow("Epoch/Iter:", self.checkpoint_epoch_edit)

        self.checkpoint_edit = PathEdit(
            parent,
            "/mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth",
            directory=False,
            compact=True,
        )
        self.form_layout.addRow("Checkpoint Path:", self.checkpoint_edit)

        self.model_combo = self._make_option_combo("model", "Model", parent)
        self.model_combo.setToolTip("Auto-filled when you pick a Train Run above; override if needed.")
        self.form_layout.addRow("Model:", self.model_combo)

        dataset_row = self._make_dataset_row(parent)
        self.form_layout.addRow("Dataset:", dataset_row)

        self.status_combo = self._make_status_combo(parent)
        self.form_layout.addRow("Status:", self.status_combo)

        self.server_combo = self._make_server_combo(parent)
        self.form_layout.addRow("Server:", self.server_combo)
        self.gpu_selector = GpuSelector(parent)
        self.gpu_selector.setVisible(False)  # 추론은 GPU 를 안 씀 - 인스턴스는 그대로 유지

        self.device_combo = self._make_option_combo("device", "Device", parent)
        self.form_layout.addRow("Device:", self.device_combo)

        self.latency_edit = QtWidgets.QLineEdit(parent)
        self.latency_edit.setPlaceholderText("ms per image")
        self.form_layout.addRow("Latency (ms):", self.latency_edit)

        self.throughput_edit = QtWidgets.QLineEdit(parent)
        self.throughput_edit.setPlaceholderText("images/sec (FPS)")
        self.form_layout.addRow("Throughput (FPS):", self.throughput_edit)

        self.input_size_edit = QtWidgets.QLineEdit(parent)
        self.input_size_edit.setPlaceholderText("auto-filled from Dataset's Image size; editable")
        self.form_layout.addRow("Input size:", self.input_size_edit)

        started_row = self._make_started_row(parent)
        self.form_layout.addRow("Started At:", started_row)

        self.duration_edit = self._make_duration_edit(parent)
        self.form_layout.addRow("Duration:", self.duration_edit)

    def _on_dataset_selected(self, row: dict[str, Any]) -> None:
        super()._on_dataset_selected(row)
        # Dataset 에 등록해 둔 이미지 크기를 Input size 기본값으로 - 그래도 손으로 고칠 수 있다.
        if row.get("image_size"):
            self.input_size_edit.setText(row["image_size"])

    def _refresh_extra_combo_sources(self) -> None:
        self.device_combo.reload()
        self.device_combo.merge_items(self.db.distinct_values("evaluation", "device", self._task_id))
        self._refresh_source_run_combo()

    def _refresh_source_run_combo(self, keep: int | None = None) -> None:
        """List Train runs from the same Task + Work only - that's the whole point."""
        current = keep if keep is not None else self.source_run_combo.currentData()
        self.source_run_combo.blockSignals(True)
        self.source_run_combo.clear()
        self.source_run_combo.addItem(self.NO_SOURCE_RUN, None)
        if self._work_id:
            for run in self.db.list_train_runs(work_id=self._work_id):
                label = f"#{run['id']} · {run.get('model') or '-'} · {run.get('status')}"
                self.source_run_combo.addItem(label, int(run["id"]))
        index = self.source_run_combo.findData(current)
        self.source_run_combo.setCurrentIndex(index if index >= 0 else 0)
        self.source_run_combo.blockSignals(False)

    def _on_source_run_changed(self, _index: int) -> None:
        run_id = self.source_run_combo.currentData()
        if run_id is None:
            return
        run = self.db.get_run("train", int(run_id))
        if run:
            self.model_combo.set_text(run.get("model"))
            self._sync_generated_command()

    def _command_values(self) -> dict[str, str]:
        values = super()._command_values()
        train_run = self._source_train_run()
        values.update(
            {
                "checkpoint_path": self.checkpoint_edit.path(),
                "checkpoint_epoch": self.checkpoint_epoch_edit.text().strip(),
                "device": self.device_combo.current_text(),
                "input_size": self.input_size_edit.text().strip(),
                # 고른 Train Run 에서 끌어오는 값들 - 체크포인트 경로 규칙이
                # 프로젝트마다 달라서, 앱이 추측하지 않고 템플릿이 정하게 둔다.
                # 예: +ckpt_path={train_result_path}/models/net_g_{checkpoint_epoch}.pth
                "train_result_path": str(train_run.get("result_path") or "") if train_run else "",
                "train_run_id": str(train_run["id"]) if train_run else "",
                "train_model": str(train_run.get("model") or "") if train_run else "",
            }
        )
        return values

    def _source_train_run(self) -> dict[str, Any] | None:
        run_id = self.source_run_combo.currentData()
        if run_id is None:
            return None
        return self.db.get_run("train", int(run_id))

    def _command_source_widgets(self) -> list[QtWidgets.QWidget]:
        return super()._command_source_widgets() + [
            self.checkpoint_edit.edit, self.checkpoint_epoch_edit,
            self.device_combo, self.input_size_edit,
        ]

    def start_evaluation_for(self, train_run_id: int) -> None:
        """Train 표에서 고른 실행을 바탕으로 새 Evaluation 폼을 연다.

        모델·데이터셋·서버를 그 학습에서 그대로 가져오고(평가는 보통 같은 서버에서
        같은 데이터 계열로 돌린다), 명령어까지 만들어 둔 상태로 폼을 띄운다.
        """
        self.open_new_run_dialog()
        index = self.source_run_combo.findData(int(train_run_id))
        if index >= 0:
            self.source_run_combo.setCurrentIndex(index)
        run = self.db.get_run("train", int(train_run_id))
        if run:
            self.server_combo.set_text(run.get("server"))
            self.dataset_combo.set_text(run.get("dataset"))
            self.dataset_path_edit.set_path(run.get("dataset_path"))
            # algo/scale 같은 사용자 정의 필드는 Hydra config group 인 경우가 많다 -
            # 같은 학습을 평가하는 것이므로 그대로 물려받아야 명령어가 맞는다.
            extra = self._row_extra(run)
            for name, combo in self._custom_widgets.items():
                if extra.get(name):
                    combo.set_text(str(extra[name]))
        self._sync_generated_command()

    def _reset_extra_fields(self) -> None:
        self._refresh_source_run_combo(keep=None)
        self.source_run_combo.setCurrentIndex(0)
        self.checkpoint_epoch_edit.clear()
        self.checkpoint_edit.clear()
        self.device_combo.set_text("")
        self.input_size_edit.clear()
        self.latency_edit.clear()
        self.throughput_edit.clear()

    def _load_extra_fields(self, row: dict[str, Any]) -> None:
        source_id = row.get("source_train_run_id")
        self._refresh_source_run_combo(keep=int(source_id) if source_id else None)
        self.checkpoint_epoch_edit.setText(str(row.get("checkpoint_epoch") or ""))
        self.checkpoint_edit.set_path(row.get("checkpoint_path"))
        self.device_combo.set_text(row.get("device"))
        self.input_size_edit.setText(str(row.get("input_size") or ""))
        self.latency_edit.setText(format_number(row.get("latency_ms")))
        self.throughput_edit.setText(format_number(row.get("throughput_fps")))

    def _collect_extra_fields(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_edit.path(),
            "device": self.device_combo.current_text(),
            "input_size": self.input_size_edit.text().strip(),
            "latency_ms": self.latency_edit.text().strip(),
            "throughput_fps": self.throughput_edit.text().strip(),
            "source_train_run_id": self.source_run_combo.currentData(),
            "checkpoint_epoch": self.checkpoint_epoch_edit.text().strip(),
        }
