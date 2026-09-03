"""Train / Inference dashboard panel.

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
from typing import Any, Sequence

from .. import constants as C
from .. import editing, theme
from ..config_store import MetricDef, OptionsConfig
from ..db import Database
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
    copy_to_clipboard,
    monospace_font,
    table_selection_to_tsv,
    toast,
)


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
    """Shared Train / Inference dashboard."""

    KIND: str = "train"
    METRIC_PRESETS: Sequence[str] = ()
    DETAIL_PATHS: Sequence[tuple[str, str]] = ()
    SAMPLE_COMMAND: str = ""

    runsChanged = Signal()
    configChanged = Signal()

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
        self._hidden_headers: set[str] = set()
        self._custom_widgets: dict[str, ManagedCombo] = {}

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
        new_run_btn.setProperty("variant", "primary")
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
        self.status_filter.currentIndexChanged.connect(
            lambda: self.proxy.set_status_filter(self.status_filter.currentData())
        )

        refresh_btn = QtWidgets.QToolButton(container)
        refresh_btn.setText("↻ Refresh")
        refresh_btn.clicked.connect(self.reload)

        export_btn = QtWidgets.QToolButton(container)
        export_btn.setText("⤓ Export CSV")
        export_btn.setToolTip("Export the currently filtered/sorted table to a CSV file.")
        export_btn.clicked.connect(self.export_csv)

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

        del_btn = QtWidgets.QToolButton(container)
        del_btn.setText("🗑 Delete")
        del_btn.clicked.connect(self.delete_selected)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addWidget(new_run_btn)
        toolbar.addWidget(self.filter_edit, 1)
        toolbar.addWidget(self.status_filter)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(export_btn)
        toolbar.addWidget(copy_btn)
        toolbar.addWidget(dup_btn)
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

        paths_box = QtWidgets.QGroupBox("Paths", container)
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

        self.detail_tabs = QtWidgets.QTabWidget(container)
        self.detail_tabs.addTab(self.detail_command, "Command")
        self.detail_tabs.addTab(self.detail_config, "config.yml")
        self.detail_tabs.addTab(self.detail_notes, "Metrics / Notes")

        edit_btn = QtWidgets.QPushButton("✎ Edit This Run", container)
        edit_btn.clicked.connect(self.open_edit_dialog)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.detail_title)
        layout.addWidget(self.detail_meta)
        layout.addWidget(paths_box)
        layout.addWidget(self.detail_tabs, 1)
        layout.addWidget(edit_btn)
        return container

    # ==================================================================
    # 3) Register / edit form (popup dialog)
    # ==================================================================
    def _build_form_dialog(self) -> None:
        self.form_dialog = QtWidgets.QDialog(self)
        self.form_dialog.setModal(True)
        self.form_dialog.setMinimumSize(440, 600)
        dialog_layout = QtWidgets.QVBoxLayout(self.form_dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(self._build_form_area())

    def open_new_run_dialog(self) -> None:
        self.reset_form()
        self._show_form_dialog()

    def open_edit_dialog(self) -> None:
        if self.load_selected_into_form():
            self._show_form_dialog()

    def _show_form_dialog(self) -> None:
        self.form_dialog.show()
        self.form_dialog.raise_()
        self.form_dialog.activateWindow()

    def _build_form_area(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self.form_dialog)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget(scroll)
        self.form_layout = QtWidgets.QFormLayout(inner)
        self.form_layout.setContentsMargins(12, 12, 12, 12)
        self.form_layout.setSpacing(6)
        self.form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # -- Common top fields -----------------------------------------------
        # Work ID comes from the DB (works table), not options config.
        self.work_combo = EditableCombo([], inner, "Work ID (new if not found)")
        self.work_combo.setToolTip("The Work ID this run belongs to. Typing a new value creates it.")

        self.server_combo = self._make_option_combo("server", "Server", inner)
        self.server_combo.currentTextChanged.connect(self._on_server_changed)
        self.model_combo = self._make_option_combo("model", "Model", inner)
        self.dataset_combo = self._make_option_combo("dataset", "Dataset", inner)

        self.status_combo = QtWidgets.QComboBox(inner)
        for status in C.STATUS_LIST:
            self.status_combo.addItem(f"●  {status}", status)
        self.status_combo.setCurrentIndex(C.STATUS_LIST.index(C.STATUS_DONE))

        self.started_edit = QtWidgets.QLineEdit(inner)
        self.started_edit.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        now_btn = QtWidgets.QToolButton(inner)
        now_btn.setText("Now")
        now_btn.clicked.connect(lambda: self.started_edit.setText(now_iso()))
        started_row = QtWidgets.QWidget(inner)
        started_layout = QtWidgets.QHBoxLayout(started_row)
        started_layout.setContentsMargins(0, 0, 0, 0)
        started_layout.setSpacing(4)
        started_layout.addWidget(self.started_edit, 1)
        started_layout.addWidget(now_btn)

        self.duration_edit = QtWidgets.QLineEdit(inner)
        self.duration_edit.setPlaceholderText("e.g. 3h 20m / 01:30:00 / 5400 (seconds)")

        self.gpu_selector = GpuSelector(inner)

        self.form_layout.addRow("Work ID:", self.work_combo)
        self.form_layout.addRow("Server:", self.server_combo)
        self.form_layout.addRow("GPU:", self.gpu_selector)
        self.form_layout.addRow("Model:", self.model_combo)
        self.form_layout.addRow("Dataset:", self.dataset_combo)
        self.form_layout.addRow("Status:", self.status_combo)
        self.form_layout.addRow("Started At:", started_row)
        self.form_layout.addRow("Duration:", self.duration_edit)

        # -- Task-specific fields (from config) -------------------------------
        self._custom_section = self._section("Task-Specific Fields", inner)
        self.form_layout.addRow(self._custom_section)
        self._custom_host = QtWidgets.QWidget(inner)
        self._custom_form = QtWidgets.QFormLayout(self._custom_host)
        self._custom_form.setContentsMargins(0, 0, 0, 0)
        self._custom_form.setSpacing(6)
        self._custom_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.form_layout.addRow(self._custom_host)

        # -- Subclass-specific fields ------------------------------------------
        self._build_extra_form_rows(inner)

        # -- Paths -------------------------------------------------------------
        self.dataset_path_edit = PathEdit(inner, "/mnt/data/DIV2K/train", compact=True)
        self.result_path_edit = PathEdit(inner, "/mnt/exp/SSL2SL/restormer_x4", compact=True)
        self.form_layout.addRow(self._section("Paths", inner))
        self.form_layout.addRow(self._dataset_path_label() + ":", self.dataset_path_edit)
        self.form_layout.addRow("Result Folder Path:", self.result_path_edit)

        # -- Metrics -----------------------------------------------------------
        self.metrics_editor = MetricsEditor(
            self.METRIC_PRESETS, inner, config=self.config, task_getter=self.current_task_name
        )
        self.metrics_editor.metricsDefined.connect(self._on_metrics_defined)
        self.form_layout.addRow(self._section("Evaluation Metrics", inner))
        self.form_layout.addRow(self.metrics_editor)

        # -- Execution command / config -------------------------------------------
        self.command_input = LabeledText(
            "Execution Command",
            inner,
            placeholder=self.SAMPLE_COMMAND,
            min_height=90,
        )
        sample_cmd_btn = QtWidgets.QToolButton(inner)
        sample_cmd_btn.setText("Sample")
        sample_cmd_btn.setToolTip("Fill in an example command.")
        sample_cmd_btn.clicked.connect(lambda: self.command_input.set_text(self.SAMPLE_COMMAND))
        self.command_input.add_header_widget(sample_cmd_btn)

        self.config_input = LabeledText(
            "config.yml", inner, placeholder="# Paste YAML config content here", min_height=140
        )
        load_cfg_btn = QtWidgets.QToolButton(inner)
        load_cfg_btn.setText("Load From File")
        load_cfg_btn.clicked.connect(self._load_config_from_file)
        self.config_input.add_header_widget(load_cfg_btn)

        self.notes_input = LabeledText(
            "Notes", inner, placeholder="Free-form notes", mono=False, min_height=70
        )

        self.form_layout.addRow(self._section("Execution Code / Config", inner))
        self.form_layout.addRow(self.command_input)
        self.form_layout.addRow(self.config_input)
        self.form_layout.addRow(self.notes_input)

        # -- Buttons ---------------------------------------------------------------
        self.save_btn = QtWidgets.QPushButton("+ Register Run", inner)
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_form)

        self.new_btn = QtWidgets.QPushButton("Clear", inner)
        self.new_btn.setToolTip("Clear the form for a new entry (keeps this dialog open).")
        self.new_btn.clicked.connect(self.reset_form)

        self.cancel_btn = QtWidgets.QPushButton("Cancel", inner)
        self.cancel_btn.clicked.connect(self.form_dialog.hide)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 6, 0, 0)
        buttons.addWidget(self.save_btn, 2)
        buttons.addWidget(self.new_btn, 1)
        buttons.addWidget(self.cancel_btn, 1)
        self.form_layout.addRow(self._wrap(buttons, inner))

        scroll.setWidget(inner)
        return scroll

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        """Subclasses add their own input fields here."""

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

    def _dataset_path_label(self) -> str:
        return "Dataset Path"

    @staticmethod
    def _section(title: str, parent: QtWidgets.QWidget) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(f"<span style='color:#1a73e8'><b>— {title} —</b></span>", parent)
        label.setContentsMargins(0, 8, 0, 0)
        return label

    @staticmethod
    def _wrap(layout: QtWidgets.QLayout, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(parent)
        widget.setLayout(layout)
        return widget

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
        return self.db.list_inference_runs(work_id=self._work_id, task_id=self._task_id)

    def _apply_column_sizing(self) -> None:
        header = self.view.horizontalHeader()
        for index, spec in enumerate(self.model.columns()):
            self.view.setColumnWidth(index, spec.width)
            header.setSectionHidden(index, spec.header in self._hidden_headers)

    def _refresh_combo_sources(self) -> None:
        """Base the list on config, but merge in legacy values only found in the DB."""
        scope = self._task_id  # keep values scoped so other Tasks' data doesn't leak in
        self.server_combo.reload()
        self.server_combo.merge_items(self.db.distinct_values(self.KIND, "server", scope))
        self.model_combo.reload()
        self.model_combo.merge_items(self.db.distinct_values(self.KIND, "model", scope))
        self.dataset_combo.reload()
        self.dataset_combo.merge_items(self.db.distinct_values(self.KIND, "dataset", scope))
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
        for edit in self._detail_path_edits.values():
            edit.clear()
        self.detail_command.clear()
        self.detail_config.clear()
        self.detail_notes.clear()

    def _show_detail(self, row: dict[str, Any]) -> None:
        title = (
            f"#{row.get('id')}  ·  {row.get('task_name', '')} ▸ {row.get('work_name', '')}  ·  "
            f"{row.get('model') or '-'}  @  {row.get('server') or '-'}"
        )
        self.detail_title.setText(title)
        self.detail_meta.setText(self._detail_meta_text(row))
        for key, edit in self._detail_path_edits.items():
            edit.setText(str(row.get(key) or ""))
        self.detail_command.set_text(row.get("exec_command") or "")
        self.detail_config.set_text(row.get("config_yaml") or "")

        metrics = loads_metrics(row.get("metrics_json"))
        notes_text = metrics_to_text(metrics, sep="\n") or "(no metrics recorded)"
        notes = (row.get("notes") or "").strip()
        self.detail_notes.set_text(
            f"[Metrics]\n{notes_text}\n\n[Notes]\n{notes or '(none)'}"
        )

    def _detail_meta_text(self, row: dict[str, Any]) -> str:
        status = str(row.get("status") or "")
        duration = format_duration(row.get("duration_sec"))
        gpus = str(row.get("gpu_indices") or "").strip()
        parts = [
            f"● {status}",
            f"{row.get('server') or '-'} · GPU {gpus or '-'}",
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
            for key, label in self.DETAIL_PATHS:
                path = str(row.get(key) or "")
                action = menu.addAction(f"📁 Open {label}")
                action.setEnabled(bool(path))
                action.triggered.connect(lambda _=False, p=path: self._open_path(p))
            menu.addSeparator()
            menu.addAction("Copy Command", lambda: copy_to_clipboard(row.get("exec_command") or "", self, "Execution Command"))
            menu.addAction("Copy config.yml", lambda: copy_to_clipboard(row.get("config_yaml") or "", self, "config.yml"))
            menu.addSeparator()
            menu.addAction("✎ Edit This Run", self.open_edit_dialog)
            menu.addAction("⎘ Duplicate", self.duplicate_selected)
            menu.addSeparator()

        menu.addAction("Copy Selected Rows (TSV)", lambda: self.copy_table(selected_only=True))
        menu.addAction("Export Entire Table to CSV", self.export_csv)
        if row is not None:
            menu.addSeparator()
            menu.addAction("🗑 Delete Selected", self.delete_selected)
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
        return spec.key not in ("id", "status", "notes")

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

    def _open_path(self, path: str) -> None:
        ok, message = open_in_file_manager(path)
        toast(self, ok, message, "Open Folder")

    def _on_double_click(self, index) -> None:
        spec = self.model.column_spec(self.proxy.mapToSource(index).column())
        row = self._current_row()
        if spec is not None and spec.key in PATH_KEYS and row is not None:
            self._open_path(str(row.get(spec.key) or ""))
            return
        self.open_edit_dialog()

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
        self.server_combo.set_text("")
        self.model_combo.set_text("")
        self.dataset_combo.set_text("")
        self.status_combo.setCurrentIndex(C.STATUS_LIST.index(C.STATUS_DONE))
        self.started_edit.setText(now_iso())
        self.duration_edit.clear()
        self.gpu_selector.clear()
        self.dataset_path_edit.clear()
        self.result_path_edit.clear()
        self.metrics_editor.clear()
        for combo in self._custom_widgets.values():
            combo.set_text("")
        self.command_input.clear()
        self.config_input.clear()
        self.notes_input.clear()
        self._reset_extra_fields()
        self._refresh_work_combo()
        self._update_form_buttons()

    def _reset_extra_fields(self) -> None:
        """Subclasses reset their own fields here."""

    def load_selected_into_form(self) -> bool:
        row = self._current_row()
        if row is None:
            toast(self, False, "Select a run in the table first.", "Load Run")
            return False
        self._editing_id = int(row["id"])
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
        self.started_edit.setText(row.get("started_at") or "")
        self.duration_edit.setText(format_duration(row.get("duration_sec")))
        self.dataset_path_edit.set_path(row.get("dataset_path"))
        self.result_path_edit.set_path(row.get("result_path"))
        self.metrics_editor.set_metrics(loads_metrics(row.get("metrics_json")))
        self.command_input.set_text(row.get("exec_command"))
        self.config_input.set_text(row.get("config_yaml"))
        self.notes_input.set_text(row.get("notes"))
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
            "metrics_json": self.metrics_editor.metrics(),
            "exec_command": self.command_input.text(),
            "config_yaml": self.config_input.text(),
            "notes": self.notes_input.text(),
        }
        data.update(self._collect_extra_fields())
        return data

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
        self.lr_edit = QtWidgets.QLineEdit(parent)
        self.lr_edit.setPlaceholderText("e.g. 3e-4")
        self.optimizer_combo = self._make_option_combo("optimizer", "Optimizer", parent)

        self.form_layout.addRow(self._section("Training Hyperparameters", parent))
        self.form_layout.addRow("Epochs / Iter:", self.epochs_edit)
        self.form_layout.addRow("Batch size:", self.batch_edit)
        self.form_layout.addRow("Learning rate:", self.lr_edit)
        self.form_layout.addRow("Optimizer:", self.optimizer_combo)

    def _refresh_extra_combo_sources(self) -> None:
        self.optimizer_combo.reload()
        self.optimizer_combo.merge_items(self.db.distinct_values("train", "optimizer", self._task_id))

    def _reset_extra_fields(self) -> None:
        self.epochs_edit.clear()
        self.batch_edit.clear()
        self.lr_edit.clear()
        self.optimizer_combo.set_text("")

    def _load_extra_fields(self, row: dict[str, Any]) -> None:
        self.epochs_edit.setText(str(row.get("epochs") or ""))
        self.batch_edit.setText(str(row.get("batch_size") or ""))
        self.lr_edit.setText(str(row.get("lr") or ""))
        self.optimizer_combo.set_text(row.get("optimizer"))

    def _collect_extra_fields(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs_edit.text().strip(),
            "batch_size": self.batch_edit.text().strip(),
            "lr": self.lr_edit.text().strip(),
            "optimizer": self.optimizer_combo.current_text(),
        }


class InferencePanel(BaseRunPanel):
    """Inference dashboard."""

    KIND = "inference"
    SAMPLE_COMMAND = C.SAMPLE_INFER_CMD
    METRIC_PRESETS = C.INFER_METRIC_PRESETS
    DETAIL_PATHS = (
        ("checkpoint_path", "Checkpoint Path"),
        ("dataset_path", "Test Dataset Path"),
        ("result_path", "Result Folder Path"),
    )

    def _dataset_path_label(self) -> str:
        return "Test Dataset Path"

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        self.checkpoint_edit = PathEdit(
            parent,
            "/mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth",
            directory=False,
            compact=True,
        )
        self.device_combo = self._make_option_combo("device", "Device", parent)
        self.input_size_edit = QtWidgets.QLineEdit(parent)
        self.input_size_edit.setPlaceholderText("e.g. 3x256x256")
        self.latency_edit = QtWidgets.QLineEdit(parent)
        self.latency_edit.setPlaceholderText("ms per image")
        self.throughput_edit = QtWidgets.QLineEdit(parent)
        self.throughput_edit.setPlaceholderText("images/sec (FPS)")

        self.form_layout.addRow(self._section("Inference Settings / Speed", parent))
        self.form_layout.addRow("Checkpoint Path:", self.checkpoint_edit)
        self.form_layout.addRow("Device:", self.device_combo)
        self.form_layout.addRow("Input size:", self.input_size_edit)
        self.form_layout.addRow("Latency (ms):", self.latency_edit)
        self.form_layout.addRow("Throughput (FPS):", self.throughput_edit)

    def _refresh_extra_combo_sources(self) -> None:
        self.device_combo.reload()
        self.device_combo.merge_items(self.db.distinct_values("inference", "device", self._task_id))

    def _reset_extra_fields(self) -> None:
        self.checkpoint_edit.clear()
        self.device_combo.set_text("")
        self.input_size_edit.clear()
        self.latency_edit.clear()
        self.throughput_edit.clear()

    def _load_extra_fields(self, row: dict[str, Any]) -> None:
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
        }
