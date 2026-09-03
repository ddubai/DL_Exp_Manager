"""Train / Inference 대시보드 패널.

레이아웃::

    ┌ 필터 · 정렬 · CSV/클립보드 툴바 ───────────────────────────┐
    │ QTableView (헤더 클릭 정렬)                                │
    ├──────────────────────────────┬─────────────────────────────┤
    │ 선택한 Run 상세 (경로 + 폴더  │  입력 폼                     │
    │ 열기 / 실행 코드 / config.yml)│  (Editable QComboBox 등)     │
    └──────────────────────────────┴─────────────────────────────┘
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
    """표에 컬럼 추가 - 기존 필드에서 고르거나 새 이름을 직접 입력."""

    def __init__(self, parent: QtWidgets.QWidget | None, candidates: Sequence[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("컬럼 추가")
        self.setMinimumWidth(340)

        self.combo = QtWidgets.QComboBox(self)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.combo.addItems(list(candidates))
        self.combo.setCurrentIndex(-1)
        self.combo.lineEdit().setPlaceholderText("기존 필드를 고르거나 새 이름 입력")

        self.metric_check = QtWidgets.QCheckBox("새 이름이면 이 Task 의 평가 지표로 등록", self)
        self.metric_check.setChecked(True)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("표에 추가할 컬럼", self))
        layout.addWidget(self.combo)
        layout.addWidget(self.metric_check)
        layout.addWidget(
            QtWidgets.QLabel(
                f"<span style='color:{theme.color('text.muted')}'>"
                "config/options.yaml 의 columns 에 저장됩니다.</span>",
                self,
            )
        )
        layout.addWidget(buttons)

    def value(self) -> str:
        return self.combo.currentText().strip()

    def as_metric(self) -> bool:
        return self.metric_check.isChecked()


class MetricSettingsDialog(QtWidgets.QDialog):
    """지표 표시 설정 - 단위 / 자릿수 / 방향."""

    def __init__(self, parent: QtWidgets.QWidget | None, metric: MetricDef) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"지표 설정 · {metric.key}")
        self._key = metric.key

        self.unit_edit = QtWidgets.QLineEdit(metric.unit, self)
        self.unit_edit.setPlaceholderText("예: dB, % (비워도 됨)")
        self.digits_spin = QtWidgets.QSpinBox(self)
        self.digits_spin.setRange(0, 8)
        self.digits_spin.setValue(metric.digits)
        self.direction = QtWidgets.QComboBox(self)
        self.direction.addItem("높을수록 좋음", True)
        self.direction.addItem("낮을수록 좋음", False)
        self.direction.setCurrentIndex(0 if metric.higher_is_better else 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("단위:", self.unit_edit)
        form.addRow("소수 자릿수:", self.digits_spin)
        form.addRow("방향:", self.direction)

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
    """Train / Inference 공통 대시보드."""

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

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        left = QtWidgets.QSplitter(Qt.Orientation.Vertical, splitter)
        left.addWidget(self._build_table_area())
        left.addWidget(self._build_detail_area())
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)
        left.setSizes([560, 340])
        splitter.addWidget(left)
        splitter.addWidget(self._build_form_area())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([860, 430])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        self._clear_detail()
        self._update_form_buttons()

    # ==================================================================
    # 1) 테이블 영역
    # ==================================================================
    def _build_table_area(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)

        self.filter_edit = QtWidgets.QLineEdit(container)
        self.filter_edit.setPlaceholderText("모든 컬럼에서 검색 (모델명, 경로, 지표 값…)")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.proxy.set_text_filter)

        self.status_filter = QtWidgets.QComboBox(container)
        self.status_filter.addItem("전체 상태", "")
        for status in C.STATUS_LIST:
            self.status_filter.addItem(f"●  {status}", status)
        self.status_filter.currentIndexChanged.connect(
            lambda: self.proxy.set_status_filter(self.status_filter.currentData())
        )

        refresh_btn = QtWidgets.QToolButton(container)
        refresh_btn.setText("↻ 새로고침")
        refresh_btn.clicked.connect(self.reload)

        export_btn = QtWidgets.QToolButton(container)
        export_btn.setText("⤓ CSV 내보내기")
        export_btn.setToolTip("현재 필터/정렬이 적용된 표를 CSV 파일로 저장합니다.")
        export_btn.clicked.connect(self.export_csv)

        copy_btn = QtWidgets.QToolButton(container)
        copy_btn.setText("⧉ 클립보드 복사")
        copy_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        copy_menu = QtWidgets.QMenu(copy_btn)
        copy_menu.addAction("선택한 행 복사", lambda: self.copy_table(selected_only=True))
        copy_menu.addAction("표 전체 복사", lambda: self.copy_table(selected_only=False))
        copy_btn.setMenu(copy_menu)

        dup_btn = QtWidgets.QToolButton(container)
        dup_btn.setText("⎘ 복제")
        dup_btn.setToolTip("선택한 실행을 같은 설정으로 복제합니다(상태는 queued).")
        dup_btn.clicked.connect(self.duplicate_selected)

        del_btn = QtWidgets.QToolButton(container)
        del_btn.setText("🗑 삭제")
        del_btn.clicked.connect(self.delete_selected)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        toolbar.addWidget(self.filter_edit, 1)
        toolbar.addWidget(self.status_filter)
        toolbar.addWidget(refresh_btn)
        toolbar.addWidget(export_btn)
        toolbar.addWidget(copy_btn)
        toolbar.addWidget(dup_btn)
        toolbar.addWidget(del_btn)

        self.view = QtWidgets.QTableView(container)
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)  # 헤더 클릭 정렬
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
        header.setToolTip("헤더를 클릭하면 정렬, 우클릭하면 컬럼 표시/숨김")

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
    # 2) 상세 영역 (실행 코드 / config.yml / 경로)
    # ==================================================================
    def _build_detail_area(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(self)

        self.detail_title = QtWidgets.QLabel("행을 선택하면 상세 정보가 표시됩니다.", container)
        self.detail_title.setStyleSheet("font-weight: bold;")
        self.detail_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.detail_meta = QtWidgets.QLabel("", container)
        self.detail_meta.setStyleSheet("color: #555;")
        self.detail_meta.setWordWrap(True)
        self.detail_meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        paths_box = QtWidgets.QGroupBox("경로", container)
        paths_layout = QtWidgets.QFormLayout(paths_box)
        paths_layout.setContentsMargins(8, 8, 8, 8)
        paths_layout.setSpacing(4)
        self._detail_path_edits: dict[str, QtWidgets.QLineEdit] = {}
        for key, label in self.DETAIL_PATHS:
            edit = QtWidgets.QLineEdit(paths_box)
            edit.setReadOnly(True)
            edit.setFont(monospace_font())
            copy_btn = QtWidgets.QToolButton(paths_box)
            copy_btn.setText("복사")
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
            "실행 코드 (Execution Command)", container, read_only=True, min_height=90
        )
        self.detail_config = LabeledText("config.yml", container, read_only=True, min_height=90)
        self.detail_notes = LabeledText("Metrics & Notes", container, read_only=True, min_height=90)

        self.detail_tabs = QtWidgets.QTabWidget(container)
        self.detail_tabs.addTab(self.detail_command, "실행 코드")
        self.detail_tabs.addTab(self.detail_config, "config.yml")
        self.detail_tabs.addTab(self.detail_notes, "Metrics / Notes")

        edit_btn = QtWidgets.QPushButton("✎ 이 실행을 폼으로 불러와 수정", container)
        edit_btn.clicked.connect(self.load_selected_into_form)

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
    # 3) 입력 폼
    # ==================================================================
    def _build_form_area(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget(scroll)
        self.form_layout = QtWidgets.QFormLayout(inner)
        self.form_layout.setContentsMargins(10, 10, 10, 10)
        self.form_layout.setSpacing(6)
        self.form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.form_title = QtWidgets.QLabel("", inner)
        self.form_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.form_layout.addRow(self.form_title)

        # -- 공통 상단 필드 -------------------------------------------------
        # Work ID 는 options.yaml 이 아니라 DB(works 테이블)에서 온다.
        self.work_combo = EditableCombo([], inner, "Work ID (없으면 새로 생성)")
        self.work_combo.setToolTip("이 실행이 속할 Work ID. 목록에 없는 값을 입력하면 새로 만듭니다.")

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
        now_btn.setText("지금")
        now_btn.clicked.connect(lambda: self.started_edit.setText(now_iso()))
        started_row = QtWidgets.QWidget(inner)
        started_layout = QtWidgets.QHBoxLayout(started_row)
        started_layout.setContentsMargins(0, 0, 0, 0)
        started_layout.setSpacing(4)
        started_layout.addWidget(self.started_edit, 1)
        started_layout.addWidget(now_btn)

        self.duration_edit = QtWidgets.QLineEdit(inner)
        self.duration_edit.setPlaceholderText("예: 3h 20m / 01:30:00 / 5400(초)")

        self.gpu_selector = GpuSelector(inner)

        self.form_layout.addRow("Work ID:", self.work_combo)
        self.form_layout.addRow("Server:", self.server_combo)
        self.form_layout.addRow("GPU:", self.gpu_selector)
        self.form_layout.addRow("Model:", self.model_combo)
        self.form_layout.addRow("Dataset:", self.dataset_combo)
        self.form_layout.addRow("Status:", self.status_combo)
        self.form_layout.addRow("시작 시각:", started_row)
        self.form_layout.addRow("실행 시간:", self.duration_edit)

        # -- Task 별 사용자 정의 필드 (options.yaml) --------------------------
        self._custom_section = self._section("Task 전용 필드", inner)
        self.form_layout.addRow(self._custom_section)
        self._custom_host = QtWidgets.QWidget(inner)
        self._custom_form = QtWidgets.QFormLayout(self._custom_host)
        self._custom_form.setContentsMargins(0, 0, 0, 0)
        self._custom_form.setSpacing(6)
        self._custom_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.form_layout.addRow(self._custom_host)

        # -- 하위 클래스 전용 필드 ------------------------------------------
        self._build_extra_form_rows(inner)

        # -- 경로 -------------------------------------------------------------
        self.dataset_path_edit = PathEdit(inner, "/mnt/data/DIV2K/train", compact=True)
        self.result_path_edit = PathEdit(inner, "/mnt/exp/SSL2SL/restormer_x4", compact=True)
        self.form_layout.addRow(self._section("경로", inner))
        self.form_layout.addRow(self._dataset_path_label() + ":", self.dataset_path_edit)
        self.form_layout.addRow("결과 폴더 경로:", self.result_path_edit)

        # -- 메트릭 -----------------------------------------------------------
        self.metrics_editor = MetricsEditor(
            self.METRIC_PRESETS, inner, config=self.config, task_getter=self.current_task_name
        )
        self.metrics_editor.metricsDefined.connect(self._on_metrics_defined)
        self.form_layout.addRow(self._section("평가 지표", inner))
        self.form_layout.addRow(self.metrics_editor)

        # -- 실행 코드 / config -------------------------------------------------
        self.command_input = LabeledText(
            "실행 코드 (Execution Command)",
            inner,
            placeholder=self.SAMPLE_COMMAND,
            min_height=90,
        )
        sample_cmd_btn = QtWidgets.QToolButton(inner)
        sample_cmd_btn.setText("샘플")
        sample_cmd_btn.setToolTip("예시 명령어를 채워 넣습니다.")
        sample_cmd_btn.clicked.connect(lambda: self.command_input.set_text(self.SAMPLE_COMMAND))
        self.command_input.add_header_widget(sample_cmd_btn)

        self.config_input = LabeledText(
            "config.yml", inner, placeholder="# YAML 설정 내용을 붙여 넣으세요", min_height=140
        )
        load_cfg_btn = QtWidgets.QToolButton(inner)
        load_cfg_btn.setText("파일에서 불러오기")
        load_cfg_btn.clicked.connect(self._load_config_from_file)
        self.config_input.add_header_widget(load_cfg_btn)

        self.notes_input = LabeledText(
            "Notes", inner, placeholder="자유 메모", mono=False, min_height=70
        )

        self.form_layout.addRow(self._section("실행 코드 / 설정", inner))
        self.form_layout.addRow(self.command_input)
        self.form_layout.addRow(self.config_input)
        self.form_layout.addRow(self.notes_input)

        # -- 버튼 ---------------------------------------------------------------
        self.save_btn = QtWidgets.QPushButton("＋ 새 실행 등록", inner)
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.save_form)

        self.new_btn = QtWidgets.QPushButton("폼 비우기 / 새 입력", inner)
        self.new_btn.clicked.connect(self.reset_form)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 6, 0, 0)
        buttons.addWidget(self.save_btn, 2)
        buttons.addWidget(self.new_btn, 1)
        self.form_layout.addRow(self._wrap(buttons, inner))

        scroll.setWidget(inner)
        return scroll

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        """하위 클래스에서 전용 입력 필드를 추가한다."""

    # -- 옵션 콤보 ------------------------------------------------------------
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
                toast(self, True, f"기존 기록 {changed} 건의 {field} 을(를) '{new}' 로 바꿨습니다.")

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
        return "Dataset 경로"

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
            self, "config 파일 선택", QtCore.QDir.homePath(), "YAML/JSON (*.yml *.yaml *.json);;모든 파일 (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                self.config_input.set_text(fp.read())
        except OSError as exc:
            toast(self, False, f"파일을 읽지 못했습니다:\n{exc}", "config 불러오기")
            return
        toast(self, True, f"config 를 불러왔습니다: {os.path.basename(path)}")

    # ==================================================================
    # 스코프 / 로딩
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
        """행은 그대로 두고 컬럼 구성만 다시 만든다(설정이 바뀐 직후)."""
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
        """설정 목록을 기준으로 하되, DB 에만 남아 있는 옛 값도 보이도록 합친다."""
        scope = self._task_id  # 다른 Task 의 값이 섞이지 않도록 범위를 건다
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

    # -- Task 전용 사용자 정의 필드 --------------------------------------------
    def _rebuild_custom_fields(self) -> None:
        """options.yaml 의 사용자 정의 필드(scale, noise_sigma …)를 폼에 만든다."""
        fields = self.config.custom_fields(self._task_name) if self.config else []
        current = {name: combo.current_text() for name, combo in self._custom_widgets.items()}

        while self._custom_form.count():
            item = self._custom_form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)   # deleteLater 만으로는 다음 이벤트 루프까지 화면에 남는다
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
        """하위 클래스 전용 콤보 값 갱신."""

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
        scope = "전체"
        if self._work_id:
            work = self.db.get_work(self._work_id)
            if work:
                scope = f"{work['task_name']} ▸ {work['name']}"
        elif self._task_id:
            task = next((t for t in self.db.list_tasks() if t["id"] == self._task_id), None)
            if task:
                scope = f"{task['name']} (Task 전체)"
        self.count_label.setText(f"범위: {scope}   ·   표시 {visible} / 전체 {total} 건")

    # ==================================================================
    # 선택 / 상세
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
        self.detail_title.setText("행을 선택하면 상세 정보가 표시됩니다.")
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
        notes_text = metrics_to_text(metrics, sep="\n") or "(등록된 지표 없음)"
        notes = (row.get("notes") or "").strip()
        self.detail_notes.set_text(
            f"[Metrics]\n{notes_text}\n\n[Notes]\n{notes or '(없음)'}"
        )

    def _detail_meta_text(self, row: dict[str, Any]) -> str:
        status = str(row.get("status") or "")
        duration = format_duration(row.get("duration_sec"))
        gpus = str(row.get("gpu_indices") or "").strip()
        parts = [
            f"● {status}",
            f"{row.get('server') or '-'} · GPU {gpus or '-'}",
            f"시작 {row.get('started_at') or '-'}",
            f"실행 시간 {duration or '-'}",
            f"등록 {row.get('created_at') or '-'}",
        ]
        return "   |   ".join(p for p in parts if p.strip())

    # ==================================================================
    # 컨텍스트 메뉴
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
                action = menu.addAction(f"📁 {label} 열기")
                action.setEnabled(bool(path))
                action.triggered.connect(lambda _=False, p=path: self._open_path(p))
            menu.addSeparator()
            menu.addAction("실행 코드 복사", lambda: copy_to_clipboard(row.get("exec_command") or "", self, "실행 코드"))
            menu.addAction("config.yml 복사", lambda: copy_to_clipboard(row.get("config_yaml") or "", self, "config.yml"))
            menu.addSeparator()
            menu.addAction("✎ 폼으로 불러와 수정", self.load_selected_into_form)
            menu.addAction("⎘ 복제", self.duplicate_selected)
            menu.addSeparator()

        menu.addAction("선택한 행 복사 (TSV)", lambda: self.copy_table(selected_only=True))
        menu.addAction("표 전체 CSV 내보내기", self.export_csv)
        if row is not None:
            menu.addSeparator()
            menu.addAction("🗑 선택 삭제", self.delete_selected)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _header_context_menu(self, pos) -> None:
        header = self.view.horizontalHeader()
        section = header.logicalIndexAt(pos)
        spec = self.model.column_spec(section)
        self._context_column = spec

        menu = QtWidgets.QMenu(self)
        if spec is not None:
            menu.addAction(f"＋ 컬럼 추가…\t{editing.ADD_KEY}", self.add_column)
            if spec.editable_label:
                menu.addAction(
                    f"'{self.model.header_text(spec)}' 이름 변경\t{editing.RENAME_KEY}",
                    lambda: self.rename_column(spec),
                )
            if self._can_remove_column(spec):
                menu.addAction(
                    f"'{self.model.header_text(spec)}' 컬럼 제거", lambda: self.remove_column(spec)
                )
            if spec.is_metric:
                menu.addAction(
                    f"'{spec.source_name}' 지표 설정…", lambda: self.edit_metric(spec)
                )
            menu.addSeparator()
        menu.addAction("이 컬럼 숨기기", lambda: self._hide_column(section))
        menu.addAction("기본 컬럼 구성으로 복원", self.reset_columns)
        menu.addSeparator()

        visibility = menu.addMenu("표시할 컬럼")
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

    # -- 컬럼 관리 (options.yaml 에 저장) ---------------------------------------
    def _require_task(self) -> str | None:
        if not self._task_name:
            toast(self, False, "좌측에서 Task 를 먼저 선택하세요.", "컬럼 관리")
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
            toast(self, False, f"'{name}' 컬럼은 이미 있습니다.", "컬럼 추가")
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
        new = editing.prompt_text(self, "컬럼 이름 변경", "새 표시 이름:", old_label)
        if not new or new == old_label:
            return
        if spec.is_metric:
            # 지표는 설정의 key 자체를 바꾼다 (컬럼 목록도 함께 갱신됨).
            self.config.rename_metric(task, spec.source_name, new)
        else:
            # 내장/사용자 필드는 표시명만 바꾼다 (데이터 키는 유지).
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
            "컬럼 제거",
            f"'{label}' 컬럼을 {task} 의 {self.KIND} 표에서 뺄까요?\n"
            "데이터는 지워지지 않고 표에서만 사라집니다.",
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
        if not editing.confirm(self, "컬럼 복원", f"{task} 의 {self.KIND} 컬럼 구성을 초기화할까요?"):
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
        toast(self, ok, message, "폴더 열기")

    def _on_double_click(self, index) -> None:
        spec = self.model.column_spec(self.proxy.mapToSource(index).column())
        row = self._current_row()
        if spec is not None and spec.key in PATH_KEYS and row is not None:
            self._open_path(str(row.get(spec.key) or ""))
            return
        self.load_selected_into_form()

    # ==================================================================
    # 내보내기 / 클립보드
    # ==================================================================
    def export_csv(self) -> None:
        if self.proxy.rowCount() == 0:
            toast(self, False, "내보낼 행이 없습니다.", "CSV 내보내기")
            return
        default_name = f"{self.KIND}_runs_{QtCore.QDate.currentDate().toString('yyyyMMdd')}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "CSV 로 내보내기",
            os.path.join(QtCore.QDir.homePath(), default_name),
            "CSV 파일 (*.csv)",
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
            toast(self, False, f"CSV 저장 실패:\n{exc}", "CSV 내보내기")
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "CSV 내보내기 완료",
            f"{count} 행을 저장했습니다.\n{path}\n\n저장 폴더를 열까요?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            open_in_file_manager(path, reveal=True)

    def copy_table(self, selected_only: bool) -> None:
        text = table_selection_to_tsv(self.view, self.model.headers(), selected_only)
        if not text:
            toast(self, False, "복사할 행이 없습니다.", "클립보드 복사")
            return
        line_count = max(0, len(text.splitlines()) - 1)
        QtWidgets.QApplication.clipboard().setText(text)
        toast(self, True, f"{line_count} 행을 클립보드에 복사했습니다. (엑셀에 바로 붙여넣기 가능)")

    # ==================================================================
    # 폼 <-> DB
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
        """하위 클래스 전용 필드 초기화."""

    def load_selected_into_form(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "먼저 표에서 실행을 선택하세요.", "불러오기")
            return
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
        """하위 클래스 전용 필드 로드."""

    def _collect_extra_fields(self) -> dict[str, Any]:
        """하위 클래스 전용 필드 수집."""
        return {}

    def _update_form_buttons(self) -> None:
        if self._editing_id is None:
            self.form_title.setText("새 실행 등록")
            self.save_btn.setText("＋ 새 실행 등록")
        else:
            self.form_title.setText(f"실행 #{self._editing_id} 수정 중")
            self.save_btn.setText(f"💾 #{self._editing_id} 저장")

    def _resolve_work_id(self) -> int | None:
        """폼의 Work ID 텍스트를 실제 work row 로 해석(없으면 생성)."""
        name = self.work_combo.current_text()
        if self._task_id is None:
            QtWidgets.QMessageBox.warning(
                self, "저장 불가", "좌측에서 DL Task 를 먼저 선택하세요."
            )
            return None
        if not name:
            QtWidgets.QMessageBox.warning(self, "저장 불가", "Work ID 를 입력하거나 선택하세요.")
            return None
        for work in self.db.list_works(self._task_id):
            if work["name"].lower() == name.lower():
                return int(work["id"])
        answer = QtWidgets.QMessageBox.question(
            self,
            "새 Work ID",
            f"'{name}' Work 가 없습니다. 새로 만들까요?",
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
            QtWidgets.QMessageBox.warning(self, "저장 불가", "Model 을 입력하세요.")
            self.model_combo.setFocus()
            return None

        duration_text = self.duration_edit.text().strip()
        duration = parse_duration(duration_text)
        if duration_text and duration is None:
            QtWidgets.QMessageBox.warning(
                self,
                "저장 불가",
                "실행 시간 형식을 해석할 수 없습니다.\n예: 3h 20m · 01:30:00 · 5400",
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
            message = f"새 실행 #{run_id} 을(를) 등록했습니다."
        else:
            run_id = self._editing_id
            self.db.update_run(self.KIND, run_id, data)
            message = f"실행 #{run_id} 을(를) 저장했습니다."

        self._work_id = int(data["work_id"]) if self._work_id is not None else self._work_id
        # 트리/서버 인디케이터를 먼저 갱신한 뒤 표를 다시 그려야 선택이 유지된다.
        self.runsChanged.emit()
        self.reload()
        self._select_run(run_id)
        toast(self, True, message, "저장")

    def duplicate_selected(self) -> None:
        row = self._current_row()
        if row is None:
            toast(self, False, "복제할 실행을 선택하세요.", "복제")
            return
        new_id = self.db.duplicate_run(self.KIND, int(row["id"]))
        self.runsChanged.emit()
        self.reload()
        if new_id:
            self._select_run(new_id)
        toast(self, True, f"실행 #{row['id']} → #{new_id} 로 복제했습니다.")

    def delete_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            toast(self, False, "삭제할 실행을 선택하세요.", "삭제")
            return
        ids = [int(r["id"]) for r in rows]
        answer = QtWidgets.QMessageBox.question(
            self,
            "삭제 확인",
            f"선택한 {len(ids)} 건의 실행 기록을 삭제할까요?\n(#{', #'.join(map(str, ids))})",
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
        toast(self, True, f"{deleted} 건을 삭제했습니다.")

    def _select_run(self, run_id: int) -> None:
        for row in range(self.proxy.rowCount()):
            index = self.proxy.index(row, 0)
            data = self.proxy.data(index, RAW_ROLE)
            if isinstance(data, dict) and int(data.get("id", -1)) == int(run_id):
                self.view.selectRow(row)
                self.view.scrollTo(index)
                return


class TrainPanel(BaseRunPanel):
    """Train 대시보드."""

    KIND = "train"
    SAMPLE_COMMAND = C.SAMPLE_TRAIN_CMD
    METRIC_PRESETS = C.TRAIN_METRIC_PRESETS
    DETAIL_PATHS = (
        ("dataset_path", "Dataset 경로"),
        ("result_path", "결과 폴더 경로"),
    )

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        self.epochs_edit = QtWidgets.QLineEdit(parent)
        self.epochs_edit.setPlaceholderText("예: 300000 iter / 200 epoch")
        self.batch_edit = QtWidgets.QLineEdit(parent)
        self.batch_edit.setPlaceholderText("예: 8")
        self.lr_edit = QtWidgets.QLineEdit(parent)
        self.lr_edit.setPlaceholderText("예: 3e-4")
        self.optimizer_combo = self._make_option_combo("optimizer", "Optimizer", parent)

        self.form_layout.addRow(self._section("학습 하이퍼파라미터", parent))
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
    """Inference 대시보드."""

    KIND = "inference"
    SAMPLE_COMMAND = C.SAMPLE_INFER_CMD
    METRIC_PRESETS = C.INFER_METRIC_PRESETS
    DETAIL_PATHS = (
        ("checkpoint_path", "체크포인트 경로"),
        ("dataset_path", "테스트 데이터셋 경로"),
        ("result_path", "결과 폴더 경로"),
    )

    def _dataset_path_label(self) -> str:
        return "테스트 데이터셋 경로"

    def _build_extra_form_rows(self, parent: QtWidgets.QWidget) -> None:
        self.checkpoint_edit = PathEdit(
            parent,
            "/mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth",
            directory=False,
            compact=True,
        )
        self.device_combo = self._make_option_combo("device", "Device", parent)
        self.input_size_edit = QtWidgets.QLineEdit(parent)
        self.input_size_edit.setPlaceholderText("예: 3x256x256")
        self.latency_edit = QtWidgets.QLineEdit(parent)
        self.latency_edit.setPlaceholderText("이미지 1장당 ms")
        self.throughput_edit = QtWidgets.QLineEdit(parent)
        self.throughput_edit.setPlaceholderText("images/sec (FPS)")

        self.form_layout.addRow(self._section("추론 설정 / 속도", parent))
        self.form_layout.addRow("체크포인트 경로:", self.checkpoint_edit)
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
