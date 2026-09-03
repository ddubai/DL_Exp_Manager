"""테이블 모델 - Task 별 컬럼 구성 + 타입에 맞는 정렬.

컬럼 구성은 `config/options.yaml` 의 `tasks.<이름>.columns` 에서 온다.
설정에 없는 Task 는 내장 기본 컬럼으로 떨어진다.
정렬은 표시 문자열이 아니라 원본 값(SORT_ROLE)을 기준으로 하므로
실행 시간·PSNR·Latency 가 사전순이 아닌 실제 크기순으로 정렬된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from . import theme
from .config_store import MetricDef, OptionsConfig
from .qt import Qt, QtCore, QtGui
from .utils import (
    elapsed_since,
    format_duration,
    format_number,
    loads_metrics,
    parse_iso,
)

SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
RAW_ROLE = int(Qt.ItemDataRole.UserRole) + 2   # 원본 row dict
COLUMN_ROLE = int(Qt.ItemDataRole.UserRole) + 3  # 이 컬럼의 ColumnSpec

METRIC_PREFIX = "metric:"
EXTRA_PREFIX = "extra:"


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    header: str
    kind: str = "text"     # text | path | number | duration | status | datetime | gpus
    width: int = 120
    metric: MetricDef | None = None
    editable_label: bool = True   # 헤더 우클릭으로 이름을 바꿀 수 있는지

    @property
    def is_metric(self) -> bool:
        return self.key.startswith(METRIC_PREFIX)

    @property
    def is_extra(self) -> bool:
        return self.key.startswith(EXTRA_PREFIX)

    @property
    def source_name(self) -> str:
        """설정 파일에 적히는 이름 (metric key / 사용자 필드명 / 내장 필드 id)."""
        if self.is_metric:
            return self.key[len(METRIC_PREFIX):]
        if self.is_extra:
            return self.key[len(EXTRA_PREFIX):]
        return self.key


# 내장 필드 레지스트리 - options.yaml 의 columns 에 이 id 를 쓸 수 있다.
FIELD_SPECS: dict[str, ColumnSpec] = {
    spec.key: spec
    for spec in (
        ColumnSpec("id", "ID", "number", 52, editable_label=False),
        ColumnSpec("status", "Status", "status", 108, editable_label=False),
        ColumnSpec("server", "Server", "text", 92),
        ColumnSpec("gpus", "GPU", "gpus", 112),
        ColumnSpec("model", "Model", "text", 126),
        ColumnSpec("task_name", "Task", "text", 84),
        ColumnSpec("work_name", "Work ID", "text", 110),
        ColumnSpec("dataset", "Dataset", "text", 124),
        ColumnSpec("dataset_path", "Dataset Path", "path", 230),
        ColumnSpec("checkpoint_path", "Checkpoint Path", "path", 250),
        ColumnSpec("result_path", "Result Folder Path", "path", 230),
        ColumnSpec("device", "Device", "text", 88),
        ColumnSpec("input_size", "Input Size", "text", 100),
        ColumnSpec("started_at", "Started At", "datetime", 140),
        ColumnSpec("duration", "Duration", "duration", 116),
        ColumnSpec("latency_ms", "Latency (ms)", "number", 108),
        ColumnSpec("throughput_fps", "Throughput (FPS)", "number", 126),
        ColumnSpec("epochs", "Epochs/Iter", "text", 96),
        ColumnSpec("batch_size", "Batch", "text", 70),
        ColumnSpec("lr", "LR", "text", 82),
        ColumnSpec("optimizer", "Optimizer", "text", 94),
        ColumnSpec("notes", "Notes", "text", 180),
    )
}

# 설정에 columns 가 없을 때 쓰는 기본 구성
FALLBACK_COLUMNS: dict[str, tuple[str, ...]] = {
    "train": ("status", "server", "gpus", "model", "dataset", "dataset_path",
              "result_path", "started_at", "duration"),
    "inference": ("status", "server", "gpus", "model", "checkpoint_path", "dataset",
                  "dataset_path", "result_path", "latency_ms", "throughput_fps", "started_at"),
}

# 항상 맨 앞/뒤에 붙는 컬럼
LEADING_COLUMNS: tuple[str, ...] = ("id",)
TRAILING_COLUMNS: tuple[str, ...] = ("notes",)

PATH_KEYS = {"dataset_path", "result_path", "checkpoint_path"}


def format_metric(value: Any, metric: MetricDef | None) -> str:
    """지표 정의의 자릿수/단위를 반영해 표시 문자열을 만든다."""
    if value in (None, ""):
        return ""
    digits = metric.digits if metric else 4
    text = format_number(value, digits)
    if text and metric and metric.unit:
        return f"{text} {metric.unit}"
    return text


def build_columns(
    config: OptionsConfig | None,
    task: str | None,
    mode: str,
    extra_metric_keys: Sequence[str] = (),
) -> list[ColumnSpec]:
    """Task + 모드(train/inference)에 맞는 컬럼 목록을 만든다.

    `extra_metric_keys` 는 데이터에는 있지만 설정에는 없는 지표로,
    설정에 없다고 값을 숨기지 않도록 뒤에 덧붙인다.
    """
    ids: list[str] = []
    if config is not None:
        ids = list(config.columns_for(task, mode))
    if not ids:
        ids = list(FALLBACK_COLUMNS.get(mode, FALLBACK_COLUMNS["train"]))
        if config is not None:
            ids += config.metric_keys(task)

    metric_keys = set(config.metric_keys(task)) if config is not None else set()
    custom_fields = set(config.custom_fields(task)) if config is not None else set()

    specs: list[ColumnSpec] = []
    seen: set[str] = set()

    def add(spec: ColumnSpec) -> None:
        if spec.key in seen:
            return
        seen.add(spec.key)
        specs.append(spec)

    for column_id in LEADING_COLUMNS:
        add(FIELD_SPECS[column_id])

    for column_id in ids:
        column_id = str(column_id).strip()
        if not column_id:
            continue
        if column_id in FIELD_SPECS:
            add(FIELD_SPECS[column_id])
        elif column_id in metric_keys:
            metric = config.metric_def(task, column_id) if config else None
            add(ColumnSpec(METRIC_PREFIX + column_id, column_id, "number", 92, metric=metric))
        elif column_id in custom_fields:
            add(ColumnSpec(EXTRA_PREFIX + column_id, column_id, "text", 100))
        else:
            # 설정에만 있고 정체를 모르는 이름 - 사용자 필드로 취급해 값이라도 보여 준다.
            add(ColumnSpec(EXTRA_PREFIX + column_id, column_id, "text", 100))

    for key in extra_metric_keys:
        if METRIC_PREFIX + key not in seen:
            add(ColumnSpec(METRIC_PREFIX + key, key, "number", 92))

    for column_id in TRAILING_COLUMNS:
        add(FIELD_SPECS[column_id])

    return specs


class RunTableModel(QtCore.QAbstractTableModel):
    """Train / Inference run 목록 모델."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._columns: list[ColumnSpec] = []
        self._rows: list[dict[str, Any]] = []
        self._labels: dict[str, str] = {}  # 사용자가 바꾼 헤더 표시명

    # -- 데이터 주입 --------------------------------------------------------
    def set_content(self, rows: Sequence[dict[str, Any]], columns: Sequence[ColumnSpec]) -> None:
        self.beginResetModel()
        self._rows = [self._prepare(dict(r)) for r in rows]
        self._columns = list(columns)
        self.endResetModel()

    @staticmethod
    def _prepare(row: dict[str, Any]) -> dict[str, Any]:
        row["_metrics"] = loads_metrics(row.get("metrics_json"))
        try:
            extra = json.loads(row.get("extra_json") or "{}")
        except (TypeError, ValueError):
            extra = {}
        row["_extra"] = extra if isinstance(extra, dict) else {}
        return row

    @staticmethod
    def metric_keys_in(rows: Sequence[dict[str, Any]]) -> list[str]:
        """행들에 실제로 들어 있는 지표 키 목록."""
        keys: set[str] = set()
        for row in rows:
            keys.update(loads_metrics(row.get("metrics_json")).keys())
        return sorted(keys)

    def set_header_label(self, key: str, label: str) -> None:
        self._labels[key] = label
        index = self.column_index(key)
        if index >= 0:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, index, index)

    # -- 조회 헬퍼 ----------------------------------------------------------
    def columns(self) -> list[ColumnSpec]:
        return list(self._columns)

    def column_spec(self, index: int) -> ColumnSpec | None:
        if 0 <= index < len(self._columns):
            return self._columns[index]
        return None

    def column_index(self, key: str) -> int:
        for i, spec in enumerate(self._columns):
            if spec.key == key:
                return i
        return -1

    def row_dict(self, row: int) -> dict[str, Any] | None:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None

    def header_text(self, spec: ColumnSpec) -> str:
        return self._labels.get(spec.key, spec.header)

    # -- QAbstractTableModel ------------------------------------------------
    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:  # noqa: N802
        if parent is not None and parent.isValid():
            return 0
        return len(self._columns)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal:
            return section + 1 if role == Qt.ItemDataRole.DisplayRole else None
        spec = self.column_spec(section)
        if spec is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.header_text(spec)
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [self.header_text(spec)]
            if spec.metric is not None:
                direction = "higher is better" if spec.metric.higher_is_better else "lower is better"
                lines.append(f"Metric · {direction} · {spec.metric.digits} decimal digits")
            elif spec.is_extra:
                lines.append("Custom field (from Task config)")
            lines.append("Click header = sort · right-click = manage columns · F2 = rename")
            return "\n".join(lines)
        if role == COLUMN_ROLE:
            return spec
        return None

    def data(self, index: QtCore.QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        spec = self._columns[index.column()]

        if role == RAW_ROLE:
            return row
        if role == COLUMN_ROLE:
            return spec
        if role == SORT_ROLE:
            return self._sort_value(row, spec)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._display(row, spec)
        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(row, spec)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if spec.kind in ("number", "duration"):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            if spec.kind == "status":
                return QtGui.QBrush(QtGui.QColor(theme.status_color(str(row.get("status") or ""))))
            if spec.kind == "path" and not str(self._raw(row, spec) or ""):
                return QtGui.QBrush(QtGui.QColor(theme.color("text.disabled")))
        return None

    # -- 값 변환 ------------------------------------------------------------
    def _raw(self, row: dict[str, Any], spec: ColumnSpec) -> Any:
        if spec.is_metric:
            return row["_metrics"].get(spec.source_name)
        if spec.is_extra:
            return row["_extra"].get(spec.source_name)
        if spec.key == "duration":
            return row.get("duration_sec")
        if spec.key == "gpus":
            return row.get("gpu_indices")
        return row.get(spec.key)

    def _display(self, row: dict[str, Any], spec: ColumnSpec) -> str:
        value = self._raw(row, spec)
        if spec.kind == "status":
            status = str(value or "")
            return f"● {status}" if status else ""
        if spec.kind == "gpus":
            text = str(value or "").strip()
            return f"GPU {text}" if text else ""
        if spec.kind == "duration":
            if value in (None, "") and str(row.get("status")) == "running":
                running = elapsed_since(row.get("started_at"))
                return f"~{format_duration(running)}" if running is not None else ""
            return format_duration(value)
        if spec.is_metric:
            return format_metric(value, spec.metric)
        if spec.kind == "number":
            return format_number(value)
        return "" if value is None else str(value)

    def _sort_value(self, row: dict[str, Any], spec: ColumnSpec) -> Any:
        value = self._raw(row, spec)
        if spec.kind in ("number", "duration") or spec.is_metric:
            if value in (None, ""):
                if spec.kind == "duration" and str(row.get("status")) == "running":
                    return elapsed_since(row.get("started_at")) or float("-inf")
                return float("-inf")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")
        if spec.kind == "datetime":
            parsed = parse_iso(value)
            return parsed.timestamp() if parsed else float("-inf")
        if spec.kind == "gpus":
            text = str(value or "").strip()
            if not text:
                return float("inf")  # 미지정은 뒤로
            try:
                return float(text.split(",")[0])
            except ValueError:
                return float("inf")
        return str(value or "").lower()

    def _tooltip(self, row: dict[str, Any], spec: ColumnSpec) -> str:
        value = self._display(row, spec)
        if spec.kind == "path":
            return f"{value}\n\nDouble-click or right-click → Open Folder" if value else "(no path set)"
        if spec.kind == "gpus":
            server = row.get("server") or "-"
            return f"{server} · {value or 'no GPU set'}"
        if spec.key == "notes":
            return str(row.get("notes") or "")
        return value

    # -- 내보내기용 ---------------------------------------------------------
    def headers(self) -> list[str]:
        return [self.header_text(spec) for spec in self._columns]

    def row_values(self, row: int) -> list[str]:
        data = self._rows[row]
        return [self._display(data, spec) for spec in self._columns]


class RunFilterProxy(QtCore.QSortFilterProxyModel):
    """모든 컬럼 대상 대소문자 무시 부분일치 + 상태 필터."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        self.setDynamicSortFilter(True)
        self._text = ""
        self._status = ""

    def set_text_filter(self, text: str) -> None:
        self._text = (text or "").strip().lower()
        self.invalidateFilter()

    def set_status_filter(self, status: str) -> None:
        self._status = (status or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if model is None:
            return True
        row = model.row_dict(source_row)  # type: ignore[attr-defined]
        if row is None:
            return False
        if self._status and str(row.get("status") or "").lower() != self._status:
            return False
        if not self._text:
            return True
        for col in range(model.columnCount()):
            index = model.index(source_row, col, source_parent)
            text = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")
            if self._text in text.lower():
                return True
        return False
