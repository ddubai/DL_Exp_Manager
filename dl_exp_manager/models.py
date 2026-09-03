"""QAbstractTableModel 구현.

- 고정 컬럼 + 실행별 메트릭(JSON)에서 파생된 동적 컬럼을 함께 노출한다.
- 정렬은 표시 문자열이 아니라 원본 값(UserRole)을 기준으로 하므로
  숫자/시간 컬럼이 사전순이 아닌 실제 크기순으로 정렬된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from . import constants as C
from .qt import Qt, QtCore, QtGui
from .utils import (
    elapsed_since,
    format_duration,
    format_number,
    loads_metrics,
    parse_iso,
)

SORT_ROLE = int(Qt.ItemDataRole.UserRole) + 1
RAW_ROLE = int(Qt.ItemDataRole.UserRole) + 2  # 원본 row dict


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    header: str
    kind: str = "text"  # text | path | number | duration | status | datetime
    width: int = 120
    metric: bool = False


TRAIN_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("id", "ID", "number", 52),
    ColumnSpec("status", "Status", "status", 90),
    ColumnSpec("server", "Server", "text", 90),
    ColumnSpec("model", "Model", "text", 120),
    ColumnSpec("task_name", "Task", "text", 80),
    ColumnSpec("work_name", "Work ID", "text", 110),
    ColumnSpec("dataset", "Dataset", "text", 110),
    ColumnSpec("dataset_path", "Dataset 경로", "path", 230),
    ColumnSpec("result_path", "결과 폴더 경로", "path", 230),
    ColumnSpec("started_at", "시작 시각", "datetime", 140),
    ColumnSpec("duration_sec", "실행 시간", "duration", 100),
    ColumnSpec("epochs", "Epochs/Iter", "text", 90),
    ColumnSpec("batch_size", "Batch", "text", 70),
    ColumnSpec("lr", "LR", "text", 80),
    ColumnSpec("optimizer", "Optimizer", "text", 90),
)

INFER_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("id", "ID", "number", 52),
    ColumnSpec("status", "Status", "status", 90),
    ColumnSpec("server", "Server", "text", 90),
    ColumnSpec("model", "Model", "text", 120),
    ColumnSpec("task_name", "Task", "text", 80),
    ColumnSpec("work_name", "Work ID", "text", 110),
    ColumnSpec("checkpoint_path", "체크포인트 경로", "path", 260),
    ColumnSpec("dataset", "Test Dataset", "text", 110),
    ColumnSpec("dataset_path", "테스트 데이터셋 경로", "path", 230),
    ColumnSpec("result_path", "결과 폴더 경로", "path", 230),
    ColumnSpec("device", "Device", "text", 90),
    ColumnSpec("input_size", "Input Size", "text", 100),
    ColumnSpec("latency_ms", "Latency (ms)", "number", 105),
    ColumnSpec("throughput_fps", "Throughput (FPS)", "number", 125),
    ColumnSpec("started_at", "실행 시각", "datetime", 140),
    ColumnSpec("duration_sec", "소요 시간", "duration", 100),
)

TRAILING_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("notes", "Notes", "text", 180),
)

# 폴더 열기 버튼/컨텍스트 메뉴가 인식하는 경로 컬럼
PATH_KEYS = {"dataset_path", "result_path", "checkpoint_path"}


class RunTableModel(QtCore.QAbstractTableModel):
    """Train / Inference run 목록 모델."""

    def __init__(self, columns: Sequence[ColumnSpec], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._base_columns = list(columns)
        self._trailing = list(TRAILING_COLUMNS)
        self._metric_columns: list[ColumnSpec] = []
        self._columns: list[ColumnSpec] = list(self._base_columns) + list(self._trailing)
        self._rows: list[dict[str, Any]] = []

    # -- 데이터 주입 --------------------------------------------------------
    def set_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = [self._prepare(dict(r)) for r in rows]
        self._rebuild_metric_columns()
        self.endResetModel()

    @staticmethod
    def _prepare(row: dict[str, Any]) -> dict[str, Any]:
        row["_metrics"] = loads_metrics(row.get("metrics_json"))
        return row

    def _rebuild_metric_columns(self) -> None:
        keys: list[str] = []
        seen: set[str] = set()
        # 자주 쓰는 메트릭을 앞쪽에 고정 배치하고, 나머지는 알파벳순.
        preferred = [k for k in C.TRAIN_METRIC_PRESETS + C.INFER_METRIC_PRESETS]
        present = {k for row in self._rows for k in row["_metrics"]}
        for key in preferred:
            if key in present and key not in seen:
                keys.append(key)
                seen.add(key)
        for key in sorted(present):
            if key not in seen:
                keys.append(key)
                seen.add(key)
        self._metric_columns = [
            ColumnSpec(f"metric:{k}", k, "number", 90, metric=True) for k in keys
        ]
        self._columns = list(self._base_columns) + self._metric_columns + list(self._trailing)

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
            if role == Qt.ItemDataRole.DisplayRole:
                return section + 1
            return None
        spec = self.column_spec(section)
        if spec is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return spec.header
        if role == Qt.ItemDataRole.ToolTipRole:
            if spec.metric:
                return f"{spec.header} (metrics JSON)\n헤더를 클릭하면 정렬됩니다."
            return f"{spec.header}\n헤더를 클릭하면 정렬됩니다."
        return None

    def data(self, index: QtCore.QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        spec = self._columns[index.column()]

        if role == RAW_ROLE:
            return row
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
        if role == Qt.ItemDataRole.ForegroundRole and spec.kind == "status":
            color = C.STATUS_COLORS.get(str(row.get("status") or ""), None)
            if color:
                return QtGui.QBrush(QtGui.QColor(color))
        if role == Qt.ItemDataRole.FontRole and spec.kind == "path":
            font = QtGui.QFont()
            font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
            return font
        return None

    # -- 값 변환 ------------------------------------------------------------
    def _raw(self, row: dict[str, Any], spec: ColumnSpec) -> Any:
        if spec.metric:
            return row["_metrics"].get(spec.header)
        return row.get(spec.key)

    def _display(self, row: dict[str, Any], spec: ColumnSpec) -> str:
        value = self._raw(row, spec)
        if spec.kind == "status":
            status = str(value or "")
            icon = C.STATUS_ICONS.get(status, "")
            return f"{icon} {status}".strip()
        if spec.kind == "duration":
            seconds = value
            if seconds in (None, "") and str(row.get("status")) == C.STATUS_RUNNING:
                running = elapsed_since(row.get("started_at"))
                return f"~{format_duration(running)}" if running is not None else ""
            return format_duration(seconds)
        if spec.kind == "number":
            return format_number(value)
        if value is None:
            return ""
        return str(value)

    def _sort_value(self, row: dict[str, Any], spec: ColumnSpec) -> Any:
        value = self._raw(row, spec)
        if spec.kind in ("number", "duration"):
            if value in (None, ""):
                if spec.kind == "duration" and str(row.get("status")) == C.STATUS_RUNNING:
                    return elapsed_since(row.get("started_at")) or float("-inf")
                return float("-inf")
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("-inf")
        if spec.kind == "datetime":
            dt = parse_iso(value)
            return dt.timestamp() if dt else float("-inf")
        return str(value or "").lower()

    def _tooltip(self, row: dict[str, Any], spec: ColumnSpec) -> str:
        value = self._display(row, spec)
        if spec.kind == "path" and value:
            return f"{value}\n\n우클릭 → '폴더 열기' 로 탐색기에서 열 수 있습니다."
        notes = str(row.get("notes") or "")
        if spec.key == "notes" and notes:
            return notes
        return value

    # -- 내보내기용 ---------------------------------------------------------
    def headers(self) -> list[str]:
        return [spec.header for spec in self._columns]

    def row_values(self, row: int) -> list[str]:
        data = self._rows[row]
        return [self._display(data, spec) for spec in self._columns]


class RunFilterProxy(QtCore.QSortFilterProxyModel):
    """모든 컬럼을 대상으로 하는 대소문자 무시 부분일치 필터 + 상태 필터."""

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
        column_count = model.columnCount()
        for col in range(column_count):
            index = model.index(source_row, col, source_parent)
            text = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")
            if self._text in text.lower():
                return True
        return False
