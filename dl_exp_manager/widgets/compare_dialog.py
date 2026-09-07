"""여러 Run 을 나란히 비교 - 지표/하이퍼파라미터 표 + 막대 그래프 + config.yaml diff.

실험 관리에서 제일 자주 하는 동작인데("이번에 뭘 바꿔서 좋아졌지?") 지금까지는
Run 을 하나씩 열어 눈으로 대조해야 했다. 표에서 여러 개를 고르면 바로 뜬다.

config.yaml diff 는 두 개를 비교할 때만 의미가 있어(그 이상은 "누가 기준인지"가
애매해진다) 2개면 좌우로 나눈 diff(SideBySideDiffWidget - 추가는 초록, 삭제는
빨강, "바뀐 줄만 보기" 토글 포함) 한 장, 그 이상이면 Run 별 config 를 따로 탭으로
보여준다. 반면 지표 표와 막대 그래프(`metrics_chart.py`)는 몇 개를 골라도 그대로 늘어나므로,
"많이 돌려 놓고 한눈에 비교" 용도로는 선택 개수를 넉넉히 열어 둔다(호출부의 RUN_LIMIT).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from .. import theme
from ..config_store import OptionsConfig
from ..qt import Qt, QtGui, QtWidgets
from ..utils import format_duration, format_number, loads_metrics, parse_gpu_count
from .common import monospace_font
from .diff_view import SideBySideDiffWidget
from .metrics_chart import MetricGroup, MetricsChartWidget


class CompareRunsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        config: OptionsConfig,
        task_name: str | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        rows = sorted(rows, key=lambda r: int(r["id"]))
        self.setWindowTitle(f"Compare {len(rows)} Runs")
        self.resize(920, 640)

        table = QtWidgets.QTableWidget(0, len(rows) + 1, self)
        table.setHorizontalHeaderLabels(
            ["Field"] + [f"#{r['id']} · {r.get('model') or '-'}" for r in rows]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        for col in range(1, len(rows) + 1):
            table.horizontalHeader().setSectionResizeMode(
                col, QtWidgets.QHeaderView.ResizeMode.Stretch
            )

        field_rows = self._build_field_rows(rows, config, task_name)
        table.setRowCount(len(field_rows))
        differ_bg = QtGui.QColor(theme.color("accent.bg"))
        for r, (label, values) in enumerate(field_rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(label))
            differs = len({v for v in values}) > 1
            for c, value in enumerate(values, start=1):
                item = QtWidgets.QTableWidgetItem(value if value else "-")
                if differs:
                    item.setBackground(differ_bg)
                table.setItem(r, c, item)

        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(table, "Metrics / Params")

        chart = MetricsChartWidget(self)
        chart.set_data(
            [f"#{r['id']} · {r.get('model') or '-'}" for r in rows],
            self._build_metric_groups(rows, config, task_name),
        )
        chart_scroll = QtWidgets.QScrollArea(self)
        chart_scroll.setWidget(chart)
        chart_scroll.setWidgetResizable(False)  # 막대 묶음이 넓어지면 가로로만 스크롤한다
        chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        chart_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tabs.addTab(chart_scroll, "📊 Chart")

        if len(rows) == 2:
            left_text = rows[0].get("config_yaml") or ""
            right_text = rows[1].get("config_yaml") or ""
            if left_text.strip() == right_text.strip():
                placeholder = QtWidgets.QLabel("(config.yaml is identical, or empty on both runs)", self)
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tabs.addTab(placeholder, "config.yaml Diff")
            else:
                diff_widget = SideBySideDiffWidget(
                    left_text, right_text, f"#{rows[0]['id']}", f"#{rows[1]['id']}", self
                )
                tabs.addTab(diff_widget, "config.yaml Diff")
        else:
            for row in rows:
                text = QtWidgets.QPlainTextEdit(self)
                text.setReadOnly(True)
                text.setFont(monospace_font())
                text.setPlainText(row.get("config_yaml") or "(no config.yaml)")
                tabs.addTab(text, f"#{row['id']} config.yaml")

        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(tabs, 1)
        layout.addLayout(footer)

    @staticmethod
    def _build_field_rows(
        rows: Sequence[dict[str, Any]], config: OptionsConfig, task_name: str | None
    ) -> list[tuple[str, list[str]]]:
        def gpu_text(row: dict[str, Any]) -> str:
            count = parse_gpu_count(row.get("gpu_indices"))
            return f"{count} GPU(s)" if count else ""

        specs: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
            ("Status", lambda r: str(r.get("status") or "")),
            ("Server", lambda r: str(r.get("server") or "")),
            ("GPU", gpu_text),
            ("Model", lambda r: str(r.get("model") or "")),
            ("Dataset", lambda r: str(r.get("dataset") or "")),
            ("Duration", lambda r: format_duration(r.get("duration_sec"))),
            ("Epochs/Iter", lambda r: str(r.get("epochs") or "")),
            ("Batch size", lambda r: str(r.get("batch_size") or "")),
            ("LR", lambda r: str(r.get("lr") or "")),
            ("Optimizer", lambda r: str(r.get("optimizer") or "")),
        ]
        out: list[tuple[str, list[str]]] = [
            (label, [fn(row) for row in rows]) for label, fn in specs
        ]

        metric_keys = list(
            dict.fromkeys(
                list(config.metric_keys(task_name) if task_name else [])
                + [key for row in rows for key in loads_metrics(row.get("metrics_json"))]
            )
        )
        for key in metric_keys:
            values = []
            for row in rows:
                metrics = loads_metrics(row.get("metrics_json"))
                values.append(format_number(metrics[key]) if key in metrics else "")
            out.append((key, values))

        custom_fields = config.custom_fields(task_name) if task_name else []
        for field_name in custom_fields:
            values = []
            for row in rows:
                try:
                    extra = json.loads(row.get("extra_json") or "{}")
                except (TypeError, ValueError):
                    extra = {}
                values.append(str(extra.get(field_name, "")) if isinstance(extra, dict) else "")
            out.append((field_name, values))

        return out

    @staticmethod
    def _build_metric_groups(
        rows: Sequence[dict[str, Any]], config: OptionsConfig, task_name: str | None
    ) -> list[MetricGroup]:
        """지표 표(§ `_build_field_rows`)와 같은 키 순서를, 문자열이 아니라 숫자로 뽑는다.

        표는 사람이 읽을 형식이 필요해서 포맷팅한 문자열을 쓰지만, 막대 그래프는
        직접 비교·정규화해야 하므로 `loads_metrics` 가 돌려주는 원래 숫자를 그대로 쓴다.
        """
        metric_keys = list(
            dict.fromkeys(
                list(config.metric_keys(task_name) if task_name else [])
                + [key for row in rows for key in loads_metrics(row.get("metrics_json"))]
            )
        )
        groups: list[MetricGroup] = []
        for key in metric_keys:
            metric_def = config.metric_def(task_name, key) if task_name else None
            values: list[float | None] = []
            for row in rows:
                metrics = loads_metrics(row.get("metrics_json"))
                raw = metrics.get(key)
                try:
                    values.append(float(raw) if raw is not None else None)
                except (TypeError, ValueError):
                    values.append(None)
            if any(v is not None for v in values):
                groups.append(
                    MetricGroup(
                        key=key,
                        unit=metric_def.unit if metric_def else "",
                        higher_is_better=metric_def.higher_is_better if metric_def else True,
                        values=values,
                    )
                )
        return groups
