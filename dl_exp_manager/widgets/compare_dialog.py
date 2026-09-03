"""2~3개 Run 을 나란히 비교 - 지표/하이퍼파라미터 표 + config.yaml diff.

실험 관리에서 제일 자주 하는 동작인데("이번에 뭘 바꿔서 좋아졌지?") 지금까지는
Run 을 하나씩 열어 눈으로 대조해야 했다. 표에서 2~3개를 고르면 바로 뜬다.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from .. import theme
from ..config_store import OptionsConfig
from ..qt import QtGui, QtWidgets
from ..utils import format_duration, format_number, loads_metrics, parse_gpu_count, unified_diff_text
from .common import monospace_font


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

        if len(rows) == 2:
            diff_view = QtWidgets.QPlainTextEdit(self)
            diff_view.setReadOnly(True)
            diff_view.setFont(monospace_font())
            diff_text = unified_diff_text(
                rows[0].get("config_yaml") or "",
                rows[1].get("config_yaml") or "",
                f"#{rows[0]['id']}",
                f"#{rows[1]['id']}",
            )
            diff_view.setPlainText(diff_text or "(config.yaml is identical, or empty on both runs)")
            tabs.addTab(diff_view, "config.yaml Diff")
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
