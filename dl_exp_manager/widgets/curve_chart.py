"""학습 곡선 - 외부 플로팅 라이브러리 없이 QPainter 로 직접 그리는 가벼운 라인 차트.

pyqtgraph/matplotlib 을 새 의존성으로 추가하지 않고, iteration vs 값 하나만
보여주면 되는 단순한 요구에 맞춘 최소 구현이다.
"""
from __future__ import annotations

import os

from .. import theme
from ..log_parser import parse_loss_log
from ..qt import Qt, QtCore, QtGui, QtWidgets
from ..utils import format_number, scan_result_folder


class CurveChartWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float]] = []
        self.setMinimumSize(360, 220)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._points = sorted(points, key=lambda p: p[0])
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(theme.color("bg.surface")))

        grid_color = QtGui.QColor(theme.color("border.subtle"))
        text_color = QtGui.QColor(theme.color("text.secondary"))
        line_color = QtGui.QColor(theme.color("accent"))

        if len(self._points) < 2:
            painter.setPen(text_color)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Not enough data points to draw a curve."
            )
            return

        area = self.rect().adjusted(60, 12, -14, -30)

        xs = [p[0] for p in self._points]
        ys = [p[1] for p in self._points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max == x_min:
            x_max += 1
        if y_max == y_min:
            y_max += 1 if y_max == 0 else abs(y_max) * 0.1

        def to_px(x: float, y: float) -> QtCore.QPointF:
            fx = (x - x_min) / (x_max - x_min)
            fy = (y - y_min) / (y_max - y_min)
            return QtCore.QPointF(
                area.left() + fx * area.width(),
                area.bottom() - fy * area.height(),
            )

        # 가로 격자선 4단 + y축 라벨
        for i in range(5):
            y_px = area.bottom() - area.height() * i / 4
            painter.setPen(QtGui.QPen(grid_color, 1))
            painter.drawLine(QtCore.QPointF(area.left(), y_px), QtCore.QPointF(area.right(), y_px))
            value = y_min + (y_max - y_min) * i / 4
            painter.setPen(text_color)
            painter.drawText(
                QtCore.QRectF(0, y_px - 8, area.left() - 6, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                format_number(value),
            )

        # x축 라벨 (시작/끝 iteration)
        painter.setPen(text_color)
        painter.drawText(
            QtCore.QRectF(area.left(), area.bottom() + 4, area.width() / 2, 20),
            Qt.AlignmentFlag.AlignLeft, format_number(x_min),
        )
        painter.drawText(
            QtCore.QRectF(area.left() + area.width() / 2, area.bottom() + 4, area.width() / 2, 20),
            Qt.AlignmentFlag.AlignRight, format_number(x_max),
        )

        # 선
        path = QtGui.QPainterPath()
        path.moveTo(to_px(*self._points[0]))
        for point in self._points[1:]:
            path.lineTo(to_px(*point))
        painter.setPen(QtGui.QPen(line_color, 2))
        painter.drawPath(path)

        # 마지막 점 강조
        painter.setBrush(QtGui.QBrush(line_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(to_px(*self._points[-1]), 3, 3)


class CurveDialog(QtWidgets.QDialog):
    """result_path 안의 로그를 파싱해 지표를 골라 곡선으로 보여준다."""

    def __init__(
        self,
        result_path: str,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Training Curve",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 540)

        self._result_path = result_path
        self._log_path: str | None = None
        self._series: dict[str, list[tuple[int, float]]] = {}

        self.path_label = QtWidgets.QLabel(self)
        self.path_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.path_label.setWordWrap(True)

        self.metric_combo = QtWidgets.QComboBox(self)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

        self.chart = CurveChartWidget(self)

        browse_btn = QtWidgets.QPushButton("Browse for Log File…", self)
        browse_btn.clicked.connect(self._browse)
        refresh_btn = QtWidgets.QPushButton("↻ Refresh", self)
        refresh_btn.clicked.connect(self.refresh)
        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Metric:", self))
        top.addWidget(self.metric_combo, 1)
        top.addWidget(refresh_btn)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(browse_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addLayout(top)
        layout.addWidget(self.chart, 1)
        layout.addLayout(buttons)

        self._auto_detect()
        self.refresh()

    # -- log discovery (LogViewerDialog 와 같은 패턴) --------------------------
    def _auto_detect(self) -> None:
        found = scan_result_folder(self._result_path) if self._result_path else {}
        self._log_path = found.get("log")

    def _browse(self) -> None:
        start = self._log_path or self._result_path or QtCore.QDir.homePath()
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Log File", start, "Log/Text Files (*.log *.txt);;All Files (*)"
        )
        if chosen:
            self._log_path = chosen
            self.refresh()

    # -- content --------------------------------------------------------------
    def refresh(self) -> None:
        if not self._log_path or not os.path.isfile(self._log_path):
            self.path_label.setText(
                f"No log file found in {self._result_path or '(no result folder set)'}."
                " Use “Browse for Log File…” to pick one."
            )
            self.metric_combo.clear()
            self._series = {}
            self.chart.set_points([])
            return

        result = parse_loss_log(self._log_path)
        self._series = {}
        for iteration, values in result.points:
            for key, value in values.items():
                self._series.setdefault(key, []).append((iteration, value))

        self.path_label.setText(
            f"{self._log_path}   ·   {len(result.points)} logged point(s)"
        )

        current = self.metric_combo.currentText()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(sorted(self._series))
        index = self.metric_combo.findText(current)
        self.metric_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_combo.blockSignals(False)
        self._on_metric_changed(self.metric_combo.currentText())

    def _on_metric_changed(self, key: str) -> None:
        self.chart.set_points(self._series.get(key, []))
