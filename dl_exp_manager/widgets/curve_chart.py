"""학습 곡선 - 외부 플로팅 라이브러리 없이 QPainter 로 직접 그리는 가벼운 라인 차트.

pyqtgraph/matplotlib 을 새 의존성으로 추가하지 않고, iteration/epoch vs 값을
보여주면 되는 단순한 요구에 맞춘 최소 구현이다. 로그 한 줄에 여러 지표가
동시에 찍히는 경우(`Average loss / Average psnr / Average ssim / ...`)가
흔해서, 지표를 하나씩 갈아 보는 대신 여러 개를 한 번에 겹쳐 볼 수 있게 한다.
"""
from __future__ import annotations

import os

from .. import theme
from ..log_parser import LogParseResult, parse_loss_log, parse_loss_log_text
from ..qt import Qt, QtCore, QtGui, QtWidgets
from ..utils import format_number, scan_result_folder


class CurveChartWidget(QtWidgets.QWidget):
    """iteration/epoch -> 값 시리즈 여러 개를 겹쳐 그린다.

    시리즈가 하나면 실제 값 축(y 라벨)을 그대로 보여준다. 둘 이상이면
    loss/PSNR/SSIM 처럼 값의 범위가 서로 달라 같은 축을 못 쓰므로, 시리즈마다
    자기 자신의 최소/최대로 0~1 정규화해 겹쳐 그린다("high"/"low" 라벨만 표시) -
    절대값은 체크박스 라벨 쪽(CurveDialog)에서 따로 보여준다.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: dict[str, list[tuple[float, float]]] = {}
        self._colors: dict[str, QtGui.QColor] = {}
        self.setMinimumSize(360, 220)

    def set_series(
        self,
        series: dict[str, list[tuple[float, float]]],
        colors: dict[str, QtGui.QColor] | None = None,
    ) -> None:
        self._series = {
            key: sorted(points, key=lambda p: p[0]) for key, points in series.items() if points
        }
        self._colors = colors or {}
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(theme.color("bg.surface")))

        grid_color = QtGui.QColor(theme.color("border.subtle"))
        text_color = QtGui.QColor(theme.color("text.secondary"))
        accent_color = QtGui.QColor(theme.color("accent"))

        drawable = {key: pts for key, pts in self._series.items() if len(pts) >= 2}
        if not drawable:
            painter.setPen(text_color)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Not enough data points to draw a curve."
            )
            return

        area = self.rect().adjusted(60, 12, -14, -30)
        all_xs = [p[0] for pts in drawable.values() for p in pts]
        x_min, x_max = min(all_xs), max(all_xs)
        if x_max == x_min:
            x_max += 1

        def x_px(x: float) -> float:
            return area.left() + (x - x_min) / (x_max - x_min) * area.width()

        def draw_grid(label_fn) -> None:
            for i in range(5):
                y_line = area.bottom() - area.height() * i / 4
                painter.setPen(QtGui.QPen(grid_color, 1))
                painter.drawLine(
                    QtCore.QPointF(area.left(), y_line), QtCore.QPointF(area.right(), y_line)
                )
                label = label_fn(i)
                if label:
                    painter.setPen(text_color)
                    painter.drawText(
                        QtCore.QRectF(0, y_line - 8, area.left() - 6, 16),
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        label,
                    )

        def draw_line(pts: list[tuple[float, float]], to_y, color: QtGui.QColor) -> None:
            path = QtGui.QPainterPath()
            path.moveTo(QtCore.QPointF(x_px(pts[0][0]), to_y(pts[0][1])))
            for x, y in pts[1:]:
                path.lineTo(QtCore.QPointF(x_px(x), to_y(y)))
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawPath(path)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QtCore.QPointF(x_px(pts[-1][0]), to_y(pts[-1][1])), 3, 3)

        if len(drawable) > 1:
            draw_grid(lambda i: {4: "high", 0: "low"}.get(i))
            for key, pts in drawable.items():
                ys = [p[1] for p in pts]
                y_min, y_max = min(ys), max(ys)
                span = (y_max - y_min) or (abs(y_max) * 0.1 or 1.0)

                def to_y(y: float, y_min: float = y_min, span: float = span) -> float:
                    return area.bottom() - (y - y_min) / span * area.height()

                draw_line(pts, to_y, self._colors.get(key, accent_color))
        else:
            key, pts = next(iter(drawable.items()))
            ys = [p[1] for p in pts]
            y_min, y_max = min(ys), max(ys)
            if y_max == y_min:
                y_max += 1 if y_max == 0 else abs(y_max) * 0.1
            draw_grid(lambda i: format_number(y_min + (y_max - y_min) * i / 4))
            draw_line(
                pts,
                lambda y: area.bottom() - (y - y_min) / (y_max - y_min) * area.height(),
                self._colors.get(key, accent_color),
            )

        # x축 라벨 (시작/끝 iteration/epoch) - 시리즈 공통
        painter.setPen(text_color)
        painter.drawText(
            QtCore.QRectF(area.left(), area.bottom() + 4, area.width() / 2, 20),
            Qt.AlignmentFlag.AlignLeft, format_number(x_min),
        )
        painter.drawText(
            QtCore.QRectF(area.left() + area.width() / 2, area.bottom() + 4, area.width() / 2, 20),
            Qt.AlignmentFlag.AlignRight, format_number(x_max),
        )


class CurveDialog(QtWidgets.QDialog):
    """result_path 안의 로그, 또는 붙여넣은 _loss_log.txt 텍스트를 파싱해 곡선으로 보여준다.

    로그 한 줄에 여러 지표가 함께 찍히는 경우(§log_parser `_AVERAGE_KV_RE`)를
    감안해, 지표 체크박스를 여러 개 동시에 켜서 한 차트에 겹쳐 볼 수 있다
    (기본값: 파싱된 지표 전부 켜짐). 체크박스 글자색이 곧 그 지표 선 색이라
    별도 범례가 필요 없다.
    """

    def __init__(
        self,
        result_path: str,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Training Curve",
        log_text: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 540)

        self._result_path = result_path
        self._log_path: str | None = None
        # 붙여넣은 텍스트가 있으면 파일 탐색 대신 그 내용을 고정으로 쓴다(New Run 폼의
        # _loss_log.txt 붙여넣기 칸 미리보기, 저장된 run 의 loss_log_text 재생용).
        self._log_text = log_text
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._colors: dict[str, QtGui.QColor] = {}
        self.metric_checks: dict[str, QtWidgets.QCheckBox] = {}

        self.path_label = QtWidgets.QLabel(self)
        self.path_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.path_label.setWordWrap(True)

        self._metrics_row = QtWidgets.QHBoxLayout()
        self._metrics_row.setContentsMargins(0, 0, 0, 0)
        self._metrics_row.setSpacing(10)
        metrics_host = QtWidgets.QWidget(self)
        metrics_host.setLayout(self._metrics_row)
        metrics_scroll = QtWidgets.QScrollArea(self)
        metrics_scroll.setWidget(metrics_host)
        metrics_scroll.setWidgetResizable(False)
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        metrics_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        metrics_scroll.setFixedHeight(34)

        select_all_btn = QtWidgets.QToolButton(self)
        select_all_btn.setText("All")
        select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        select_none_btn = QtWidgets.QToolButton(self)
        select_none_btn.setText("None")
        select_none_btn.clicked.connect(lambda: self._set_all_checked(False))

        self.chart = CurveChartWidget(self)

        browse_btn = QtWidgets.QPushButton("Browse for Log File…", self)
        browse_btn.clicked.connect(self._browse)
        browse_btn.setVisible(self._log_text is None)
        refresh_btn = QtWidgets.QPushButton("↻ Refresh", self)
        refresh_btn.clicked.connect(self.refresh)
        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Metrics:", self))
        top.addWidget(metrics_scroll, 1)
        top.addWidget(select_all_btn)
        top.addWidget(select_none_btn)
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
        if self._log_text is not None:
            return
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
        if self._log_text is not None:
            self._apply_result(
                parse_loss_log_text(self._log_text), "(pasted _loss_log.txt content)"
            )
            return

        if not self._log_path or not os.path.isfile(self._log_path):
            self.path_label.setText(
                f"No log file found in {self._result_path or '(no result folder set)'}."
                " Use “Browse for Log File…” to pick one."
            )
            self._series = {}
            self._rebuild_metric_checks()
            self.chart.set_series({})
            return

        self._apply_result(parse_loss_log(self._log_path), self._log_path)

    def _apply_result(self, result: LogParseResult, source_label: str) -> None:
        self._series = {}
        for iteration, values in result.points:
            for key, value in values.items():
                self._series.setdefault(key, []).append((iteration, value))

        self.path_label.setText(f"{source_label}   ·   {len(result.points)} logged point(s)")
        self._rebuild_metric_checks()

    def _rebuild_metric_checks(self) -> None:
        """지표 체크박스를 다시 만든다 - 이미 켜/꺼 둔 것은 그대로 유지한다.

        기본값은 전부 켜짐("Plot 을 같이") - 새로 처음 보는 지표만 켠 채로 추가한다.
        색은 정렬된 키 순서로 고정 배정해서, 체크를 껐다 켜도 같은 지표는 항상
        같은 색을 유지한다.
        """
        previous_checked = {
            key for key, box in self.metric_checks.items() if box.isChecked()
        }
        is_first_build = not self.metric_checks

        while self._metrics_row.count():
            item = self._metrics_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.metric_checks = {}

        keys = sorted(self._series)
        self._colors = {key: QtGui.QColor(theme.series_color(i)) for i, key in enumerate(keys)}
        for key in keys:
            box = QtWidgets.QCheckBox(key, self)
            box.setChecked(key in previous_checked if not is_first_build else True)
            box.setStyleSheet(f"QCheckBox {{ color: {self._colors[key].name()}; }}")
            box.toggled.connect(self._on_selection_changed)
            self._metrics_row.addWidget(box)
            self.metric_checks[key] = box
        self._metrics_row.addStretch(1)
        self._on_selection_changed()

    def _set_all_checked(self, checked: bool) -> None:
        for box in self.metric_checks.values():
            box.setChecked(checked)

    def _on_selection_changed(self, *_: object) -> None:
        selected = {
            key: self._series[key] for key, box in self.metric_checks.items() if box.isChecked()
        }
        self.chart.set_series(selected, self._colors)
