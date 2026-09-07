"""여러 Run 을 나란히 비교하는 막대 그래프 - 외부 플로팅 라이브러리 없이 QPainter 로 직접 그린다.

`curve_chart.py` 와 같은 이유로 새 의존성을 추가하지 않는다(ROADMAP §16.7). 지표마다
스케일이 다르므로(PSNR ~30 vs SSIM ~0.9) 축을 하나로 공유하지 않고, **지표별로 그
지표 안에서만 정규화**한 작은 막대 묶음을 지표 개수만큼 나란히 그린다 - 값 라벨이
막대 위에 바로 붙어 있어 실제 수치는 항상 눈에 보인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import theme
from ..qt import Qt, QtCore, QtGui, QtWidgets
from ..utils import format_number

_BAR_W = 22
_BAR_GAP = 6
_GROUP_GAP = 30
_TOP_PAD = 22       # 값 라벨 자리
_BOTTOM_PAD = 34    # 지표 이름 자리
_LEGEND_H = 24


@dataclass
class MetricGroup:
    key: str
    unit: str = ""
    higher_is_better: bool = True
    values: list[float | None] = field(default_factory=list)  # run 순서와 동일


class MetricsChartWidget(QtWidgets.QWidget):
    """`run_labels` 개수만큼의 막대를, `groups` 안 지표 하나마다 하나씩 묶어 그린다."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._run_labels: list[str] = []
        self._groups: list[MetricGroup] = []
        self.setMinimumHeight(260)

    def set_data(self, run_labels: list[str], groups: list[MetricGroup]) -> None:
        self._run_labels = list(run_labels)
        self._groups = groups
        self._update_size()
        self.update()

    def _legend_font(self) -> QtGui.QFont:
        font = QtGui.QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        return font

    def _legend_width(self) -> float:
        """범례 한 줄이 실제로 차지하는 폭.

        막대만 보고 폭을 잡으면(Run 이 적고 지표도 적을 때) 범례 글자가 위젯보다
        넓어져 그대로 잘려 나간다 - `#12 · Restormer` 처럼 라벨이 길수록 잘 드러난다.
        그래서 최종 폭은 막대가 필요한 만큼과 범례가 필요한 만큼 중 큰 쪽을 쓴다.
        """
        metrics = QtGui.QFontMetrics(self._legend_font())
        total = 8.0
        for label in self._run_labels:
            total += 15 + metrics.horizontalAdvance(label) + 18
        return total

    def _update_size(self) -> None:
        n = max(1, len(self._run_labels))
        group_w = n * _BAR_W + (n - 1) * _BAR_GAP
        bars_w = len(self._groups) * (group_w + _GROUP_GAP) + _GROUP_GAP
        total_w = max(320, bars_w, self._legend_width())
        self.setMinimumWidth(int(total_w) + 1)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(self.minimumWidth() or 320, 300)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(theme.color("bg.surface")))

        text_color = QtGui.QColor(theme.color("text.secondary"))
        muted_color = QtGui.QColor(theme.color("text.muted"))
        best_color = QtGui.QColor(theme.color("metric.best"))

        if not self._groups or not self._run_labels:
            painter.setPen(text_color)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Select runs that share at least one metric to see a chart.",
            )
            return

        colors = [QtGui.QColor(theme.series_color(i)) for i in range(len(self._run_labels))]

        # -- 범례 (맨 위 한 줄) --------------------------------------------------
        legend_x = 8.0
        legend_y = 6.0
        painter.setFont(self._legend_font())
        for i, label in enumerate(self._run_labels):
            painter.setBrush(QtGui.QBrush(colors[i]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QtCore.QRectF(legend_x, legend_y + 3, 10, 10), 2, 2)
            painter.setPen(text_color)
            text_w = painter.fontMetrics().horizontalAdvance(label)
            painter.drawText(QtCore.QRectF(legend_x + 15, legend_y, text_w + 4, _LEGEND_H),
                              Qt.AlignmentFlag.AlignVCenter, label)
            legend_x += 15 + text_w + 18

        area = self.rect().adjusted(8, _LEGEND_H + 10, -8, -_BOTTOM_PAD)
        if area.height() <= 0 or area.width() <= 0:
            return

        n = len(self._run_labels)
        group_w = n * _BAR_W + (n - 1) * _BAR_GAP
        x = float(area.left()) + _GROUP_GAP / 2

        value_font = painter.font()
        value_font.setPointSizeF(max(7.5, value_font.pointSizeF() - 1.5))

        for group in self._groups:
            present = [v for v in group.values if v is not None]
            group_max = max(present) if present else 1.0
            if group_max <= 0:
                group_max = 1.0
            best = (max(present) if group.higher_is_better else min(present)) if present else None

            bx = x
            for i in range(n):
                value = group.values[i] if i < len(group.values) else None
                bar_rect = QtCore.QRectF(bx, 0, _BAR_W, 0)
                if value is None:
                    # 이 run 에는 없는 지표 - 값 라벨 자리에 "–" 를 놓고, 그 아래
                    # 세로 점선으로 "막대가 없다"는 걸 눈에 띄게 표시한다(1px 가로줄은
                    # 배경에 묻혀 안 보였다).
                    painter.setFont(value_font)
                    painter.setPen(muted_color)
                    label_rect = QtCore.QRectF(bx - 10, area.bottom() - 30, _BAR_W + 20, 14)
                    painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "–")
                    painter.setPen(QtGui.QPen(muted_color, 1.5, Qt.PenStyle.DashLine))
                    painter.drawLine(
                        QtCore.QPointF(bx + _BAR_W / 2, area.bottom() - 14),
                        QtCore.QPointF(bx + _BAR_W / 2, area.bottom()),
                    )
                else:
                    fraction = min(1.0, max(0.02, value / group_max))
                    bar_h = area.height() * fraction
                    bar_rect = QtCore.QRectF(bx, area.bottom() - bar_h, _BAR_W, bar_h)
                    is_best = best is not None and value == best and len(present) > 1
                    painter.setBrush(QtGui.QBrush(colors[i]))
                    painter.setPen(
                        QtGui.QPen(best_color, 2) if is_best else Qt.PenStyle.NoPen
                    )
                    painter.drawRoundedRect(bar_rect, 3, 3)

                    painter.setFont(value_font)
                    painter.setPen(best_color if is_best else text_color)
                    label_rect = QtCore.QRectF(bx - 10, bar_rect.top() - 16, _BAR_W + 20, 14)
                    painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, format_number(value))
                bx += _BAR_W + _BAR_GAP

            # 지표 이름 (+ 단위, + 방향 화살표)
            arrow = "↑" if group.higher_is_better else "↓"
            caption = f"{group.key} ({group.unit}) {arrow}" if group.unit else f"{group.key} {arrow}"
            painter.setFont(self.font())
            painter.setPen(text_color)
            painter.drawText(
                QtCore.QRectF(x - _GROUP_GAP / 2, area.bottom() + 6, group_w + _GROUP_GAP, _BOTTOM_PAD - 6),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                caption,
            )
            x += group_w + _GROUP_GAP
