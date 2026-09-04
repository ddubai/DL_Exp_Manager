"""작은 벡터 아이콘 - QPainter 로 그려서 이모지 대신 쓴다.

이모지(📁 ✎ 🗑)는 OS/폰트마다 굵기·색·정렬이 달라 다크 테마 안에서 붕 뜬다.
여기 아이콘들은 한 번 그려서 어디서든 같은 굵기·같은 색으로 보인다.
"""
from __future__ import annotations

from ..qt import QtCore, QtGui

_GRID = 20  # 좌표는 이 기준으로 잡고, 실제 렌더는 requested size 로 스케일한다.


def _painter(size: int) -> tuple[QtGui.QPixmap, QtGui.QPainter]:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    scale = size / _GRID
    painter.scale(scale, scale)
    return pixmap, painter


def _folder_path() -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    path.moveTo(2, 4.5)
    path.lineTo(8.2, 4.5)
    path.lineTo(9.8, 6.5)
    path.lineTo(18, 6.5)
    path.lineTo(18, 16)
    path.lineTo(2, 16)
    path.closeSubpath()
    return path


def _draw_folder(painter: QtGui.QPainter, color: QtGui.QColor) -> None:
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawPath(_folder_path())


def _draw_edit(painter: QtGui.QPainter, color: QtGui.QColor) -> None:
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.save()
    painter.translate(10, 10)
    painter.rotate(45)
    # 지우개 쪽(위)은 살짝 각지게, 촉(아래)은 뾰족하게 - 실루엣만으로 연필임을 알아보게.
    cap = QtCore.QRectF(-1.6, -8, 3.2, 2.4)
    painter.drawRoundedRect(cap, 0.6, 0.6)
    body = QtGui.QPolygonF(
        [
            QtCore.QPointF(-1.6, -5.8),
            QtCore.QPointF(1.6, -5.8),
            QtCore.QPointF(1.6, 3),
            QtCore.QPointF(0, 3),
            QtCore.QPointF(-1.6, 3),
        ]
    )
    painter.drawPolygon(body)
    tip = QtGui.QPolygonF(
        [QtCore.QPointF(-1.6, 3), QtCore.QPointF(1.6, 3), QtCore.QPointF(0, 8.2)]
    )
    painter.drawPolygon(tip)
    painter.restore()


def _draw_delete(painter: QtGui.QPainter, color: QtGui.QColor) -> None:
    pen = QtGui.QPen(color)
    pen.setWidthF(1.4)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawLine(QtCore.QPointF(3, 6), QtCore.QPointF(17, 6))
    painter.drawLine(QtCore.QPointF(8, 6), QtCore.QPointF(8, 4))
    painter.drawLine(QtCore.QPointF(8, 4), QtCore.QPointF(12, 4))
    painter.drawLine(QtCore.QPointF(12, 4), QtCore.QPointF(12, 6))
    body = QtGui.QPolygonF(
        [
            QtCore.QPointF(4.3, 6.5),
            QtCore.QPointF(15.7, 6.5),
            QtCore.QPointF(14.8, 17),
            QtCore.QPointF(5.2, 17),
        ]
    )
    painter.drawPolyline(body)
    painter.drawLine(QtCore.QPointF(4.3, 6.5), QtCore.QPointF(5.2, 17))
    painter.drawLine(QtCore.QPointF(15.7, 6.5), QtCore.QPointF(14.8, 17))
    for x in (7.5, 10, 12.5):
        painter.drawLine(QtCore.QPointF(x, 9), QtCore.QPointF(x, 14.5))


def _draw_add(painter: QtGui.QPainter, color: QtGui.QColor) -> None:
    pen = QtGui.QPen(color)
    pen.setWidthF(1.8)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(10, 4), QtCore.QPointF(10, 16))
    painter.drawLine(QtCore.QPointF(4, 10), QtCore.QPointF(16, 10))


_DRAWERS = {
    "folder": _draw_folder,
    "edit": _draw_edit,
    "delete": _draw_delete,
    "add": _draw_add,
}


def icon(name: str, color: str, size: int = 16) -> QtGui.QIcon:
    """`name` 아이콘을 `color` 로 그려 `QIcon` 으로 돌려준다.

    호출마다 새로 그린다 - 아이콘 자체가 몇 개 안 되고 매번 그려도 비용이 무시할
    만한 수준이라, 테마 전환 시 캐시를 무효화하는 번거로움보다 단순함을 택했다.
    """
    drawer = _DRAWERS.get(name)
    if drawer is None:
        raise KeyError(f"Unknown icon: {name!r}")
    pixmap, painter = _painter(size)
    drawer(painter, QtGui.QColor(color))
    painter.end()
    return QtGui.QIcon(pixmap)
