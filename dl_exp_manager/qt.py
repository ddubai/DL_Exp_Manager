"""Qt 바인딩 추상화 레이어.

PyQt6 를 우선 사용하고, 없으면 PySide6 로 폴백한다.
두 바인딩의 API 차이(signal 데코레이터 이름, exec 반환 등)는 여기서 흡수한다.
"""
from __future__ import annotations

QT_BINDING: str

try:  # pragma: no cover - 환경에 따라 분기
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore

    QT_BINDING = "PyQt6"
    Signal = QtCore.pyqtSignal
    Slot = QtCore.pyqtSlot
    Property = QtCore.pyqtProperty
except ImportError:  # pragma: no cover
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyQt6 또는 PySide6 가 필요합니다.  `pip install -r requirements.txt` 를 실행하세요."
        ) from exc

    QT_BINDING = "PySide6"
    Signal = QtCore.Signal
    Slot = QtCore.Slot
    Property = QtCore.Property

Qt = QtCore.Qt

__all__ = [
    "QT_BINDING",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "Qt",
    "Signal",
    "Slot",
    "Property",
]
