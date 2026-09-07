"""Qt 바인딩 - PySide6 (LGPL).

예전엔 PyQt6 를 우선 쓰고 PySide6 로 폴백했다. GPLv3/상용 라이선스인 PyQt6 대신
Qt 공식 배포판이자 LGPL 인 PySide6 하나로 정리했다 - 사내 서버 IP·실험 메타데이터가
들어가는 이 앱을 동료에게 그대로 넘겨도 라이선스 문제가 없어야 하기 때문이다.

이 파일이 유일한 import 지점이라는 원칙은 그대로 유지한다 - 나머지 코드는 전부
`from .qt import QtCore, QtWidgets, ...` 로만 Qt 를 참조하고, `PySide6` 를 직접
import 하지 않는다. 다시 바인딩을 바꿔야 할 일이 생기면 이 파일만 고치면 된다.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

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
