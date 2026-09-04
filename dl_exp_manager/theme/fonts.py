"""폰트 스택 해석.

번들 폰트를 두지 않고 시스템에 설치된 것 중 앞 순위를 고른다.
`assets/fonts/` 에 폰트 파일을 넣어 두면 자동으로 등록해서 함께 후보로 삼는다.
"""
from __future__ import annotations

import os

from ..qt import QtGui

UI_FONT_STACK: tuple[str, ...] = (
    "Pretendard",           # 1순위 — 한글+라틴 한 벌, OFL
    "Pretendard Variable",
    "Apple SD Gothic Neo",  # macOS
    "Malgun Gothic",        # Windows
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "SUIT",
    "Inter",
    "Segoe UI Variable",
    "Segoe UI",
)

MONO_FONT_STACK: tuple[str, ...] = (
    "JetBrains Mono",
    "D2Coding",
    "SF Mono",
    "Menlo",
    "Cascadia Code",
    "Consolas",
    "DejaVu Sans Mono",
    "Courier New",
)

_ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets",
    "fonts",
)


def load_bundled_fonts() -> list[str]:
    """assets/fonts 안의 ttf/otf 를 애플리케이션 폰트로 등록한다."""
    loaded: list[str] = []
    if not os.path.isdir(_ASSET_DIR):
        return loaded
    for name in sorted(os.listdir(_ASSET_DIR)):
        if not name.lower().endswith((".ttf", ".otf")):
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(os.path.join(_ASSET_DIR, name))
        if font_id != -1:
            loaded.extend(QtGui.QFontDatabase.applicationFontFamilies(font_id))
    return loaded


def _first_available(stack: tuple[str, ...], fallback: str) -> str:
    available = {f.lower() for f in QtGui.QFontDatabase.families()}
    for family in stack:
        if family.lower() in available:
            return family
    return fallback


def resolve_ui_family() -> str:
    return _first_available(UI_FONT_STACK, QtGui.QFont().defaultFamily())


def resolve_mono_family() -> str:
    fallback = QtGui.QFontDatabase.systemFont(
        QtGui.QFontDatabase.SystemFont.FixedFont
    ).family()
    return _first_available(MONO_FONT_STACK, fallback)


def ui_font(size: float, weight: int = 400) -> QtGui.QFont:
    font = QtGui.QFont(resolve_ui_family())
    font.setPointSizeF(size)
    font.setWeight(QtGui.QFont.Weight(weight))
    return font


def mono_font(size: float) -> QtGui.QFont:
    font = QtGui.QFont(resolve_mono_family())
    font.setPointSizeF(size)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font


def apply_tabular_figures(font: QtGui.QFont) -> QtGui.QFont:
    """숫자 폭을 고정해 지표 비교가 쉽도록 한다 (Qt 6.7+ 에서만 동작)."""
    setter = getattr(font, "setFeature", None)
    if callable(setter):
        try:
            setter("tnum", 1)
        except (TypeError, ValueError):
            pass
    return font
