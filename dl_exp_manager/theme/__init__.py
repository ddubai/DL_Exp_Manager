"""테마 적용 - QPalette + QSS + 폰트.

QSS 만으로는 툴팁이나 일부 네이티브 요소가 OS 기본 밝은 색으로 남기 때문에
QPalette 도 함께 설정한다.
"""
from __future__ import annotations

import os
import re

from ..qt import QtGui
from . import fonts as _fonts
from .tokens import FONT_SIZES, METRICS, SERIES, palette, series_color

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dark.qss.tpl")
_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.\-]+)\}\}")

_current: dict[str, str] = palette("dark")
_current_name = "dark"


def color(token: str, fallback: str = "#FF00FF") -> str:
    """토큰 이름으로 현재 테마의 색을 얻는다."""
    return _current.get(token, fallback)


def status_color(status: str) -> str:
    return _current.get(f"status.{status}", _current["text.secondary"])


def current_theme() -> str:
    return _current_name


def render_qss(theme: str = "dark") -> str:
    values: dict[str, str] = dict(palette(theme))
    values.update({k: str(v) for k, v in METRICS.items()})
    with open(_TEMPLATE, "r", encoding="utf-8") as fp:
        template = fp.read()

    missing: list[str] = []

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            missing.append(key)
            return match.group(0)
        return values[key]

    qss = _PLACEHOLDER.sub(substitute, template)
    if missing:  # 토큰 오타를 조용히 넘기지 않는다.
        raise KeyError(f"QSS 템플릿에 정의되지 않은 토큰: {sorted(set(missing))}")
    return qss


def build_palette(theme: str = "dark") -> QtGui.QPalette:
    values = palette(theme)
    qp = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    group = QtGui.QPalette.ColorGroup

    def set_color(color_role, token: str) -> None:
        qp.setColor(color_role, QtGui.QColor(values[token]))

    set_color(role.Window, "bg.base")
    set_color(role.WindowText, "text.primary")
    set_color(role.Base, "bg.input")
    set_color(role.AlternateBase, "bg.surface.alt")
    set_color(role.ToolTipBase, "bg.elevated")
    set_color(role.ToolTipText, "text.primary")
    set_color(role.Text, "text.primary")
    set_color(role.Button, "bg.elevated")
    set_color(role.ButtonText, "text.primary")
    set_color(role.BrightText, "danger")
    set_color(role.Link, "accent")
    set_color(role.Highlight, "accent.bg")
    set_color(role.HighlightedText, "text.primary")
    set_color(role.PlaceholderText, "text.muted")

    for disabled_role in (role.Text, role.ButtonText, role.WindowText):
        qp.setColor(group.Disabled, disabled_role, QtGui.QColor(values["text.disabled"]))
    return qp


def apply_theme(app, theme: str = "dark") -> None:
    """QApplication 에 테마를 적용한다."""
    global _current, _current_name
    _current = palette(theme)
    _current_name = theme

    _fonts.load_bundled_fonts()

    base_font = _fonts.ui_font(FONT_SIZES["body"])
    _fonts.apply_tabular_figures(base_font)
    app.setFont(base_font)

    app.setStyle("Fusion")  # QSS 가 플랫폼 스타일과 싸우지 않도록 고정
    app.setPalette(build_palette(theme))
    app.setStyleSheet(render_qss(theme))


__all__ = [
    "apply_theme",
    "build_palette",
    "color",
    "current_theme",
    "render_qss",
    "series_color",
    "status_color",
    "FONT_SIZES",
    "METRICS",
    "SERIES",
]
