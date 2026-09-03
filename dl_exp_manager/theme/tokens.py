"""디자인 토큰 - 색/치수의 단일 출처.

색을 바꾸고 싶으면 **이 파일만** 고치면 된다.
값의 근거와 컴포넌트 규격은 docs/STYLE_GUIDE.md 참고.
"""
from __future__ import annotations

DARK: dict[str, str] = {
    # -- 배경 / 표면 ---------------------------------------------------------
    "bg.base": "#16181D",
    "bg.surface": "#1C1F26",
    "bg.surface.alt": "#20242C",
    "bg.elevated": "#232730",
    "bg.input": "#12141A",
    "bg.hover": "#2A2F3A",
    # -- 선 -------------------------------------------------------------------
    "border.subtle": "#2A2F3A",
    "border.default": "#333A47",
    "border.strong": "#3A4150",
    # -- 텍스트 ---------------------------------------------------------------
    "text.primary": "#E6E9EF",
    "text.secondary": "#9BA3B4",
    "text.muted": "#6B7488",
    "text.disabled": "#4E5666",
    "text.on-accent": "#0E1116",
    # -- 강조 -----------------------------------------------------------------
    "accent": "#5B9DFF",
    "accent.hover": "#7FB4FF",
    "accent.pressed": "#3F82E6",
    "accent.bg": "#1E2A3D",
    "accent.border": "#2F4A78",
    # -- 상태 -----------------------------------------------------------------
    "status.queued": "#8B93A7",
    "status.running": "#4DA3FF",
    "status.done": "#3FBF7F",
    "status.failed": "#FF6B6B",
    # -- 의미 -----------------------------------------------------------------
    "metric.best": "#3FBF7F",
    "metric.worst": "#FF8A8A",
    "danger": "#FF6B6B",
    "warning": "#FFB454",
}

LIGHT: dict[str, str] = {
    "bg.base": "#F4F5F7",
    "bg.surface": "#FFFFFF",
    "bg.surface.alt": "#F7F8FA",
    "bg.elevated": "#FFFFFF",
    "bg.input": "#FFFFFF",
    "bg.hover": "#EDF0F4",
    "border.subtle": "#E3E6EB",
    "border.default": "#D3D8E0",
    "border.strong": "#B9C0CC",
    "text.primary": "#1B1F27",
    "text.secondary": "#5A6474",
    "text.muted": "#8A93A2",
    "text.disabled": "#B0B7C2",
    "text.on-accent": "#FFFFFF",
    "accent": "#2C6FE0",
    "accent.hover": "#1F5FCC",
    "accent.pressed": "#1A52B0",
    "accent.bg": "#E6EFFC",
    "accent.border": "#A9C7F5",
    "status.queued": "#7A8394",
    "status.running": "#2C6FE0",
    "status.done": "#1D9A5F",
    "status.failed": "#D9483F",
    "metric.best": "#1D9A5F",
    "metric.worst": "#D9483F",
    "danger": "#D9483F",
    "warning": "#C87B12",
}

# 시각화 시리즈 / GPU 슬롯이 공유하는 팔레트 (순환 사용)
SERIES: tuple[str, ...] = (
    "#5B9DFF",
    "#3FBF7F",
    "#FFB454",
    "#C792EA",
    "#4DD0E1",
    "#FF6B6B",
    "#A3BE8C",
    "#E5C07B",
)

# -- 치수 (4px 그리드) --------------------------------------------------------
METRICS: dict[str, int] = {
    "radius.small": 4,
    "radius.medium": 6,
    "radius.large": 8,
    "row.height": 26,
    "row.height.dense": 22,
    "space.xs": 4,
    "space.sm": 6,
    "space.md": 8,
    "space.lg": 12,
    "space.xl": 16,
}

# -- 폰트 크기 (pt) -----------------------------------------------------------
FONT_SIZES: dict[str, float] = {
    "title": 12.5,
    "section": 9.0,
    "header": 9.0,
    "body": 9.5,
    "mono": 9.0,
    "caption": 8.5,
}

THEMES: dict[str, dict[str, str]] = {"dark": DARK, "light": LIGHT}


def palette(name: str = "dark") -> dict[str, str]:
    return dict(THEMES.get(name, DARK))


def series_color(index: int) -> str:
    return SERIES[index % len(SERIES)]
