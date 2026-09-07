"""좌우로 나란히 놓고 보는 텍스트 diff - 추가는 초록, 삭제는 빨강.

코드 리뷰 도구(GitHub split view)처럼: 같은 줄은 나란히, 한쪽에만 있는 줄은
반대쪽을 빈 줄로 채워 줄 번호가 어긋나지 않게 하고, 스크롤은 양쪽이 같이 움직인다.
"바뀐 부분만" 보기를 켜면 동일한 줄 구간을 "⋯ N unchanged line(s) ⋯" 한 줄로 접는다.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from .. import theme
from ..qt import QtGui, QtWidgets
from .common import monospace_font


@dataclass
class _Line:
    text: str
    kind: str  # equal | add | remove | blank | sep


class SideBySideDiffWidget(QtWidgets.QWidget):
    def __init__(
        self,
        left_text: str,
        right_text: str,
        left_label: str,
        right_label: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._left_full, self._right_full, self._added, self._removed = self._diff_lines(
            left_text, right_text
        )

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(0)
        for label in (left_label, right_label):
            lbl = QtWidgets.QLabel(label, self)
            lbl.setStyleSheet(f"font-weight: 600; padding: 2px 6px; color: {theme.color('text.secondary')};")
            header.addWidget(lbl, 1)

        self.summary_label = QtWidgets.QLabel(self)
        self.summary_label.setStyleSheet("padding: 0 6px;")

        self.changed_only_check = QtWidgets.QCheckBox("Changed lines only", self)
        self.changed_only_check.setChecked(True)
        self.changed_only_check.toggled.connect(self._render)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(self.summary_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.changed_only_check)

        self.left_view = QtWidgets.QPlainTextEdit(self)
        self.right_view = QtWidgets.QPlainTextEdit(self)
        for view in (self.left_view, self.right_view):
            view.setReadOnly(True)
            view.setFont(monospace_font())
            view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

        splitter = QtWidgets.QSplitter(self)
        splitter.addWidget(self.left_view)
        splitter.addWidget(self.right_view)
        splitter.setSizes([1, 1])
        splitter.setChildrenCollapsible(False)

        # 스크롤은 좌우가 같이 움직여야 같은 줄이 계속 나란히 보인다.
        self._syncing = False
        self.left_view.verticalScrollBar().valueChanged.connect(
            lambda v: self._sync(self.right_view.verticalScrollBar(), v)
        )
        self.right_view.verticalScrollBar().valueChanged.connect(
            lambda v: self._sync(self.left_view.verticalScrollBar(), v)
        )
        self.left_view.horizontalScrollBar().valueChanged.connect(
            lambda v: self._sync(self.right_view.horizontalScrollBar(), v)
        )
        self.right_view.horizontalScrollBar().valueChanged.connect(
            lambda v: self._sync(self.left_view.horizontalScrollBar(), v)
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addLayout(toolbar)
        layout.addWidget(splitter, 1)

        self._render()

    def _sync(self, bar: QtWidgets.QScrollBar, value: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        bar.setValue(value)
        self._syncing = False

    # -- diff 계산 --------------------------------------------------------------
    @staticmethod
    def _diff_lines(left_text: str, right_text: str) -> tuple[list[_Line], list[_Line], int, int]:
        left_src = (left_text or "").splitlines()
        right_src = (right_text or "").splitlines()
        matcher = difflib.SequenceMatcher(a=left_src, b=right_src, autojunk=False)
        left_lines: list[_Line] = []
        right_lines: list[_Line] = []
        added = removed = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    left_lines.append(_Line(left_src[i1 + k], "equal"))
                    right_lines.append(_Line(right_src[j1 + k], "equal"))
            elif tag == "delete":
                for k in range(i1, i2):
                    left_lines.append(_Line(left_src[k], "remove"))
                    right_lines.append(_Line("", "blank"))
                removed += i2 - i1
            elif tag == "insert":
                for k in range(j1, j2):
                    left_lines.append(_Line("", "blank"))
                    right_lines.append(_Line(right_src[k], "add"))
                added += j2 - j1
            elif tag == "replace":
                left_seg = left_src[i1:i2]
                right_seg = right_src[j1:j2]
                for k in range(max(len(left_seg), len(right_seg))):
                    left_lines.append(_Line(left_seg[k], "remove") if k < len(left_seg) else _Line("", "blank"))
                    right_lines.append(_Line(right_seg[k], "add") if k < len(right_seg) else _Line("", "blank"))
                removed += len(left_seg)
                added += len(right_seg)

        return left_lines, right_lines, added, removed

    # -- "바뀐 부분만" 접기 --------------------------------------------------------
    def _collapsed(self) -> tuple[list[_Line], list[_Line]]:
        left, right = [], []
        n = len(self._left_full)
        i = 0
        while i < n:
            if self._left_full[i].kind != "equal" or self._right_full[i].kind != "equal":
                left.append(self._left_full[i])
                right.append(self._right_full[i])
                i += 1
                continue
            start = i
            while i < n and self._left_full[i].kind == "equal" and self._right_full[i].kind == "equal":
                i += 1
            skipped = i - start
            note = f"⋯ {skipped} unchanged line(s) ⋯"
            left.append(_Line(note, "sep"))
            right.append(_Line(note, "sep"))
        return left, right

    # -- 그리기 ------------------------------------------------------------------
    def _render(self) -> None:
        if self.changed_only_check.isChecked():
            left, right = self._collapsed()
        else:
            left, right = self._left_full, self._right_full
        self._paint(self.left_view, left)
        self._paint(self.right_view, right)
        self.summary_label.setText(f"+{self._added}  -{self._removed}")

    def _paint(self, view: QtWidgets.QPlainTextEdit, lines: list[_Line]) -> None:
        view.setPlainText("\n".join(ln.text for ln in lines))
        colors = self._line_colors()
        selections: list[QtWidgets.QTextEdit.ExtraSelection] = []
        doc = view.document()
        for i, ln in enumerate(lines):
            color = colors.get(ln.kind)
            if color is None:
                continue
            block = doc.findBlockByNumber(i)
            if not block.isValid():
                continue
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.cursor = QtGui.QTextCursor(block)
            sel.format.setBackground(color)
            sel.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            selections.append(sel)
        view.setExtraSelections(selections)

    @staticmethod
    def _line_colors() -> dict[str, QtGui.QColor]:
        add = QtGui.QColor(theme.color("metric.best"))   # 이미 앱 전역에서 "좋음/추가"=초록
        add.setAlpha(55)
        remove = QtGui.QColor(theme.color("metric.worst"))  # "나쁨/삭제"=빨강
        remove.setAlpha(55)
        blank = QtGui.QColor(theme.color("border.subtle"))
        blank.setAlpha(90)
        sep = QtGui.QColor(theme.color("bg.hover"))
        sep.setAlpha(200)
        return {"add": add, "remove": remove, "blank": blank, "sep": sep}
