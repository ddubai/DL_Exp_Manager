"""UI 항목 편집의 공통 규약.

앱 어디서나 같은 방식으로 동작하도록 한곳에 모은다.

    F2  이름 변경        Del  삭제        Ins / ＋  추가        우클릭  전체 메뉴
"""
from __future__ import annotations

from typing import Callable

from .qt import Qt, QtGui, QtWidgets

RENAME_KEY = "F2"
DELETE_KEY = "Del"
ADD_KEY = "Ins"


def prompt_text(
    parent: QtWidgets.QWidget | None,
    title: str,
    label: str,
    text: str = "",
) -> str | None:
    """한 줄 입력. 취소하거나 공백이면 None."""
    value, ok = QtWidgets.QInputDialog.getText(parent, title, label, text=text)
    if not ok:
        return None
    value = value.strip()
    return value or None


def confirm(parent: QtWidgets.QWidget | None, title: str, message: str) -> bool:
    button = QtWidgets.QMessageBox.StandardButton
    answer = QtWidgets.QMessageBox.question(
        parent, title, message, button.Yes | button.No, button.No
    )
    return answer == button.Yes


def confirm_delete(
    parent: QtWidgets.QWidget | None,
    what: str,
    usage_count: int = 0,
    usage_note: str = "{n} existing run(s) use this value.\nRecords are kept; it just leaves the option list.",
) -> bool:
    lines = [f"Delete '{what}'?"]
    if usage_count:
        lines.append("")
        lines.append(usage_note.format(n=usage_count))
    return confirm(parent, "Confirm Delete", "\n".join(lines))


def install_shortcuts(
    widget: QtWidgets.QWidget,
    on_rename: Callable[[], None] | None = None,
    on_delete: Callable[[], None] | None = None,
    on_add: Callable[[], None] | None = None,
) -> None:
    """위젯에 F2 / Del / Ins 단축키를 붙인다(위젯에 포커스가 있을 때만 동작)."""
    context = Qt.ShortcutContext.WidgetWithChildrenShortcut
    for key, handler in ((RENAME_KEY, on_rename), (DELETE_KEY, on_delete), (ADD_KEY, on_add)):
        if handler is None:
            continue
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), widget)
        shortcut.setContext(context)
        shortcut.activated.connect(handler)


def build_item_menu(
    parent: QtWidgets.QWidget,
    *,
    add_label: str | None = None,
    on_add: Callable[[], None] | None = None,
    rename_label: str = "Rename",
    on_rename: Callable[[], None] | None = None,
    delete_label: str = "Delete",
    on_delete: Callable[[], None] | None = None,
    extra_top: list[tuple[str, Callable[[], None]]] | None = None,
) -> QtWidgets.QMenu:
    """추가 / 이름변경 / 삭제 순서와 단축키 표기를 통일한 컨텍스트 메뉴."""
    menu = QtWidgets.QMenu(parent)
    for label, handler in extra_top or []:
        menu.addAction(label, handler)
    if extra_top and (on_add or on_rename or on_delete):
        menu.addSeparator()
    if on_add is not None:
        menu.addAction(f"+ {add_label or 'Add'}\t{ADD_KEY}", on_add)
    if on_rename is not None:
        menu.addAction(f"{rename_label}\t{RENAME_KEY}", on_rename)
    if on_delete is not None:
        menu.addSeparator()
        action = menu.addAction(f"{delete_label}\t{DELETE_KEY}", on_delete)
        action.setIcon(QtGui.QIcon())
    return menu


class AddOptionDialog(QtWidgets.QDialog):
    """콤보박스 항목 추가 - 값과 적용 범위를 함께 받는다."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        field_label: str,
        task: str | None,
        initial: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Add {field_label}")
        self.setMinimumWidth(340)

        self.value_edit = QtWidgets.QLineEdit(initial, self)
        self.value_edit.setPlaceholderText(f"New {field_label} value")

        self.task_radio = QtWidgets.QRadioButton(
            f"This Task only  ({task})" if task else "This Task only", self
        )
        self.global_radio = QtWidgets.QRadioButton("Shared across all Tasks (defaults)", self)
        self.task_radio.setChecked(bool(task))
        self.global_radio.setChecked(not task)
        self.task_radio.setEnabled(bool(task))

        scope_box = QtWidgets.QGroupBox("Scope", self)
        scope_layout = QtWidgets.QVBoxLayout(scope_box)
        scope_layout.addWidget(self.task_radio)
        scope_layout.addWidget(self.global_radio)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Add to the <b>{field_label}</b> option list.", self))
        layout.addWidget(self.value_edit)
        layout.addWidget(scope_box)
        layout.addWidget(
            QtWidgets.QLabel(
                "<span style='color:#9BA3B4'>Saved to this Task's config file.</span>", self
            )
        )
        layout.addWidget(buttons)
        self.value_edit.setFocus()

    def value(self) -> str:
        return self.value_edit.text().strip()

    def is_task_scope(self) -> bool:
        return self.task_radio.isChecked()

    @classmethod
    def run(
        cls, parent: QtWidgets.QWidget | None, field_label: str, task: str | None, initial: str = ""
    ) -> tuple[str, bool] | None:
        dialog = cls(parent, field_label, task, initial)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        value = dialog.value()
        if not value:
            return None
        return value, dialog.is_task_scope()
