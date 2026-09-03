"""Log tail viewer (#11) - the last N lines of a run's log file.

Looks for a *.log (or *-ish .txt) file directly inside the run's result
folder via `scan_result_folder`, and lets the user browse for one manually
if none is found or a different file is wanted.
"""
from __future__ import annotations

import os

from .. import theme
from ..qt import QtCore, QtWidgets
from ..utils import open_in_file_manager, scan_result_folder, tail_file
from .common import copy_to_clipboard, monospace_font, toast

MAX_LINES = 500


class LogViewerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        result_path: str,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Log",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(760, 560)

        self._result_path = result_path
        self._log_path: str | None = None

        self.path_label = QtWidgets.QLabel(self)
        self.path_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(monospace_font())

        browse_btn = QtWidgets.QPushButton("Browse for Log File…", self)
        browse_btn.clicked.connect(self._browse)

        self.refresh_btn = QtWidgets.QPushButton("↻ Refresh", self)
        self.refresh_btn.clicked.connect(self.refresh)

        copy_btn = QtWidgets.QPushButton("Copy", self)
        copy_btn.clicked.connect(lambda: copy_to_clipboard(self.text.toPlainText(), self, "log"))

        self.open_folder_btn = QtWidgets.QPushButton("📁 Open Folder", self)
        self.open_folder_btn.clicked.connect(self._open_folder)

        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(browse_btn)
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(copy_btn)
        buttons.addWidget(self.open_folder_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(self.text, 1)
        layout.addLayout(buttons)

        self._auto_detect()
        self.refresh()

    # -- log discovery ----------------------------------------------------------
    def _auto_detect(self) -> None:
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

    def _open_folder(self) -> None:
        target = self._log_path or self._result_path
        ok, message = open_in_file_manager(target, reveal=bool(self._log_path))
        toast(self, ok, message, "Open Folder")

    # -- content -----------------------------------------------------------------
    def refresh(self) -> None:
        if not self._log_path:
            self.path_label.setText(
                f"No log file found in {self._result_path or '(no result folder set)'}."
                " Use “Browse for Log File…” to pick one."
            )
            self.text.setPlainText("")
            self.open_folder_btn.setEnabled(bool(self._result_path))
            return

        if not os.path.isfile(self._log_path):
            self.path_label.setText(f"{self._log_path}\n\n(file not found - not mounted, or removed)")
            self.text.setPlainText("")
            self.open_folder_btn.setEnabled(bool(self._result_path))
            return

        size = os.path.getsize(self._log_path)
        self.path_label.setText(
            f"{self._log_path}   ·   {size / 1024:.1f} KB   ·   last {MAX_LINES} lines"
        )
        self.text.setPlainText(tail_file(self._log_path, max_lines=MAX_LINES))
        scrollbar = self.text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self.open_folder_btn.setEnabled(True)
