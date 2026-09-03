"""대표 이미지 뷰어 - 결과 폴더에서 이미지 하나만 골라 크게 보여준다.

SR/Denoising 은 숫자보다 눈으로 보는 게 진짜 판단 기준인데, 지금까지는 결과
폴더를 직접 열어야 했다. 여러 장을 나란히 비교하는 화면 대신, 대표 이미지
한 장만 빠르게 확인하는 용도로 최소한으로 만든다.
"""
from __future__ import annotations

import os

from .. import theme
from ..qt import Qt, QtCore, QtGui, QtWidgets
from ..utils import find_representative_image, open_in_file_manager
from .common import toast


class ImageViewerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        result_path: str,
        parent: QtWidgets.QWidget | None = None,
        title: str = "Representative Image",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 600)

        self._result_path = result_path
        self._image_path: str | None = None
        self._pixmap: QtGui.QPixmap | None = None

        self.path_label = QtWidgets.QLabel(self)
        self.path_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet(
            f"background-color: {theme.color('bg.surface')}; color: {theme.color('text.muted')};"
        )

        browse_btn = QtWidgets.QPushButton("Browse for Image…", self)
        browse_btn.clicked.connect(self._browse)

        self.open_folder_btn = QtWidgets.QPushButton("📁 Open Folder", self)
        self.open_folder_btn.clicked.connect(self._open_folder)

        close_btn = QtWidgets.QPushButton("Close", self)
        close_btn.clicked.connect(self.accept)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(browse_btn)
        buttons.addWidget(self.open_folder_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.path_label)
        layout.addWidget(self.image_label, 1)
        layout.addLayout(buttons)

        self._auto_detect()
        self._render()

    # -- discovery ----------------------------------------------------------
    def _auto_detect(self) -> None:
        self._image_path = find_representative_image(self._result_path) if self._result_path else None

    def _browse(self) -> None:
        start = self._image_path or self._result_path or QtCore.QDir.homePath()
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", start,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
        )
        if chosen:
            self._image_path = chosen
            self._render()

    def _open_folder(self) -> None:
        target = self._image_path or self._result_path
        ok, message = open_in_file_manager(target, reveal=bool(self._image_path))
        toast(self, ok, message, "Open Folder")

    # -- rendering ------------------------------------------------------------
    def _render(self) -> None:
        if not self._image_path or not os.path.isfile(self._image_path):
            self._pixmap = None
            self.image_label.setPixmap(QtGui.QPixmap())
            self.image_label.setText(
                "No image found in this result folder.\nUse “Browse for Image…” to pick one."
            )
            self.path_label.setText(self._result_path or "(no result folder set)")
            self.open_folder_btn.setEnabled(bool(self._result_path))
            return

        pixmap = QtGui.QPixmap(self._image_path)
        if pixmap.isNull():
            self._pixmap = None
            self.image_label.setText(f"Could not load image:\n{self._image_path}")
            self.path_label.setText(self._image_path)
            self.open_folder_btn.setEnabled(True)
            return

        self._pixmap = pixmap
        self.path_label.setText(
            f"{self._image_path}   ·   {pixmap.width()}×{pixmap.height()}"
        )
        self.open_folder_btn.setEnabled(True)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()
