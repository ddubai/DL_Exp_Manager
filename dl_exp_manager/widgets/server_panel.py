"""서버 상태 패널 - 서버당 GPU N장, 동시 학습 다수를 전제로 한다.

한 줄 요약(접힘)과 실행 중인 코드까지 보이는 상세(펼침)를 토글한다.
GPU 슬롯은 실행(run) 단위로 색을 달리해서, 어떤 학습이 어떤 GPU 를 잡고 있는지 보인다.
"""
from __future__ import annotations

from typing import Any

from .. import editing, theme
from ..config_store import GpuDef, OptionsConfig, ServerDef
from ..db import Database
from ..qt import Qt, QtCore, QtGui, QtWidgets, Signal
from ..utils import elapsed_since, format_duration, open_in_file_manager
from .common import copy_to_clipboard, monospace_font, toast

SLOT_W = 16
SLOT_H = 13
SLOT_GAP = 3


class GpuStrip(QtWidgets.QWidget):
    """GPU 점유 상태를 슬롯 막대로 그린다."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._gpus: list[GpuDef] = []
        self._assign: dict[int, dict[str, Any]] = {}
        self._conflicts: set[int] = set()
        self.setMouseTracking(True)
        self.setFixedHeight(SLOT_H + 4)

    def set_state(
        self,
        gpus: list[GpuDef],
        assign: dict[int, dict[str, Any]],
        conflicts: set[int] | None = None,
    ) -> None:
        self._gpus = list(gpus)
        self._assign = dict(assign)
        self._conflicts = set(conflicts or ())
        count = max(1, len(self._gpus))
        self.setFixedWidth(count * SLOT_W + (count - 1) * SLOT_GAP)
        self.update()

    def _slot_at(self, pos) -> GpuDef | None:
        if not self._gpus:
            return None
        index = int(pos.x() // (SLOT_W + SLOT_GAP))
        if 0 <= index < len(self._gpus):
            return self._gpus[index]
        return None

    def event(self, event) -> bool:
        if event.type() == QtCore.QEvent.Type.ToolTip:
            gpu = self._slot_at(event.pos())
            if gpu is None:
                QtWidgets.QToolTip.hideText()
            else:
                job = self._assign.get(gpu.index)
                lines = [f"GPU {gpu.index} · {gpu.type}"]
                if gpu.memory_gb:
                    lines[0] += f" · {gpu.memory_gb:g}GB"
                if job:
                    lines.append(f"{job['model']} · {job['task_name']}/{job['work_name']}")
                    elapsed = elapsed_since(job.get("started_at"))
                    if elapsed is not None:
                        lines.append(f"경과 {format_duration(elapsed)}")
                    if job.get("exec_command"):
                        lines.append("")
                        lines.append(str(job["exec_command"]))
                else:
                    lines.append("사용 안 함")
                if gpu.index in self._conflicts:
                    lines.append("")
                    lines.append("⚠ 여러 실행이 이 GPU 를 동시에 잡고 있습니다.")
                QtWidgets.QToolTip.showText(event.globalPos(), "\n".join(lines), self)
            return True
        return super().event(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        empty = QtGui.QColor(theme.color("border.subtle"))
        conflict = QtGui.QColor(theme.color("danger"))

        for position, gpu in enumerate(self._gpus):
            x = position * (SLOT_W + SLOT_GAP)
            rect = QtCore.QRectF(x, 2, SLOT_W, SLOT_H)
            job = self._assign.get(gpu.index)
            if job is not None:
                fill = QtGui.QColor(job["color"])
                painter.setBrush(fill)
                painter.setPen(QtGui.QPen(fill.darker(120), 1))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(empty, 1))
            painter.drawRoundedRect(rect, 3, 3)

            if gpu.index in self._conflicts:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QtGui.QPen(conflict, 1.6))
                painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 4, 4)
        painter.end()


class ServerRow(QtWidgets.QWidget):
    """서버 한 대 - 요약 줄 + 펼침 상세."""

    editRequested = Signal(str)
    removeRequested = Signal(str)
    renameRequested = Signal(str)
    addRequested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = ""
        self._jobs: list[dict[str, Any]] = []

        self.toggle = QtWidgets.QToolButton(self)
        self.toggle.setText("⌄")
        self.toggle.setCheckable(True)
        self.toggle.setFixedWidth(22)
        self.toggle.setToolTip("실행 중인 코드 펼치기")
        self.toggle.toggled.connect(self._on_toggled)

        self.name_label = QtWidgets.QLabel(self)
        self.name_label.setMinimumWidth(84)
        self.gpu_label = QtWidgets.QLabel(self)
        self.gpu_label.setMinimumWidth(96)
        self.gpu_label.setStyleSheet(f"color: {theme.color('text.secondary')};")
        self.strip = GpuStrip(self)
        self.usage_label = QtWidgets.QLabel(self)
        self.usage_label.setStyleSheet(f"color: {theme.color('text.secondary')};")

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)
        head.addWidget(self.toggle)
        head.addWidget(self.name_label)
        head.addWidget(self.gpu_label)
        head.addWidget(self.strip)
        head.addWidget(self.usage_label)
        head.addStretch(1)

        self.detail = QtWidgets.QWidget(self)
        self.detail_layout = QtWidgets.QVBoxLayout(self.detail)
        self.detail_layout.setContentsMargins(30, 2, 0, 6)
        self.detail_layout.setSpacing(3)
        self.detail.setVisible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.detail)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # -- 내용 --------------------------------------------------------------
    def set_content(
        self,
        server: ServerDef,
        jobs: list[dict[str, Any]],
        assign: dict[int, dict[str, Any]],
        conflicts: set[int],
        online: bool = True,
    ) -> None:
        self._name = server.name
        self._jobs = jobs

        muted = theme.color("text.disabled") if not online else theme.color("text.primary")
        self.name_label.setText(f"<b>{server.name}</b>")
        self.name_label.setStyleSheet(f"color: {muted};")
        host = f" · {server.host}" if server.host else ""
        self.gpu_label.setText(f"{server.gpu_summary}{host}")

        self.strip.set_state(server.gpus, assign, conflicts)

        busy = len(assign)
        total = len(server.gpus)
        if busy:
            self.usage_label.setText(
                f"<span style='color:{theme.color('status.running')}'>{busy}/{total} 사용 중</span>"
            )
        else:
            self.usage_label.setText(f"{'0' if total else '-'}/{total} idle" if total else "GPU 미등록")

        self.toggle.setEnabled(bool(jobs))
        self.toggle.setToolTip("실행 중인 코드 펼치기" if jobs else "실행 중인 학습이 없습니다")
        if not jobs and self.toggle.isChecked():
            self.toggle.setChecked(False)
        self._rebuild_detail()

    def _rebuild_detail(self) -> None:
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        for job in self._jobs:
            self.detail_layout.addWidget(self._job_widget(job))

    def _job_widget(self, job: dict[str, Any]) -> QtWidgets.QWidget:
        holder = QtWidgets.QWidget(self.detail)
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        gpus = str(job.get("gpu_indices") or "").strip()
        swatch = QtWidgets.QLabel("■", holder)
        swatch.setStyleSheet(f"color: {job['color']};")

        head = QtWidgets.QLabel(
            f"<b>GPU {gpus or '-'}</b>  {job.get('model') or '-'} · "
            f"{job.get('task_name')}/{job.get('work_name')}",
            holder,
        )
        elapsed = elapsed_since(job.get("started_at"))
        time_label = QtWidgets.QLabel(format_duration(elapsed) if elapsed is not None else "", holder)
        time_label.setStyleSheet(f"color: {theme.color('text.secondary')};")

        command = str(job.get("exec_command") or "")
        cmd_label = QtWidgets.QLabel(holder)
        cmd_label.setFont(monospace_font(-1))
        cmd_label.setStyleSheet(f"color: {theme.color('text.muted')};")
        cmd_label.setText(self._elide(command, 96) if command else "(실행 코드 미등록)")
        cmd_label.setToolTip(command or "")

        copy_btn = QtWidgets.QToolButton(holder)
        copy_btn.setText("복사")
        copy_btn.setEnabled(bool(command))
        copy_btn.clicked.connect(lambda: copy_to_clipboard(command, self, "실행 코드"))

        folder_btn = QtWidgets.QToolButton(holder)
        folder_btn.setText("📁")
        result_path = str(job.get("result_path") or "")
        folder_btn.setEnabled(bool(result_path))
        folder_btn.setToolTip(result_path or "결과 폴더 미등록")
        folder_btn.clicked.connect(
            lambda: toast(self, *open_in_file_manager(result_path), "폴더 열기")
        )

        row.addWidget(swatch)
        row.addWidget(head)
        row.addWidget(time_label)
        row.addWidget(cmd_label, 1)
        row.addWidget(copy_btn)
        row.addWidget(folder_btn)
        return holder

    @staticmethod
    def _elide(text: str, limit: int) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"

    def _on_toggled(self, checked: bool) -> None:
        self.detail.setVisible(checked)
        self.toggle.setText("⌃" if checked else "⌄")

    def _context_menu(self, pos) -> None:
        menu = editing.build_item_menu(
            self,
            add_label="서버 추가",
            on_add=self.addRequested.emit,
            rename_label=f"'{self._name}' 이름 변경",
            on_rename=lambda: self.renameRequested.emit(self._name),
            delete_label=f"'{self._name}' 삭제",
            on_delete=lambda: self.removeRequested.emit(self._name),
            extra_top=[(f"'{self._name}' GPU 구성 편집…", lambda: self.editRequested.emit(self._name))],
        )
        menu.exec(self.mapToGlobal(pos))


class ServerEditDialog(QtWidgets.QDialog):
    """서버 이름 / 호스트 / GPU 구성 편집."""

    def __init__(self, parent: QtWidgets.QWidget | None, server: ServerDef | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("서버 편집" if server else "서버 추가")
        self.setMinimumWidth(420)

        self.name_edit = QtWidgets.QLineEdit(server.name if server else "", self)
        self.name_edit.setPlaceholderText("예: Server 5")
        self.host_edit = QtWidgets.QLineEdit(server.host if server else "", self)
        self.host_edit.setPlaceholderText("예: 192.168.0.105 (선택)")

        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Index", "종류 (V100/H100 …)", "메모리(GB)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        for gpu in (server.gpus if server else []):
            self._append(gpu)

        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("＋ GPU 추가")
        add_btn.clicked.connect(lambda: self._append(None))
        del_btn = QtWidgets.QToolButton(self)
        del_btn.setText("− 선택 삭제")
        del_btn.clicked.connect(self._remove_selected)
        editing.install_shortcuts(self.table, on_delete=self._remove_selected, on_add=lambda: self._append(None))

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(add_btn)
        controls.addWidget(del_btn)
        controls.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QtWidgets.QFormLayout()
        form.addRow("이름:", self.name_edit)
        form.addRow("호스트:", self.host_edit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("<b>GPU 구성</b>", self))
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)
        layout.addWidget(
            QtWidgets.QLabel(
                f"<span style='color:{theme.color('text.muted')}'>"
                "config/options.yaml 의 servers 에 저장됩니다.</span>",
                self,
            )
        )
        layout.addWidget(buttons)

    def _append(self, gpu: GpuDef | None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        index = gpu.index if gpu else row
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(index)))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(gpu.type if gpu else "H100"))
        memory = "" if not gpu or gpu.memory_gb is None else f"{gpu.memory_gb:g}"
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(memory))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def result_server(self) -> ServerDef | None:
        name = self.name_edit.text().strip()
        if not name:
            return None
        gpus: list[GpuDef] = []
        for row in range(self.table.rowCount()):
            def cell(col: int) -> str:
                item = self.table.item(row, col)
                return item.text().strip() if item else ""

            raw_index = cell(0)
            try:
                index = int(raw_index)
            except ValueError:
                index = row
            memory_text = cell(2)
            try:
                memory = float(memory_text) if memory_text else None
            except ValueError:
                memory = None
            gpus.append(GpuDef(index=index, type=cell(1) or "GPU", memory_gb=memory))
        return ServerDef(name=name, host=self.host_edit.text().strip(), gpus=gpus)


class ServerStatusPanel(QtWidgets.QWidget):
    """모든 서버의 GPU 점유 현황."""

    configChanged = Signal()

    def __init__(
        self,
        db: Database,
        config: OptionsConfig,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db = db
        self.config = config
        self._rows: dict[str, ServerRow] = {}

        self.title = QtWidgets.QLabel("<b>서버 상태</b>", self)
        self.summary = QtWidgets.QLabel("", self)
        self.summary.setStyleSheet(f"color: {theme.color('text.secondary')};")

        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("＋ 서버")
        add_btn.setToolTip("서버 추가 (options.yaml 에 저장)")
        add_btn.clicked.connect(self.add_server)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.title)
        header.addWidget(self.summary, 1)
        header.addWidget(add_btn)

        self.rows_host = QtWidgets.QWidget(self)
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_host)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)
        layout.addLayout(header)
        layout.addWidget(self.rows_host)

        self.setStyleSheet(
            f"ServerStatusPanel {{ background: {theme.color('bg.surface')};"
            f" border-bottom: 1px solid {theme.color('border.subtle')}; }}"
        )
        self.refresh()

    # -- 갱신 --------------------------------------------------------------
    def refresh(self) -> None:
        running = self.db.running_by_server()
        servers = self.config.servers

        # 설정에 없는 서버 이름이 DB 에 있으면 GPU 없는 서버로 임시 표시한다.
        known = {s.name for s in servers}
        for name in running:
            if name not in known:
                servers = servers + [ServerDef(name=name)]

        colors: dict[int, str] = {}
        order = 0
        for name in sorted(running):
            for job in running[name]:
                colors[int(job["id"])] = theme.series_color(order)
                order += 1

        busy_total = free_total = 0
        for name, row in list(self._rows.items()):
            if name not in {s.name for s in servers}:
                row.setParent(None)
                row.deleteLater()
                self._rows.pop(name)

        for position, server in enumerate(servers):
            jobs: list[dict[str, Any]] = []
            assign: dict[int, dict[str, Any]] = {}
            conflicts: set[int] = set()

            for job in running.get(server.name, []):
                job = dict(job)
                job["color"] = colors.get(int(job["id"]), theme.color("status.running"))
                jobs.append(job)
                for index in self._parse_indices(job.get("gpu_indices")):
                    if index in assign:
                        conflicts.add(index)
                    assign[index] = job

            busy_total += len(assign)
            free_total += max(0, len(server.gpus) - len(assign))

            row = self._rows.get(server.name)
            if row is None:
                row = ServerRow(self.rows_host)
                row.addRequested.connect(self.add_server)
                row.editRequested.connect(self.edit_server)
                row.removeRequested.connect(self.remove_server)
                row.renameRequested.connect(self.rename_server)
                self._rows[server.name] = row
            self.rows_layout.insertWidget(position, row)
            row.set_content(server, jobs, assign, conflicts)

        total = busy_total + free_total
        running_count = sum(len(v) for v in running.values())
        parts = [f"{busy_total}/{total} GPU 사용 중"] if total else []
        if running_count:
            parts.append(f"학습 {running_count} 건 진행")
        self.summary.setText("  ·  ".join(parts) if parts else "진행 중인 학습 없음")

    @staticmethod
    def _parse_indices(text: Any) -> list[int]:
        out: list[int] = []
        for part in str(text or "").split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return out

    # -- 편집 --------------------------------------------------------------
    def add_server(self) -> None:
        dialog = ServerEditDialog(self, None)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        server = dialog.result_server()
        if server is None:
            toast(self, False, "서버 이름을 입력하세요.", "서버 추가")
            return
        self.config.upsert_server(server.name, server.host, server.gpus)
        self.refresh()
        self.configChanged.emit()

    def edit_server(self, name: str) -> None:
        server = self.config.server(name)
        dialog = ServerEditDialog(self, server)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        updated = dialog.result_server()
        if updated is None:
            return
        if updated.name != name:
            self.config.rename_server(name, updated.name)
        self.config.upsert_server(updated.name, updated.host, updated.gpus)
        self.refresh()
        self.configChanged.emit()

    def rename_server(self, name: str) -> None:
        new = editing.prompt_text(self, "서버 이름 변경", "새 이름:", name)
        if not new or new == name:
            return
        self.config.rename_server(name, new)
        self.refresh()
        self.configChanged.emit()

    def remove_server(self, name: str) -> None:
        used = self.db.count_runs_using("server", name)
        if not editing.confirm_delete(self, name, used):
            return
        self.config.remove_server(name)
        self.refresh()
        self.configChanged.emit()
