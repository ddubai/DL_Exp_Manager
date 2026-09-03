"""Server status bar - one compact line: server name and busy/idle only.

Click a server chip for a details menu (running jobs, GPU indices, command,
copy/open-folder). Right-click for management (add/rename/edit GPUs/remove).
Full GPU accounting (indices, conflicts) still runs underneath; it is just
not painted as a big block anymore - it surfaces in the tooltip and menu.
"""
from __future__ import annotations

from typing import Any

from .. import editing, theme
from ..config_store import GpuDef, OptionsConfig, ServerDef
from ..db import Database
from ..qt import QtWidgets, Signal
from ..utils import elapsed_since, format_duration, open_in_file_manager, parse_gpu_count
from .common import copy_to_clipboard, toast


class ServerEditDialog(QtWidgets.QDialog):
    """Edit a server's name / host / GPU inventory."""

    def __init__(self, parent: QtWidgets.QWidget | None, server: ServerDef | None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Server" if server else "Add Server")
        self.setMinimumWidth(420)

        self.name_edit = QtWidgets.QLineEdit(server.name if server else "", self)
        self.name_edit.setPlaceholderText("e.g. Server 5")
        self.host_edit = QtWidgets.QLineEdit(server.host if server else "", self)
        self.host_edit.setPlaceholderText("e.g. 192.168.0.105 (optional)")

        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Index", "Type (V100/H100 ...)", "Memory (GB)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        for gpu in (server.gpus if server else []):
            self._append(gpu)

        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("+ Add GPU")
        add_btn.clicked.connect(lambda: self._append(None))
        del_btn = QtWidgets.QToolButton(self)
        del_btn.setText("- Remove Selected")
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
        form.addRow("Name:", self.name_edit)
        form.addRow("Host:", self.host_edit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("<b>GPU Inventory</b>", self))
        layout.addWidget(self.table, 1)
        layout.addLayout(controls)
        layout.addWidget(
            QtWidgets.QLabel(
                f"<span style='color:{theme.color('text.muted')}'>"
                "Saved to config/servers.yaml</span>",
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
    """One-line summary: a chip per server showing name + busy/idle."""

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
        self._chips: dict[str, QtWidgets.QToolButton] = {}
        self._state: dict[str, dict[str, Any]] = {}

        self.chips_layout = QtWidgets.QHBoxLayout()
        self.chips_layout.setContentsMargins(0, 0, 0, 0)
        self.chips_layout.setSpacing(4)

        add_btn = QtWidgets.QToolButton(self)
        add_btn.setText("+")
        add_btn.setToolTip("Add server")
        add_btn.clicked.connect(self.add_server)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addWidget(QtWidgets.QLabel("Servers:", self))
        layout.addLayout(self.chips_layout)
        layout.addWidget(add_btn)
        layout.addStretch(1)

        self.setStyleSheet(
            f"ServerStatusPanel {{ background: {theme.color('bg.surface')};"
            f" border-bottom: 1px solid {theme.color('border.subtle')}; }}"
        )
        self.refresh()

    # -- refresh --------------------------------------------------------------
    def refresh(self) -> None:
        running = self.db.running_by_server()
        servers = self.config.servers

        # A server name that only exists in DB records (not in config) still
        # gets a chip, just with no GPU inventory.
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

        current_names = {s.name for s in servers}
        for name in list(self._chips):
            if name not in current_names:
                self._chips.pop(name).deleteLater()
                self._state.pop(name, None)

        for position, server in enumerate(servers):
            jobs: list[dict[str, Any]] = []
            used = 0
            for job in running.get(server.name, []):
                job = dict(job)
                job["color"] = colors.get(int(job["id"]), theme.color("status.running"))
                jobs.append(job)
                used += self._parse_count(job.get("gpu_indices"))

            total = len(server.gpus)
            self._state[server.name] = {
                "server": server,
                "jobs": jobs,
                "used": used,
                "over_capacity": bool(total) and used > total,
            }

            chip = self._chips.get(server.name)
            if chip is None:
                chip = QtWidgets.QToolButton(self)
                chip.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
                self._chips[server.name] = chip
            self.chips_layout.insertWidget(position, chip)
            self._paint_chip(chip, server.name)

    def _paint_chip(self, chip: QtWidgets.QToolButton, name: str) -> None:
        from ..qt import Qt

        state = self._state[name]
        server: ServerDef = state["server"]
        busy = state["used"]
        total = len(server.gpus)

        if not total:
            text = f"○ {name}"
            color = theme.color("text.muted")
        elif busy:
            text = f"● {name} ({busy}/{total})"
            color = theme.color("status.running")
        else:
            text = f"○ {name} (0/{total})"
            color = theme.color("text.secondary")

        chip.setText(text)
        chip.setStyleSheet(
            f"QToolButton {{ color: {color}; border: none; background: transparent;"
            f" padding: 2px 4px; }}"
            f" QToolButton:hover {{ background: {theme.color('bg.hover')};"
            f" border-radius: {theme.METRICS['radius.small']}px; }}"
        )
        chip.setToolTip(self._tooltip_text(name))

        chip.setMenu(self._build_menu(name))

        chip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        try:
            chip.customContextMenuRequested.disconnect()
        except TypeError:
            pass
        chip.customContextMenuRequested.connect(
            lambda pos, n=name, c=chip: self._management_menu(n).exec(c.mapToGlobal(pos))
        )

    def _tooltip_text(self, name: str) -> str:
        state = self._state[name]
        server: ServerDef = state["server"]
        lines = [f"{name}  ({server.gpu_summary})" if server.gpus else name]
        if server.host:
            lines.append(server.host)
        if not state["jobs"]:
            lines.append("No running jobs")
        else:
            for job in state["jobs"]:
                gpu_text = self._gpu_label(job.get("gpu_indices"))
                elapsed = elapsed_since(job.get("started_at"))
                time_text = f" · {format_duration(elapsed)}" if elapsed is not None else ""
                lines.append(f"{gpu_text}  {job.get('model') or '-'}{time_text}")
        if state["over_capacity"]:
            lines.append("⚠ More GPUs claimed than this server has")
        return "\n".join(lines)

    def _build_menu(self, name: str) -> QtWidgets.QMenu:
        state = self._state[name]
        menu = QtWidgets.QMenu(self)
        jobs = state["jobs"]
        if not jobs:
            action = menu.addAction("No running jobs")
            action.setEnabled(False)
        else:
            for job in jobs:
                self._add_job_section(menu, job)
        menu.addSeparator()
        menu.addAction("Edit server / GPUs...", lambda: self.edit_server(name))
        menu.addAction("+ Add server...", self.add_server)
        return menu

    def _add_job_section(self, menu: QtWidgets.QMenu, job: dict[str, Any]) -> None:
        gpu_text = self._gpu_label(job.get("gpu_indices"))
        elapsed = elapsed_since(job.get("started_at"))
        time_text = f"  ·  {format_duration(elapsed)}" if elapsed is not None else ""
        header = menu.addAction(
            f"{gpu_text}   {job.get('model') or '-'}   "
            f"{job.get('task_name')}/{job.get('work_name')}{time_text}"
        )
        header.setEnabled(False)

        command = str(job.get("exec_command") or "")
        copy_action = menu.addAction("  Copy run command")
        copy_action.setEnabled(bool(command))
        copy_action.triggered.connect(lambda: copy_to_clipboard(command, self, "Run command"))

        result_path = str(job.get("result_path") or "")
        open_action = menu.addAction("  Open result folder")
        open_action.setEnabled(bool(result_path))
        open_action.triggered.connect(
            lambda: toast(self, *open_in_file_manager(result_path), "Open Folder")
        )
        menu.addSeparator()

    def _management_menu(self, name: str) -> QtWidgets.QMenu:
        return editing.build_item_menu(
            self,
            add_label="Add server",
            on_add=self.add_server,
            rename_label=f"Rename '{name}'",
            on_rename=lambda: self.rename_server(name),
            delete_label=f"Delete '{name}'",
            on_delete=lambda: self.remove_server(name),
            extra_top=[(f"Edit '{name}' GPUs...", lambda: self.edit_server(name))],
        )

    @staticmethod
    def _parse_count(text: Any) -> int:
        return parse_gpu_count(text)

    @staticmethod
    def _gpu_label(text: Any) -> str:
        count = parse_gpu_count(text)
        return f"{count} GPU(s)" if count else "GPU count not set"

    # -- editing --------------------------------------------------------------
    def add_server(self) -> None:
        dialog = ServerEditDialog(self, None)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        server = dialog.result_server()
        if server is None:
            toast(self, False, "Enter a server name.", "Add Server")
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
        new = editing.prompt_text(self, "Rename Server", "New name:", name)
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
