"""메인 윈도우 - 좌측 네비게이션 + Train/Inference 탭 + 서버 상태 인디케이터."""
from __future__ import annotations

import os

from . import APP_NAME, ORG_NAME, __version__
from . import constants as C
from .db import Database
from .qt import QT_BINDING, Qt, QtCore, QtGui, QtWidgets
from .utils import (
    format_duration,
    elapsed_since,
    open_in_file_manager,
    platform_label,
)
from .widgets.common import toast
from .widgets.nav_panel import NavigationPanel
from .widgets.run_panel import InferencePanel, TrainPanel

STATUS_REFRESH_MS = 15_000


class ServerStatusBar(QtWidgets.QWidget):
    """Server 1~4 의 현재 학습 진행 상황을 한 줄로 보여 주는 인디케이터."""

    def __init__(self, db: Database, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.db = db

        self.label = QtWidgets.QLabel("", self)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(QtWidgets.QLabel("<b>서버 상태</b>", self))
        layout.addWidget(self.label, 1)

        self.setStyleSheet("QWidget { background: palette(alternate-base); }")
        self.refresh()

    def refresh(self) -> None:
        running = self.db.running_by_server()
        chunks: list[str] = []
        for name in self.db.server_names():
            jobs = running.get(name, [])
            if jobs:
                job = jobs[0]
                elapsed = elapsed_since(job.get("started_at"))
                extra = f" · {format_duration(elapsed)}" if elapsed is not None else ""
                more = f" (+{len(jobs) - 1})" if len(jobs) > 1 else ""
                chunks.append(
                    f"<span style='color:{C.STATUS_COLORS[C.STATUS_RUNNING]}'>🔵 {name}: "
                    f"{job['model']} @ {job['task_name']}/{job['work_name']}{extra}{more}</span>"
                )
            else:
                chunks.append(f"<span style='color:#888'>⚪ {name}: idle</span>")

        orphan = [s for s in running if s not in set(self.db.server_names())]
        for name in orphan:
            chunks.append(f"<span style='color:#b06000'>🔵 {name}: {len(running[name])} running</span>")

        total = sum(len(v) for v in running.values())
        summary = f"&nbsp;&nbsp;|&nbsp;&nbsp;<b>학습 중 {total} 건</b>" if total else ""
        self.label.setText("&nbsp;&nbsp;·&nbsp;&nbsp;".join(chunks) + summary)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, db_path: str | None = None) -> None:
        super().__init__()
        self.db = Database(db_path)
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)

        self.setWindowTitle(f"{APP_NAME}  —  {os.path.basename(self.db.path)}")
        self.resize(1500, 940)

        # -- 중앙 위젯 ------------------------------------------------------
        self.nav = NavigationPanel(self.db, self)
        self.train_panel = TrainPanel(self.db, self)
        self.inference_panel = InferencePanel(self.db, self)

        self.tabs = QtWidgets.QTabWidget(self)
        self.tabs.addTab(self.train_panel, "🏋  Train")
        self.tabs.addTab(self.inference_panel, "🔎  Inference")
        self.tabs.setDocumentMode(True)

        self.scope_label = QtWidgets.QLabel("", self)
        self.scope_label.setStyleSheet("font-size: 14px; font-weight: 600; padding: 2px 6px;")

        right = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)
        right_layout.addWidget(self.scope_label)
        right_layout.addWidget(self.tabs, 1)

        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.addWidget(self.nav)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([300, 1200])

        self.server_bar = ServerStatusBar(self.db, self)

        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.server_bar)
        central_layout.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

        # -- 시그널 ---------------------------------------------------------
        self.nav.selectionChanged.connect(self._on_scope_changed)
        self.train_panel.runsChanged.connect(self._on_runs_changed)
        self.inference_panel.runsChanged.connect(self._on_runs_changed)

        self._build_menu()
        self.statusBar().showMessage(
            f"{platform_label()} · {QT_BINDING} · DB: {self.db.path}"
        )

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(STATUS_REFRESH_MS)
        self._timer.timeout.connect(self.server_bar.refresh)
        self._timer.start()

        self._restore_state()
        self.nav.refresh()

    # ==================================================================
    # 메뉴
    # ==================================================================
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("파일(&F)")
        file_menu.addAction(self._action("DB 열기…", self.open_database, "Ctrl+O"))
        file_menu.addAction(self._action("DB 폴더 열기", self.open_db_folder))
        file_menu.addSeparator()
        file_menu.addAction(self._action("현재 탭 CSV 내보내기", self.export_current, "Ctrl+E"))
        file_menu.addSeparator()
        file_menu.addAction(self._action("종료", self.close, "Ctrl+Q"))

        edit_menu = self.menuBar().addMenu("편집(&E)")
        edit_menu.addAction(self._action("DL Task 추가", self.nav.add_task, "Ctrl+Shift+T"))
        edit_menu.addAction(self._action("Work ID 추가", self.nav.add_work, "Ctrl+Shift+W"))
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("선택 행 복사", self.copy_current_selection, "Ctrl+C"))
        edit_menu.addAction(self._action("표 전체 복사", self.copy_current_all, "Ctrl+Shift+C"))
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("선택 실행 삭제", self.delete_current, "Del"))

        view_menu = self.menuBar().addMenu("보기(&V)")
        view_menu.addAction(self._action("새로고침", self.refresh_all, "F5"))
        view_menu.addAction(self._action("Train 탭", lambda: self.tabs.setCurrentIndex(0), "Ctrl+1"))
        view_menu.addAction(self._action("Inference 탭", lambda: self.tabs.setCurrentIndex(1), "Ctrl+2"))

        tools_menu = self.menuBar().addMenu("도구(&T)")
        tools_menu.addAction(self._action("서버 추가…", self.add_server))
        tools_menu.addAction(self._action("샘플 데이터 넣기", self.insert_sample_data))

        help_menu = self.menuBar().addMenu("도움말(&H)")
        help_menu.addAction(self._action("정보", self.show_about))

    def _action(self, text: str, slot, shortcut: str = "") -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        action.triggered.connect(lambda *_: slot())
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        return action

    # ==================================================================
    # 슬롯
    # ==================================================================
    def _current_panel(self):
        return self.tabs.currentWidget()

    def _on_scope_changed(self, task_id: int, work_id: int) -> None:
        self.train_panel.set_scope(task_id, work_id)
        self.inference_panel.set_scope(task_id, work_id)

        if work_id > 0:
            work = self.db.get_work(work_id)
            if work:
                desc = f"  —  {work['description']}" if work.get("description") else ""
                self.scope_label.setText(f"{work['task_name']}  ▸  {work['name']}{desc}")
                return
        if task_id > 0:
            task = next((t for t in self.db.list_tasks() if t["id"] == task_id), None)
            if task:
                self.scope_label.setText(f"{task['name']}  (Task 전체 보기)")
                return
        self.scope_label.setText("좌측에서 Task / Work 를 선택하세요.")

    def _on_runs_changed(self) -> None:
        self.server_bar.refresh()
        work_id = self.nav.current_work_id()
        task_id = self.nav.current_task_id()
        self.nav.refresh(select_work_id=work_id, select_task_id=None if work_id else task_id)

    def refresh_all(self) -> None:
        self.train_panel.reload()
        self.inference_panel.reload()
        self.server_bar.refresh()
        self.statusBar().showMessage("새로고침했습니다.", 2000)

    def export_current(self) -> None:
        self._current_panel().export_csv()

    def copy_current_selection(self) -> None:
        self._current_panel().copy_table(selected_only=True)

    def copy_current_all(self) -> None:
        self._current_panel().copy_table(selected_only=False)

    def delete_current(self) -> None:
        self._current_panel().delete_selected()

    def open_db_folder(self) -> None:
        ok, message = open_in_file_manager(self.db.path, reveal=True)
        toast(self, ok, message, "DB 폴더 열기")

    def open_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "experiments.db 선택 또는 새로 만들기",
            os.path.dirname(self.db.path),
            "SQLite DB (*.db *.sqlite);;모든 파일 (*)",
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        if not path:
            return
        try:
            new_db = Database(path)
        except Exception as exc:  # noqa: BLE001 - 사용자에게 그대로 보여 준다
            toast(self, False, f"DB 를 열지 못했습니다:\n{exc}", "DB 열기")
            return

        old = self.db
        self.db = new_db
        for widget in (self.nav, self.train_panel, self.inference_panel, self.server_bar):
            widget.db = new_db
        old.close()

        self.setWindowTitle(f"{APP_NAME}  —  {os.path.basename(new_db.path)}")
        self.statusBar().showMessage(f"DB: {new_db.path}")
        self.nav.refresh()
        self.server_bar.refresh()

    def add_server(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "서버 추가", "서버 이름 (예: Server 5):")
        if not ok or not name.strip():
            return
        host, _ = QtWidgets.QInputDialog.getText(self, "서버 추가", "호스트/IP (선택):")
        gpu, _ = QtWidgets.QInputDialog.getText(self, "서버 추가", "GPU 정보 (선택):")
        self.db.add_server(name.strip(), host.strip(), gpu.strip())
        self.train_panel.reload()
        self.inference_panel.reload()
        self.server_bar.refresh()
        toast(self, True, f"서버 '{name.strip()}' 를 추가했습니다.")

    def insert_sample_data(self) -> None:
        """빈 상태에서 UI 를 둘러볼 수 있도록 예시 실행 몇 건을 넣는다."""
        from .sample_data import populate

        answer = QtWidgets.QMessageBox.question(
            self,
            "샘플 데이터",
            "SR/SSL2SL 등에 예시 Train·Inference 기록을 추가합니다. 계속할까요?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        added = populate(self.db)
        self.nav.refresh()
        self.refresh_all()
        toast(self, True, f"샘플 {added} 건을 추가했습니다.")

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            f"{APP_NAME} 정보",
            f"<h3>{APP_NAME} v{__version__}</h3>"
            "<p>4대 학습 서버의 실험을 로컬에서 아카이빙·관리하는 데스크톱 앱.</p>"
            "<p><b>계층</b>: DL Task ▸ Work ID ▸ Train / Inference</p>"
            f"<p><b>DB</b>: {self.db.path}<br>"
            f"<b>Qt</b>: {QT_BINDING}<br>"
            f"<b>환경</b>: {platform_label()}</p>",
        )

    # ==================================================================
    # 창 상태 저장/복원
    # ==================================================================
    def _restore_state(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter = self.settings.value("window/splitter")
        if splitter is not None:
            self.splitter.restoreState(splitter)
        tab = self.settings.value("window/tab")
        if tab is not None:
            try:
                self.tabs.setCurrentIndex(int(tab))
            except (TypeError, ValueError):
                pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/splitter", self.splitter.saveState())
        self.settings.setValue("window/tab", self.tabs.currentIndex())
        self.db.close()
        super().closeEvent(event)
