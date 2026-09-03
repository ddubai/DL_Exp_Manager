"""메인 윈도우 - 좌측 네비게이션 + Train/Inference 탭 + 서버 상태 인디케이터."""
from __future__ import annotations

import os

from . import APP_NAME, ORG_NAME, __version__
from . import theme
from .config_store import OptionsConfig, backend_name, preserves_comments
from .db import Database
from .qt import QT_BINDING, Qt, QtCore, QtGui, QtWidgets
from .utils import open_in_file_manager, platform_label
from .widgets.common import toast
from .widgets.nav_panel import NavigationPanel
from .widgets.run_panel import InferencePanel, TrainPanel
from .widgets.server_panel import ServerStatusPanel

STATUS_REFRESH_MS = 15_000


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, db_path: str | None = None, config_path: str | None = None) -> None:
        super().__init__()
        self.db = Database(db_path)
        self.config = OptionsConfig(config_path)
        self.settings = QtCore.QSettings(ORG_NAME, APP_NAME)

        self.setWindowTitle(f"{APP_NAME}  —  {os.path.basename(self.db.path)}")
        self.resize(1500, 940)

        # -- 중앙 위젯 ------------------------------------------------------
        self.nav = NavigationPanel(self.db, self.config, self)
        self.train_panel = TrainPanel(self.db, self.config, self)
        self.inference_panel = InferencePanel(self.db, self.config, self)

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

        self.server_bar = ServerStatusPanel(self.db, self.config, self)

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
        self.train_panel.configChanged.connect(self._on_config_changed)
        self.inference_panel.configChanged.connect(self._on_config_changed)
        self.server_bar.configChanged.connect(self._on_config_changed)

        self._build_menu()
        self._show_startup_status()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(STATUS_REFRESH_MS)
        self._timer.timeout.connect(self.server_bar.refresh)
        self._timer.start()

        # 외부 편집기로 options.yaml 을 고쳐도 즉시 반영한다.
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._watch_config()
        self._watcher.fileChanged.connect(self._on_config_file_changed)
        self._reload_pending = QtCore.QTimer(self)
        self._reload_pending.setSingleShot(True)
        self._reload_pending.setInterval(400)
        self._reload_pending.timeout.connect(self._reload_config_from_disk)

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
        tools_menu.addAction(self._action("서버 / GPU 추가…", self.server_bar.add_server))
        tools_menu.addSeparator()
        tools_menu.addAction(self._action("설정 폴더 열기", self.open_config_file))
        tools_menu.addAction(
            self._action("현재 Task 설정 파일 열기", self.open_task_config, "Ctrl+Shift+O")
        )
        tools_menu.addAction(self._action("설정 다시 읽기", self.reload_config, "Ctrl+R"))
        tools_menu.addAction(self._action("설정 상태 보기", self.show_config_status))
        tools_menu.addSeparator()
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
            f"<b>설정</b>: {self.config.config_dir}<br>"
            f"<b>Qt</b>: {QT_BINDING} · 테마: {theme.current_theme()}<br>"
            f"<b>환경</b>: {platform_label()}</p>",
        )


    # ==================================================================
    # 설정 파일
    # ==================================================================
    def _watch_config(self) -> None:
        """설정이 여러 파일로 나뉘어 있으므로 전부 감시한다."""
        existing = self._watcher.files()
        if existing:
            self._watcher.removePaths(existing)
        paths = self.config.watch_paths()
        if paths:
            self._watcher.addPaths(paths)

    def _on_config_file_changed(self, _path: str) -> None:
        # 에디터가 원자적 교체로 저장하면 감시가 끊기므로 다시 걸어 준다.
        self._reload_pending.start()

    def _reload_config_from_disk(self) -> None:
        self.config.load()
        self._watch_config()
        self._apply_config_everywhere()
        if self.config.errors:
            self.statusBar().showMessage("설정: " + self.config.errors[0], 8000)
        else:
            count = len(self.config.watch_paths())
            self.statusBar().showMessage(f"설정 파일 {count} 개를 다시 읽었습니다.", 3000)

    def _on_config_changed(self) -> None:
        """UI 에서 설정을 바꾼 직후 - 파일은 이미 저장돼 있다."""
        self._watch_config()
        self._apply_config_everywhere()

    def _apply_config_everywhere(self) -> None:
        self.train_panel.reload_columns()
        self.inference_panel.reload_columns()
        self.server_bar.refresh()

    def _show_startup_status(self) -> None:
        if self.config.errors:
            self.statusBar().showMessage("⚠ 설정: " + self.config.errors[0], 12000)
            return
        comment_note = "" if preserves_comments() else " (주석 미보존: ruamel.yaml 미설치)"
        self.statusBar().showMessage(
            f"{platform_label()} · {QT_BINDING} · YAML: {backend_name()}{comment_note}"
            f" · 설정: {self.config.config_dir}"
            f" · DB: {self.db.path}"
        )

    def open_config_file(self) -> None:
        ok, message = open_in_file_manager(self.config.config_dir)
        toast(self, ok, message, "설정 폴더")

    def open_task_config(self) -> None:
        """지금 보고 있는 Task 의 설정 파일을 탐색기에서 선택한 채로 연다."""
        task = self._current_panel().current_task_name()
        if not task:
            toast(self, False, "좌측에서 Task 를 먼저 선택하세요.", "Task 설정")
            return
        ok, message = open_in_file_manager(self.config.task_path(task), reveal=True)
        toast(self, ok, message, f"{task} 설정 파일")

    def reload_config(self) -> None:
        self._reload_config_from_disk()

    def show_config_status(self) -> None:
        task = self.train_panel.current_task_name()
        files = "<br>".join(
            f"· {role}: <code>{os.path.relpath(path, self.config.config_dir)}</code>"
            for role, path in self.config.files_summary()
        )
        lines = [
            f"<b>설정 폴더</b><br>{self.config.config_dir}<br>{files}",
            f"<b>YAML 백엔드</b>: {backend_name()}"
            + ("  (주석 보존)" if preserves_comments() else "  (주석 미보존)"),
        ]
        if task:
            lines.append(
                f"<b>{task}</b><br>"
                f"· 지표: {', '.join(self.config.metric_keys(task)) or '(없음)'}<br>"
                f"· 사용자 필드: {', '.join(self.config.custom_fields(task)) or '(없음)'}<br>"
                f"· Train 컬럼: {', '.join(self.config.columns_for(task, 'train')) or '(기본값)'}"
            )
        if self.config.errors:
            lines.append("<b style='color:#FF6B6B'>문제</b><br>" + "<br>".join(self.config.errors))
        QtWidgets.QMessageBox.information(self, "설정 상태", "<br><br>".join(lines))

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
