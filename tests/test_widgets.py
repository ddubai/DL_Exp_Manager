"""위젯 동작 테스트 (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="Qt 바인딩 필요")

from dl_exp_manager.config_store import OptionsConfig


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from dl_exp_manager import theme

    theme.apply_theme(app, "dark")
    return app


@pytest.fixture
def config():
    return OptionsConfig(os.path.join(tempfile.mkdtemp(), "options.yaml"))


# --- 테마 --------------------------------------------------------------------
def test_qss_renders_for_both_themes(qapp):
    from dl_exp_manager import theme

    assert len(theme.render_qss("dark")) > 1000
    assert len(theme.render_qss("light")) > 1000


def test_unknown_token_in_template_is_an_error(qapp, monkeypatch):
    """토큰 오타를 조용히 넘기지 않는지."""
    from dl_exp_manager import theme

    monkeypatch.setattr(theme, "_TEMPLATE", theme._TEMPLATE)
    original = open(theme._TEMPLATE, encoding="utf-8").read()
    path = os.path.join(tempfile.mkdtemp(), "broken.qss.tpl")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(original + "\nQWidget { color: {{nope.not.a.token}}; }")
    monkeypatch.setattr(theme, "_TEMPLATE", path)
    with pytest.raises(KeyError):
        theme.render_qss("dark")


def test_status_colors_come_from_theme(qapp):
    from dl_exp_manager import theme

    assert theme.status_color("running") == theme.color("status.running")
    assert theme.status_color("failed") != theme.status_color("done")


# --- ManagedCombo ------------------------------------------------------------
def test_combo_lists_task_options_with_sentinel_last(qapp, config):
    from dl_exp_manager.widgets.common import SENTINEL_TEXT, ManagedCombo

    combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: "SR")
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items[-1] == SENTINEL_TEXT
    assert combo._is_sentinel(combo.count() - 1)
    assert "HAT" in items


def test_combo_switches_list_when_task_changes(qapp, config):
    from dl_exp_manager.widgets.common import ManagedCombo

    task = {"name": "SR"}
    combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: task["name"])
    assert "HAT" in [combo.itemText(i) for i in range(combo.count())]
    task["name"] = "Classification"
    combo.reload()
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "ResNet-50" in items and "HAT" not in items


def test_sentinel_is_never_taken_as_a_value(qapp, config):
    from dl_exp_manager.widgets.common import ManagedCombo

    combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: "SR")
    combo.setCurrentIndex(combo.count() - 1)
    assert combo.current_text() == ""


def test_activating_sentinel_restores_previous_value(qapp, config):
    from dl_exp_manager.widgets.common import ManagedCombo

    combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: "SR")
    combo.set_text("SwinIR")
    combo._on_activated(combo.count() - 1)
    assert combo.current_text() == "SwinIR"


def test_merge_items_keeps_sentinel_last(qapp, config):
    from dl_exp_manager.widgets.common import ManagedCombo

    combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: "SR")
    combo.merge_items(["LegacyNet"])
    assert combo._is_sentinel(combo.count() - 1)
    assert "LegacyNet" in [combo.itemText(i) for i in range(combo.count())]


def test_combo_scope_detection(qapp, config):
    from dl_exp_manager.widgets.common import ManagedCombo

    model_combo = ManagedCombo("model", "Model", config=config, task_getter=lambda: "SR")
    optimizer_combo = ManagedCombo("optimizer", "Optimizer", config=config, task_getter=lambda: "SR")
    assert model_combo._is_task_scoped("SR") is True       # SR 이 직접 정의
    assert optimizer_combo._is_task_scoped("SR") is False  # defaults 상속


# --- GpuSelector -------------------------------------------------------------
def test_gpu_selector_reflects_server_inventory(qapp, config):
    from dl_exp_manager.widgets.common import GpuSelector

    selector = GpuSelector()
    selector.set_server(config.server("Server 1"))
    assert len(selector._boxes) == 4
    selector.set_value("0,2")
    assert selector.value() == "0,2"
    assert selector.hint.text() == "CUDA_VISIBLE_DEVICES=0,2"


def test_gpu_selector_without_server(qapp):
    from dl_exp_manager.widgets.common import GpuSelector

    selector = GpuSelector()
    selector.set_server(None)
    assert selector.value() == ""
    assert selector._empty.isVisibleTo(selector)


# --- 서버 패널 ---------------------------------------------------------------
def _panel_with_runs(config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.server_panel import ServerStatusPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    db.insert_run("train", {"work_id": work_id, "server": "Server 3", "model": "MambaIR",
                            "gpu_indices": "0,1", "status": "running"})
    db.insert_run("train", {"work_id": work_id, "server": "Server 3", "model": "EDSR",
                            "gpu_indices": "2,3", "status": "running"})
    return db, ServerStatusPanel(db, config)


def test_server_panel_shows_multiple_jobs_per_server(qapp, config):
    db, panel = _panel_with_runs(config)
    state = panel._state["Server 3"]
    assert len(state["jobs"]) == 2
    assert state["jobs"][0]["color"] != state["jobs"][1]["color"]
    assert len(state["assign"]) == 4  # GPU 0,1,2,3 all claimed
    chip = panel._chips["Server 3"]
    assert "Server 3" in chip.text()
    assert "4/4" in chip.text()
    db.close()


def test_server_panel_flags_gpu_conflict(qapp, config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.server_panel import ServerStatusPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    for model in ("A", "B"):
        db.insert_run("train", {"work_id": work_id, "server": "Server 1", "model": model,
                                "gpu_indices": "0", "status": "running"})
    panel = ServerStatusPanel(db, config)
    assert panel._state["Server 1"]["conflicts"] == {0}
    assert "⚠" in panel._tooltip_text("Server 1")
    db.close()


def test_server_panel_shows_unknown_server_from_db(qapp, config):
    """A server name that only exists in DB records still gets a chip."""
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.server_panel import ServerStatusPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    db.insert_run("train", {"work_id": work_id, "server": "Ghost", "model": "M",
                            "status": "running"})
    panel = ServerStatusPanel(db, config)
    assert "Ghost" in panel._chips
    db.close()


def test_gpu_index_parsing():
    from dl_exp_manager.widgets.server_panel import ServerStatusPanel

    assert ServerStatusPanel._parse_indices("0,1,3") == [0, 1, 3]
    assert ServerStatusPanel._parse_indices("") == []
    assert ServerStatusPanel._parse_indices("a,1") == [1]
    assert ServerStatusPanel._parse_indices(None) == []


# --- #7 Favorites / tags / failure reason (run_panel wiring) -----------------
def _panel_with_one_run(config):
    """Parented under a real QMainWindow so `toast()` finds a status bar.

    Without one, `toast(ok=True)` falls back to a blocking QMessageBox.information()
    that never returns under the offscreen platform - any test calling a panel method
    that succeeds and reports via toast() needs this, not a bare `TrainPanel(db, config)`.
    """
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import TrainPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    run_id = db.insert_run("train", {"work_id": work_id, "model": "Restormer"})
    window = QtWidgets.QMainWindow()
    panel = TrainPanel(db, config, parent=window)
    window.setCentralWidget(panel)
    panel._test_window = window  # keep it alive as long as panel is referenced
    panel.set_scope(task_id, work_id)
    return db, panel, run_id


def test_toggle_favorite_updates_db_and_reselects_row(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.view.selectRow(0)
    assert panel._current_row()["favorite"] == 0

    panel.toggle_favorite()
    assert db.get_run("train", run_id)["favorite"] == 1
    assert panel._current_row() is not None and panel._current_row()["id"] == run_id

    panel.toggle_favorite()
    assert db.get_run("train", run_id)["favorite"] == 0
    db.close()


def test_favorites_only_toolbar_filter(qapp, config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import TrainPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    db.insert_run("train", {"work_id": work_id, "model": "A", "favorite": True})
    db.insert_run("train", {"work_id": work_id, "model": "B", "favorite": False})
    panel = TrainPanel(db, config)
    panel.set_scope(task_id, work_id)

    assert panel.proxy.rowCount() == 2
    panel.favorites_btn.setChecked(True)
    assert panel.proxy.rowCount() == 1
    panel.favorites_btn.setChecked(False)
    assert panel.proxy.rowCount() == 2
    db.close()


def test_editing_a_run_preserves_favorite_state(qapp, config):
    """Saving the edit form must not silently clear favorite - it has no field for it."""
    db, panel, run_id = _panel_with_one_run(config)
    db.toggle_favorite("train", run_id)
    panel.reload()
    panel.view.selectRow(0)

    assert panel.load_selected_into_form() is True
    assert panel._editing_favorite is True
    panel.model_combo.set_text("SwinIR")
    panel.save_form()

    assert db.get_run("train", run_id)["favorite"] == 1
    assert db.get_run("train", run_id)["model"] == "SwinIR"
    db.close()


def test_form_collects_tags_and_failure_reason(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.view.selectRow(0)
    panel.load_selected_into_form()
    panel.tags_edit.setText("baseline, x4")
    panel.status_combo.setCurrentIndex(panel.status_combo.findData("failed"))
    panel.failure_reason_edit.setText("CUDA OOM")
    panel.save_form()

    row = db.get_run("train", run_id)
    assert row["tags"] == "baseline, x4"
    assert row["status"] == "failed"
    assert row["failure_reason"] == "CUDA OOM"
    db.close()


def test_failure_reason_row_visibility_follows_status(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()
    assert panel.form_layout.isRowVisible(panel.failure_reason_edit) is False

    panel.status_combo.setCurrentIndex(panel.status_combo.findData("failed"))
    assert panel.form_layout.isRowVisible(panel.failure_reason_edit) is True

    panel.status_combo.setCurrentIndex(panel.status_combo.findData("done"))
    assert panel.form_layout.isRowVisible(panel.failure_reason_edit) is False
    db.close()


def test_new_run_defaults_to_not_favorite(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()
    panel.work_combo.set_text("W")
    panel.model_combo.set_text("NewModel")
    panel.save_form()

    rows = db.list_train_runs()
    new_row = next(r for r in rows if r["model"] == "NewModel")
    assert new_row["favorite"] == 0
    db.close()


# --- #8 Column presets --------------------------------------------------------
def test_column_preset_simple_hides_paths_and_hyperparams(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.apply_preset_simple()
    visible = {h for h in panel.model.headers() if h not in panel._hidden_headers}
    assert "Result Folder Path" not in visible
    assert "Dataset Path" not in visible
    assert "Notes" not in visible
    assert "Model" in visible and "PSNR" in visible and "Status" in visible
    db.close()


def test_column_preset_paper_keeps_only_model_and_metrics(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.apply_preset_paper()
    visible = {h for h in panel.model.headers() if h not in panel._hidden_headers}
    assert visible == {"Model", "scale", "PSNR", "SSIM", "LPIPS"}
    db.close()


def test_column_preset_full_clears_hidden_set(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.apply_preset_simple()
    assert panel._hidden_headers  # sanity: simple actually hid something
    panel.apply_preset_full()
    assert panel._hidden_headers == set()
    db.close()


# --- #6 Global search dialog --------------------------------------------------
def _search_env(config):
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    sr = db.add_task("SR")
    w1 = db.add_work(sr, "SSL2SL")
    db.add_work(sr, "EmptyWork")  # no runs yet - should still be searchable
    db.insert_run("train", {"work_id": w1, "model": "Restormer", "notes": "baseline"})
    db.insert_run("inference", {"work_id": w1, "model": "SwinIR", "checkpoint_path": "/mnt/x/net.pth"})
    return db, sr, w1


def test_search_dialog_empty_query_lists_tasks_and_works(qapp, config):
    from dl_exp_manager.qt import Qt
    from dl_exp_manager.widgets.search_dialog import GlobalSearchDialog

    db, sr, w1 = _search_env(config)
    dialog = GlobalSearchDialog(db, lambda payload: None)
    kinds = {
        dialog.list.item(i).data(Qt.ItemDataRole.UserRole)["kind"]
        for i in range(dialog.list.count())
    }
    assert kinds == {"task", "work"}
    # the Work with no runs yet must still be findable
    assert any("EmptyWork" in dialog.list.item(i).text() for i in range(dialog.list.count()))
    db.close()


def test_search_dialog_finds_runs_by_model(qapp, config):
    from dl_exp_manager.qt import Qt
    from dl_exp_manager.widgets.search_dialog import GlobalSearchDialog

    db, sr, w1 = _search_env(config)
    dialog = GlobalSearchDialog(db, lambda payload: None)
    dialog.query_edit.setText("SwinIR")
    payloads = [dialog.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(dialog.list.count())]
    run_hits = [p for p in payloads if p["kind"] == "run"]
    assert len(run_hits) == 1
    assert run_hits[0]["run_kind"] == "inference"
    db.close()


def test_search_dialog_finds_runs_by_path_and_notes(qapp, config):
    from dl_exp_manager.qt import Qt
    from dl_exp_manager.widgets.search_dialog import GlobalSearchDialog

    db, sr, w1 = _search_env(config)
    dialog = GlobalSearchDialog(db, lambda payload: None)

    dialog.query_edit.setText("net.pth")
    assert any(
        dialog.list.item(i).data(Qt.ItemDataRole.UserRole)["kind"] == "run"
        for i in range(dialog.list.count())
    )

    dialog.query_edit.setText("baseline")
    assert any(
        "Restormer" in dialog.list.item(i).text() for i in range(dialog.list.count())
    )
    db.close()


def test_search_dialog_activation_calls_back_with_payload(qapp, config):
    from dl_exp_manager.widgets.search_dialog import GlobalSearchDialog

    db, sr, w1 = _search_env(config)
    received = []
    dialog = GlobalSearchDialog(db, received.append)
    dialog.query_edit.setText("Restormer")
    item = dialog.list.item(0)
    dialog._activate(item)
    assert len(received) == 1
    assert received[0]["kind"] == "run"
    db.close()


def test_search_dialog_escape_rejects(qapp, config):
    from dl_exp_manager.qt import Qt, QtCore, QtGui
    from dl_exp_manager.widgets.search_dialog import GlobalSearchDialog

    db, sr, w1 = _search_env(config)
    dialog = GlobalSearchDialog(db, lambda payload: None)
    event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    dialog.keyPressEvent(event)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected
    db.close()


def test_main_window_navigates_to_run_from_search(qapp, config):
    from dl_exp_manager.main_window import MainWindow

    d = tempfile.mkdtemp()
    window = MainWindow(os.path.join(d, "e.db"), os.path.join(d, "options.yaml"))
    sr = window.db.add_task("SR")
    work_id = window.db.add_work(sr, "SSL2SL")
    run_id = window.db.insert_run("train", {"work_id": work_id, "model": "FindThisRun"})
    window.nav.refresh()

    window._navigate_to_search_result(
        {"kind": "run", "run_kind": "train", "task_id": sr, "work_id": work_id, "run_id": run_id}
    )
    assert window.tabs.currentWidget() is window.train_panel
    current = window.train_panel._current_row()
    assert current is not None and current["id"] == run_id
    window.close()


# --- #9 Drag-and-drop folder registration ------------------------------------
def _drop_folder(edit, path):
    from dl_exp_manager.qt import Qt, QtCore, QtGui

    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
    event = QtGui.QDropEvent(
        QtCore.QPointF(5, 5), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    edit.dropEvent(event)


def test_path_edit_accepts_dropped_folder(qapp):
    from dl_exp_manager.widgets.common import PathEdit

    folder = tempfile.mkdtemp()
    edit = PathEdit(None, directory=True)
    received = []
    edit.folderDropped.connect(received.append)

    _drop_folder(edit, folder)
    assert edit.path() == folder
    assert received == [folder]


def test_path_edit_dropped_file_uses_parent_dir(qapp):
    from dl_exp_manager.widgets.common import PathEdit

    folder = tempfile.mkdtemp()
    file_path = os.path.join(folder, "config.yml")
    open(file_path, "w").write("x")

    edit = PathEdit(None, directory=True)
    _drop_folder(edit, file_path)
    assert edit.path() == folder


def test_result_folder_drop_autofills_empty_config(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()

    folder = tempfile.mkdtemp()
    open(os.path.join(folder, "config.yml"), "w").write("model: DroppedNet\n")

    assert panel.config_input.text().strip() == ""
    panel._on_result_folder_dropped(folder)
    assert "DroppedNet" in panel.config_input.text()
    db.close()


def test_result_folder_drop_does_not_overwrite_existing_config(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()
    panel.config_input.set_text("# my own content")

    folder = tempfile.mkdtemp()
    open(os.path.join(folder, "config.yml"), "w").write("model: DroppedNet\n")

    panel._on_result_folder_dropped(folder)
    assert panel.config_input.text() == "# my own content"
    db.close()
