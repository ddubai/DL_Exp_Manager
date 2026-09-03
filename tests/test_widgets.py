"""위젯 동작 테스트 (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="Qt 바인딩 필요")
QtGui = pytest.importorskip("PyQt6.QtGui", reason="Qt 바인딩 필요")

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
    assert selector.spin.maximum() == 4
    selector.spin.setValue(2)
    assert selector.value() == "2"
    assert selector.hint.text() == "2 of 4 GPU(s) on this server"

    # legacy comma-index data still round-trips to a sensible count
    selector.set_value("0,2")
    assert selector.value() == "2"


def test_gpu_selector_without_server(qapp):
    from dl_exp_manager.widgets.common import GpuSelector

    selector = GpuSelector()
    selector.set_server(None)
    assert selector.value() == ""
    assert selector.spin.maximum() == 64


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
    assert state["used"] == 4  # 2 + 2 GPUs claimed
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
                                "gpu_indices": "3", "status": "running"})
    panel = ServerStatusPanel(db, config)
    assert panel._state["Server 1"]["over_capacity"] is True  # 3+3=6 > 4 GPUs on Server 1
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


# --- ServerEditDialog: "+ Add GPU" defaults to Index 0's Type/Memory ----------
def test_add_gpu_defaults_to_index0_content(qapp):
    from dl_exp_manager.widgets.server_panel import ServerEditDialog

    dialog = ServerEditDialog(None, None)  # "Add Server" - starts empty
    dialog._append(None)  # first row -> Index 0, falls back to H100/blank
    dialog.table.item(0, 1).setText("A100")
    dialog.table.item(0, 2).setText("40")

    dialog._append(None)  # second row should copy Index 0's Type/Memory
    assert dialog.table.item(1, 0).text() == "1"
    assert dialog.table.item(1, 1).text() == "A100"
    assert dialog.table.item(1, 2).text() == "40"

    dialog._append(None)  # a third row keeps following Index 0, not row 1
    assert dialog.table.item(2, 1).text() == "A100"
    assert dialog.table.item(2, 2).text() == "40"


def test_add_gpu_falls_back_when_no_index0_row(qapp):
    from dl_exp_manager.widgets.server_panel import ServerEditDialog

    dialog = ServerEditDialog(None, None)
    dialog._append(None)
    dialog.table.item(0, 0).setText("5")  # no row is literally Index 0

    dialog._append(None)
    assert dialog.table.item(1, 1).text() == "H100"
    assert dialog.table.item(1, 2).text() == ""


def test_gpu_count_parsing():
    from dl_exp_manager.widgets.server_panel import ServerStatusPanel

    assert ServerStatusPanel._parse_count("0,1,3") == 3  # legacy index list -> count
    assert ServerStatusPanel._parse_count("2") == 2       # new format: a plain count
    assert ServerStatusPanel._parse_count("") == 0
    assert ServerStatusPanel._parse_count(None) == 0


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


def test_new_run_defaults_to_queued_status(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()
    assert panel.status_combo.currentData() == "queued"
    db.close()


def test_server_combo_only_lists_configured_servers(qapp, config):
    """#1: Server can only be picked from the top Servers list, not typed in."""
    db, panel, run_id = _panel_with_one_run(config)
    assert panel.server_combo.isEditable() is False
    names = {panel.server_combo.itemText(i) for i in range(panel.server_combo.count())}
    assert names - {""} == set(config.server_names())
    db.close()


def test_metrics_are_shared_within_a_task(qapp, config):
    """#6: a metric typed on one Run becomes the Task's default, prefilled (empty)
    on the next New Run - without needing the explicit "register" menu action."""
    db, panel, run_id = _panel_with_one_run(config)
    panel.view.selectRow(0)
    panel.load_selected_into_form()
    panel.metrics_editor.set_metrics({"CustomMetric": 12.5})
    panel.save_form()

    assert "CustomMetric" in config.metric_keys("SR")

    panel.reset_form()
    assert panel.metrics_editor.metrics() == {}  # empty prefilled row isn't "set"
    keys = {
        panel.metrics_editor.table.item(r, 0).text()
        for r in range(panel.metrics_editor.table.rowCount())
    }
    assert "CustomMetric" in keys
    db.close()


def test_paths_and_command_hidden_until_row_selected(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    assert panel.paths_box.isVisibleTo(panel) is False
    assert panel.detail_tabs.isVisibleTo(panel) is False

    panel.view.selectRow(0)
    assert panel.paths_box.isVisibleTo(panel) is True
    assert panel.detail_tabs.isVisibleTo(panel) is True
    db.close()


def test_inference_hides_server_and_gpu_rows(qapp, config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import InferencePanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    window = QtWidgets.QMainWindow()
    panel = InferencePanel(db, config, parent=window)
    window.setCentralWidget(panel)
    panel.set_scope(task_id, work_id)

    assert panel.form_layout.getWidgetPosition(panel.server_combo)[0] == -1
    assert panel.form_layout.getWidgetPosition(panel.gpu_selector)[0] == -1
    # Not just absent from the layout - actually hidden, or they float unpositioned
    # at (0,0) on top of the form instead of taking no space.
    assert panel.server_combo.isVisibleTo(panel) is False
    assert panel.gpu_selector.isVisibleTo(panel) is False
    db.close()


def test_inference_source_train_run_prefills_model(qapp, config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import InferencePanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    train_id = db.insert_run("train", {"work_id": work_id, "model": "Restormer"})
    window = QtWidgets.QMainWindow()
    panel = InferencePanel(db, config, parent=window)
    window.setCentralWidget(panel)
    panel.set_scope(task_id, work_id)
    panel.reset_form()

    index = panel.source_run_combo.findData(train_id)
    assert index >= 0
    panel.source_run_combo.setCurrentIndex(index)
    assert panel.model_combo.current_text() == "Restormer"

    panel.checkpoint_epoch_edit.setText("300000")
    panel.result_path_edit.set_path("/mnt/exp/x")
    panel.save_form()

    row = db.list_inference_runs(work_id=work_id)[0]
    assert row["source_train_run_id"] == train_id
    assert row["checkpoint_epoch"] == "300000"
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


# --- Nav panel: Task ▸ Work drill-down (Option A) -----------------------------
def _nav_env(config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.nav_panel import NavigationPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"), seed=False)
    sr = db.add_task("SR")
    dn = db.add_task("DN")
    ssl2sl = db.add_work(sr, "SSL2SL", "transfer experiment")
    bsr = db.add_work(sr, "BSR-x4")
    n2n = db.add_work(dn, "N2N-Base")
    db.insert_run("train", {"work_id": ssl2sl, "model": "Restormer"})
    nav = NavigationPanel(db, config)
    return db, nav, sr, dn, ssl2sl, bsr, n2n


def test_nav_auto_drills_into_first_task_and_work_on_first_load(qapp, config):
    # list_tasks()/list_works() sort by name, so "DN" < "SR" alphabetically -
    # the first Task/Work is DN / N2N-Base, not insertion order.
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    assert nav.current_task_id() == dn
    assert nav.current_work_id() == n2n
    db.close()


def test_nav_go_root_then_enter_task_shows_works_only(qapp, config):
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    received = []
    nav.selectionChanged.connect(lambda t, w: received.append((t, w)))

    nav._go_root()
    assert nav.current_task_id() is None and nav.current_work_id() is None
    assert received[-1] == (-1, -1)

    nav._enter_task(sr)
    assert nav.current_task_id() == sr and nav.current_work_id() is None
    assert received[-1] == (sr, -1)

    nav._enter_work(bsr)
    assert nav.current_work_id() == bsr
    assert received[-1] == (sr, bsr)
    db.close()


def test_nav_refresh_with_explicit_ids_jumps_directly(qapp, config):
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    nav.refresh(select_task_id=dn)
    assert nav.current_task_id() == dn and nav.current_work_id() is None

    nav.refresh(select_work_id=bsr)
    assert nav.current_task_id() == sr and nav.current_work_id() == bsr
    db.close()


def test_nav_bare_refresh_preserves_current_position(qapp, config):
    """A no-arg refresh() (called after every save/config change) must not
    yank the user back to the first Task/Work - only the very first load does that."""
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    nav._go_root()
    nav.refresh()
    assert nav.current_task_id() is None and nav.current_work_id() is None

    nav._enter_task(dn)
    nav.refresh()
    assert nav.current_task_id() == dn and nav.current_work_id() is None
    db.close()


def test_nav_refresh_falls_back_when_current_work_deleted(qapp, config):
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    nav.refresh(select_work_id=ssl2sl)
    db.delete_work(ssl2sl)
    nav.refresh()
    assert nav.current_task_id() == sr
    assert nav.current_work_id() is None  # dropped back to the Works list, not left dangling
    db.close()


def test_nav_shows_registered_datasets_for_selected_work(qapp, config):
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    db.add_dataset(ssl2sl, "DIV2K", "Full Pair", "/mnt/data/DIV2K/train")
    db.add_dataset(ssl2sl, "DF2K")
    nav.refresh(select_work_id=ssl2sl)

    text_blob = " ".join(
        label.text()
        for label in nav.list_host.findChildren(QtWidgets.QLabel)
    )
    assert "DIV2K" in text_blob and "Full Pair" in text_blob and "DF2K" in text_blob
    db.close()


def test_nav_add_dataset_via_inline_dialog(qapp, config, monkeypatch):
    from dl_exp_manager.widgets.dataset_dialog import DatasetEditDialog

    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    nav.refresh(select_work_id=ssl2sl)

    monkeypatch.setattr(DatasetEditDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted)
    monkeypatch.setattr(
        DatasetEditDialog, "result_values", lambda self: ("DIV2K", "Full Pair", "/mnt/data/DIV2K", "")
    )
    nav._add_dataset()

    datasets = db.list_datasets(ssl2sl)
    assert len(datasets) == 1 and datasets[0]["name"] == "DIV2K"
    db.close()


def test_nav_add_task_and_work_drill_into_the_new_one(qapp, config, monkeypatch):
    db, nav, sr, dn, ssl2sl, bsr, n2n = _nav_env(config)
    nav.refresh(select_work_id=ssl2sl)

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText",
        staticmethod(lambda *a, **k: ("NewTask", True)),
    )
    nav.add_task()
    assert nav.current_work_id() is None
    new_task_id = nav.current_task_id()
    assert new_task_id != sr

    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getText",
        staticmethod(lambda *a, **k: ("NewWork", True)),
    )
    nav.add_work()
    assert nav.current_task_id() == new_task_id
    work = db.get_work(nav.current_work_id())
    assert work is not None and work["name"] == "NewWork"
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


# --- #11 Log tail viewer -----------------------------------------------------
def test_log_viewer_auto_detects_and_tails_log(qapp):
    from dl_exp_manager.widgets.log_viewer import LogViewerDialog

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "train.log"), "w") as fp:
        fp.write("\n".join(f"line {i}" for i in range(1000)))

    dialog = LogViewerDialog(folder, title="Test Log")
    assert "line 999" in dialog.text.toPlainText()
    assert "line 0" not in dialog.text.toPlainText()


def test_log_viewer_reports_missing_log_without_crashing(qapp):
    from dl_exp_manager.widgets.log_viewer import LogViewerDialog

    dialog = LogViewerDialog(tempfile.mkdtemp(), title="Empty")
    assert "No log file found" in dialog.path_label.text()
    assert dialog.text.toPlainText() == ""


def test_log_viewer_handles_blank_result_path(qapp):
    from dl_exp_manager.widgets.log_viewer import LogViewerDialog

    dialog = LogViewerDialog("", title="No Path")
    assert "no result folder set" in dialog.path_label.text()
    assert not dialog.open_folder_btn.isEnabled()


def test_log_viewer_browse_switches_to_chosen_file(qapp, monkeypatch):
    from dl_exp_manager.widgets.log_viewer import LogViewerDialog

    folder = tempfile.mkdtemp()
    other = os.path.join(folder, "custom.txt")
    with open(other, "w") as fp:
        fp.write("hello from custom file")

    dialog = LogViewerDialog(folder, title="Browse Test")
    assert dialog._log_path is None

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (other, "")),
    )
    dialog._browse()
    assert dialog._log_path == other
    assert "hello from custom file" in dialog.text.toPlainText()


def test_view_log_requires_selection(qapp, config, monkeypatch):
    from dl_exp_manager.qt import QtCore

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    db, panel, run_id = _panel_with_one_run(config)
    panel.view.clearSelection()
    panel.view.setCurrentIndex(QtCore.QModelIndex())
    panel.view_log()  # must not raise even with nothing selected
    db.close()


# --- #10 Markdown / HTML report export ---------------------------------------
def test_export_report_writes_markdown_file(qapp, config, monkeypatch, tmp_path):
    db, panel, run_id = _panel_with_one_run(config)

    out_path = str(tmp_path / "report.md")
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out_path, "Markdown (*.md)"))
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No)
    )

    panel.export_report()

    with open(out_path, encoding="utf-8") as fp:
        content = fp.read()
    assert content.startswith("# Train Runs Report")
    assert "Restormer" in content
    db.close()


def test_export_report_writes_html_file_when_html_filter_chosen(qapp, config, monkeypatch, tmp_path):
    db, panel, run_id = _panel_with_one_run(config)

    out_path = str(tmp_path / "report.html")
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (out_path, "HTML (*.html)"))
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No)
    )

    panel.export_report()

    with open(out_path, encoding="utf-8") as fp:
        content = fp.read()
    assert "<title>Train Runs Report</title>" in content
    assert "Restormer" in content
    db.close()


def test_export_report_with_no_rows_warns_instead_of_writing(qapp, config, monkeypatch, tmp_path):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import TrainPanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    window = QtWidgets.QMainWindow()
    panel = TrainPanel(db, config, parent=window)
    window.setCentralWidget(panel)
    panel._test_window = window
    panel.set_scope(task_id, work_id)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    called = {}
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: called.setdefault("called", True) or ("", "")),
    )

    panel.export_report()
    assert "called" not in called  # dialog never opened because there were no rows
    db.close()


def test_view_log_opens_dialog_for_selected_run(qapp, config, monkeypatch):
    db, panel, run_id = _panel_with_one_run(config)
    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "train.log"), "w") as fp:
        fp.write("run log content\n")
    db.update_run("train", run_id, {**db.get_run("train", run_id), "result_path": folder})
    panel.reload()
    panel.view.selectRow(0)

    opened = {}

    def fake_exec(self):
        opened["path_label"] = self.path_label.text()
        opened["content"] = self.text.toPlainText()
        return 0

    from dl_exp_manager.widgets.log_viewer import LogViewerDialog

    monkeypatch.setattr(LogViewerDialog, "exec", fake_exec)
    panel.view_log()
    assert "run log content" in opened["content"]
    db.close()


# --- Image viewer (representative image) --------------------------------------
def test_image_viewer_auto_detects_and_renders_image(qapp):
    from dl_exp_manager.widgets.image_viewer import ImageViewerDialog

    folder = tempfile.mkdtemp()
    pixmap = QtGui.QPixmap(40, 20)
    pixmap.fill(QtGui.QColor("red"))
    pixmap.save(os.path.join(folder, "restormer_output.png"))
    open(os.path.join(folder, "input.png"), "w").close()  # 0바이트 - 대표 이미지 후보에서 밀림

    dialog = ImageViewerDialog(folder, title="Test Image")
    assert dialog._image_path == os.path.join(folder, "restormer_output.png")
    assert dialog._pixmap is not None and not dialog._pixmap.isNull()


def test_image_viewer_reports_missing_image_without_crashing(qapp):
    from dl_exp_manager.widgets.image_viewer import ImageViewerDialog

    dialog = ImageViewerDialog(tempfile.mkdtemp(), title="Test Image")
    assert dialog._image_path is None
    assert "No image found" in dialog.image_label.text()


def test_view_image_opens_dialog_for_selected_run(qapp, config, monkeypatch):
    db, panel, run_id = _panel_with_one_run(config)
    folder = tempfile.mkdtemp()
    pixmap = QtGui.QPixmap(10, 10)
    pixmap.fill(QtGui.QColor("blue"))
    pixmap.save(os.path.join(folder, "out.png"))
    db.update_run("train", run_id, {**db.get_run("train", run_id), "result_path": folder})
    panel.reload()
    panel.view.selectRow(0)

    opened = {}

    def fake_exec(self):
        opened["image_path"] = self._image_path
        return 0

    from dl_exp_manager.widgets.image_viewer import ImageViewerDialog

    monkeypatch.setattr(ImageViewerDialog, "exec", fake_exec)
    panel.view_image()
    assert opened["image_path"] == os.path.join(folder, "out.png")
    db.close()


# --- Training curve -------------------------------------------------------------
def test_curve_dialog_parses_log_into_selectable_metric_series(qapp):
    from dl_exp_manager.widgets.curve_chart import CurveDialog

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "loss.log"), "w") as fp:
        fp.write(
            "2024-01-01 00:00:00,000 INFO: [iter: 100] l_pix: 5.0e-02\n"
            "2024-01-01 01:00:00,000 INFO: [iter: 200] l_pix: 2.0e-02\n"
        )

    dialog = CurveDialog(folder, title="Test Curve")
    items = [dialog.metric_combo.itemText(i) for i in range(dialog.metric_combo.count())]
    assert "l_pix" in items
    dialog.metric_combo.setCurrentText("l_pix")
    assert dialog.chart._points == [(100, 5.0e-02), (200, 2.0e-02)]


def test_curve_dialog_no_log_shows_message(qapp):
    from dl_exp_manager.widgets.curve_chart import CurveDialog

    dialog = CurveDialog(tempfile.mkdtemp(), title="Test Curve")
    assert dialog.metric_combo.count() == 0
    assert "No log file found" in dialog.path_label.text()


def test_inference_panel_has_no_training_curve_button(qapp, config):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.run_panel import InferencePanel

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "W")
    window = QtWidgets.QMainWindow()
    panel = InferencePanel(db, config, parent=window)
    window.setCentralWidget(panel)
    panel.set_scope(task_id, work_id)
    assert panel.SHOW_TRAINING_CURVE is False
    db.close()


# --- Compare runs ---------------------------------------------------------------
def _fake_run(run_id, model, psnr, config_yaml="", **extra):
    row = {
        "id": run_id, "work_id": 1, "status": "done", "server": "Server 1",
        "gpu_indices": "2", "model": model, "dataset": "DIV2K",
        "duration_sec": 3600, "epochs": "100", "batch_size": "8", "lr": "3e-4",
        "optimizer": "AdamW", "metrics_json": f'{{"PSNR": {psnr}}}',
        "extra_json": "{}", "config_yaml": config_yaml,
    }
    row.update(extra)
    return row


def test_compare_dialog_highlights_differing_fields(qapp, config):
    from dl_exp_manager.qt import Qt
    from dl_exp_manager.widgets.compare_dialog import CompareRunsDialog

    rows = [_fake_run(1, "Restormer", 30.0, "a: 1\nb: 2\n"), _fake_run(2, "SwinIR", 32.0, "a: 1\nb: 3\n")]
    dialog = CompareRunsDialog(rows, config, "SR")

    table = dialog.findChild(QtWidgets.QTableWidget)
    labels = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "Model" in labels and "PSNR" in labels

    model_row = labels.index("Model")
    assert table.item(model_row, 1).text() == "Restormer"
    assert table.item(model_row, 2).text() == "SwinIR"
    # 값이 다른 필드(Model)는 배경이 칠해지고, 같은 필드(Optimizer)는 칠해지지 않는다
    assert table.item(model_row, 1).background().style() != Qt.BrushStyle.NoBrush
    optimizer_row = labels.index("Optimizer")
    assert table.item(optimizer_row, 1).background().style() == Qt.BrushStyle.NoBrush

    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert tabs.tabText(1) == "config.yaml Diff"


def test_compare_dialog_three_runs_shows_separate_config_tabs(qapp, config):
    from dl_exp_manager.widgets.compare_dialog import CompareRunsDialog

    rows = [_fake_run(i, f"Model{i}", 30.0 + i) for i in (1, 2, 3)]
    dialog = CompareRunsDialog(rows, config, "SR")
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    tab_labels = [tabs.tabText(i) for i in range(tabs.count())]
    assert tab_labels == ["Metrics / Params", "#1 config.yaml", "#2 config.yaml", "#3 config.yaml"]


def test_compare_selected_opens_dialog_for_two_or_three_rows(qapp, config, monkeypatch):
    from dl_exp_manager.qt import QtCore

    db, panel, run_id = _panel_with_one_run(config)
    dup_id = db.duplicate_run("train", run_id)
    panel.reload()
    panel.view.selectRow(0)
    panel.view.selectionModel().select(
        panel.proxy.index(1, 0),
        QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )

    opened = {}

    def fake_exec(self):
        opened["columns"] = self.findChild(QtWidgets.QTableWidget).columnCount()
        return 0

    from dl_exp_manager.widgets.compare_dialog import CompareRunsDialog

    monkeypatch.setattr(CompareRunsDialog, "exec", fake_exec)
    panel.compare_selected()
    assert opened["columns"] == 3  # Field + 2 selected runs
    db.close()


# --- Auto-logging: parse config.yaml + training log ---------------------------
_TRAIN_CONFIG_YAML = """\
network_g:
  type: Restormer
datasets:
  train:
    name: DIV2K
    dataroot_gt: /mnt/data/DIV2K/train
    batch_size_per_gpu: 8
train:
  total_iter: 300000
  optim_g:
    type: AdamW
    lr: !!float 3e-4
"""

_TRAIN_LOG = """\
2024-01-01 00:00:00,000 INFO: start
2024-01-01 01:00:00,000 INFO: [iter: 1,000] l_pix: 1.0e-02
2024-01-01 02:00:00,000 INFO: Validation
\t # psnr: 31.5000\tBest: 31.5 @ 1000 iter
\t # ssim: 0.9000\tBest: 0.9 @ 1000 iter
"""


def test_parse_result_folder_fills_train_form(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "config.yml"), "w") as fp:
        fp.write(_TRAIN_CONFIG_YAML)
    with open(os.path.join(folder, "loss.log"), "w") as fp:
        fp.write(_TRAIN_LOG)
    panel.result_path_edit.set_path(folder)

    panel._parse_result_folder()

    assert panel.model_combo.current_text() == "Restormer"
    assert panel.dataset_combo.current_text() == "DIV2K"
    assert panel.dataset_path_edit.path() == "/mnt/data/DIV2K/train"
    assert panel.epochs_edit.text() == "300000"
    assert panel.batch_edit.text() == "8"
    assert panel.lr_edit.text() == "0.0003"
    assert panel.optimizer_combo.current_text() == "AdamW"
    assert panel.metrics_editor.metrics() == {"PSNR": 31.5, "SSIM": 0.9}
    assert panel.duration_edit.text() == "02:00:00"
    db.close()


def test_parse_result_folder_does_not_clobber_manual_dataset_path(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    panel.reset_form()
    panel.dataset_path_edit.set_path("/my/own/path")

    folder = tempfile.mkdtemp()
    with open(os.path.join(folder, "config.yml"), "w") as fp:
        fp.write(_TRAIN_CONFIG_YAML)
    panel.result_path_edit.set_path(folder)

    panel._parse_result_folder()
    assert panel.dataset_path_edit.path() == "/my/own/path"
    db.close()


# --- Work-scoped dataset registry -----------------------------------------------
def test_dataset_edit_dialog_round_trips_fields(qapp):
    from dl_exp_manager.widgets.dataset_dialog import DatasetEditDialog

    dialog = DatasetEditDialog(dataset={"name": "DIV2K", "variant": "Full Pair", "path": "/a", "notes": "n"})
    assert dialog.name_edit.text() == "DIV2K"
    assert dialog.variant_edit.text() == "Full Pair"
    assert dialog.path_edit.path() == "/a"
    assert dialog.result_values() == ("DIV2K", "Full Pair", "/a", "n")


def test_dataset_manager_dialog_add_edit_remove(qapp, config, monkeypatch):
    from dl_exp_manager.db import Database
    from dl_exp_manager.widgets.dataset_dialog import DatasetEditDialog, DatasetManagerDialog

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "BSR-x4")
    window = QtWidgets.QMainWindow()
    dialog = DatasetManagerDialog(db, work_id, "BSR-x4", parent=window)
    window.setCentralWidget(dialog)

    monkeypatch.setattr(
        DatasetEditDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(
        DatasetEditDialog, "result_values", lambda self: ("DIV2K", "Full Pair", "/mnt/a", "")
    )
    dialog._add()
    assert dialog.table.rowCount() == 1
    assert db.list_datasets(work_id)[0]["path"] == "/mnt/a"

    dialog.table.selectRow(0)
    monkeypatch.setattr(
        DatasetEditDialog, "result_values", lambda self: ("DIV2K", "Full Pair", "/mnt/b", "moved")
    )
    dialog._edit_selected()
    assert db.list_datasets(work_id)[0]["path"] == "/mnt/b"
    assert db.list_datasets(work_id)[0]["notes"] == "moved"

    monkeypatch.setattr("dl_exp_manager.editing.confirm_delete", lambda *a, **k: True)
    dialog.table.selectRow(0)
    dialog._remove_selected()
    assert db.list_datasets(work_id) == []
    db.close()


def test_registered_dataset_combo_scoped_to_work(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    other_work = db.add_work(panel._task_id, "OtherWork")
    db.add_dataset(panel._work_id, "DIV2K", "Full Pair", "/mnt/data/div2k")
    db.add_dataset(other_work, "Set5", "", "/mnt/data/set5")

    panel.reset_form()
    labels = [
        panel.dataset_registry_combo.itemText(i)
        for i in range(panel.dataset_registry_combo.count())
    ]
    assert "DIV2K · Full Pair" in labels
    assert not any("Set5" in label for label in labels)
    db.close()


def test_picking_registered_dataset_fills_dataset_and_path(qapp, config):
    db, panel, run_id = _panel_with_one_run(config)
    dataset_id = db.add_dataset(panel._work_id, "DIV2K", "Full Pair", "/mnt/data/div2k")
    panel.reset_form()

    index = panel.dataset_registry_combo.findData(dataset_id)
    assert index >= 0
    panel.dataset_registry_combo.setCurrentIndex(index)

    assert panel.dataset_combo.current_text() == "DIV2K · Full Pair"
    assert panel.dataset_path_edit.path() == "/mnt/data/div2k"
    db.close()


def test_open_dataset_manager_opens_dialog_scoped_to_current_work(qapp, config, monkeypatch):
    db, panel, run_id = _panel_with_one_run(config)

    opened = {}

    def fake_exec(self):
        opened["work_id"] = self.work_id
        return 0

    from dl_exp_manager.widgets.dataset_dialog import DatasetManagerDialog

    monkeypatch.setattr(DatasetManagerDialog, "exec", fake_exec)
    panel._open_dataset_manager()
    assert opened["work_id"] == panel._work_id
    db.close()
