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
