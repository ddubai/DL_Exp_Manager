"""Task 별 컬럼 구성과 DB v1→v2 마이그레이션 테스트."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from dl_exp_manager.config_store import OptionsConfig

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", reason="Qt 바인딩 필요")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from dl_exp_manager import theme

    theme.apply_theme(app, "dark")
    return app


@pytest.fixture
def config():
    return OptionsConfig(os.path.join(tempfile.mkdtemp(), "options.yaml"))


SAMPLE_ROW = {
    "id": 1,
    "status": "done",
    "server": "Server 1",
    "gpu_indices": "0,1",
    "model": "Restormer",
    "dataset": "DIV2K",
    "duration_sec": 5400,
    "result_path": "/mnt/exp/a",
    "metrics_json": '{"PSNR": 32.4123, "SSIM": 0.8993, "LPIPS": 0.121}',
    "extra_json": '{"scale": "x4"}',
    "notes": "ok",
}


def test_columns_follow_task(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns

    model = RunTableModel()
    model.set_content([SAMPLE_ROW], build_columns(config, "SR", "train"))
    sr_headers = model.headers()
    assert "LPIPS" in sr_headers and "scale" in sr_headers

    model.set_content([SAMPLE_ROW], build_columns(config, "Classification", "train"))
    cls_headers = model.headers()
    assert "Top-1" in cls_headers
    assert "LPIPS" not in cls_headers and "scale" not in cls_headers


def test_metric_display_uses_unit_and_digits(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns

    model = RunTableModel()
    model.set_content([SAMPLE_ROW], build_columns(config, "SR", "train"))
    values = dict(zip(model.headers(), model.row_values(0)))
    assert values["PSNR"] == "32.41 dB"   # digits=2 + unit
    assert values["SSIM"] == "0.8993"     # digits=4, 단위 없음
    assert values["GPU"] == "2 GPU(s)"
    assert values["scale"] == "x4"        # extra_json 에서 옴


def test_metrics_present_in_data_are_not_hidden(qapp, config):
    """설정에 없는 지표라도 값이 있으면 컬럼으로 덧붙여 보여 준다."""
    from dl_exp_manager.models import RunTableModel, build_columns

    extra_keys = RunTableModel.metric_keys_in([SAMPLE_ROW])
    columns = build_columns(config, "Classification", "train", extra_keys)
    headers = [c.header for c in columns]
    assert "PSNR" in headers and "Top-1" in headers


def test_numeric_sorting_is_by_value_not_text(qapp, config):
    from dl_exp_manager.models import SORT_ROLE, RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    rows = [
        dict(SAMPLE_ROW, id=1, duration_sec=5400, metrics_json='{"PSNR": 9.5}'),
        dict(SAMPLE_ROW, id=2, duration_sec=93600, metrics_json='{"PSNR": 32.4}'),
        dict(SAMPLE_ROW, id=3, duration_sec=600, metrics_json='{"PSNR": 100.0}'),
    ]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))

    from dl_exp_manager.models import RunFilterProxy

    proxy = RunFilterProxy()
    proxy.setSourceModel(model)
    col = model.column_index("duration")
    proxy.sort(col, Qt.SortOrder.AscendingOrder)
    order = [
        proxy.data(proxy.index(r, col), SORT_ROLE) for r in range(proxy.rowCount())
    ]
    assert order == sorted(order)
    assert order == [600.0, 5400.0, 93600.0]


def test_running_row_shows_elapsed_when_duration_missing(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.utils import now_iso

    row = dict(SAMPLE_ROW, status="running", duration_sec=None, started_at=now_iso())
    model = RunTableModel()
    model.set_content([row], build_columns(config, "SR", "train"))
    values = dict(zip(model.headers(), model.row_values(0)))
    assert values["Duration"].startswith("~")


def _make_v1_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE servers (id INTEGER PRIMARY KEY, name TEXT UNIQUE, host TEXT, gpu TEXT,
            note TEXT, updated_at TEXT);
        CREATE TABLE tasks (id INTEGER PRIMARY KEY, name TEXT UNIQUE, description TEXT, created_at TEXT);
        CREATE TABLE works (id INTEGER PRIMARY KEY, task_id INTEGER, name TEXT, description TEXT,
            created_at TEXT);
        CREATE TABLE train_runs (id INTEGER PRIMARY KEY, work_id INTEGER, server TEXT, model TEXT,
            dataset TEXT, dataset_path TEXT, result_path TEXT, status TEXT, started_at TEXT,
            duration_sec REAL, epochs TEXT, batch_size TEXT, lr TEXT, optimizer TEXT,
            metrics_json TEXT, exec_command TEXT, config_yaml TEXT, notes TEXT,
            created_at TEXT, updated_at TEXT);
        CREATE TABLE inference_runs (id INTEGER PRIMARY KEY, work_id INTEGER, server TEXT, model TEXT,
            checkpoint_path TEXT, dataset TEXT, dataset_path TEXT, result_path TEXT, device TEXT,
            input_size TEXT, latency_ms REAL, throughput_fps REAL, status TEXT, started_at TEXT,
            duration_sec REAL, metrics_json TEXT, exec_command TEXT, config_yaml TEXT, notes TEXT,
            created_at TEXT, updated_at TEXT);
        PRAGMA user_version = 1;
        """
    )
    conn.execute("INSERT INTO tasks(id,name,description,created_at) VALUES (1,'SR','',datetime())")
    conn.execute(
        "INSERT INTO works(id,task_id,name,description,created_at) VALUES (1,1,'SSL2SL','',datetime())"
    )
    conn.execute(
        "INSERT INTO train_runs(work_id,model,metrics_json,created_at,updated_at) "
        "VALUES (1,'Restormer','{\"PSNR\": 31.2}',datetime(),datetime())"
    )
    conn.execute(
        "INSERT INTO inference_runs(work_id,model,checkpoint_path,created_at,updated_at) "
        "VALUES (1,'SwinIR','/mnt/x/net_g.pth',datetime(),datetime())"
    )
    conn.commit()
    conn.close()


def test_v1_database_migrates_and_keeps_data():
    from dl_exp_manager.db import Database

    path = os.path.join(tempfile.mkdtemp(), "old.db")
    _make_v1_db(path)

    db = Database(path, seed=False)
    assert db.migrated_from == 1
    from dl_exp_manager.db import SCHEMA_VERSION

    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    rows = db.list_train_runs()
    assert len(rows) == 1
    assert rows[0]["model"] == "Restormer"
    assert rows[0]["metrics_json"] == '{"PSNR": 31.2}'
    assert rows[0]["gpu_indices"] == ""
    assert rows[0]["extra_json"] == "{}"

    # inference_runs -> evaluation_runs 로 이름이 바뀌어도 기록은 그대로 살아 있어야 한다.
    evaluations = db.list_evaluation_runs()
    assert len(evaluations) == 1
    assert evaluations[0]["model"] == "SwinIR"
    assert evaluations[0]["checkpoint_path"] == "/mnt/x/net_g.pth"
    db.close()


def test_v7_inference_tables_and_history_are_renamed_to_evaluation():
    """v7 -> v8: 테이블 이름과 run_history 의 run_kind 문자열을 함께 옮긴다."""
    from dl_exp_manager.db import Database

    path = os.path.join(tempfile.mkdtemp(), "v7.db")
    db = Database(path, seed=False)
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run("evaluation", {"work_id": work_id, "model": "NAFNet"})
    db.close()

    # 실제 v7 DB 모양으로 되돌려 둔다 (옛 테이블 이름 + 옛 run_kind + user_version).
    conn = sqlite3.connect(path)
    conn.executescript(
        f"""
        ALTER TABLE evaluation_runs RENAME TO inference_runs;
        UPDATE run_history SET run_kind = 'inference' WHERE run_id = {run_id};
        PRAGMA user_version = 7;
        """
    )
    conn.commit()
    conn.close()

    db = Database(path, seed=False)
    assert db.migrated_from == 7
    tables = {
        row[0]
        for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "evaluation_runs" in tables and "inference_runs" not in tables

    rows = db.list_evaluation_runs()
    assert len(rows) == 1 and rows[0]["model"] == "NAFNet"
    assert {h["run_kind"] for h in db.list_history("evaluation", run_id)} == {"evaluation"}
    db.close()


def test_migration_is_idempotent():
    from dl_exp_manager.db import Database

    path = os.path.join(tempfile.mkdtemp(), "old.db")
    _make_v1_db(path)
    Database(path, seed=False).close()
    db = Database(path, seed=False)
    assert db.migrated_from is None
    db.close()


def test_gpu_and_extra_round_trip():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run(
        "train",
        {"work_id": work_id, "model": "M", "gpu_indices": "0,3", "extra_json": {"scale": "x4"}},
    )
    row = db.get_run("train", run_id)
    assert row["gpu_indices"] == "0,3"
    assert row["extra_json"] == '{"scale": "x4"}'
    assert db.count_extra_value("scale", "x4") == 1
    db.close()


def test_distinct_values_scoped_to_task():
    """Task 별 옵션이므로 다른 Task 의 값이 섞이면 안 된다."""
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    sr = db.add_task("SR")
    cls = db.add_task("Classification")
    db.insert_run("train", {"work_id": db.add_work(sr, "A"), "model": "SwinIR"})
    db.insert_run("train", {"work_id": db.add_work(cls, "B"), "model": "ResNet-50"})

    assert db.distinct_values("train", "model", sr) == ["SwinIR"]
    assert db.distinct_values("train", "model", cls) == ["ResNet-50"]
    assert set(db.distinct_values("train", "model")) == {"SwinIR", "ResNet-50"}
    db.close()


def test_bulk_rename_updates_both_tables():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    db.insert_run("train", {"work_id": work_id, "model": "Old"})
    db.insert_run("evaluation", {"work_id": work_id, "model": "Old"})
    assert db.count_runs_using("model", "Old") == 2
    assert db.rename_value_in_runs("model", "Old", "New") == 2
    assert db.count_runs_using("model", "Old") == 0
    assert db.count_runs_using("model", "New") == 2
    db.close()


# --- v3: favorites / tags / failure_reason ---------------------------------
def test_v3_columns_present_and_default():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run("train", {"work_id": work_id, "model": "M"})
    row = db.get_run("train", run_id)
    assert row["favorite"] == 0
    assert row["tags"] == ""
    assert row["failure_reason"] == ""
    db.close()


def test_toggle_favorite_round_trips():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run("train", {"work_id": work_id, "model": "M"})
    assert db.toggle_favorite("train", run_id) is True
    assert db.get_run("train", run_id)["favorite"] == 1
    assert db.toggle_favorite("train", run_id) is False
    assert db.get_run("train", run_id)["favorite"] == 0
    assert db.toggle_favorite("train", 99999) is False
    db.close()


def test_duplicate_resets_favorite_but_keeps_tags():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run(
        "train", {"work_id": work_id, "model": "M", "tags": "keep-me", "favorite": True}
    )
    dup_id = db.duplicate_run("train", run_id)
    dup = db.get_run("train", dup_id)
    assert dup["favorite"] == 0
    assert dup["tags"] == "keep-me"
    db.close()


def test_search_runs_matches_notes_tags_and_names():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "SSL2SL")
    db.insert_run("train", {"work_id": work_id, "model": "Restormer", "notes": "OOM crash here"})
    db.insert_run("train", {"work_id": work_id, "model": "SwinIR", "tags": "paper-final"})
    db.insert_run("evaluation", {"work_id": work_id, "model": "NAFNet", "checkpoint_path": "/mnt/x/net.pth"})

    assert {r["model"] for r in db.search_runs("OOM")} == {"Restormer"}
    assert {r["model"] for r in db.search_runs("paper-final")} == {"SwinIR"}
    assert {r["model"] for r in db.search_runs("SSL2SL")} == {"Restormer", "SwinIR", "NAFNet"}
    assert {r["model"] for r in db.search_runs("net.pth")} == {"NAFNet"}
    assert db.search_runs("   ") == []
    db.close()


def test_search_runs_marks_kind():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    work_id = db.add_work(db.add_task("SR"), "W")
    db.insert_run("train", {"work_id": work_id, "model": "FindMe"})
    db.insert_run("evaluation", {"work_id": work_id, "model": "FindMe"})
    results = db.search_runs("FindMe")
    assert {r["kind"] for r in results} == {"train", "evaluation"}
    db.close()


# --- backup ------------------------------------------------------------------
def test_backup_creates_file_and_prunes():
    import time

    from dl_exp_manager.db import Database

    d = tempfile.mkdtemp()
    db = Database(os.path.join(d, "e.db"))
    paths = []
    for _ in range(4):
        p = db.backup(keep=2)
        assert p is not None and os.path.exists(p)
        paths.append(p)
        time.sleep(1.01)  # timestamp filename has second resolution
    assert len(db.list_backups()) == 2
    # the two survivors are the most recent
    assert set(db.list_backups()) == set(sorted(paths)[-2:])
    db.close()


def test_backup_keep_zero_removes_all():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    db.backup(keep=0)
    assert db.list_backups() == []
    db.close()


def test_backup_missing_db_file_returns_none():
    from dl_exp_manager.db import Database

    db = Database(os.path.join(tempfile.mkdtemp(), "e.db"))
    os.remove(db.path)
    assert db.backup() is None
    db.close()


# --- #1 Best-value highlight & #4 Path existence badge -----------------------
def test_best_value_highlight_respects_higher_is_better(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    rows = [
        dict(SAMPLE_ROW, id=1, work_id=10, metrics_json='{"PSNR": 30.0, "LPIPS": 0.20}'),
        dict(SAMPLE_ROW, id=2, work_id=10, metrics_json='{"PSNR": 32.0, "LPIPS": 0.05}'),
    ]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    psnr_col = model.column_index("metric:PSNR")
    lpips_col = model.column_index("metric:LPIPS")

    # PSNR: higher is better -> row 1 (32.0) wins
    assert model.data(model.index(0, psnr_col), Qt.ItemDataRole.FontRole) is None
    assert model.data(model.index(1, psnr_col), Qt.ItemDataRole.FontRole) is not None
    # LPIPS: lower is better -> row 1 (0.05) wins
    assert model.data(model.index(0, lpips_col), Qt.ItemDataRole.FontRole) is None
    assert model.data(model.index(1, lpips_col), Qt.ItemDataRole.FontRole) is not None


def test_best_value_not_highlighted_for_lone_row_in_work(qapp, config):
    """A Work with only one run has nothing to compare against."""
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    rows = [
        dict(SAMPLE_ROW, id=1, work_id=10, metrics_json='{"PSNR": 30.0}'),
        dict(SAMPLE_ROW, id=2, work_id=20, metrics_json='{"PSNR": 999.0}'),  # different Work, alone
    ]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    col = model.column_index("metric:PSNR")
    assert model.data(model.index(1, col), Qt.ItemDataRole.FontRole) is None


def test_best_value_grouped_per_work_not_globally(qapp, config):
    """Two different Works shouldn't have their metrics compared against each other."""
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    rows = [
        dict(SAMPLE_ROW, id=1, work_id=10, metrics_json='{"PSNR": 30.0}'),
        dict(SAMPLE_ROW, id=2, work_id=10, metrics_json='{"PSNR": 28.0}'),
        dict(SAMPLE_ROW, id=3, work_id=20, metrics_json='{"PSNR": 10.0}'),
        dict(SAMPLE_ROW, id=4, work_id=20, metrics_json='{"PSNR": 5.0}'),
    ]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    col = model.column_index("metric:PSNR")
    best = [
        model.data(model.index(r, col), Qt.ItemDataRole.FontRole) is not None
        for r in range(4)
    ]
    assert best == [True, False, True, False]  # winner in each Work, not overall


def test_path_badge_flags_missing_path(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    row = dict(SAMPLE_ROW, result_path="/definitely/does/not/exist/xyz")
    model = RunTableModel()
    model.set_content([row], build_columns(config, "SR", "train"))
    col = model.column_index("result_path")
    color = model.data(model.index(0, col), Qt.ItemDataRole.ForegroundRole)
    tooltip = model.data(model.index(0, col), Qt.ItemDataRole.ToolTipRole)
    assert color is not None
    assert "not reachable" in tooltip


def test_path_badge_does_not_flag_existing_path(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    row = dict(SAMPLE_ROW, result_path="/tmp")
    model = RunTableModel()
    model.set_content([row], build_columns(config, "SR", "train"))
    col = model.column_index("result_path")
    tooltip = model.data(model.index(0, col), Qt.ItemDataRole.ToolTipRole)
    assert "not reachable" not in tooltip


def test_path_badge_ignores_empty_path(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns
    from dl_exp_manager.qt import Qt

    row = dict(SAMPLE_ROW, result_path="")
    model = RunTableModel()
    model.set_content([row], build_columns(config, "SR", "train"))
    col = model.column_index("result_path")
    tooltip = model.data(model.index(0, col), Qt.ItemDataRole.ToolTipRole)
    assert tooltip == "(no path set)"


# --- #7 Favorites column + filter --------------------------------------------
def test_favorite_column_always_present_and_toggleable_display(qapp, config):
    from dl_exp_manager.models import RunTableModel, build_columns

    rows = [dict(SAMPLE_ROW, id=1, favorite=1), dict(SAMPLE_ROW, id=2, favorite=0)]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    assert "★" in model.headers()
    col = model.column_index("favorite")
    values = [model.data(model.index(r, col)) for r in range(2)]
    assert values == ["★", "☆"]


def test_favorite_sorts_true_first_descending(qapp, config):
    from dl_exp_manager.models import RunFilterProxy, RunTableModel, SORT_ROLE, build_columns
    from dl_exp_manager.qt import Qt

    rows = [dict(SAMPLE_ROW, id=1, favorite=0), dict(SAMPLE_ROW, id=2, favorite=1)]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    proxy = RunFilterProxy()
    proxy.setSourceModel(model)
    col = model.column_index("favorite")
    proxy.sort(col, Qt.SortOrder.DescendingOrder)
    ordered_ids = [proxy.data(proxy.index(r, model.column_index("id")), SORT_ROLE) for r in range(2)]
    assert ordered_ids == [2.0, 1.0]


def test_favorites_only_filter(qapp, config):
    from dl_exp_manager.models import RunFilterProxy, RunTableModel, build_columns

    rows = [dict(SAMPLE_ROW, id=1, favorite=1), dict(SAMPLE_ROW, id=2, favorite=0)]
    model = RunTableModel()
    model.set_content(rows, build_columns(config, "SR", "train"))
    proxy = RunFilterProxy()
    proxy.setSourceModel(model)
    assert proxy.rowCount() == 2
    proxy.set_favorites_only(True)
    assert proxy.rowCount() == 1
    proxy.set_favorites_only(False)
    assert proxy.rowCount() == 2


def test_tags_and_failure_reason_columns_available(qapp, config):
    from dl_exp_manager.models import FIELD_SPECS

    assert "tags" in FIELD_SPECS and FIELD_SPECS["tags"].header == "Tags"
    assert "failure_reason" in FIELD_SPECS
