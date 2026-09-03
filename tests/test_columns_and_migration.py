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
    assert values["GPU"] == "GPU 0,1"
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
    assert values["실행 시간"].startswith("~")


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
    conn.commit()
    conn.close()


def test_v1_database_migrates_and_keeps_data():
    from dl_exp_manager.db import Database

    path = os.path.join(tempfile.mkdtemp(), "old.db")
    _make_v1_db(path)

    db = Database(path, seed=False)
    assert db.migrated_from == 1
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 2

    rows = db.list_train_runs()
    assert len(rows) == 1
    assert rows[0]["model"] == "Restormer"
    assert rows[0]["metrics_json"] == '{"PSNR": 31.2}'
    assert rows[0]["gpu_indices"] == ""
    assert rows[0]["extra_json"] == "{}"
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
    db.insert_run("inference", {"work_id": work_id, "model": "Old"})
    assert db.count_runs_using("model", "Old") == 2
    assert db.rename_value_in_runs("model", "Old", "New") == 2
    assert db.count_runs_using("model", "Old") == 0
    assert db.count_runs_using("model", "New") == 2
    db.close()
