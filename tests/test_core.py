"""GUI 없이 돌아가는 핵심 로직 테스트.

    pip install pytest && pytest -q
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_exp_manager import constants as C
from dl_exp_manager.db import Database
from dl_exp_manager.utils import (
    dumps_metrics,
    format_duration,
    format_number,
    loads_metrics,
    open_in_file_manager,
    parse_duration,
    rows_to_tsv,
    write_csv,
)


def make_db() -> Database:
    return Database(os.path.join(tempfile.mkdtemp(), "experiments.db"))


# --- utils ------------------------------------------------------------------
def test_parse_duration_formats():
    assert parse_duration("3h 20m") == 12000
    assert parse_duration("01:30:00") == 5400
    assert parse_duration("12:30") == 750
    assert parse_duration("5400") == 5400
    assert parse_duration("1d2h") == 93600
    assert parse_duration("nonsense") is None
    assert parse_duration("") is None


def test_format_duration_roundtrip():
    assert format_duration(5400) == "01:30:00"
    assert format_duration(93600) == "1d 02:00:00"
    assert format_duration(None) == ""


def test_format_number():
    assert format_number(31.240000) == "31.24"
    assert format_number(0.0000123) == "1.230e-05"
    assert format_number("n/a") == "n/a"
    assert format_number(None) == ""


def test_metrics_json_roundtrip():
    raw = dumps_metrics({"PSNR": "31.24", " SSIM ": 0.9, "": 1})
    parsed = loads_metrics(raw)
    assert parsed == {"PSNR": 31.24, "SSIM": 0.9}
    assert loads_metrics("not json") == {}


def test_open_in_file_manager_rejects_missing_path():
    ok, message = open_in_file_manager("")
    assert not ok and "empty" in message
    ok, _ = open_in_file_manager("/definitely/does/not/exist/xyz")
    assert not ok


def test_csv_and_tsv(tmp_path=None):
    target = os.path.join(tempfile.mkdtemp(), "out.csv")
    written = write_csv(target, ["a", "b"], [[1, 2], [3, None]])
    assert written == 2
    with open(target, encoding="utf-8-sig") as fp:
        assert fp.read().splitlines()[0] == "a,b"
    assert rows_to_tsv(["a"], [["x"]]) == "a\nx"


# --- db ---------------------------------------------------------------------
def test_seed_creates_tasks_and_servers():
    db = make_db()
    assert {t["name"] for t in db.list_tasks()} >= {"SR", "DN", "Clustering", "Classification"}
    assert len(db.server_names()) == len(C.DEFAULT_SERVERS)
    db.close()


def test_task_work_run_crud():
    db = make_db()
    task_id = db.add_task("SR")
    work_id = db.add_work(task_id, "SSL2SL")

    run_id = db.insert_run(
        "train",
        {
            "work_id": work_id,
            "server": "Server 1",
            "model": "Restormer",
            "duration_sec": "5400",
            "metrics_json": {"PSNR": 31.2},
            "status": C.STATUS_RUNNING,
        },
    )
    rows = db.list_train_runs(work_id=work_id)
    assert len(rows) == 1
    assert rows[0]["duration_sec"] == 5400.0
    assert loads_metrics(rows[0]["metrics_json"]) == {"PSNR": 31.2}
    assert rows[0]["task_name"] == "SR"

    db.update_run("train", run_id, {**rows[0], "model": "SwinIR", "duration_sec": None})
    assert db.get_run("train", run_id)["model"] == "SwinIR"
    assert db.get_run("train", run_id)["duration_sec"] is None

    assert db.counts_for_work(work_id) == (1, 0)
    assert "Server 1" in db.running_by_server()

    dup_id = db.duplicate_run("train", run_id)
    assert dup_id != run_id and db.counts_for_work(work_id)[0] == 2

    assert db.delete_runs("train", [run_id, dup_id]) == 2
    assert db.counts_for_work(work_id) == (0, 0)
    db.close()


def test_duplicate_names_are_not_inserted_twice():
    db = make_db()
    first = db.add_task("SR")
    assert db.add_task("SR") == first
    work = db.add_work(first, "SSL2SL")
    assert db.add_work(first, "SSL2SL") == work
    db.close()


def test_delete_task_cascades_to_runs():
    db = make_db()
    task_id = db.add_task("Temp")
    work_id = db.add_work(task_id, "W1")
    db.insert_run("inference", {"work_id": work_id, "model": "M", "latency_ms": "12.5"})
    assert db.summary()["inference"] == 1
    db.delete_task(task_id)
    assert db.summary()["inference"] == 0
    db.close()


def test_inference_numeric_fields():
    db = make_db()
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run(
        "inference",
        {"work_id": work_id, "model": "M", "latency_ms": "41.7", "throughput_fps": "", "device": "cuda:0"},
    )
    row = db.get_run("inference", run_id)
    assert row["latency_ms"] == 41.7
    assert row["throughput_fps"] is None
    assert db.distinct_values("inference", "device") == ["cuda:0"]
    db.close()


def test_run_history_records_create_update_duplicate():
    db = make_db()
    work_id = db.add_work(db.add_task("SR"), "W")
    run_id = db.insert_run("train", {"work_id": work_id, "model": "Restormer"})

    history = db.list_history("train", run_id)
    assert len(history) == 1 and history[0]["action"] == "created"

    db.update_run("train", run_id, {"work_id": work_id, "model": "SwinIR"})
    history = db.list_history("train", run_id)
    assert history[0]["action"] == "updated"
    assert "model" in history[0]["detail"] and "Restormer" in history[0]["detail"]

    # Saving with no actual change should not add a no-op history entry.
    before = len(db.list_history("train", run_id))
    db.update_run("train", run_id, {"work_id": work_id, "model": "SwinIR"})
    assert len(db.list_history("train", run_id)) == before

    dup_id = db.duplicate_run("train", run_id)
    dup_history = db.list_history("train", dup_id)
    assert dup_history[0]["action"] == "duplicated"
    assert f"#{run_id}" in dup_history[0]["detail"]

    db.delete_runs("train", [run_id])
    assert db.list_history("train", run_id) == []
    db.close()


def test_inference_source_train_run_and_epoch_round_trip():
    db = make_db()
    work_id = db.add_work(db.add_task("SR"), "W")
    train_id = db.insert_run("train", {"work_id": work_id, "model": "Restormer"})
    infer_id = db.insert_run(
        "inference",
        {
            "work_id": work_id,
            "model": "Restormer",
            "source_train_run_id": train_id,
            "checkpoint_epoch": "300000",
        },
    )
    row = db.get_run("inference", infer_id)
    assert row["source_train_run_id"] == train_id
    assert row["checkpoint_epoch"] == "300000"
    db.close()


def test_task_scope_lists_runs_of_all_works():
    db = make_db()
    task_id = db.add_task("SR")
    for name in ("A", "B"):
        work_id = db.add_work(task_id, name)
        db.insert_run("train", {"work_id": work_id, "model": name})
    assert len(db.list_train_runs(task_id=task_id)) == 2
    db.close()


# --- #9 scan_result_folder / tail_file ---------------------------------------
def test_scan_result_folder_finds_config_and_log():
    from dl_exp_manager.utils import scan_result_folder

    d = tempfile.mkdtemp()
    open(os.path.join(d, "config.yml"), "w").write("model: X\n")
    open(os.path.join(d, "train.log"), "w").write("line\n")
    open(os.path.join(d, "checkpoint.pth"), "wb").write(b"x")

    found = scan_result_folder(d)
    assert os.path.basename(found["config"]) == "config.yml"
    assert os.path.basename(found["log"]) == "train.log"


def test_scan_result_folder_does_not_recurse():
    from dl_exp_manager.utils import scan_result_folder

    d = tempfile.mkdtemp()
    sub = os.path.join(d, "sub")
    os.makedirs(sub)
    open(os.path.join(sub, "config.yml"), "w").write("x")

    found = scan_result_folder(d)
    assert found["config"] is None


def test_scan_result_folder_missing_dir_is_safe():
    from dl_exp_manager.utils import scan_result_folder

    assert scan_result_folder("/does/not/exist") == {"config": None, "log": None}


def test_tail_file_returns_last_n_lines():
    from dl_exp_manager.utils import tail_file

    d = tempfile.mkdtemp()
    path = os.path.join(d, "big.log")
    with open(path, "w") as fp:
        fp.write("\n".join(f"line {i}" for i in range(1000)))

    tail = tail_file(path, max_lines=3)
    assert tail == "line 997\nline 998\nline 999"


def test_tail_file_missing_file_reports_error_not_exception():
    from dl_exp_manager.utils import tail_file

    result = tail_file("/does/not/exist.log")
    assert "could not read file" in result


# --- #10 Markdown / HTML report export ---------------------------------------
def test_render_markdown_report_escapes_pipes_and_lists_rows():
    from dl_exp_manager.utils import render_markdown_report

    md = render_markdown_report("Train Runs Report", ["id", "model"], [["1", "A|B"], ["2", "SwinIR"]])
    assert "# Train Runs Report" in md
    assert "2 row(s)" in md
    assert "| id | model |" in md
    assert "| 1 | A\\|B |" in md
    assert "| 2 | SwinIR |" in md


def test_render_markdown_report_handles_no_rows():
    from dl_exp_manager.utils import render_markdown_report

    md = render_markdown_report("Empty", ["id"], [])
    assert "0 row(s)" in md
    assert "(no rows)" in md


def test_render_html_report_escapes_html_and_lists_rows():
    from dl_exp_manager.utils import render_html_report

    out = render_html_report("Train Runs Report", ["id", "model"], [["1", "<script>x</script>"]])
    assert "<title>Train Runs Report</title>" in out
    assert "<th>id</th>" in out
    assert "&lt;script&gt;x&lt;/script&gt;" in out
    assert "<script>x</script>" not in out


def test_render_html_report_handles_no_rows():
    from dl_exp_manager.utils import render_html_report

    out = render_html_report("Empty", ["id"], [])
    assert "(no rows)" in out
