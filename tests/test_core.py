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
    assert {t["name"] for t in db.list_tasks()} >= {"Super-Resolution", "Denoising", "Clustering", "Classification"}
    assert len(db.server_names()) == len(C.DEFAULT_SERVERS)
    db.close()


def test_task_work_run_crud():
    db = make_db()
    task_id = db.add_task("Super-Resolution")
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
    assert rows[0]["task_name"] == "Super-Resolution"

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
    first = db.add_task("Super-Resolution")
    assert db.add_task("Super-Resolution") == first
    work = db.add_work(first, "SSL2SL")
    assert db.add_work(first, "SSL2SL") == work
    db.close()


def test_delete_task_cascades_to_runs():
    db = make_db()
    task_id = db.add_task("Temp")
    work_id = db.add_work(task_id, "W1")
    db.insert_run("evaluation", {"work_id": work_id, "model": "M", "latency_ms": "12.5"})
    assert db.summary()["evaluation"] == 1
    db.delete_task(task_id)
    assert db.summary()["evaluation"] == 0
    db.close()


def test_dataset_registry_is_scoped_per_work():
    db = make_db()
    task_id = db.add_task("Super-Resolution")
    work1 = db.add_work(task_id, "BSR-x4")
    work2 = db.add_work(task_id, "SSL2SL")

    db.add_dataset(work1, "DIV2K", "Full Pair", "/mnt/data/DIV2K/full")
    db.add_dataset(work1, "DIV2K", "Subset A", "/mnt/data/DIV2K/subset_a")
    db.add_dataset(work2, "DIV2K", "", "/mnt/data/DIV2K/other")

    work1_sets = db.list_datasets(work1)
    assert len(work1_sets) == 2
    assert {d["variant"] for d in work1_sets} == {"Full Pair", "Subset A"}
    assert len(db.list_datasets(work2)) == 1

    full_pair = next(d for d in work1_sets if d["variant"] == "Full Pair")
    assert full_pair["path"] == "/mnt/data/DIV2K/full"
    db.close()


def test_dataset_add_is_idempotent_per_name_and_variant():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    first = db.add_dataset(work_id, "DIV2K", "Full Pair", "/a")
    again = db.add_dataset(work_id, "DIV2K", "Full Pair", "/b")
    assert first == again
    assert len(db.list_datasets(work_id)) == 1
    db.close()


def test_dataset_update_and_delete():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    dataset_id = db.add_dataset(work_id, "DIV2K", path="/old/path")

    db.update_dataset(dataset_id, "DIV2K", "Full Pair", "/new/path", "updated notes")
    row = db.get_dataset(dataset_id)
    assert row["variant"] == "Full Pair"
    assert row["path"] == "/new/path"
    assert row["notes"] == "updated notes"

    db.delete_dataset(dataset_id)
    assert db.get_dataset(dataset_id) is None
    assert db.list_datasets(work_id) == []
    db.close()


def test_dataset_registered_date_is_editable_but_defaults_to_now():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")

    auto_id = db.add_dataset(work_id, "DIV2K", path="/a")
    assert db.get_dataset(auto_id)["created_at"]  # now() 로 채워짐

    dated_id = db.add_dataset(work_id, "DF2K", path="/b", created_at="2024-01-15")
    assert db.get_dataset(dated_id)["created_at"] == "2024-01-15"

    db.update_dataset(dated_id, "DF2K", path="/b", created_at="2024-02-20")
    assert db.get_dataset(dated_id)["created_at"] == "2024-02-20"

    # created_at 을 안 주면 기존 값을 그대로 둔다 (다른 필드만 고칠 때 날짜가 안 튐)
    db.update_dataset(dated_id, "DF2K", path="/c")
    assert db.get_dataset(dated_id)["created_at"] == "2024-02-20"
    db.close()


def test_dataset_sample_count_round_trip():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    dataset_id = db.add_dataset(work_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K", sample_count=900)
    row = db.get_dataset(dataset_id)
    assert row["sample_count"] == 900

    db.update_dataset(dataset_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K", "", sample_count=1000)
    assert db.get_dataset(dataset_id)["sample_count"] == 1000

    # 지정하지 않으면(=None) 비워 둔다 - "모름"과 "0개"를 구분한다
    other_id = db.add_dataset(work_id, "DF2K")
    assert db.get_dataset(other_id)["sample_count"] is None
    db.close()


def test_dataset_device_and_abbreviation_round_trip():
    """명령어의 data=<device>/<abbreviation> 자리에 쓰는 두 필드."""
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    dataset_id = db.add_dataset(
        work_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K", device="server1", abbreviation="div2k"
    )
    row = db.get_dataset(dataset_id)
    assert row["device"] == "server1"
    assert row["abbreviation"] == "div2k"

    db.update_dataset(
        dataset_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K", device="server2", abbreviation="div2k-v2"
    )
    updated = db.get_dataset(dataset_id)
    assert updated["device"] == "server2"
    assert updated["abbreviation"] == "div2k-v2"

    # 안 주면(구버전 등록 방식) 빈 문자열 - "명령어에서 그 자리가 통째로 빠진다"의 전제
    other_id = db.add_dataset(work_id, "DF2K")
    other = db.get_dataset(other_id)
    assert other["device"] == "" and other["abbreviation"] == ""
    db.close()


def test_dataset_image_size_and_extension_round_trip():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    dataset_id = db.add_dataset(
        work_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K",
        image_size="256x256", extension="tiff",
    )
    row = db.get_dataset(dataset_id)
    assert row["image_size"] == "256x256"
    assert row["extension"] == "tiff"

    db.update_dataset(dataset_id, "DIV2K", "Full Pair", "/mnt/data/DIV2K", image_size="512x512", extension="png")
    row = db.get_dataset(dataset_id)
    assert row["image_size"] == "512x512"
    assert row["extension"] == "png"
    db.close()


def test_dataset_deleted_when_work_deleted():
    db = make_db()
    task_id = db.add_task("Super-Resolution")
    work_id = db.add_work(task_id, "W")
    db.add_dataset(work_id, "DIV2K", path="/a")
    db.delete_work(work_id)
    assert db.list_datasets(work_id) == []
    db.close()


def test_count_runs_using_dataset():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    db.add_dataset(work_id, "DIV2K", "Full Pair", "/a")
    db.insert_run("train", {"work_id": work_id, "model": "M", "dataset": "DIV2K · Full Pair"})
    db.insert_run("evaluation", {"work_id": work_id, "model": "M", "dataset": "DIV2K · Full Pair"})
    db.insert_run("train", {"work_id": work_id, "model": "M", "dataset": "Other"})

    assert db.count_runs_using_dataset(work_id, "DIV2K", "Full Pair") == 2
    assert db.count_runs_using_dataset(work_id, "DIV2K", "") == 0
    db.close()


def test_evaluation_numeric_fields():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    run_id = db.insert_run(
        "evaluation",
        {"work_id": work_id, "model": "M", "latency_ms": "41.7", "throughput_fps": "", "device": "cuda:0"},
    )
    row = db.get_run("evaluation", run_id)
    assert row["latency_ms"] == 41.7
    assert row["throughput_fps"] is None
    assert db.distinct_values("evaluation", "device") == ["cuda:0"]
    db.close()


def test_run_history_records_create_update_duplicate():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
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


def test_evaluation_source_train_run_and_epoch_round_trip():
    db = make_db()
    work_id = db.add_work(db.add_task("Super-Resolution"), "W")
    train_id = db.insert_run("train", {"work_id": work_id, "model": "Restormer"})
    eval_id = db.insert_run(
        "evaluation",
        {
            "work_id": work_id,
            "model": "Restormer",
            "source_train_run_id": train_id,
            "checkpoint_epoch": "300000",
        },
    )
    row = db.get_run("evaluation", eval_id)
    assert row["source_train_run_id"] == train_id
    assert row["checkpoint_epoch"] == "300000"
    db.close()


def test_task_scope_lists_runs_of_all_works():
    db = make_db()
    task_id = db.add_task("Super-Resolution")
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



# --- find_representative_image -----------------------------------------------
def test_find_representative_image_prefers_hinted_name():
    from dl_exp_manager.utils import find_representative_image

    d = tempfile.mkdtemp()
    open(os.path.join(d, "input.png"), "w").close()
    open(os.path.join(d, "restormer_output.png"), "w").close()
    assert find_representative_image(d) == os.path.join(d, "restormer_output.png")


def test_find_representative_image_falls_back_to_first_and_subdir():
    from dl_exp_manager.utils import find_representative_image

    d = tempfile.mkdtemp()
    open(os.path.join(d, "b.png"), "w").close()
    open(os.path.join(d, "a.png"), "w").close()
    assert find_representative_image(d) == os.path.join(d, "a.png")

    d2 = tempfile.mkdtemp()
    os.makedirs(os.path.join(d2, "visualization"))
    open(os.path.join(d2, "visualization", "x.jpg"), "w").close()
    assert find_representative_image(d2) == os.path.join(d2, "visualization", "x.jpg")



# --- log_parser: train.py 의 config.yaml / loss.log 파싱 ----------------------
_BASICSR_CONFIG = """\
name: SSL2SL_Restormer_x4
scale: 4
network_g:
  type: Restormer
datasets:
  train:
    name: DIV2K
    dataroot_gt: /mnt/data/DIV2K/train
    batch_size_per_gpu: 8
    gt_size: 256
train:
  total_iter: 300000
  optim_g:
    type: AdamW
    lr: !!float 3e-4
"""

_BASICSR_LOG = """\
2024-01-15 10:00:00,000 INFO: Start training.
2024-01-15 10:05:00,000 INFO: [epoch:  1, iter:     100, lr:(3.000e-04,)] l_pix: 5.4321e-02
2024-01-15 12:00:00,000 INFO: [epoch: 10, iter:  10,000, lr:(3.000e-04,)] l_pix: 1.2345e-02
2024-01-15 12:00:05,000 INFO: Validation Set5
\t # psnr: 30.1200\tBest: 30.1200 @ 10000 iter
\t # ssim: 0.8800\tBest: 0.8800 @ 10000 iter
2024-01-15 14:00:00,000 INFO: [epoch: 20, iter:  20,000, lr:(3.000e-04,)] l_pix: 9.8760e-03
2024-01-15 14:00:05,000 INFO: Validation Set5
\t # psnr: 32.4123\tBest: 32.4123 @ 20000 iter
\t # ssim: 0.9012\tBest: 0.9012 @ 20000 iter
2024-01-15 16:00:00,000 INFO: End of training.
"""


def test_parse_train_config_reads_basicsr_style_yaml():
    from dl_exp_manager.log_parser import parse_train_config

    path = os.path.join(tempfile.mkdtemp(), "config.yml")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_BASICSR_CONFIG)

    fields = parse_train_config(path)
    assert fields["model"] == "Restormer"
    assert fields["dataset"] == "DIV2K"
    assert fields["dataset_path"] == "/mnt/data/DIV2K/train"
    assert fields["batch_size"] == "8"
    assert fields["crop_size"] == "256"
    assert fields["lr"] == "0.0003"
    assert fields["optimizer"] == "AdamW"
    assert fields["epochs"] == "300000"
    assert fields["scale"] == "4"


_DATA_TRAIN_CONFIG = """\
name: my_exp
model:
  type: NAFNet
data:
  train:
    name: MyDataset
    dataroot: /mnt/data/mydata/train
    batch_size: 16
    imagesize: 256
train:
  total_epoch: 200
  optim:
    type: AdamW
    lr: 0.0002
"""


def test_parse_train_config_reads_data_train_nested_yaml():
    """`datasets:` 대신 `data:` 를 쓰고, crop 크기를 `imagesize` 로 부르는 스키마도 잡아야 한다."""
    from dl_exp_manager.log_parser import parse_train_config

    path = os.path.join(tempfile.mkdtemp(), "config.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_DATA_TRAIN_CONFIG)

    fields = parse_train_config(path)
    assert fields["model"] == "NAFNet"
    assert fields["dataset"] == "MyDataset"
    assert fields["dataset_path"] == "/mnt/data/mydata/train"
    assert fields["batch_size"] == "16"
    assert fields["crop_size"] == "256"
    assert fields["lr"] == "0.0002"
    assert fields["optimizer"] == "AdamW"
    assert fields["epochs"] == "200"



def test_parse_loss_log_extracts_curve_and_latest_metrics():
    from dl_exp_manager.log_parser import parse_loss_log

    path = os.path.join(tempfile.mkdtemp(), "loss.log")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_BASICSR_LOG)

    result = parse_loss_log(path)

    # 최종(가장 마지막) 검증 지표만 남는다
    assert result.latest_metrics == {"psnr": 32.4123, "ssim": 0.9012}

    # l_pix 곡선: iter 순으로 3개 학습 손실 포인트
    l_pix_points = [(i, v["l_pix"]) for i, v in result.points if "l_pix" in v]
    assert l_pix_points == [(100, 5.4321e-02), (10000, 1.2345e-02), (20000, 9.876e-03)]

    # psnr 곡선: 검증 시점(@ N iter)에 찍힌 포인트도 곡선에 들어간다
    psnr_points = [(i, v["psnr"]) for i, v in result.points if "psnr" in v]
    assert psnr_points == [(10000, 30.12), (20000, 32.4123)]

    # 첫/마지막 타임스탬프 차이로 소요 시간을 추정한다 (10:00 -> 16:00 = 6시간)
    assert result.duration_sec == 6 * 3600



def test_canonical_metric_name():
    from dl_exp_manager.log_parser import canonical_metric_name

    assert canonical_metric_name("psnr") == "PSNR"
    assert canonical_metric_name("Top-5") == "Top-5"
    assert canonical_metric_name("SomeCustomThing") == "SomeCustomThing"


# --- command_builder: Task 템플릿 -> 실행 명령어 -------------------------------
def test_render_command_fills_placeholders_and_drops_empty_tokens():
    from dl_exp_manager.command_builder import render_command

    template = (
        "python train.py algo={task_lower}/{algo} data={task_lower}/{dataset}"
        " model={task_lower}/{model} +batch_size={batch_size} +crop_size={crop_size}"
    )
    result = render_command(
        template,
        {
            "task_lower": "dn", "algo": "noise2noise", "dataset": "dataset1",
            "model": "UNet", "batch_size": "16", "crop_size": "",
        },
    )
    # 빈 값(crop_size)은 `+crop_size=` 를 남기지 않고 토큰째 사라져야 한다.
    assert result.text == (
        "python train.py algo=dn/noise2noise data=dn/dataset1 model=dn/UNet +batch_size=16"
    )
    assert result.dropped == ["crop_size"]
    assert result.unknown == []


def test_render_command_reports_unknown_placeholders():
    from dl_exp_manager.command_builder import render_command

    result = render_command("python train.py +foo={nope} model={model}", {"model": "UNet"})
    assert result.text == "python train.py model=UNet"
    assert result.unknown == ["nope"]


def test_render_command_quotes_values_with_spaces():
    from dl_exp_manager.command_builder import render_command

    result = render_command(
        "python eval.py +ckpt={checkpoint_path}", {"checkpoint_path": "/mnt/my runs/net.pth"}
    )
    assert result.text == "python eval.py +ckpt='/mnt/my runs/net.pth'"


def test_render_command_empty_template_is_safe():
    from dl_exp_manager.command_builder import render_command

    assert render_command("", {"model": "UNet"}).text == ""
    assert not render_command("   ", {})


def test_placeholders_in_lists_names_once_in_order():
    from dl_exp_manager.command_builder import placeholders_in

    assert placeholders_in("a={x} b={y} c={x}") == ["x", "y"]


# --- <name>: params.yaml 이 정하는 인자 표기법 ---------------------------------
def test_angle_placeholder_renders_the_whole_argument():
    from dl_exp_manager.command_builder import ParamStyle, render_command

    style = ParamStyle.from_dict(
        {"style": {"prefix": "+", "separator": "="}, "params": {"epochs": {"name": "max_epoch"}}}
    )
    result = render_command(
        "python train.py model={model} <batch_size> <epochs> <crop_size>",
        {"model": "UNet", "batch_size": "16", "epochs": "200", "crop_size": ""},
        style,
    )
    # 등록 안 된 batch_size 는 공통 스타일 + 제 이름으로, epochs 는 max_epoch 으로.
    assert result.text == "python train.py model=UNet +batch_size=16 +max_epoch=200"
    assert result.dropped == ["crop_size"]


def test_param_style_switches_every_task_at_once():
    """`+batch_size=16` -> `+batchsize=16` 은 params.yaml 한 줄로 끝나야 한다."""
    from dl_exp_manager.command_builder import ParamStyle, render_command

    template = "python train.py <batch_size>"
    hydra = ParamStyle.from_dict({"params": {"batch_size": {"name": "batchsize"}}})
    assert render_command(template, {"batch_size": "16"}, hydra).text == (
        "python train.py +batchsize=16"
    )

    argparse = ParamStyle.from_dict(
        {"style": {"prefix": "--", "separator": " "},
         "params": {"batch_size": {"name": "batch-size"}}}
    )
    assert render_command(template, {"batch_size": "16"}, argparse).text == (
        "python train.py --batch-size 16"
    )


def test_param_spec_overrides_and_template_beat_the_common_style():
    from dl_exp_manager.command_builder import ParamStyle, render_command

    style = ParamStyle.from_dict(
        {
            "style": {"prefix": "+", "separator": "="},
            "params": {
                "lr": {"prefix": "", "name": "optim.lr"},      # hydra override (기존 키 덮어쓰기)
                "gpus": {"template": "--gpu-ids {value}"},     # 아예 다른 모양
                "epochs": "max_epoch",                          # 이름만 바꾸는 지름길
            },
        }
    )
    result = render_command(
        "python train.py <lr> <gpus> <epochs>",
        {"lr": "0.0003", "gpus": "2", "epochs": "200"},
        style,
    )
    assert result.text == "python train.py optim.lr=0.0003 --gpu-ids 2 +max_epoch=200"


def test_angle_placeholder_quotes_values_with_spaces():
    from dl_exp_manager.command_builder import render_command

    result = render_command("python eval.py <result_path>", {"result_path": "/mnt/my runs"})
    assert result.text == "python eval.py +result_path='/mnt/my runs'"


def test_default_param_style_needs_no_config():
    """params.yaml 이 없어도 <name> 은 동작해야 한다(공통 스타일 + 제 이름)."""
    from dl_exp_manager.command_builder import render_command

    assert render_command("t <scale>", {"scale": "x4"}).text == "t +scale=x4"


def test_placeholders_in_counts_both_forms():
    from dl_exp_manager.command_builder import placeholders_in

    assert placeholders_in("a={x} <y> c={x} <y>") == ["x", "y"]


def test_shell_redirection_is_not_mistaken_for_a_placeholder():
    from dl_exp_manager.command_builder import render_command

    result = render_command("python train.py <model> 2>&1 | tee log.txt", {"model": "UNet"})
    assert result.text == "python train.py +model=UNet 2>&1 | tee log.txt"
    assert result.unknown == []


# --- sample_data: 예시 데이터 생성기 --------------------------------------------
def _make_sample_env():
    from dl_exp_manager.config_store import OptionsConfig

    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"), seed=False)
    config = OptionsConfig(os.path.join(tmp, "config", "options.yaml"))
    return tmp, db, config


def test_populate_fills_all_four_tasks_with_linked_train_and_eval_runs():
    from dl_exp_manager.sample_data import populate

    tmp, db, config = _make_sample_env()
    added = populate(db, config)
    assert added > 50  # "많이" - 최소한 수동으로 넣던 5건보다 훨씬 많아야 한다

    summary = db.summary()
    assert summary["train"] > 25
    assert summary["evaluation"] > 15
    assert {t["name"] for t in db.list_tasks()} == {
        "Super-Resolution", "Denoising", "Clustering", "Classification",
    }

    evaluations = db.list_evaluation_runs()
    linked = [r for r in evaluations if r["source_train_run_id"]]
    assert linked  # 최소 하나는 Train Run 에 연결돼 있어야 한다("이 학습으로 평가 만들기" 재현)
    for row in linked:
        source = db.get_run("train", row["source_train_run_id"])
        assert source is not None  # 링크가 실제로 존재하는 Train Run 을 가리켜야 한다

    db.close()


def test_populate_generates_commands_from_the_real_task_templates():
    """손으로 쓴 문자열이 아니라 config.command_template() 로 실제 생성된 명령어여야 한다."""
    from dl_exp_manager.sample_data import populate

    tmp, db, config = _make_sample_env()
    populate(db, config)

    denoising_runs = [r for r in db.list_train_runs() if r["task_name"] == "Denoising"]
    assert denoising_runs
    for row in denoising_runs:
        if row["status"] == C.STATUS_QUEUED:
            continue  # queued 는 batch_size 등이 비어 있을 수 있어 명령어가 짧아도 된다
        assert row["exec_command"].startswith("python train.py")
        assert "model=dn/" in row["exec_command"]  # short: dn 이 반영돼야 한다
        assert "algo=dn/noise2noise" in row["exec_command"]  # Denoising 은 algo 옵션이 있다

    db.close()




def test_populate_with_local_assets_writes_real_parseable_result_folders():
    from dl_exp_manager.log_parser import parse_loss_log, parse_train_config
    from dl_exp_manager.sample_data import populate
    from dl_exp_manager.utils import find_representative_image, scan_result_folder

    tmp, db, config = _make_sample_env()
    results_root = os.path.join(tmp, "results")
    populate(db, config, with_local_assets=True, results_root=results_root)

    found_any = False
    for row in db.list_train_runs():
        if not row["result_path"].startswith(results_root):
            continue
        found_any = True
        scanned = scan_result_folder(row["result_path"])
        assert scanned["config"] and scanned["log"]

        parsed = parse_train_config(scanned["config"])
        assert parsed.get("model") == row["model"]

        log_result = parse_loss_log(scanned["log"])
        assert len(log_result.points) > 5
        assert log_result.latest_metrics  # 검증 체크포인트가 최소 하나는 파싱돼야 한다

        image = find_representative_image(row["result_path"])
        assert image and os.path.isfile(image)
        with open(image, "rb") as fp:
            assert fp.read(8) == b"\x89PNG\r\n\x1a\n"  # 진짜 PNG 여야 한다

    assert found_any  # with_local_assets=True 인데 실제 폴더가 하나도 없으면 이 옵션이 죽어 있는 것
    db.close()


