"""options.yaml 로더/라이터 테스트 (GUI 불필요)."""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from dl_exp_manager.config_store import MetricDef, OptionsConfig, GpuDef


def make_config() -> OptionsConfig:
    return OptionsConfig(os.path.join(tempfile.mkdtemp(), "options.yaml"))


def test_creates_file_with_builtin_defaults():
    config = make_config()
    assert os.path.exists(config.path)
    assert config.errors == []
    assert set(config.task_names) >= {"SR", "DN", "Clustering", "Classification"}


def test_task_options_replace_defaults():
    config = make_config()
    # SR 은 model 을 직접 정의하므로 defaults 를 대체한다.
    assert "HAT" in config.options_for("SR", "model")
    # optimizer 는 SR 에 없으므로 defaults 를 상속한다.
    assert config.options_for("SR", "optimizer") == config.options_for("DN", "optimizer")
    # Task 마다 목록이 다르다.
    assert config.options_for("SR", "model") != config.options_for("Classification", "model")


def test_metrics_differ_per_task():
    assert [m.key for m in make_config().metrics_for("SR")] == ["PSNR", "SSIM", "LPIPS"]
    assert [m.key for m in make_config().metrics_for("Classification")] == ["Top-1", "Top-5"]


def test_metric_definition_carries_display_rules():
    psnr = make_config().metric_def("SR", "PSNR")
    assert psnr is not None
    assert (psnr.unit, psnr.digits, psnr.higher_is_better) == ("dB", 2, True)
    lpips = make_config().metric_def("SR", "LPIPS")
    assert lpips is not None and lpips.higher_is_better is False


def test_columns_differ_per_task():
    config = make_config()
    assert "LPIPS" in config.columns_for("SR", "train")
    assert "Top-1" in config.columns_for("Classification", "train")
    assert "LPIPS" not in config.columns_for("Classification", "train")


def test_custom_fields_are_non_native_options():
    config = make_config()
    assert config.custom_fields("SR") == ["scale"]
    assert config.custom_fields("DN") == ["noise_sigma"]


def test_add_option_task_scope_and_global_scope():
    config = make_config()
    assert config.add_option("SR", "model", "MyNet")
    assert "MyNet" in config.options_for("SR", "model")
    assert "MyNet" not in config.options_for("DN", "model")

    assert config.add_option(None, "optimizer", "Adan")
    assert "Adan" in config.options_for("SR", "optimizer")
    assert "Adan" in config.options_for("DN", "optimizer")


def test_add_option_is_idempotent():
    config = make_config()
    assert config.add_option("SR", "model", "Dup")
    assert not config.add_option("SR", "model", "Dup")


def test_first_task_scoped_edit_seeds_from_defaults():
    config = make_config()
    before = config.options_for("Clustering", "optimizer")   # defaults 상속
    config.add_option("Clustering", "optimizer", "Custom")
    after = config.options_for("Clustering", "optimizer")
    assert after[:-1] == before and after[-1] == "Custom"


def test_rename_and_remove_option():
    config = make_config()
    config.add_option("SR", "model", "A")
    assert config.rename_option("SR", "model", "A", "B")
    assert "B" in config.options_for("SR", "model")
    assert config.remove_option("SR", "model", "B")
    assert "B" not in config.options_for("SR", "model")
    assert not config.remove_option("SR", "model", "NotThere")


def test_metric_removal_cleans_columns():
    config = make_config()
    assert "LPIPS" in config.columns_for("SR", "train")
    assert config.remove_metric("SR", "LPIPS")
    assert "LPIPS" not in config.columns_for("SR", "train")
    assert "LPIPS" not in config.columns_for("SR", "inference")


def test_metric_rename_updates_columns():
    config = make_config()
    assert config.rename_metric("SR", "PSNR", "PSNR-Y")
    assert "PSNR-Y" in config.metric_keys("SR")
    assert "PSNR-Y" in config.columns_for("SR", "train")
    assert "PSNR" not in config.columns_for("SR", "train")


def test_update_metric_display_rules():
    config = make_config()
    assert config.update_metric("SR", "SSIM", digits=2, unit="x")
    updated = config.metric_def("SR", "SSIM")
    assert updated is not None and updated.digits == 2 and updated.unit == "x"


def test_changes_persist_to_disk():
    config = make_config()
    config.add_option("SR", "model", "Persisted")
    config.add_metric("SR", MetricDef("NIQE", digits=3, higher_is_better=False))
    reloaded = OptionsConfig(config.path)
    assert "Persisted" in reloaded.options_for("SR", "model")
    assert "NIQE" in reloaded.metric_keys("SR")


def test_servers_and_gpu_inventory():
    config = make_config()
    server = config.server("Server 1")
    assert server is not None
    assert len(server.gpus) == 4
    assert server.gpus[0].type == "H100"
    assert "H100 ×4" in server.gpu_summary


def test_server_crud():
    config = make_config()
    config.upsert_server("Server 9", "10.0.0.9", [GpuDef(0, "H100", 80), GpuDef(1, "H100", 80)])
    assert config.server("Server 9") is not None
    assert config.rename_server("Server 9", "Server 10")
    assert config.server("Server 10") is not None
    assert config.remove_server("Server 10")
    assert config.server("Server 10") is None


def test_ensure_task_adds_slot_for_new_task():
    config = make_config()
    config.ensure_task("Segmentation")
    assert "Segmentation" in config.task_names
    assert config.options_for("Segmentation", "optimizer")  # defaults 상속


def test_broken_yaml_keeps_app_usable():
    path = os.path.join(tempfile.mkdtemp(), "bad.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("tasks: [\n  broken: :\n")
    config = OptionsConfig(path, auto_create=False)
    assert config.errors, "문법 오류를 보고해야 한다"
    assert config.task_names, "기본값으로라도 동작해야 한다"


def test_non_mapping_task_is_dropped_with_error():
    path = os.path.join(tempfile.mkdtemp(), "odd.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"tasks": {"SR": ["not", "a", "mapping"]}, "defaults": {}, "servers": []}, fp)
    config = OptionsConfig(path, auto_create=False)
    assert config.errors
    assert "SR" not in config.task_names


def test_external_edit_is_picked_up_on_reload():
    config = make_config()
    with open(config.path, encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    data["tasks"]["SR"]["options"]["model"].append("HandEdited")
    with open(config.path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)
    config.load()
    assert "HandEdited" in config.options_for("SR", "model")


def test_save_keeps_backup():
    config = make_config()
    config.add_option("SR", "model", "One")
    config.add_option("SR", "model", "Two")
    assert os.path.exists(config.path + ".bak")
