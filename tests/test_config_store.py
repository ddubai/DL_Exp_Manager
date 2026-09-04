"""options.yaml 로더/라이터 테스트 (GUI 불필요)."""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from dl_exp_manager.config_store import MetricDef, OptionsConfig, GpuDef


def make_config() -> OptionsConfig:
    return OptionsConfig(os.path.join(tempfile.mkdtemp(), "config", "options.yaml"))


def test_creates_split_files_with_builtin_defaults():
    config = make_config()
    assert config.errors == []
    assert os.path.exists(config.path)
    assert os.path.exists(config.servers_path)
    assert os.path.exists(config.defaults_path)
    assert os.path.isdir(config.tasks_dir)
    assert set(config.task_names) >= {"SR", "DN", "Clustering", "Classification"}


def test_each_task_gets_its_own_file():
    config = make_config()
    for name in ("SR", "DN", "Classification"):
        path = config.task_path(name)
        assert os.path.exists(path)
        assert os.path.basename(path) == f"{name}.yaml"


def test_split_files_stay_short():
    """한 파일이 길어서 못 보겠다는 게 분할의 이유다."""
    config = make_config()
    for path in config.watch_paths():
        with open(path, encoding="utf-8") as fp:
            assert len(fp.readlines()) < 60, f"{path} 가 너무 깁니다"


def test_watch_paths_covers_every_file():
    config = make_config()
    paths = set(config.watch_paths())
    assert config.path in paths
    assert config.servers_path in paths
    assert config.defaults_path in paths
    assert config.task_path("SR") in paths


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
    assert "LPIPS" not in config.columns_for("SR", "evaluation")


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
    """정의가 깨졌으면 내장 Task 로 덮어쓰지 않는다.

    같은 이름의 내장 정의를 채워 넣으면 사용자의 SR 을 가리게 되고,
    다음 UI 편집이 원본 파일을 덮어써 버린다.
    """
    path = os.path.join(tempfile.mkdtemp(), "odd.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"tasks": {"SR": ["not", "a", "mapping"]}, "defaults": {}, "servers": []}, fp)
    config = OptionsConfig(path, auto_create=False)
    assert config.errors
    assert "SR" not in config.task_names


def test_external_edit_is_picked_up_on_reload():
    """에디터로 tasks/SR.yaml 을 고치면 다시 읽었을 때 반영돼야 한다."""
    config = make_config()
    path = config.task_path("SR")
    with open(path, encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    data["options"]["model"].append("HandEdited")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)
    config.load()
    assert "HandEdited" in config.options_for("SR", "model")


def test_save_keeps_backup():
    config = make_config()
    config.add_option("SR", "model", "One")
    config.add_option("SR", "model", "Two")
    assert os.path.exists(config.task_path("SR") + ".bak")


# --- 기능별 분할 -------------------------------------------------------------
def _mtimes(config: OptionsConfig) -> dict[str, float]:
    return {p: os.path.getmtime(p) for p in config.watch_paths()}


def _changed(config: OptionsConfig, before: dict[str, float]) -> set[str]:
    after = _mtimes(config)
    return {os.path.basename(p) for p in after if after[p] != before.get(p)}


def test_task_edit_touches_only_that_task_file():
    config = make_config()
    time.sleep(0.02)
    before = _mtimes(config)
    config.add_option("SR", "model", "OnlyHere")
    assert _changed(config, before) == {"SR.yaml"}


def test_global_option_edit_touches_only_defaults_file():
    config = make_config()
    time.sleep(0.02)
    before = _mtimes(config)
    config.add_option(None, "optimizer", "Adan")
    assert _changed(config, before) == {"defaults.yaml"}


def test_server_edit_touches_only_servers_file():
    config = make_config()
    time.sleep(0.02)
    before = _mtimes(config)
    config.upsert_server("Server 9", "10.0.0.9", [GpuDef(0, "H200", 141)])
    assert _changed(config, before) == {"servers.yaml"}


def test_new_task_creates_its_file():
    config = make_config()
    config.ensure_task("Segmentation")
    assert os.path.exists(config.task_path("Segmentation"))
    assert "Segmentation" in OptionsConfig(config.path).task_names


def test_remove_task_deletes_its_file():
    config = make_config()
    path = config.task_path("Clustering")
    assert config.remove_task("Clustering")
    assert not os.path.exists(path)
    assert "Clustering" not in OptionsConfig(config.path).task_names


def test_legacy_single_file_is_split_automatically():
    """예전처럼 options.yaml 한 파일에 전부 들어 있으면 나눠 준다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    path = os.path.join(directory, "options.yaml")
    legacy = {
        "version": 2,
        "servers": [{"name": "Old1", "host": "1.1.1.1", "gpus": [{"index": 0, "type": "V100"}]}],
        "defaults": {"optimizer": ["AdamW"]},
        "tasks": {
            "SR": {
                "label": "SR",
                "options": {"model": ["LegacyNet"]},
                "metrics": [{"key": "PSNR", "digits": 2}],
                "columns": {"train": ["status", "model", "PSNR"]},
            }
        },
    }
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(legacy, fp, allow_unicode=True, sort_keys=False)

    config = OptionsConfig(path)
    assert os.path.exists(os.path.join(directory, "servers.yaml"))
    assert os.path.exists(os.path.join(directory, "defaults.yaml"))
    assert os.path.exists(os.path.join(directory, "tasks", "SR.yaml"))
    # 내용이 그대로 살아 있어야 한다
    assert config.options_for("SR", "model") == ["LegacyNet"]
    assert config.metric_keys("SR") == ["PSNR"]
    assert [s.name for s in config.servers] == ["Old1"]
    assert config.options_for("SR", "optimizer") == ["AdamW"]
    # 원본은 백업된다
    assert os.path.exists(path + ".bak")


def test_split_runs_only_once():
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    path = os.path.join(directory, "options.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"version": 2, "tasks": {"SR": {"options": {"model": ["A"]}}}}, fp)

    OptionsConfig(path)
    second = OptionsConfig(path)
    assert second.errors == []
    assert second.task_names == ["SR"]


def test_missing_servers_yaml_falls_back_without_writing_it():
    """servers.yaml 은 실서버 정보라 gitignore 대상. 없어도 앱은 안 죽고, 알아서 만들지도 않는다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    path = os.path.join(directory, "options.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"version": 2}, fp)
    # servers.yaml 은 없고 template 만 있는, 실제 저장소를 클론한 상태를 흉내낸다.
    with open(os.path.join(directory, "servers.template.yaml"), "w", encoding="utf-8") as fp:
        yaml.safe_dump({"servers": [{"name": "Example", "host": "0.0.0.0", "gpus": []}]}, fp)

    config = OptionsConfig(path)

    assert not os.path.exists(os.path.join(directory, "servers.yaml"))
    assert [s.name for s in config.servers] == ["Server 1", "Server 2", "Server 3", "Server 4"]
    assert any("servers.template.yaml" in e for e in config.errors)


def test_broken_task_file_does_not_break_the_rest():
    config = make_config()
    with open(config.task_path("DN"), "w", encoding="utf-8") as fp:
        fp.write("options: [\n  broken: :\n")
    reloaded = OptionsConfig(config.path)
    assert any("DN.yaml" in e for e in reloaded.errors)
    assert "SR" in reloaded.task_names
    assert reloaded.metric_keys("SR") == ["PSNR", "SSIM", "LPIPS"]


def test_task_file_name_key_wins_over_filename():
    config = make_config()
    path = os.path.join(config.tasks_dir, "custom_file.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"name": "Detection", "options": {"model": ["YOLO"]}}, fp)
    reloaded = OptionsConfig(config.path)
    assert "Detection" in reloaded.task_names
    assert reloaded.options_for("Detection", "model") == ["YOLO"]


# --- Missing YAML backend: save must degrade gracefully, not crash --------
def test_save_without_backend_does_not_raise(monkeypatch):
    """Deleting/editing something with no YAML lib installed must not crash the app.

    Regression test: previously `_write` raised a bare OSError that propagated
    out of `save()` (and out of things like ServerStatusPanel.remove_server),
    surfacing as an unhandled Korean error instead of a clean message.
    """
    import dl_exp_manager.config_store as config_store

    config = make_config()  # created normally, with a real backend
    assert config.remove_server("Server 1")  # in-memory change succeeds

    # Now simulate the environment having no usable YAML library at all.
    monkeypatch.setattr(config_store, "_BACKEND", "none")
    monkeypatch.setattr(config, "_yaml", None)

    ok = config.add_option("SR", "model", "WontPersist")
    assert ok is True  # the in-memory list is still updated...
    assert "WontPersist" in config.options_for("SR", "model")
    assert config.last_save_error is not None  # ...but the failure is recorded
    assert "YAML" in config.last_save_error or "yaml" in config.last_save_error


def test_save_return_value_reflects_success():
    config = make_config()
    assert config.save() is True  # nothing dirty -> trivially true
    assert config.add_option("SR", "model", "X") is True
    assert config.last_save_error is None
