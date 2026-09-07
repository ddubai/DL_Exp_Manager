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
    assert set(config.task_names) >= {"Super-Resolution", "Denoising", "Clustering", "Classification"}


def test_each_task_gets_its_own_file():
    config = make_config()
    for name in ("Super-Resolution", "Denoising", "Classification"):
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
    assert config.task_path("Super-Resolution") in paths


def test_task_options_replace_defaults():
    config = make_config()
    # SR 은 model 을 직접 정의하므로 defaults 를 대체한다.
    assert "HAT" in config.options_for("Super-Resolution", "model")
    # optimizer 는 SR 에 없으므로 defaults 를 상속한다.
    assert config.options_for("Super-Resolution", "optimizer") == config.options_for("Denoising", "optimizer")
    # Task 마다 목록이 다르다.
    assert config.options_for("Super-Resolution", "model") != config.options_for("Classification", "model")


def test_metrics_differ_per_task():
    assert [m.key for m in make_config().metrics_for("Super-Resolution")] == ["PSNR", "SSIM", "LPIPS"]
    assert [m.key for m in make_config().metrics_for("Classification")] == ["Top-1", "Top-5"]


def test_metric_definition_carries_display_rules():
    psnr = make_config().metric_def("Super-Resolution", "PSNR")
    assert psnr is not None
    assert (psnr.unit, psnr.digits, psnr.higher_is_better) == ("dB", 2, True)
    lpips = make_config().metric_def("Super-Resolution", "LPIPS")
    assert lpips is not None and lpips.higher_is_better is False


def test_columns_differ_per_task():
    config = make_config()
    assert "LPIPS" in config.columns_for("Super-Resolution", "train")
    assert "Top-1" in config.columns_for("Classification", "train")
    assert "LPIPS" not in config.columns_for("Classification", "train")


def test_custom_fields_are_non_native_options():
    config = make_config()
    assert config.custom_fields("Super-Resolution") == ["scale"]
    assert config.custom_fields("Denoising") == ["noise_sigma"]


def test_add_option_task_scope_and_global_scope():
    config = make_config()
    assert config.add_option("Super-Resolution", "model", "MyNet")
    assert "MyNet" in config.options_for("Super-Resolution", "model")
    assert "MyNet" not in config.options_for("Denoising", "model")

    assert config.add_option(None, "optimizer", "Adan")
    assert "Adan" in config.options_for("Super-Resolution", "optimizer")
    assert "Adan" in config.options_for("Denoising", "optimizer")


def test_add_option_is_idempotent():
    config = make_config()
    assert config.add_option("Super-Resolution", "model", "Dup")
    assert not config.add_option("Super-Resolution", "model", "Dup")


def test_first_task_scoped_edit_seeds_from_defaults():
    config = make_config()
    before = config.options_for("Clustering", "optimizer")   # defaults 상속
    config.add_option("Clustering", "optimizer", "Custom")
    after = config.options_for("Clustering", "optimizer")
    assert after[:-1] == before and after[-1] == "Custom"


def test_rename_and_remove_option():
    config = make_config()
    config.add_option("Super-Resolution", "model", "A")
    assert config.rename_option("Super-Resolution", "model", "A", "B")
    assert "B" in config.options_for("Super-Resolution", "model")
    assert config.remove_option("Super-Resolution", "model", "B")
    assert "B" not in config.options_for("Super-Resolution", "model")
    assert not config.remove_option("Super-Resolution", "model", "NotThere")


def test_metric_removal_cleans_columns():
    config = make_config()
    assert "LPIPS" in config.columns_for("Super-Resolution", "train")
    assert config.remove_metric("Super-Resolution", "LPIPS")
    assert "LPIPS" not in config.columns_for("Super-Resolution", "train")
    assert "LPIPS" not in config.columns_for("Super-Resolution", "evaluation")


def test_metric_rename_updates_columns():
    config = make_config()
    assert config.rename_metric("Super-Resolution", "PSNR", "PSNR-Y")
    assert "PSNR-Y" in config.metric_keys("Super-Resolution")
    assert "PSNR-Y" in config.columns_for("Super-Resolution", "train")
    assert "PSNR" not in config.columns_for("Super-Resolution", "train")


def test_update_metric_display_rules():
    config = make_config()
    assert config.update_metric("Super-Resolution", "SSIM", digits=2, unit="x")
    updated = config.metric_def("Super-Resolution", "SSIM")
    assert updated is not None and updated.digits == 2 and updated.unit == "x"


def test_changes_persist_to_disk():
    config = make_config()
    config.add_option("Super-Resolution", "model", "Persisted")
    config.add_metric("Super-Resolution", MetricDef("NIQE", digits=3, higher_is_better=False))
    reloaded = OptionsConfig(config.path)
    assert "Persisted" in reloaded.options_for("Super-Resolution", "model")
    assert "NIQE" in reloaded.metric_keys("Super-Resolution")


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
        yaml.safe_dump({"tasks": {"Super-Resolution": ["not", "a", "mapping"]}, "defaults": {}, "servers": []}, fp)
    config = OptionsConfig(path, auto_create=False)
    assert config.errors
    assert "Super-Resolution" not in config.task_names


def test_external_edit_is_picked_up_on_reload():
    """에디터로 task-defs/Super-Resolution.yaml 을 고치면 다시 읽었을 때 반영돼야 한다."""
    config = make_config()
    path = config.task_path("Super-Resolution")
    with open(path, encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    data["options"]["model"].append("HandEdited")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(data, fp, allow_unicode=True, sort_keys=False)
    config.load()
    assert "HandEdited" in config.options_for("Super-Resolution", "model")


def test_save_keeps_backup():
    config = make_config()
    config.add_option("Super-Resolution", "model", "One")
    config.add_option("Super-Resolution", "model", "Two")
    assert os.path.exists(config.task_path("Super-Resolution") + ".bak")


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
    config.add_option("Super-Resolution", "model", "OnlyHere")
    assert _changed(config, before) == {"Super-Resolution.yaml"}


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
            "Super-Resolution": {
                "label": "Super-Resolution",
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
    assert os.path.exists(os.path.join(directory, "task-defs", "Super-Resolution.yaml"))
    # 내용이 그대로 살아 있어야 한다
    assert config.options_for("Super-Resolution", "model") == ["LegacyNet"]
    assert config.metric_keys("Super-Resolution") == ["PSNR"]
    assert [s.name for s in config.servers] == ["Old1"]
    assert config.options_for("Super-Resolution", "optimizer") == ["AdamW"]
    # 원본은 백업된다
    assert os.path.exists(path + ".bak")


def test_split_runs_only_once():
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    path = os.path.join(directory, "options.yaml")
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump({"version": 2, "tasks": {"Super-Resolution": {"options": {"model": ["A"]}}}}, fp)

    OptionsConfig(path)
    second = OptionsConfig(path)
    assert second.errors == []
    assert second.task_names == ["Super-Resolution"]


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


# --- config/ 의 모든 YAML 은 template 뿐이다 - 실제 파일은 로컬에서 그걸 복사해 만든다 ---
def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(text)


def test_options_defaults_params_are_seeded_from_their_templates():
    """git clone 직후(template 만 있고 실제 파일은 없는) 상태를 흉내낸다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    tasks_dir = os.path.join(directory, "task-defs")
    os.makedirs(tasks_dir)
    path = os.path.join(directory, "options.yaml")

    _write(os.path.join(directory, "options.template.yaml"), "version: 2\n# 손으로 단 주석\n")
    _write(os.path.join(directory, "defaults.template.yaml"),
           "defaults:\n  model: [MyNet]\n")
    _write(os.path.join(directory, "params.template.yaml"),
           "style: {prefix: '--', separator: ' '}\nparams: {}\n")
    _write(
        os.path.join(tasks_dir, "Demo.template.yaml"),
        "name: Demo\noptions: {}\nmetrics: []\ncolumns: {}\ncommands: {}\n",
    )

    config = OptionsConfig(path)

    assert os.path.exists(path)
    assert os.path.exists(os.path.join(directory, "defaults.yaml"))
    assert os.path.exists(os.path.join(directory, "params.yaml"))
    # servers.template.yaml 을 안 뒀으니 그 안내 하나만 남고, 나머지는 조용히 채워져야 한다.
    assert len(config.errors) == 1 and "servers.template.yaml" in config.errors[0]
    assert config.options_for(None, "model") == ["MyNet"]
    assert config.param_style().prefix == "--"
    # 바이트 그대로 복사하므로 손으로 단 주석도 살아 있어야 한다
    with open(path, encoding="utf-8") as fp:
        assert "손으로 단 주석" in fp.read()


def test_task_yaml_is_seeded_from_its_template_including_custom_options():
    directory = os.path.join(tempfile.mkdtemp(), "config")
    tasks_dir = os.path.join(directory, "task-defs")
    os.makedirs(tasks_dir)
    _write(os.path.join(directory, "options.template.yaml"), "version: 2\n")
    _write(
        os.path.join(tasks_dir, "Denoising.template.yaml"),
        "name: Denoising\nlabel: Denoising\nshort: dn\n"
        "options:\n  model: [NAFNet]\n  algo: [noise2noise]\n"
        "metrics: []\ncolumns: {}\ncommands: {}\n",
    )

    config = OptionsConfig(os.path.join(directory, "options.yaml"))

    assert os.path.exists(os.path.join(tasks_dir, "Denoising.yaml"))
    assert "Denoising" in config.task_names
    assert config.options_for("Denoising", "algo") == ["noise2noise"]
    assert config.task("Denoising").short == "dn"


def test_template_files_are_not_read_as_task_definitions():
    """`Foo.template.yaml` 자체가 "Foo" 라는 Task 정의로 잘못 읽히면 안 된다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    tasks_dir = os.path.join(directory, "task-defs")
    os.makedirs(tasks_dir)
    _write(os.path.join(directory, "options.template.yaml"), "version: 2\n")
    _write(
        os.path.join(tasks_dir, "Denoising.template.yaml"),
        "name: Denoising\noptions: {}\nmetrics: []\ncolumns: {}\ncommands: {}\n",
    )

    config = OptionsConfig(os.path.join(directory, "options.yaml"))

    assert config.task_names.count("Denoising") == 1
    assert config.task_path("Denoising") == os.path.join(tasks_dir, "Denoising.yaml")


def test_only_servers_yaml_is_excluded_from_auto_seeding():
    """나머지는 template 만 있으면 다 채워지는데, servers.yaml 만 그대로 안내만 남긴다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    _write(os.path.join(directory, "options.template.yaml"), "version: 2\n")
    _write(os.path.join(directory, "servers.template.yaml"),
           "servers:\n- {name: RealServer, host: 10.0.0.1, gpus: []}\n")

    config = OptionsConfig(os.path.join(directory, "options.yaml"))

    assert os.path.exists(os.path.join(directory, "options.yaml"))
    assert not os.path.exists(os.path.join(directory, "servers.yaml"))
    assert any("servers.template.yaml" in e for e in config.errors)


def test_seeding_does_not_overwrite_an_existing_real_file():
    """이미 사용자가 고쳐 쓴 실제 파일이 있으면, template 이 있어도 덮어쓰지 않는다."""
    directory = os.path.join(tempfile.mkdtemp(), "config")
    os.makedirs(directory)
    _write(os.path.join(directory, "options.template.yaml"), "version: 2\nfrom_template: true\n")
    _write(os.path.join(directory, "options.yaml"), "version: 2\nfrom_template: false\n")

    OptionsConfig(os.path.join(directory, "options.yaml"))

    with open(os.path.join(directory, "options.yaml"), encoding="utf-8") as fp:
        assert "from_template: false" in fp.read()


def test_broken_task_file_does_not_break_the_rest():
    config = make_config()
    with open(config.task_path("Denoising"), "w", encoding="utf-8") as fp:
        fp.write("options: [\n  broken: :\n")
    reloaded = OptionsConfig(config.path)
    assert any("Denoising.yaml" in e for e in reloaded.errors)
    assert "Super-Resolution" in reloaded.task_names
    assert reloaded.metric_keys("Super-Resolution") == ["PSNR", "SSIM", "LPIPS"]


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

    ok = config.add_option("Super-Resolution", "model", "WontPersist")
    assert ok is True  # the in-memory list is still updated...
    assert "WontPersist" in config.options_for("Super-Resolution", "model")
    assert config.last_save_error is not None  # ...but the failure is recorded
    assert "YAML" in config.last_save_error or "yaml" in config.last_save_error


def test_save_return_value_reflects_success():
    config = make_config()
    assert config.save() is True  # nothing dirty -> trivially true
    assert config.add_option("Super-Resolution", "model", "X") is True
    assert config.last_save_error is None


def test_command_templates_are_per_task_and_editable():
    config = make_config()
    template = config.command_template("Denoising", "train")
    assert "{model}" in template and "train.py" in template

    config.set_command_template("Denoising", "train", "python mytrain.py model={model}")
    assert config.command_template("Denoising", "train") == "python mytrain.py model={model}"
    # 파일로 저장되고, 다른 Task 는 그대로여야 한다.
    reloaded = OptionsConfig(config.path)
    assert reloaded.command_template("Denoising", "train") == "python mytrain.py model={model}"
    assert reloaded.command_template("Super-Resolution", "train") != "python mytrain.py model={model}"


def test_command_template_falls_back_to_builtin_when_task_has_none():
    from dl_exp_manager.config_store import DEFAULT_COMMANDS

    config = make_config()
    config.ensure_task("Segmentation")  # commands 를 안 쓴 새 Task
    assert config.command_template("Segmentation", "evaluation") == DEFAULT_COMMANDS["evaluation"]
    assert config.command_template(None, "train") == DEFAULT_COMMANDS["train"]


# --- params.yaml: 명령어에 파라미터를 적는 방식 --------------------------------
def test_params_file_is_created_with_the_other_config_files():
    config = make_config()
    assert os.path.exists(config.params_path)
    assert config.params_path in config.watch_paths()
    assert config.param_cli_name("epochs") == "max_epoch"
    # 등록 안 된 이름도 제 이름 그대로 쓴다.
    assert config.param_cli_name("scale") == "scale"


def test_params_file_appears_in_a_config_folder_that_predates_it():
    """이미 쓰고 있던 설정 폴더에도 params.yaml 이 생겨야 한다(새로 만들 뿐 덮어쓰진 않는다)."""
    config = make_config()
    os.remove(config.params_path)

    reloaded = OptionsConfig(config.path)
    assert os.path.exists(reloaded.params_path)
    assert reloaded.errors == []


def test_editing_params_yaml_changes_every_task_command():
    from dl_exp_manager.command_builder import render_command

    config = make_config()
    with open(config.params_path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(
            {"style": {"prefix": "--", "separator": " "},
             "params": {"batch_size": {"name": "batch-size"}}},
            fp,
        )

    reloaded = OptionsConfig(config.path)
    for task in ("Super-Resolution", "Denoising"):
        template = reloaded.command_template(task, "train")
        text = render_command(template, {"batch_size": "16"}, reloaded.param_style()).text
        assert "--batch-size 16" in text


def test_set_param_name_writes_only_params_yaml():
    config = make_config()
    before = {p: os.path.getmtime(p) for p in config.watch_paths()}
    time.sleep(0.01)

    config.set_param_name("epochs", "num_epochs")
    assert config.param_cli_name("epochs") == "num_epochs"
    assert OptionsConfig(config.path).param_cli_name("epochs") == "num_epochs"
    assert _changed(config, before) == {"params.yaml"}


def test_broken_params_yaml_falls_back_to_the_builtin_style():
    config = make_config()
    with open(config.params_path, "w", encoding="utf-8") as fp:
        fp.write("style: [\n  broken: :\n")

    reloaded = OptionsConfig(config.path)
    assert any("params.yaml" in e for e in reloaded.errors)
    assert reloaded.param_style().prefix == "+"          # 앱은 계속 동작한다


# --- short: 표시 이름과 명령어 경로 분리 ---------------------------------------
def test_task_short_name_is_read_but_optional():
    config = make_config()
    assert config.task("Denoising").short == "dn"
    assert config.task("Clustering").short == ""         # 없으면 빈 문자열
