"""설정 로더/라이터 - 기능별로 나뉜 YAML 파일들을 하나의 뷰로 합친다.

    config/
      options.yaml          진입점 (버전 + 안내). 세부 설정은 아래 파일들에 있다.
      servers.yaml          서버 & GPU 인벤토리 (실서버 정보라 gitignore 대상 - 직접 만들어야 한다)
      servers.template.yaml servers.yaml 이 없을 때 복사해서 쓰는 예시 (git 추적)
      defaults.yaml         모든 Task 공통 선택지
      tasks/
        SR.yaml             Task 별 선택지 / 지표 / 컬럼
        DN.yaml
        ...

읽을 때는 전부 합쳐 하나의 딕셔너리로 보고, 쓸 때는 **그 값이 원래 있던 파일에만** 저장한다.
(SR 모델을 추가하면 tasks/SR.yaml 만 바뀐다.)

- `ruamel.yaml` 이 있으면 주석과 순서를 보존하며 저장한다. 없으면 PyYAML 로 동작한다.
- 파일 하나가 깨져도 나머지는 살리고, 무엇이 문제인지 `errors` 에 남긴다.
- 예전처럼 options.yaml 한 파일에 전부 들어 있으면 첫 실행 때 자동으로 나눠 준다.
"""
from __future__ import annotations

import copy
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable

CONFIG_VERSION = 2

# --- YAML 백엔드 선택 --------------------------------------------------------
_BACKEND = "none"
try:  # pragma: no cover - 환경 의존
    from ruamel.yaml import YAML as _RuamelYAML  # type: ignore

    _BACKEND = "ruamel"
except ImportError:  # pragma: no cover
    try:
        import yaml as _pyyaml  # type: ignore

        _BACKEND = "pyyaml"
    except ImportError:
        _pyyaml = None  # type: ignore


def backend_name() -> str:
    return _BACKEND


def preserves_comments() -> bool:
    return _BACKEND == "ruamel"



# --- YAML 출력 모양 다듬기 ----------------------------------------------------
# 스칼라만 든 리스트(`[DIV2K, DF2K]`)와 리스트 안의 작은 매핑(`{key: PSNR, digits: 2}`)은
# 한 줄로 뽑는다. 한 줄에 하나씩 늘어놓으면 Task 파일이 금세 수십 줄이 된다.
class _FlowList(list):
    pass


class _FlowDict(dict):
    pass


if _BACKEND == "pyyaml":  # pragma: no branch

    class _Dumper(_pyyaml.SafeDumper):
        pass

    def _represent_flow_list(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def _represent_flow_dict(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)

    _Dumper.add_representer(_FlowList, _represent_flow_list)
    _Dumper.add_representer(_FlowDict, _represent_flow_dict)
else:  # pragma: no cover
    _Dumper = None  # type: ignore


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _flowify(value: Any) -> Any:
    """중첩 구조를 훑으며 한 줄로 뽑아도 되는 곳만 표시한다."""
    if isinstance(value, dict):
        return {k: _flowify(v) for k, v in value.items()}
    if isinstance(value, list):
        if value and all(_is_scalar(v) for v in value):
            return _FlowList(value)
        if value and all(
            isinstance(v, dict) and all(_is_scalar(x) for x in v.values()) for v in value
        ):
            # 항목 하나가 한 줄이 되도록 바깥 리스트는 블록으로 둔다.
            #   metrics:
            #     - {key: PSNR, unit: dB, digits: 2, higher_is_better: true}
            return [_FlowDict(v) for v in value]
        return [_flowify(v) for v in value]
    return value


# --- 데이터 구조 --------------------------------------------------------------
@dataclass
class GpuDef:
    index: int
    type: str = "GPU"
    memory_gb: float | None = None

    @property
    def label(self) -> str:
        return f"GPU {self.index} · {self.type}"


@dataclass
class ServerDef:
    name: str
    host: str = ""
    gpus: list[GpuDef] = field(default_factory=list)

    @property
    def gpu_summary(self) -> str:
        if not self.gpus:
            return "No GPU info"
        kinds: dict[str, int] = {}
        for gpu in self.gpus:
            kinds[gpu.type] = kinds.get(gpu.type, 0) + 1
        return " + ".join(f"{kind} ×{n}" for kind, n in kinds.items())


@dataclass
class MetricDef:
    key: str
    unit: str = ""
    digits: int = 4
    higher_is_better: bool = True


@dataclass
class TaskDef:
    name: str
    label: str = ""
    options: dict[str, list[str]] = field(default_factory=dict)
    metrics: list[MetricDef] = field(default_factory=list)
    columns: dict[str, list[str]] = field(default_factory=dict)


# --- 파일 레이아웃 ------------------------------------------------------------
ROOT_FILE = "options.yaml"
SERVERS_FILE = "servers.yaml"
SERVERS_TEMPLATE_FILE = "servers.template.yaml"
DEFAULTS_FILE = "defaults.yaml"
TASKS_DIR = "tasks"

# 옵션 필드 중 DB 에 전용 컬럼이 있는 것들. 이외의 필드는 extra_json 으로 간다.
NATIVE_OPTION_FIELDS = {"model", "dataset", "optimizer", "server"}


# --- 내장 기본값 --------------------------------------------------------------
BUILTIN: dict[str, Any] = {
    "version": CONFIG_VERSION,
    "servers": [
        {"name": "Server 1", "host": "192.168.0.101",
         "gpus": [{"index": i, "type": "H100", "memory_gb": 80} for i in range(4)]},
        {"name": "Server 2", "host": "192.168.0.102",
         "gpus": [{"index": i, "type": "V100", "memory_gb": 32} for i in range(4)]},
        {"name": "Server 3", "host": "192.168.0.103",
         "gpus": [{"index": i, "type": "A6000", "memory_gb": 48} for i in range(4)]},
        {"name": "Server 4", "host": "192.168.0.104",
         "gpus": [{"index": i, "type": "A100", "memory_gb": 80} for i in range(4)]},
    ],
    "defaults": {
        "model": ["Restormer", "SwinIR", "MambaIR", "NAFNet"],
        "dataset": ["DIV2K", "DF2K"],
        "optimizer": ["AdamW", "Adam", "SGD", "Lion"],
    },
    "tasks": {
        "SR": {
            "label": "Super Resolution",
            "options": {
                "model": ["Restormer", "SwinIR", "MambaIR", "HAT", "EDSR", "RCAN"],
                "dataset": ["DIV2K", "DF2K", "Flickr2K", "Set5", "Set14", "Urban100"],
                "scale": ["x2", "x3", "x4"],
            },
            "metrics": [
                {"key": "PSNR", "unit": "dB", "digits": 2, "higher_is_better": True},
                {"key": "SSIM", "digits": 4, "higher_is_better": True},
                {"key": "LPIPS", "digits": 3, "higher_is_better": False},
            ],
            "columns": {
                "train": ["status", "server", "gpus", "model", "dataset", "scale",
                          "duration", "PSNR", "SSIM", "LPIPS", "result_path"],
                "inference": ["status", "server", "gpus", "model", "checkpoint_path",
                              "dataset", "latency_ms", "throughput_fps",
                              "PSNR", "SSIM", "LPIPS", "result_path"],
            },
        },
        "DN": {
            "label": "Denoising",
            "options": {
                "model": ["Restormer", "NAFNet", "SCUNet", "Uformer"],
                "dataset": ["SIDD", "DND", "BSD68", "Kodak24"],
                "noise_sigma": ["15", "25", "50"],
            },
            "metrics": [
                {"key": "PSNR", "unit": "dB", "digits": 2, "higher_is_better": True},
                {"key": "SSIM", "digits": 4, "higher_is_better": True},
            ],
            "columns": {
                "train": ["status", "server", "gpus", "model", "dataset", "noise_sigma",
                          "duration", "PSNR", "SSIM", "result_path"],
                "inference": ["status", "server", "gpus", "model", "checkpoint_path",
                              "dataset", "noise_sigma", "latency_ms", "PSNR", "SSIM",
                              "result_path"],
            },
        },
        "Clustering": {
            "label": "Unsupervised Clustering",
            "options": {
                "model": ["DeepCluster", "SCAN", "SwAV"],
                "dataset": ["CIFAR-10", "STL-10", "ImageNet-50"],
            },
            "metrics": [
                {"key": "NMI", "digits": 4, "higher_is_better": True},
                {"key": "ARI", "digits": 4, "higher_is_better": True},
                {"key": "ACC", "unit": "%", "digits": 2, "higher_is_better": True},
            ],
            "columns": {
                "train": ["status", "server", "gpus", "model", "dataset", "duration",
                          "NMI", "ARI", "ACC", "result_path"],
                "inference": ["status", "server", "model", "checkpoint_path", "dataset",
                              "NMI", "ARI", "ACC", "result_path"],
            },
        },
        "Classification": {
            "label": "Image Classification",
            "options": {
                "model": ["ResNet-50", "ViT-B/16", "ConvNeXt-T", "Swin-T"],
                "dataset": ["ImageNet-1k", "CIFAR-100", "Food-101"],
            },
            "metrics": [
                {"key": "Top-1", "unit": "%", "digits": 2, "higher_is_better": True},
                {"key": "Top-5", "unit": "%", "digits": 2, "higher_is_better": True},
            ],
            "columns": {
                "train": ["status", "server", "gpus", "model", "dataset", "duration",
                          "Top-1", "Top-5", "result_path"],
                "inference": ["status", "server", "model", "checkpoint_path", "dataset",
                              "latency_ms", "throughput_fps", "Top-1", "Top-5", "result_path"],
            },
        },
    },
}


# --- 파일별 머리말 주석 --------------------------------------------------------
ROOT_HEADER = """\
# DL Experiment Manager - config entry point
#
# Settings are split by function. Open only the file you want to change.
#
#   servers.yaml        Servers and GPU inventory (type / count / memory)
#   defaults.yaml       Options shared by every Task
#   tasks/<name>.yaml   Per-Task options, metrics, and table columns
#
# Edit by hand or through the app UI - both write to the same files.
# Saving is picked up by the app immediately, and a UI edit only touches
# the file the value already lived in.
#
# ── How to write tasks/<name>.yaml ──────────────────────────────────────────
#   options : That Task's combo-box choices. A name here 'replaces' the same
#             name in defaults.yaml. Any name other than model / dataset /
#             optimizer becomes a custom field with its own combo box in the
#             form, stored in the DB's extra_json.
#   metrics : The table's metric columns. digits sets decimal places shown,
#             higher_is_better marks whether a bigger value is better.
#   columns : Which columns appear (and in what order) in the Train /
#             Inference tables. Allowed values:
#     built-in    status, server, gpus, model, dataset, dataset_path, result_path,
#                 checkpoint_path, device, input_size, duration, started_at,
#                 latency_ms, throughput_fps, epochs, batch_size, lr, optimizer, notes
#     metric      any key defined under metrics
#     custom field  any name defined under options
"""

SERVERS_HEADER = """\
# Servers & GPU inventory
# This file holds real server addresses, so it is gitignored - only
# servers.template.yaml is tracked in git. Nothing here leaves this machine.
# The GPUs listed here become the checkboxes in the run form and the slots
# in the server status bar.
#   index = CUDA_VISIBLE_DEVICES number · type = V100 / H100 / A100 ... · memory_gb = informational
"""

DEFAULTS_HEADER = """\
# Options shared by every Task
# If tasks/<name>.yaml defines the same name under options, that 'replaces'
# this value. (Never merged - so it's always clear why an item is in the list.)
"""

TASK_HEADER_TEMPLATE = """\
# Task: {name}
# options = combo-box choices · metrics = table metric columns · columns = table layout
# See the header of options.yaml for the full syntax.
"""


def default_config_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "config", ROOT_FILE)


def _safe_filename(name: str) -> str:
    """Task 이름을 파일명으로. 경로 구분자 등만 걷어낸다."""
    cleaned = "".join("_" if ch in '/\\:*?"<>|' else ch for ch in name).strip()
    return (cleaned or "task") + ".yaml"


class OptionsConfig:
    """여러 설정 파일을 하나의 뷰로 합쳐 읽고, 쓸 때는 원래 파일로 되돌린다."""

    def __init__(self, path: str | None = None, auto_create: bool = True) -> None:
        self.path = os.path.abspath(path or default_config_path())
        self.errors: list[str] = []
        self.last_save_error: str | None = None
        self._data: dict[str, Any] = copy.deepcopy(BUILTIN)
        # 값의 출처: 저장할 때 어느 파일로 되돌릴지 결정한다.
        self._servers_file: str = self.servers_path
        self._defaults_file: str = self.defaults_path
        self._task_files: dict[str, str] = {}
        self._dirty: set[str] = set()

        self._yaml = _RuamelYAML() if _BACKEND == "ruamel" else None
        if self._yaml is not None:  # pragma: no cover - 환경 의존
            self._yaml.preserve_quotes = True
            self._yaml.indent(mapping=2, sequence=4, offset=2)

        self.load(auto_create=auto_create)

    # -- 경로 ----------------------------------------------------------------
    @property
    def config_dir(self) -> str:
        return os.path.dirname(self.path)

    @property
    def servers_path(self) -> str:
        return os.path.join(os.path.dirname(self.path), SERVERS_FILE)

    @property
    def defaults_path(self) -> str:
        return os.path.join(os.path.dirname(self.path), DEFAULTS_FILE)

    @property
    def tasks_dir(self) -> str:
        return os.path.join(os.path.dirname(self.path), TASKS_DIR)

    def task_path(self, task: str) -> str:
        known = self._task_files.get(task)
        if known:
            return known
        return os.path.join(self.tasks_dir, _safe_filename(task))

    def watch_paths(self) -> list[str]:
        """외부 편집을 감지하기 위해 지켜봐야 할 파일 목록."""
        paths = {self.path, self._servers_file, self._defaults_file}
        paths.update(self._task_files.values())
        return sorted(p for p in paths if os.path.exists(p))

    def files_summary(self) -> list[tuple[str, str]]:
        """(역할, 경로) - 설정 상태 보기용."""
        rows = [
            ("진입점", self.path),
            ("서버/GPU", self._servers_file),
            ("공통 선택지", self._defaults_file),
        ]
        for name in sorted(self._task_files):
            rows.append((f"Task · {name}", self._task_files[name]))
        return rows

    # -- 읽기 ----------------------------------------------------------------
    def _read(self, path: str) -> dict[str, Any] | None:
        """YAML 한 개 읽기. 없으면 None, 깨졌으면 errors 에 남기고 None."""
        if not os.path.exists(path):
            return None
        if _BACKEND == "none":
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                if self._yaml is not None:  # pragma: no cover
                    loaded = self._yaml.load(fp)
                else:
                    loaded = _pyyaml.safe_load(fp)
        except OSError as exc:
            self.errors.append(f"Could not read {os.path.basename(path)}: {exc}")
            return None
        except Exception as exc:  # 손으로 고치는 파일이라 문법 오류는 상시 발생한다
            self.errors.append(f"Syntax error in {os.path.basename(path)}: {exc}")
            return None
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            self.errors.append(f"The top level of {os.path.basename(path)} is not a mapping.")
            return None
        return loaded

    def load(self, auto_create: bool = True) -> None:
        self.errors = []
        self._dirty = set()
        self._task_files = {}
        self._servers_file = self.servers_path
        self._defaults_file = self.defaults_path

        root = self._read(self.path)
        if root is None and not os.path.exists(self.path):
            if auto_create:
                self._data = copy.deepcopy(BUILTIN)
                self._assign_default_origins()
                if not self.save(force_all=True) and self.last_save_error:
                    self.errors.append(
                        f"Could not create the default config files: {self.last_save_error}"
                    )
            else:
                self._assign_default_origins()
            return

        if _BACKEND == "none":
            self.errors.append(
                "PyYAML or ruamel.yaml is missing, so built-in defaults are used. "
                "`pip install -r requirements.txt` 로 설치하세요."
            )
            self._assign_default_origins()
            return

        root = root or {}
        merged: dict[str, Any] = {"version": root.get("version", CONFIG_VERSION)}

        # 1) 서버 - 전용 파일이 우선, 없으면 진입점의 인라인 값(구버전 형식)
        servers_doc = self._read(self.servers_path)
        if servers_doc is not None and isinstance(servers_doc.get("servers"), list):
            merged["servers"] = servers_doc["servers"]
            self._servers_file = self.servers_path
        elif isinstance(root.get("servers"), list):
            merged["servers"] = root["servers"]
            self._servers_file = self.path
        else:
            merged["servers"] = copy.deepcopy(BUILTIN["servers"])
            self.errors.append(
                "No servers.yaml found; showing placeholder servers. "
                f"Copy {SERVERS_TEMPLATE_FILE} to {SERVERS_FILE} (or use the server bar's "
                "+ button) to add your real servers."
            )

        # 2) 공통 선택지
        defaults_doc = self._read(self.defaults_path)
        if defaults_doc is not None and isinstance(defaults_doc.get("defaults"), dict):
            merged["defaults"] = defaults_doc["defaults"]
            self._defaults_file = self.defaults_path
        elif isinstance(root.get("defaults"), dict):
            merged["defaults"] = root["defaults"]
            self._defaults_file = self.path
        else:
            merged["defaults"] = copy.deepcopy(BUILTIN["defaults"])

        # 3) Task - tasks/*.yaml 을 먼저 읽고, 진입점 인라인은 없는 것만 채운다
        tasks: dict[str, Any] = {}
        task_files, had_task_sources = self._read_task_files()
        for name, body, origin in task_files:
            tasks[name] = body
            self._task_files[name] = origin

        inline = root.get("tasks")
        if isinstance(inline, dict):
            had_task_sources = had_task_sources or bool(inline)
            for name, body in inline.items():
                if name in tasks:
                    self.errors.append(
                        f"Task '{name}' is defined in both options.yaml and tasks/; using tasks/."
                    )
                    continue
                if not isinstance(body, dict):
                    self.errors.append(f"Task '{name}' definition is not a mapping; ignored it.")
                    continue
                tasks[name] = body
                self._task_files[name] = self.path

        if not tasks and not had_task_sources:
            # 정의가 아예 없을 때만 기본값을 채운다.
            # 정의가 있었는데 전부 깨진 경우까지 채우면, 사용자의 SR 을 내장 SR 이 가려 버리고
            # 다음 UI 편집이 원본 파일을 덮어쓴다.
            tasks = copy.deepcopy(BUILTIN["tasks"])
            for name in tasks:
                self._task_files[name] = os.path.join(self.tasks_dir, _safe_filename(name))
            self.errors.append("No Task definitions found; added the default Tasks.")
        elif not tasks:
            self.errors.append(
                "Could not read any Task definitions. Tables will use default columns."
            )

        merged["tasks"] = tasks
        self._data = merged
        self._normalize_tasks()

        # 구버전(한 파일에 전부) 이면 기능별로 나눠 준다.
        if auto_create:
            self._split_legacy_layout(root)

    def _read_task_files(self) -> tuple[list[tuple[str, dict[str, Any], str]], bool]:
        """(읽은 Task 목록, 파일이 하나라도 있었는지)."""
        out: list[tuple[str, dict[str, Any], str]] = []
        if not os.path.isdir(self.tasks_dir):
            return out, False
        seen_any = False
        for filename in sorted(os.listdir(self.tasks_dir)):
            if not filename.lower().endswith((".yaml", ".yml")):
                continue
            seen_any = True
            full = os.path.join(self.tasks_dir, filename)
            body = self._read(full)
            if body is None:
                continue  # 깨진 파일 - 이유는 이미 errors 에 있다
            name = str(body.get("name") or os.path.splitext(filename)[0]).strip()
            if not name:
                self.errors.append(f"{filename} has an empty Task name; skipped it.")
                continue
            body.pop("name", None)  # 이름은 파일명/키로 관리한다
            out.append((name, body, full))
        return out, seen_any

    def _assign_default_origins(self) -> None:
        self._servers_file = self.servers_path
        self._defaults_file = self.defaults_path
        self._task_files = {
            name: os.path.join(self.tasks_dir, _safe_filename(name))
            for name in self._data.get("tasks", {})
        }

    def _normalize_tasks(self) -> None:
        for name, raw in list(self._data["tasks"].items()):
            if not isinstance(raw, dict):
                self.errors.append(f"Task '{name}' definition is not a mapping; ignored it.")
                self._data["tasks"].pop(name)
                self._task_files.pop(name, None)
                continue
            raw.setdefault("options", {})
            raw.setdefault("metrics", [])
            raw.setdefault("columns", {})
            if not isinstance(raw["options"], dict):
                raw["options"] = {}
                self.errors.append(f"Task '{name}' options is not a mapping; cleared it.")
            if not isinstance(raw["metrics"], list):
                raw["metrics"] = []
                self.errors.append(f"Task '{name}' metrics is not a list; cleared it.")
            if not isinstance(raw["columns"], dict):
                raw["columns"] = {}

    def _split_legacy_layout(self, root: dict[str, Any]) -> None:
        """options.yaml 한 파일에 전부 들어 있던 구버전을 기능별 파일로 나눈다."""
        inline_sections = [k for k in ("servers", "defaults", "tasks") if k in root]
        if not inline_sections:
            return
        self._servers_file = self.servers_path
        self._defaults_file = self.defaults_path
        self._task_files = {
            name: os.path.join(self.tasks_dir, _safe_filename(name))
            for name in self._data["tasks"]
        }
        if not self.save(force_all=True):
            if self.last_save_error:
                self.errors.append(f"Could not split the config file: {self.last_save_error}")
            return
        self.errors.append(
            "Split the settings that used to live in options.yaml into "
            "servers.yaml / defaults.yaml / tasks/."
        )

    # -- 쓰기 ----------------------------------------------------------------
    def _write(self, path: str, document: dict[str, Any], header: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(path):  # 손으로 쓴 파일을 덮어쓰기 전에 한 벌 남긴다
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            if self._yaml is not None:  # pragma: no cover - 주석 보존 경로
                self._yaml.dump(document, fp)
            elif _BACKEND == "pyyaml":
                fp.write(header)
                fp.write("\n")
                _pyyaml.dump(
                    _flowify(document),
                    fp,
                    Dumper=_Dumper,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                    width=120,
                )
            else:
                raise OSError(
                    "No YAML backend available (PyYAML or ruamel.yaml). "
                    "Install one, e.g. `pip install pyyaml`, then reload the config."
                )
        os.replace(tmp, path)

    def save(self, force_all: bool = False) -> bool:
        """변경된 파일만 저장한다. force_all 이면 전부 다시 쓴다.

        디스크 쓰기가 실패해도(예: YAML 백엔드 없음) 예외를 올리지 않는다.
        메모리 상의 변경은 이미 반영돼 있으므로 UI 는 정상 동작하고,
        실패 사유는 `last_save_error` 에 담아 호출부(주로 main_window)가
        한 번 알려 주고 넘어가게 한다.
        """
        targets = set(self._dirty)
        if force_all:
            targets = {self.path, self._servers_file, self._defaults_file}
            targets.update(self._task_files.values())

        if not targets:
            return True

        self.last_save_error = None
        try:
            for path in sorted(targets):
                self._write_one(path, force_all=force_all)
        except OSError as exc:
            self.last_save_error = str(exc)
            return False

        self._dirty -= targets
        return True

    def _write_one(self, path: str, force_all: bool = False) -> None:
        same = os.path.normpath

        if same(path) == same(self.path):
            document: dict[str, Any] = {"version": CONFIG_VERSION}
            # 진입점에 남아 있는 인라인 섹션(구버전)이 있으면 그대로 유지한다.
            if same(self._servers_file) == same(self.path):
                document["servers"] = self._data.get("servers", [])
            if same(self._defaults_file) == same(self.path):
                document["defaults"] = self._data.get("defaults", {})
            inline_tasks = {
                name: body
                for name, body in self._data.get("tasks", {}).items()
                if same(self._task_files.get(name, "")) == same(self.path)
            }
            if inline_tasks:
                document["tasks"] = inline_tasks
            self._write(path, document, ROOT_HEADER)
            return

        if same(path) == same(self._servers_file):
            self._write(path, {"servers": self._data.get("servers", [])}, SERVERS_HEADER)
            return

        if same(path) == same(self._defaults_file):
            self._write(path, {"defaults": self._data.get("defaults", {})}, DEFAULTS_HEADER)
            return

        for name, origin in self._task_files.items():
            if same(origin) == same(path):
                body = dict(self._data["tasks"].get(name, {}))
                document = {"name": name}
                document.update(body)
                self._write(path, document, TASK_HEADER_TEMPLATE.format(name=name))
                return

    # -- 변경 표시 -------------------------------------------------------------
    def _touch_task(self, task: str) -> None:
        self._dirty.add(self.task_path(task))
        self.save()

    def _touch_defaults(self) -> None:
        self._dirty.add(self._defaults_file)
        self.save()

    def _touch_servers(self) -> None:
        self._dirty.add(self._servers_file)
        self.save()

    # -- 조회 ----------------------------------------------------------------
    @property
    def task_names(self) -> list[str]:
        return list(self._data.get("tasks", {}).keys())

    def has_task(self, task: str | None) -> bool:
        return bool(task) and task in self._data.get("tasks", {})

    def _task_raw(self, task: str | None) -> dict[str, Any] | None:
        if not task:
            return None
        raw = self._data.get("tasks", {}).get(task)
        return raw if isinstance(raw, dict) else None

    def task(self, task: str | None) -> TaskDef | None:
        raw = self._task_raw(task)
        if raw is None:
            return None
        return TaskDef(
            name=str(task),
            label=str(raw.get("label") or task),
            options={
                k: [str(v) for v in (vals or [])]
                for k, vals in (raw.get("options") or {}).items()
            },
            metrics=[self._metric_def(m) for m in (raw.get("metrics") or []) if m],
            columns={
                k: [str(c) for c in (v or [])] for k, v in (raw.get("columns") or {}).items()
            },
        )

    @staticmethod
    def _metric_def(raw: Any) -> MetricDef:
        if isinstance(raw, str):
            return MetricDef(key=raw)
        if not isinstance(raw, dict):
            return MetricDef(key=str(raw))
        try:
            digits = int(raw.get("digits", 4))
        except (TypeError, ValueError):
            digits = 4
        return MetricDef(
            key=str(raw.get("key", "")),
            unit=str(raw.get("unit", "") or ""),
            digits=digits,
            higher_is_better=bool(raw.get("higher_is_better", True)),
        )

    def option_fields(self, task: str | None) -> list[str]:
        fields = list((self._data.get("defaults") or {}).keys())
        raw = self._task_raw(task)
        if raw:
            for key in (raw.get("options") or {}):
                if key not in fields:
                    fields.append(key)
        return fields

    def options_for(self, task: str | None, field_name: str) -> list[str]:
        """Task 전용 목록이 있으면 그것을, 없으면 defaults 를 돌려준다(대체 규칙)."""
        raw = self._task_raw(task)
        if raw:
            values = (raw.get("options") or {}).get(field_name)
            if isinstance(values, list):
                return [str(v) for v in values]
        values = (self._data.get("defaults") or {}).get(field_name)
        if isinstance(values, list):
            return [str(v) for v in values]
        return []

    def custom_fields(self, task: str | None) -> list[str]:
        return [f for f in self.option_fields(task) if f not in NATIVE_OPTION_FIELDS]

    def metrics_for(self, task: str | None) -> list[MetricDef]:
        task_def = self.task(task)
        return list(task_def.metrics) if task_def else []

    def metric_keys(self, task: str | None) -> list[str]:
        return [m.key for m in self.metrics_for(task) if m.key]

    def metric_def(self, task: str | None, key: str) -> MetricDef | None:
        for metric in self.metrics_for(task):
            if metric.key == key:
                return metric
        return None

    def columns_for(self, task: str | None, mode: str) -> list[str]:
        task_def = self.task(task)
        if task_def is None:
            return []
        return list(task_def.columns.get(mode, []))

    @property
    def servers(self) -> list[ServerDef]:
        out: list[ServerDef] = []
        for raw in self._data.get("servers") or []:
            if not isinstance(raw, dict):
                continue
            gpus: list[GpuDef] = []
            for gpu_raw in raw.get("gpus") or []:
                if not isinstance(gpu_raw, dict):
                    continue
                try:
                    index = int(gpu_raw.get("index", len(gpus)))
                except (TypeError, ValueError):
                    index = len(gpus)
                memory = gpu_raw.get("memory_gb")
                try:
                    memory_gb = float(memory) if memory not in (None, "") else None
                except (TypeError, ValueError):
                    memory_gb = None
                gpus.append(
                    GpuDef(index=index, type=str(gpu_raw.get("type", "GPU")), memory_gb=memory_gb)
                )
            out.append(
                ServerDef(
                    name=str(raw.get("name", "")),
                    host=str(raw.get("host", "") or ""),
                    gpus=sorted(gpus, key=lambda g: g.index),
                )
            )
        return [s for s in out if s.name]

    def server(self, name: str) -> ServerDef | None:
        for srv in self.servers:
            if srv.name == name:
                return srv
        return None

    def server_names(self) -> list[str]:
        return [s.name for s in self.servers]

    # -- 변경 (모두 저장까지 수행) --------------------------------------------
    def ensure_task(self, task: str) -> None:
        tasks = self._data.setdefault("tasks", {})
        if task not in tasks:
            tasks[task] = {"label": task, "options": {}, "metrics": [], "columns": {}}
            self._task_files[task] = os.path.join(self.tasks_dir, _safe_filename(task))
            self._touch_task(task)

    def remove_task(self, task: str) -> bool:
        """Task 정의와 그 파일을 지운다(실행 기록은 DB 쪽 문제라 건드리지 않는다)."""
        if task not in self._data.get("tasks", {}):
            return False
        self._data["tasks"].pop(task)
        path = self._task_files.pop(task, "")
        if path and os.path.normpath(path) != os.path.normpath(self.path):
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            self._dirty.add(self.path)
            self.save()
        return True

    def add_option(self, task: str | None, field_name: str, value: str) -> bool:
        """task 가 None 이면 defaults(전체 공통)에 추가한다."""
        value = value.strip()
        if not value:
            return False
        container = self._options_container(task, field_name, create=True)
        if container is None:
            return False
        if value in container:
            return False
        container.append(value)
        self._touch_option_scope(task)
        return True

    def remove_option(self, task: str | None, field_name: str, value: str) -> bool:
        container = self._options_container(task, field_name, create=False)
        if not container or value not in container:
            return False
        container.remove(value)
        self._touch_option_scope(task)
        return True

    def rename_option(self, task: str | None, field_name: str, old: str, new: str) -> bool:
        new = new.strip()
        container = self._options_container(task, field_name, create=False)
        if not container or not new or old not in container:
            return False
        if new in container and new != old:
            container.remove(old)
        else:
            container[container.index(old)] = new
        self._touch_option_scope(task)
        return True

    def _touch_option_scope(self, task: str | None) -> None:
        if task:
            self._touch_task(task)
        else:
            self._touch_defaults()

    def _options_container(
        self, task: str | None, field_name: str, create: bool
    ) -> list[Any] | None:
        if task:
            raw = self._task_raw(task)
            if raw is None:
                if not create:
                    return None
                self.ensure_task(task)
                raw = self._task_raw(task)
                if raw is None:
                    return None
            options = raw.setdefault("options", {})
            if field_name not in options:
                if not create:
                    return None
                # Task 전용 목록을 처음 만들 때는 현재 보이던 값(defaults)에서 출발한다.
                options[field_name] = list(self.options_for(task, field_name))
            return options[field_name]

        defaults = self._data.setdefault("defaults", {})
        if field_name not in defaults:
            if not create:
                return None
            defaults[field_name] = []
        return defaults[field_name]

    # -- 지표 -----------------------------------------------------------------
    def add_metric(self, task: str, metric: MetricDef) -> bool:
        raw = self._task_raw(task)
        if raw is None or not metric.key.strip():
            return False
        metrics = raw.setdefault("metrics", [])
        if any(self._metric_def(m).key == metric.key for m in metrics):
            return False
        metrics.append(
            {
                "key": metric.key,
                "unit": metric.unit,
                "digits": metric.digits,
                "higher_is_better": metric.higher_is_better,
            }
        )
        self._touch_task(task)
        return True

    def remove_metric(self, task: str, key: str) -> bool:
        raw = self._task_raw(task)
        if raw is None:
            return False
        metrics = raw.get("metrics") or []
        for i, item in enumerate(list(metrics)):
            if self._metric_def(item).key == key:
                metrics.pop(i)
                self._drop_column_everywhere(task, key)
                self._touch_task(task)
                return True
        return False

    def rename_metric(self, task: str, old: str, new: str) -> bool:
        raw = self._task_raw(task)
        new = new.strip()
        if raw is None or not new:
            return False
        for item in raw.get("metrics") or []:
            if self._metric_def(item).key == old:
                if isinstance(item, dict):
                    item["key"] = new
                    self._rename_column_everywhere(task, old, new)
                    self._touch_task(task)
                    return True
        return False

    def update_metric(self, task: str, key: str, **changes: Any) -> bool:
        raw = self._task_raw(task)
        if raw is None:
            return False
        for item in raw.get("metrics") or []:
            if isinstance(item, dict) and item.get("key") == key:
                item.update(changes)
                self._touch_task(task)
                return True
        return False

    # -- 컬럼 -----------------------------------------------------------------
    def set_columns(self, task: str, mode: str, columns: Iterable[str]) -> None:
        raw = self._task_raw(task)
        if raw is None:
            self.ensure_task(task)
            raw = self._task_raw(task)
            if raw is None:
                return
        raw.setdefault("columns", {})[mode] = [str(c) for c in columns]
        self._touch_task(task)

    def _drop_column_everywhere(self, task: str, column: str) -> None:
        raw = self._task_raw(task) or {}
        for mode, cols in (raw.get("columns") or {}).items():
            raw["columns"][mode] = [c for c in cols if c != column]

    def _rename_column_everywhere(self, task: str, old: str, new: str) -> None:
        raw = self._task_raw(task) or {}
        for mode, cols in (raw.get("columns") or {}).items():
            raw["columns"][mode] = [new if c == old else c for c in cols]

    # -- 서버 / GPU ------------------------------------------------------------
    def upsert_server(self, name: str, host: str = "", gpus: list[GpuDef] | None = None) -> None:
        servers = self._data.setdefault("servers", [])
        payload = {
            "name": name,
            "host": host,
            "gpus": [
                {"index": g.index, "type": g.type, "memory_gb": g.memory_gb}
                for g in (gpus or [])
            ],
        }
        for i, raw in enumerate(servers):
            if isinstance(raw, dict) and raw.get("name") == name:
                servers[i] = payload
                self._touch_servers()
                return
        servers.append(payload)
        self._touch_servers()

    def remove_server(self, name: str) -> bool:
        servers = self._data.get("servers") or []
        for i, raw in enumerate(list(servers)):
            if isinstance(raw, dict) and raw.get("name") == name:
                servers.pop(i)
                self._touch_servers()
                return True
        return False

    def rename_server(self, old: str, new: str) -> bool:
        new = new.strip()
        if not new:
            return False
        for raw in self._data.get("servers") or []:
            if isinstance(raw, dict) and raw.get("name") == old:
                raw["name"] = new
                self._touch_servers()
                return True
        return False
