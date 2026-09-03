"""`config/options.yaml` 로더 / 라이터.

콤보박스 선택지, Task 별 평가 지표와 테이블 컬럼, 서버·GPU 인벤토리를 한 파일에서 관리한다.
손으로 편집해도 되고 UI 에서 바꿔도 되며, 두 경로가 같은 파일을 쓴다.

- `ruamel.yaml` 이 설치돼 있으면 주석과 순서를 보존하며 저장한다.
- 없으면 PyYAML 로 동작한다(주석은 사라진다).
- 파일이 깨져 있어도 앱은 내장 기본값으로 계속 뜨고, `errors` 에 이유가 담긴다.
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
            return "GPU 정보 없음"
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
                              "dataset", "noise_sigma", "latency_ms", "PSNR", "SSIM", "result_path"],
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

# 옵션 필드 중 DB 에 전용 컬럼이 있는 것들. 이외의 필드는 extra_json 으로 간다.
NATIVE_OPTION_FIELDS = {"model", "dataset", "optimizer", "server"}

HEADER_COMMENT = """\
# DL Experiment Manager - 선택지 / Task 정의
#
# 이 파일은 손으로 편집해도 되고, 앱 UI 에서 바꿔도 됩니다(같은 파일을 씁니다).
#   - defaults : 모든 Task 가 공유하는 기본 선택지
#   - tasks.<이름>.options : 해당 Task 전용 선택지. 있으면 defaults 를 '대체'합니다.
#   - tasks.<이름>.metrics : 표의 지표 컬럼 + 표시 자릿수 + 높을수록 좋은지 여부
#   - tasks.<이름>.columns : Train / Inference 표에 보일 컬럼과 순서
#
# columns 에 쓸 수 있는 값
#   내장 필드 : status, server, gpus, model, dataset, dataset_path, result_path,
#              checkpoint_path, device, input_size, duration, started_at,
#              latency_ms, throughput_fps, epochs, batch_size, lr, optimizer, notes
#   지표      : metrics 에 정의한 key (예: PSNR)
#   사용자 필드: options 에 직접 만든 이름 (예: scale) -> DB extra_json 에 저장됩니다
"""


def default_config_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "config", "options.yaml")


class OptionsConfig:
    """options.yaml 의 인메모리 표현. 변경 메서드는 즉시 파일에 저장한다."""

    def __init__(self, path: str | None = None, auto_create: bool = True) -> None:
        self.path = os.path.abspath(path or default_config_path())
        self.errors: list[str] = []
        self._data: dict[str, Any] = copy.deepcopy(BUILTIN)
        self._yaml = _RuamelYAML() if _BACKEND == "ruamel" else None
        if self._yaml is not None:  # pragma: no cover - 환경 의존
            self._yaml.preserve_quotes = True
            self._yaml.indent(mapping=2, sequence=4, offset=2)
        self.load(auto_create=auto_create)

    # -- 입출력 --------------------------------------------------------------
    def load(self, auto_create: bool = True) -> None:
        self.errors = []
        if not os.path.exists(self.path):
            if auto_create:
                self._data = copy.deepcopy(BUILTIN)
                try:
                    self.save()
                except OSError as exc:
                    self.errors.append(f"기본 설정 파일을 만들지 못했습니다: {exc}")
            return

        if _BACKEND == "none":
            self.errors.append(
                "PyYAML 또는 ruamel.yaml 이 없어 내장 기본값으로 실행합니다. "
                "`pip install -r requirements.txt` 로 설치하세요."
            )
            return

        try:
            with open(self.path, "r", encoding="utf-8") as fp:
                if self._yaml is not None:  # pragma: no cover
                    loaded = self._yaml.load(fp)
                else:
                    loaded = _pyyaml.safe_load(fp)
        except OSError as exc:
            self.errors.append(f"설정 파일을 읽지 못했습니다: {exc}")
            return
        except Exception as exc:  # YAML 문법 오류 - 사용자가 손으로 고치는 파일이다
            self.errors.append(f"설정 파일 문법 오류 ({os.path.basename(self.path)}): {exc}")
            return

        if not isinstance(loaded, dict):
            self.errors.append("설정 파일의 최상위가 매핑이 아닙니다. 기본값으로 실행합니다.")
            return

        self._data = loaded
        self._validate()

    def _validate(self) -> None:
        """치명적이지 않은 문제는 고쳐 넣고, 무엇을 고쳤는지 errors 에 남긴다."""
        data = self._data
        if not isinstance(data.get("tasks"), dict):
            data["tasks"] = copy.deepcopy(BUILTIN["tasks"])
            self.errors.append("`tasks` 항목이 없어 기본 Task 를 넣었습니다.")
        if not isinstance(data.get("defaults"), dict):
            data["defaults"] = copy.deepcopy(BUILTIN["defaults"])
        if not isinstance(data.get("servers"), list):
            data["servers"] = copy.deepcopy(BUILTIN["servers"])
            self.errors.append("`servers` 항목이 없어 기본 서버를 넣었습니다.")

        for name, raw in list(data["tasks"].items()):
            if not isinstance(raw, dict):
                self.errors.append(f"Task '{name}' 정의가 매핑이 아니라 무시했습니다.")
                data["tasks"].pop(name)
                continue
            raw.setdefault("options", {})
            raw.setdefault("metrics", [])
            raw.setdefault("columns", {})
            if not isinstance(raw["options"], dict):
                raw["options"] = {}
                self.errors.append(f"Task '{name}' 의 options 가 매핑이 아니라 비웠습니다.")
            if not isinstance(raw["metrics"], list):
                raw["metrics"] = []
                self.errors.append(f"Task '{name}' 의 metrics 가 리스트가 아니라 비웠습니다.")

    def save(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(self.path):  # 손으로 쓴 파일을 덮어쓰기 전에 한 벌 남긴다
            try:
                shutil.copy2(self.path, self.path + ".bak")
            except OSError:
                pass

        self._data["version"] = CONFIG_VERSION
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            if self._yaml is not None:  # pragma: no cover - 주석 보존 경로
                self._yaml.dump(self._data, fp)
            elif _BACKEND == "pyyaml":
                fp.write(HEADER_COMMENT)
                fp.write("\n")
                _pyyaml.safe_dump(
                    self._data, fp, allow_unicode=True, sort_keys=False, default_flow_style=False
                )
            else:
                raise OSError("YAML 백엔드가 없어 저장할 수 없습니다.")
        os.replace(tmp, self.path)

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
            options={k: [str(v) for v in (vals or [])] for k, vals in (raw.get("options") or {}).items()},
            metrics=[self._metric_def(m) for m in (raw.get("metrics") or []) if m],
            columns={k: [str(c) for c in (v or [])] for k, v in (raw.get("columns") or {}).items()},
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
        """해당 Task 에서 콤보박스를 만들 필드 목록 (기본값 + Task 전용)."""
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
        """DB 전용 컬럼이 없어 extra_json 으로 저장할 사용자 정의 필드."""
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
                gpus.append(GpuDef(index=index, type=str(gpu_raw.get("type", "GPU")), memory_gb=memory_gb))
            out.append(
                ServerDef(name=str(raw.get("name", "")), host=str(raw.get("host", "") or ""),
                          gpus=sorted(gpus, key=lambda g: g.index))
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
            self.save()

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
        self.save()
        return True

    def remove_option(self, task: str | None, field_name: str, value: str) -> bool:
        container = self._options_container(task, field_name, create=False)
        if not container or value not in container:
            return False
        container.remove(value)
        self.save()
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
        self.save()
        return True

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
        self.save()
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
                self.save()
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
                    self.save()
                    return True
        return False

    def update_metric(self, task: str, key: str, **changes: Any) -> bool:
        raw = self._task_raw(task)
        if raw is None:
            return False
        for item in raw.get("metrics") or []:
            if isinstance(item, dict) and item.get("key") == key:
                item.update(changes)
                self.save()
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
        self.save()

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
                self.save()
                return
        servers.append(payload)
        self.save()

    def remove_server(self, name: str) -> bool:
        servers = self._data.get("servers") or []
        for i, raw in enumerate(list(servers)):
            if isinstance(raw, dict) and raw.get("name") == name:
                servers.pop(i)
                self.save()
                return True
        return False

    def rename_server(self, old: str, new: str) -> bool:
        new = new.strip()
        if not new:
            return False
        for raw in self._data.get("servers") or []:
            if isinstance(raw, dict) and raw.get("name") == old:
                raw["name"] = new
                self.save()
                return True
        return False
