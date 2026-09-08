"""train.py 의 run_meta.json / config.yaml / 학습 로그(loss.log 등)에서 값을 뽑아내는 파서.

프로젝트마다 config·로그 포맷이 다르므로 정답을 보장할 수 없다. 여러 흔한 스키마를
관대하게 시도하고, 못 찾으면 그냥 비워 두며(예외를 던지지 않는다), 결과는 항상
사용자가 폼에서 눈으로 확인하고 저장하는 구조라 오탐이 있어도 되돌리기 쉽다.

- `parse_run_meta_text` : run_meta.json (붙여넣은 텍스트) -> {run_id, started_at, command,
  git_commit, algo, model, dataset}. Started At 자동 연동에 쓴다.
- `parse_train_config`  : config.yaml -> {model, dataset, batch_size, crop_size, lr, ...}.
  BasicSR 류(`datasets.train.*`) 를 기본으로 삼되, 최상위 섹션 이름(`datasets`/`data`/`dataset`)과
  crop 크기 필드 이름(`gt_size`/`crop_size`/`imagesize`/`image_size`/...) 이 다른 스키마도
  `_section_candidates()` 로 함께 시도한다.
- `parse_loss_log`      : 파일 경로로부터 학습 로그를 읽어 `parse_loss_log_text` 에 위임한다.
- `parse_loss_log_text` : 텍스트(파일이든 붙여넣기든) -> 곡선용 (iter/epoch, {지표: 값}) 목록 +
  최근 검증 지표 + 소요 시간. `iter: N` 관례와 `# key: value` (BasicSR) 외에
  `[Epoch N/Total] Average key: value / Average key2: value2 ...` 형태도 인식한다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - requirements.txt 에 PyYAML 이 있어 보통 없다
    _yaml = None


# ---------------------------------------------------------------------------
# run_meta.json
# ---------------------------------------------------------------------------
_RUN_META_KEYS: tuple[str, ...] = (
    "run_id", "started_at", "command", "git_commit", "algo", "model", "dataset",
)


def parse_run_meta_text(text: str) -> dict[str, str]:
    """run_meta.json 을 붙여넣은 텍스트에서 흔히 쓰는 키를 뽑는다.

    JSON 이 아니거나 깨졌거나, dict 가 아니면 예외 없이 빈 dict 를 돌려준다 -
    사용자가 아직 붙여넣기를 끝내지 않은 중간 상태일 수도 있어서다.
    """
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key in _RUN_META_KEYS:
        value = data.get(key)
        if value not in (None, ""):
            out[key] = str(value)
    return out


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------
def _get(data: dict[str, Any], *path: str) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first(data: dict[str, Any], candidates: list[tuple[str, ...]]) -> Any:
    for path in candidates:
        value = _get(data, *path)
        if value not in (None, ""):
            return value
    return None


# config.yaml 은 프로젝트마다 최상위 섹션 이름과 중첩 방식이 제각각이다
# (BasicSR 은 `datasets.train.*`, 어떤 프로젝트는 `data.train.*`, 또 어떤 건
# 섹션을 아예 안 나누고 `train.*` 에 다 넣는다). 필드 하나마다 경로를 일일이
# 손으로 나열하는 대신, "섹션 이름 후보 × train 중첩 여부 × 필드 이름 후보"를
# 조합해서 흔한 변형을 한 번에 시도한다.
_SECTION_ALIASES: tuple[str, ...] = ("datasets", "data", "dataset")


def _section_candidates(
    field_aliases: tuple[str, ...], bare_train: bool = False
) -> list[tuple[str, ...]]:
    """`{section}.train.{field}` 와 `{section}.{field}` 형태의 후보 경로들을 만든다.

    section 은 datasets/data/dataset 을 다 시도한다 - 예를 들어
    `data: {train: {imagesize: 256}}` 처럼 `data` 아래 `train` 을 두는 스키마도
    BasicSR 의 `datasets: {train: {gt_size: 256}}` 과 같은 방식으로 잡힌다.
    `bare_train=True` 면 섹션 없이 최상위 `train.{field}` 도 마지막으로 시도한다
    (섹션을 아예 안 나누는 config 용).
    """
    paths: list[tuple[str, ...]] = []
    for section in _SECTION_ALIASES:
        for field in field_aliases:
            paths.append((section, "train", field))
    for section in _SECTION_ALIASES:
        for field in field_aliases:
            paths.append((section, field))
    if bare_train:
        for field in field_aliases:
            paths.append(("train", field))
    return paths


def parse_train_config(path: str) -> dict[str, str]:
    """config.yaml 에서 흔히 쓰는 필드를 추정해 뽑는다 (BasicSR 스키마를 우선 시도).

    찾은 필드만 문자열로 채워 돌려준다. 못 찾은 필드는 아예 키에 없다 -
    호출부가 "찾은 것만 채우기"를 하기 쉽도록.
    """
    if _yaml is None or not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = _yaml.safe_load(fp)
    except Exception:  # noqa: BLE001 - 손으로 쓴 파일, 문법 오류는 항상 있을 수 있다
        return {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, str] = {}

    model = _first(
        data,
        [
            ("network_g", "type"), ("model", "type"), ("network", "type"), ("arch",),
            ("model", "name"), ("model", "arch"),
        ],
    )
    if isinstance(model, str):
        out["model"] = model

    dataset = _first(data, _section_candidates(("name", "dataset_name", "dataset")))
    if isinstance(dataset, str):
        out["dataset"] = dataset

    dataset_path = _first(
        data,
        _section_candidates(
            ("dataroot_gt", "dataroot", "root", "path", "dir", "data_root", "data_dir")
        ),
    )
    if isinstance(dataset_path, str):
        out["dataset_path"] = dataset_path

    batch = _first(
        data,
        _section_candidates(("batch_size_per_gpu", "batch_size", "batch"), bare_train=True),
    )
    if batch is not None:
        out["batch_size"] = str(batch)

    # "imagesize"/"image_size" 처럼 학습 crop 크기를 부르는 이름이 프로젝트마다 다르다
    # (예: `data: {train: {imagesize: 256}}`) - 흔한 표기를 전부 후보에 넣는다.
    crop_size = _first(
        data,
        _section_candidates(
            (
                "gt_size", "crop_size", "patch_size",
                "imagesize", "image_size", "img_size", "input_size",
            ),
            bare_train=True,
        ),
    )
    if crop_size is not None:
        out["crop_size"] = str(crop_size)

    lr = _first(
        data,
        [
            ("train", "optim_g", "lr"), ("train", "optim", "lr"), ("optim", "lr"),
            ("train", "lr"), ("train", "learning_rate"), ("optimizer", "lr"), ("learning_rate",),
        ],
    )
    if lr is not None:
        out["lr"] = str(lr)

    optimizer = _first(
        data,
        [
            ("train", "optim_g", "type"), ("train", "optim", "type"), ("optim", "type"),
            ("train", "optimizer"), ("optimizer", "type"), ("optimizer", "name"),
            ("train", "optim_type"),
        ],
    )
    if isinstance(optimizer, str):
        out["optimizer"] = optimizer

    epochs = _first(
        data,
        [
            ("train", "total_iter"), ("train", "total_epoch"), ("train", "num_epoch"),
            ("train", "epochs"), ("train", "num_epochs"), ("train", "max_epochs"),
            ("epochs",), ("num_epochs",),
        ],
    )
    if epochs is not None:
        out["epochs"] = str(epochs)

    scale = _first(data, [("scale",), *_section_candidates(("scale",))])
    if scale is not None:
        out["scale"] = str(scale)

    return out


# ---------------------------------------------------------------------------
# 학습 로그 (loss.log 등)
# ---------------------------------------------------------------------------
_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
_ITER_RE = re.compile(r"\biter[:=]\s*([\d,]+)", re.IGNORECASE)
_AT_ITER_RE = re.compile(r"@\s*([\d,]+)\s*iter", re.IGNORECASE)
_KV_RE = re.compile(r"([A-Za-z][\w\-]*)\s*[:=]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_HASH_METRIC_RE = re.compile(r"#\s*([A-Za-z][\w\-]*)\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
# `[Epoch 33/1000]` 처럼 대괄호 안에 현재/전체 epoch 을 적는 관례 (요청자의 _loss_log.txt 형식).
_EPOCH_RE = re.compile(r"\[?\s*epoch\s*[:=]?\s*(\d+)\s*/\s*\d+\s*\]?", re.IGNORECASE)
# `Average loss:14.4 / Average psnr: 15 / Average ssim:0.3 / ...` - "Average " 뒤의 이름을 키로 쓴다.
# 지표 이름 뒤에 방향 표시(`psnr(↑)`, `ssim(윗 화살표)`, 단위(`(dB)`) 등)가 붙기도 하므로,
# 콜론 앞까지 "/", ":", "=", 줄바꿈이 아닌 건 뭐든 건너뛰고 값을 찾는다. 반각(ASCII)
# 괄호뿐 아니라 한글 입력기로 흔히 붙는 전각 괄호(（）)나 공백이 낀 덧말도 이 방식이면
# 별도 패턴을 추가하지 않고 그대로 통과한다 - 이전엔 괄호/대괄호 쌍만 명시적으로
# 허용해서 전각 괄호가 섞이면 그 지표 하나만 조용히 파싱에서 빠졌다.
_AVERAGE_KV_RE = re.compile(
    r"average\s+([A-Za-z][\w\-]*)[^/:=\n]*?[:=]\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

# 학습 곡선에 남길 만한 흔한 손실/지표 이름 (iter: 줄에서 KV 로 잡히는 잡음을 거른다)
_CURVE_KEYS = {
    "loss", "l_pix", "l_g", "l_d", "l_total", "l_percep", "l_style",
    "psnr", "ssim", "lpips", "niqe", "lr",
    "top-1", "top1", "top-5", "top5", "acc", "accuracy", "nmi", "ari", "miou",
}

# 대표적인 축약을 화면에 보여줄 이름으로. 모르는 키는 그대로(제목만 다듬어) 쓴다.
_CANONICAL_NAMES = {
    "psnr": "PSNR", "ssim": "SSIM", "lpips": "LPIPS", "niqe": "NIQE",
    "top-1": "Top-1", "top1": "Top-1", "top-5": "Top-5", "top5": "Top-5",
    "acc": "Accuracy", "accuracy": "Accuracy",
    "nmi": "NMI", "ari": "ARI", "miou": "mIoU",
    "loss": "Loss", "l_pix": "l_pix", "lr": "LR",
}


def canonical_metric_name(key: str) -> str:
    return _CANONICAL_NAMES.get(key.strip().lower(), key.strip())


@dataclass
class LogParseResult:
    points: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    latest_metrics: dict[str, float] = field(default_factory=dict)
    duration_sec: float | None = None


def parse_loss_log(path: str, max_bytes: int = 4_000_000) -> LogParseResult:
    """파일 경로로부터 학습 로그를 읽어 `parse_loss_log_text` 에 위임한다.

    형식을 못 알아봐도, 파일이 없어도 예외 없이 빈 결과를 돌려준다.
    """
    if not path or not os.path.isfile(path):
        return LogParseResult()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read(max_bytes)
    except OSError:
        return LogParseResult()
    return parse_loss_log_text(text)


def parse_loss_log_text(text: str) -> LogParseResult:
    """학습 로그 텍스트(파일 내용이든 붙여넣기든)를 관대하게 파싱한다.

    - `... iter: 10,000 ... l_pix: 1.23e-02` 같은 학습 loss 줄  -> 곡선 포인트
    - `# psnr: 32.41  Best: ... @ 10000 iter` 같은 BasicSR 검증 줄
      -> 곡선 포인트(있으면) + 최근 검증 지표(latest_metrics)
    - `[Epoch 33/1000] Average loss:14.4 / Average psnr: 15 / Average ssim:0.3 / ...`
      같은 epoch 단위 요약 줄 -> 곡선 포인트(x=epoch) + 최근 검증 지표
    - 맨 앞/뒤 줄의 타임스탬프 차이 -> 대략적인 소요 시간

    형식을 못 알아봐도 예외 없이 빈 결과를 돌려준다.
    """
    result = LogParseResult()
    if not text:
        return result

    first_ts: str | None = None
    last_ts: str | None = None

    for line in text.splitlines():
        ts_match = _TIMESTAMP_RE.search(line)
        if ts_match:
            first_ts = first_ts or ts_match.group(1)
            last_ts = ts_match.group(1)

        hash_match = _HASH_METRIC_RE.search(line)
        if hash_match:
            key, value_text = hash_match.group(1), hash_match.group(2)
            try:
                value = float(value_text)
            except ValueError:
                continue
            result.latest_metrics[key] = value
            at_iter = _AT_ITER_RE.search(line)
            if at_iter:
                try:
                    iteration = int(at_iter.group(1).replace(",", ""))
                    result.points.append((iteration, {key: value}))
                except ValueError:
                    pass
            continue

        epoch_match = _EPOCH_RE.search(line)
        if epoch_match:
            values = {
                key: float(value_text)
                for key, value_text in _AVERAGE_KV_RE.findall(line)
                if _is_float(value_text)
            }
            if values:
                try:
                    epoch = int(epoch_match.group(1))
                    result.points.append((epoch, values))
                except ValueError:
                    pass
                result.latest_metrics.update(values)
            continue

        iter_match = _ITER_RE.search(line)
        if not iter_match:
            continue
        try:
            iteration = int(iter_match.group(1).replace(",", ""))
        except ValueError:
            continue

        values = {}
        for key, value_text in _KV_RE.findall(line):
            key_norm = key.lower()
            if key_norm in ("iter", "epoch") or key_norm not in _CURVE_KEYS:
                continue
            try:
                values[key] = float(value_text)
            except ValueError:
                continue
        if values:
            result.points.append((iteration, values))

    if first_ts and last_ts and first_ts != last_ts:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                start = datetime.strptime(first_ts, fmt)
                end = datetime.strptime(last_ts, fmt)
                result.duration_sec = max(0.0, (end - start).total_seconds())
                break
            except ValueError:
                continue

    return result


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
