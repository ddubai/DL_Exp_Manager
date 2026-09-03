"""train.py 의 config.yaml / 학습 로그(loss.log 등)에서 값을 뽑아내는 파서.

프로젝트마다 config·로그 포맷이 다르므로 정답을 보장할 수 없다. 여러 흔한 스키마를
관대하게 시도하고, 못 찾으면 그냥 비워 두며(예외를 던지지 않는다), 결과는 항상
사용자가 폼에서 눈으로 확인하고 저장하는 구조라 오탐이 있어도 되돌리기 쉽다.

- `parse_train_config`  : BasicSR 류 config.yaml -> {model, dataset, batch_size, lr, ...}
- `parse_loss_log`      : 학습 로그 -> 곡선용 (iter, {지표: 값}) 목록 + 최근 검증 지표 + 소요 시간
"""
from __future__ import annotations

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

    model = _first(data, [("network_g", "type"), ("model", "type"), ("network", "type"), ("arch",)])
    if isinstance(model, str):
        out["model"] = model

    dataset = _first(data, [("datasets", "train", "name"), ("dataset", "name"), ("data", "name")])
    if isinstance(dataset, str):
        out["dataset"] = dataset

    dataset_path = _first(
        data,
        [
            ("datasets", "train", "dataroot_gt"),
            ("datasets", "train", "dataroot"),
            ("dataset", "path"),
            ("data", "root"),
        ],
    )
    if isinstance(dataset_path, str):
        out["dataset_path"] = dataset_path

    batch = _first(
        data,
        [
            ("datasets", "train", "batch_size_per_gpu"),
            ("datasets", "train", "batch_size"),
            ("train", "batch_size"),
        ],
    )
    if batch is not None:
        out["batch_size"] = str(batch)

    lr = _first(data, [("train", "optim_g", "lr"), ("train", "optim", "lr"), ("optim", "lr")])
    if lr is not None:
        out["lr"] = str(lr)

    optimizer = _first(
        data, [("train", "optim_g", "type"), ("train", "optim", "type"), ("optim", "type")]
    )
    if isinstance(optimizer, str):
        out["optimizer"] = optimizer

    epochs = _first(
        data,
        [("train", "total_iter"), ("train", "total_epoch"), ("train", "num_epoch")],
    )
    if epochs is not None:
        out["epochs"] = str(epochs)

    scale = _get(data, "scale")
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
    """학습 로그를 관대하게 파싱한다.

    - `... iter: 10,000 ... l_pix: 1.23e-02` 같은 학습 loss 줄  -> 곡선 포인트
    - `# psnr: 32.41  Best: ... @ 10000 iter` 같은 BasicSR 검증 줄
      -> 곡선 포인트(있으면) + 최근 검증 지표(latest_metrics)
    - 맨 앞/뒤 줄의 타임스탬프 차이 -> 대략적인 소요 시간

    형식을 못 알아봐도 예외 없이 빈 결과를 돌려준다.
    """
    result = LogParseResult()
    if not path or not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fp:
            text = fp.read(max_bytes)
    except OSError:
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

        iter_match = _ITER_RE.search(line)
        if not iter_match:
            continue
        try:
            iteration = int(iter_match.group(1).replace(",", ""))
        except ValueError:
            continue

        values: dict[str, float] = {}
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
