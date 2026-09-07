"""예시 실험 데이터 - 처음 실행 시 UI 를 바로 확인하거나, 기능을 눈으로 훑어볼 때 쓴다.

4개 Task(SuperResolution/Denoising/Clustering/Classification) 아래 여러 Work 에 걸쳐
학습/평가 기록을 채운다. 실제 앱이 쓰는 기능을 최대한 그대로 통과시킨다:

- 실행 명령어는 손으로 쓴 문자열이 아니라 `config.command_template()` +
  `command_builder.render_command()` 로 **진짜 생성**한다(Task 파일을 손으로
  고치면 여기서 만드는 명령어도 그대로 바뀐다).
- Evaluation 은 `source_train_run_id`/`checkpoint_epoch` 로 Train Run 에 연결해
  "이 학습으로 평가 만들기" 이후의 상태를 재현한다.
- 즐겨찾기/태그/실패 사유/중복(`duplicate_run`)/수정(`update_run`) 을 실제로 한 번씩
  호출해 History 탭에 created/updated/duplicated 항목이 고르게 남게 한다.

`with_local_assets=True` 를 주면 몇 개의 대표 Run 에는 **실제로 디스크에 존재하는**
결과 폴더(config.yaml + 학습 로그 + 결과 이미지 한 장)를 만들어 result_path 를
그리로 돌린다 - ⇪ Parse / 📈 Training Curve / 🖼 View Image / 로그 tail 뷰어가
가짜 경로가 아니라 진짜 파일을 상대로 동작하는 걸 확인할 수 있다. 이미지는 외부
이미징 라이브러리 없이 PNG 를 손으로 인코딩한다(`_write_gradient_png`) - 이 모듈은
Qt 를 포함해 아무 의존성도 없는 순수 데이터 계층으로 남겨 둔다.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from datetime import datetime, timedelta
from typing import Any

from . import constants as C
from .command_builder import ParamStyle, render_command
from .config_store import DEFAULT_COMMANDS, OptionsConfig
from .db import Database
from .utils import parse_gpu_count

_RESULTS_DIRNAME = "sample_results"


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_results_root() -> str:
    return os.path.join(_project_root(), _RESULTS_DIRNAME)


def _ts(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).replace(microsecond=0).isoformat(sep=" ")


# ---------------------------------------------------------------------------
# 실행 명령어 - 진짜 Task 템플릿으로 만든다
# ---------------------------------------------------------------------------
def _command_values(
    config: OptionsConfig | None, task: str, work_name: str, row: dict[str, Any], mode: str
) -> dict[str, Any]:
    task_def = config.task(task) if config else None
    task_short = (task_def.short if task_def else "") or task
    server_name = str(row.get("server") or "")
    server = config.server(server_name) if config and server_name else None
    gpu_count = parse_gpu_count(row.get("gpu_indices"))

    values: dict[str, Any] = {
        "task": task,
        "task_lower": task.lower(),
        "task_short": task_short,
        "work": work_name,
        "model": row.get("model", ""),
        "dataset": row.get("dataset", ""),
        "dataset_path": row.get("dataset_path", ""),
        "result_path": row.get("result_path", ""),
        "server": server_name,
        "host": server.host if server else "",
        "gpus": str(gpu_count) if gpu_count else "",
        "cuda_devices": ",".join(str(i) for i in range(gpu_count)) if gpu_count else "",
        "status": row.get("status", ""),
    }
    if mode == "train":
        values.update(
            {
                "epochs": row.get("epochs", ""),
                "batch_size": row.get("batch_size", ""),
                "crop_size": row.get("crop_size", ""),
                "lr": row.get("lr", ""),
                "optimizer": row.get("optimizer", ""),
            }
        )
    else:
        values.update(
            {
                "checkpoint_path": row.get("checkpoint_path", ""),
                "checkpoint_epoch": row.get("checkpoint_epoch", ""),
                "device": row.get("device", ""),
                "input_size": row.get("input_size", ""),
            }
        )
    values.update(row.get("extra_json") or {})
    return values


def _build_command(
    config: OptionsConfig | None, task: str, work_name: str, mode: str, row: dict[str, Any]
) -> str:
    """이 Run 이 실제로 만들어졌을 때 "⚙ Generate" 가 만들었을 명령어를 재현한다."""
    if config is not None:
        template = config.command_template(task, mode)
        style = config.param_style()
    else:
        template = DEFAULT_COMMANDS.get(mode, "")
        style = ParamStyle()
    return render_command(template, _command_values(config, task, work_name, row, mode), style).text


# ---------------------------------------------------------------------------
# 로컬 결과 폴더 - Parse / Training Curve / View Image 를 진짜로 시험해 볼 수 있게
# ---------------------------------------------------------------------------
def _write_png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _write_gradient_png(
    path: str, width: int, height: int, c1: tuple[int, int, int], c2: tuple[int, int, int]
) -> None:
    """대각선 그라디언트 PNG 를 손으로 인코딩한다(외부 이미징 라이브러리 없이)."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # 스캔라인 필터: None
        ty = y / max(1, height - 1)
        for x in range(width):
            tx = x / max(1, width - 1)
            t = (tx + ty) / 2
            raw += bytes(
                int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)
            )
    png = b"\x89PNG\r\n\x1a\n"
    png += _write_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _write_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _write_png_chunk(b"IEND", b"")
    with open(path, "wb") as fp:
        fp.write(png)


def _write_demo_config(path: str, model: str, dataset: str, dataset_path: str, lr: str, batch_size: str,
                        crop_size: str, total_iter: str, scale: str = "") -> None:
    scale_line = f"scale: {scale}\n" if scale else ""
    content = f"""\
network_g:
  type: {model}
datasets:
  train:
    name: {dataset}
    dataroot_gt: {dataset_path}
    batch_size_per_gpu: {batch_size or 8}
    gt_size: {crop_size or 256}
  val:
    name: {dataset}-val
train:
  total_iter: {total_iter or 300000}
  optim_g:
    type: AdamW
    lr: !!float {lr or "3e-4"}
{scale_line}"""
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


def _write_demo_log(
    path: str, metric_a: tuple[str, float], metric_b: tuple[str, float],
    total_iter: int, days_ago: float,
) -> None:
    """`log_parser.parse_loss_log()` 가 곡선으로 뽑을 수 있는 형태의 학습 로그.

    `metric_a`/`metric_b` 는 (지표 이름, 최종값) - Task 마다 이름이 다르므로
    (PSNR/SSIM, Top-1/Top-5, NMI/ARI, ...) 호출부가 지정한다. 둘 다
    `log_parser._CURVE_KEYS` 에 있는 이름이어야 곡선으로 잡힌다.
    """
    start = datetime.now() - timedelta(days=days_ago)
    lines: list[str] = []
    step = max(1, total_iter // 30)
    loss = 0.085
    (name_a, final_a), (name_b, final_b) = metric_a, metric_b
    for i, it in enumerate(range(step, total_iter + 1, step)):
        loss *= 0.93
        ts = (start + timedelta(minutes=4 * i)).replace(microsecond=0).isoformat(sep=" ")
        lines.append(f"{ts} iter: {it:,} lr: 3.000e-04 l_pix: {loss:.4e}")
        if i % 5 == 4:  # 다섯 스텝마다 검증 체크포인트
            frac = it / total_iter
            value_a = final_a * (0.85 + 0.15 * frac)
            value_b = final_b * (0.85 + 0.15 * frac)
            lines.append(f"{ts} # {name_a}: {value_a:.4g} Best: {final_a:.4g} @ {it:,} iter")
            lines.append(f"{ts} # {name_b}: {value_b:.4g} Best: {final_b:.4g} @ {it:,} iter")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def _slug(*parts: str) -> str:
    cleaned = "_".join(p.strip().lower().replace(" ", "-").replace("/", "-") for p in parts if p)
    return cleaned or "run"


def _make_local_asset(
    results_root: str, task: str, work: str, row: dict[str, Any],
    total_iter: int, days_ago: float, curve_metrics: tuple[tuple[str, float], tuple[str, float]],
) -> str:
    """대표 Run 하나의 결과 폴더를 실제로 만들고, 그 경로를 돌려준다.

    `row` 는 `db.get_run()` 이 돌려준 원본 행이라 `metrics_json`/`extra_json` 이 아직
    JSON 문자열이다(폼에 채울 때만 파싱된 dict). 여기서 직접 걷어낸다.
    `curve_metrics` 는 로그 곡선에 쓸 두 지표 - Task 마다 이름이 달라(PSNR/SSIM,
    Top-1/Top-5, NMI/ARI, ...) 자동으로 추측하는 대신 호출부가 명시한다.
    """
    folder = os.path.join(results_root, task, work, _slug(row["model"], row.get("dataset", "")))
    os.makedirs(folder, exist_ok=True)

    extra = json.loads(row.get("extra_json") or "{}")

    _write_demo_config(
        os.path.join(folder, "config.yaml"),
        model=row["model"], dataset=row.get("dataset", ""), dataset_path=row.get("dataset_path", ""),
        lr=row.get("lr", ""), batch_size=row.get("batch_size", ""), crop_size=row.get("crop_size", ""),
        total_iter=str(total_iter), scale=str(extra.get("scale", "")),
    )
    _write_demo_log(os.path.join(folder, "train.log"), curve_metrics[0], curve_metrics[1], total_iter, days_ago)
    _write_gradient_png(
        os.path.join(folder, "result_comparison.png"), 512, 320,
        c1=(24, 34, 64), c2=(45, 212, 191),  # bg -> cta 톤, 이 앱의 테마 색과 맞춘다
    )
    return folder


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
def populate(
    db: Database,
    config: OptionsConfig | None = None,
    with_local_assets: bool = False,
    results_root: str | None = None,
) -> int:
    """샘플 Train / Evaluation 기록을 넣고 추가한 건수를 돌려준다.

    `config` 를 주면 실제 Task 템플릿으로 실행 명령어를 생성한다(권장) - 안 주면
    내장 기본 템플릿으로 대체한다. `with_local_assets=True` 면 대표 Run 몇 개에
    한해 실제 결과 폴더(config.yaml + 로그 + 이미지)도 디스크에 만든다.
    """
    results_root = results_root or default_results_root()
    count = 0
    local_asset_targets: list[tuple[str, str, int, dict[str, Any], int, float]] = []

    def add_train(task: str, work_id: int, work_name: str, row: dict[str, Any]) -> int:
        nonlocal count
        row = dict(row)
        row["exec_command"] = _build_command(config, task, work_name, "train", row)
        run_id = db.insert_run("train", {**row, "work_id": work_id})
        count += 1
        return run_id

    def add_eval(task: str, work_id: int, work_name: str, row: dict[str, Any]) -> int:
        nonlocal count
        row = dict(row)
        row["exec_command"] = _build_command(config, task, work_name, "evaluation", row)
        run_id = db.insert_run("evaluation", {**row, "work_id": work_id})
        count += 1
        return run_id

    # == Task: SuperResolution ===============================================
    sr = db.add_task("SuperResolution", "Super Resolution")
    ssl2sl = db.add_work(sr, "SSL2SL", "Self-supervised -> Supervised transfer experiment")
    bsr = db.add_work(sr, "BSR-x4", "Blind SR x4 baseline")
    light = db.add_work(sr, "LightSR-Mobile", "Lightweight models for on-device inference")

    db.add_dataset(ssl2sl, "DIV2K", path="/mnt/data/DIV2K/train", sample_count=800,
                    image_size="varies", extension="png", created_at=_ts(20))
    db.add_dataset(ssl2sl, "DF2K", path="/mnt/data/DF2K/train", sample_count=3450,
                    image_size="varies", extension="png", notes="DIV2K + Flickr2K merged.", created_at=_ts(15))
    db.add_dataset(bsr, "DF2K", path="/mnt/data/DF2K/train", sample_count=3450,
                    image_size="varies", extension="png", created_at=_ts(14))
    db.add_dataset(light, "DIV2K", path="/mnt/data/DIV2K/train", sample_count=800,
                    image_size="varies", extension="png", created_at=_ts(8))

    sr_train = {
        "restormer": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 1", "model": "Restormer", "dataset": "DIV2K",
            "gpu_indices": "0,1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/SSL2SL/restormer_x4",
            "status": C.STATUS_DONE, "started_at": _ts(6), "duration_sec": 19 * 3600 + 42 * 60,
            "epochs": "300000", "batch_size": "8", "crop_size": "256", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 32.41, "SSIM": 0.8993, "LPIPS": 0.121},
            "config_yaml": C.SAMPLE_CONFIG_YML,
            "notes": "Baseline. Best result at iter 285000 (val_freq 5000).",
            "favorite": 1, "tags": "baseline",
        }),
        "swinir": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 2", "model": "SwinIR", "dataset": "DF2K",
            "gpu_indices": "0,1,2,3", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/swinir_x4",
            "status": C.STATUS_DONE, "started_at": _ts(4), "duration_sec": 27 * 3600,
            "epochs": "500000", "batch_size": "16", "crop_size": "256", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 32.72, "SSIM": 0.9031, "LPIPS": 0.118},
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "SwinIR"),
            "notes": "Expanded to DF2K data. +0.31dB over Restormer.", "tags": "ablation,data-scale",
        }),
        "mambair": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 3", "model": "MambaIR", "dataset": "DF2K",
            "gpu_indices": "0,1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/mambair_x4",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.35), "duration_sec": None,
            "epochs": "400000", "batch_size": "8", "crop_size": "192", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 32.55, "SSIM": 0.9012},
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "MambaIR"),
            "notes": "In progress. Intermediate metrics at 220k iter.",
        }),
        "hat": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 4", "model": "HAT", "dataset": "DF2K",
            "gpu_indices": "0,1,2,3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/hat_x2",
            "status": C.STATUS_FAILED, "started_at": _ts(2), "duration_sec": 41 * 60,
            "epochs": "800000", "batch_size": "32", "crop_size": "192", "lr": "1e-4", "optimizer": "AdamW",
            "metrics_json": {},
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "HAT"),
            "notes": "CUDA OOM (batch 32). Needs retry with batch 16.",
            "failure_reason": "CUDA out of memory (rank 2, batch_size_per_gpu=32).",
        }),
        "edsr": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 3", "model": "EDSR", "dataset": "DIV2K",
            "gpu_indices": "2,3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/SSL2SL/edsr_x2",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.1), "duration_sec": None,
            "epochs": "300000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 34.02},
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "EDSR"),
            "notes": "Running alongside MambaIR on Server 3 (GPU 2,3).",
        }),
        "rcan": add_train("SuperResolution", ssl2sl, "SSL2SL", {
            "server": "Server 1", "model": "RCAN", "dataset": "DF2K",
            "gpu_indices": "1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/rcan_x4",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "500000", "batch_size": "16", "crop_size": "192", "lr": "1e-4", "optimizer": "Adam",
            "metrics_json": {}, "notes": "Queued behind Restormer/HAT on Server 1.",
        }),

        "bsr_restormer": add_train("SuperResolution", bsr, "BSR-x4", {
            "server": "Server 2", "model": "Restormer", "dataset": "DF2K",
            "gpu_indices": "0,1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/BSR-x4/restormer",
            "status": C.STATUS_DONE, "started_at": _ts(9), "duration_sec": 22 * 3600,
            "epochs": "300000", "batch_size": "8", "crop_size": "256", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 31.98, "SSIM": 0.8890, "LPIPS": 0.134},
            "notes": "Blind degradation model (random kernel + noise + JPEG).",
            "tags": "blind-degradation",
        }),
        "bsr_swinir": add_train("SuperResolution", bsr, "BSR-x4", {
            "server": "Server 4", "model": "SwinIR", "dataset": "DF2K",
            "gpu_indices": "0,1,2,3", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/BSR-x4/swinir",
            "status": C.STATUS_DONE, "started_at": _ts(7), "duration_sec": 26 * 3600,
            "epochs": "500000", "batch_size": "16", "crop_size": "256", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 32.10, "SSIM": 0.8905, "LPIPS": 0.130},
            "notes": "Best blind-degradation result so far.", "favorite": 1,
        }),
        "bsr_mambair": add_train("SuperResolution", bsr, "BSR-x4", {
            "server": "Server 1", "model": "MambaIR", "dataset": "DIV2K",
            "gpu_indices": "2", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/BSR-x4/mambair",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "400000", "batch_size": "8", "crop_size": "192", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {}, "notes": "Queued - waiting for Server 1 GPU 2.",
        }),
        "bsr_edsr": add_train("SuperResolution", bsr, "BSR-x4", {
            "server": "Server 3", "model": "EDSR", "dataset": "DIV2K",
            "gpu_indices": "0", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/BSR-x4/edsr_x2",
            "status": C.STATUS_DONE, "started_at": _ts(11), "duration_sec": 8 * 3600 + 20 * 60,
            "epochs": "300000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 34.02, "SSIM": 0.9210, "LPIPS": 0.098},
            "notes": "Lighter model, quick benchmark run.",
        }),

        "light_edsr": add_train("SuperResolution", light, "LightSR-Mobile", {
            "server": "Server 2", "model": "EDSR", "dataset": "DIV2K",
            "gpu_indices": "3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/LightSR-Mobile/edsr_x2",
            "status": C.STATUS_DONE, "started_at": _ts(3), "duration_sec": 5 * 3600 + 10 * 60,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 33.55, "SSIM": 0.9145, "LPIPS": 0.142},
            "notes": "Mobile-friendly baseline, ~1.5M params.",
        }),
        "light_rcan": add_train("SuperResolution", light, "LightSR-Mobile", {
            "server": "Server 2", "model": "RCAN", "dataset": "DIV2K",
            "gpu_indices": "3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/LightSR-Mobile/rcan_x2",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.2), "duration_sec": None,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {}, "notes": "Running right after EDSR on the same GPU.",
        }),
        "light_restormer": add_train("SuperResolution", light, "LightSR-Mobile", {
            "server": "Server 4", "model": "Restormer", "dataset": "DIV2K",
            "gpu_indices": "2", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/LightSR-Mobile/restormer_x2",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "200000", "batch_size": "8", "crop_size": "128", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {}, "notes": "Queued - control run for the mobile study.",
        }),
    }

    def _sr_eval(work_id: int, work_name: str, source_key: str, bench: str, latency: float,
                 fps: float, metrics: dict[str, float], extra: dict[str, Any] | None = None,
                 favorite: int = 0) -> None:
        source = sr_train[source_key]
        src_row = db.get_run("train", source)
        source_extra = json.loads(src_row.get("extra_json") or "{}")
        add_eval("SuperResolution", work_id, work_name, {
            "server": src_row["server"], "model": src_row["model"],
            "checkpoint_path": f"{src_row['result_path']}/models/net_g_{src_row['epochs']}.pth",
            "checkpoint_epoch": src_row["epochs"], "source_train_run_id": source,
            "gpu_indices": "0", "extra_json": extra if extra is not None else source_extra,
            "dataset": bench, "dataset_path": f"/mnt/data/benchmark/{bench}/LR",
            "result_path": f"{src_row['result_path']}/results/{bench}",
            "device": "cuda:0", "input_size": "3x256x256", "latency_ms": latency, "throughput_fps": fps,
            "status": C.STATUS_DONE, "started_at": _ts(2), "duration_sec": 96, "metrics_json": metrics,
            "notes": f"{bench} benchmark.", "favorite": favorite,
        })

    _sr_eval(ssl2sl, "SSL2SL", "restormer", "Set5", 41.7, 23.98,
             {"PSNR": 32.41, "SSIM": 0.8993, "LPIPS": 0.121}, favorite=1)
    _sr_eval(ssl2sl, "SSL2SL", "restormer", "Urban100", 168.0, 5.95,
             {"PSNR": 27.02, "SSIM": 0.8115, "LPIPS": 0.169})
    _sr_eval(ssl2sl, "SSL2SL", "swinir", "Set5", 88.2, 11.34,
             {"PSNR": 32.66, "SSIM": 0.9022, "LPIPS": 0.117})
    _sr_eval(ssl2sl, "SSL2SL", "swinir", "Urban100", 210.5, 4.75,
             {"PSNR": 27.05, "SSIM": 0.8142, "LPIPS": 0.163})
    _sr_eval(bsr, "BSR-x4", "bsr_restormer", "Set14", 44.1, 22.68,
             {"PSNR": 28.55, "SSIM": 0.7920, "LPIPS": 0.201})
    _sr_eval(bsr, "BSR-x4", "bsr_swinir", "Set14", 90.4, 11.06,
             {"PSNR": 28.80, "SSIM": 0.7965, "LPIPS": 0.192}, favorite=1)
    _sr_eval(bsr, "BSR-x4", "bsr_edsr", "BSD100", 18.2, 54.9,
             {"PSNR": 27.30, "SSIM": 0.7610, "LPIPS": 0.230})
    _sr_eval(light, "LightSR-Mobile", "light_edsr", "Set5", 6.4, 156.3,
             {"PSNR": 33.20, "SSIM": 0.9098, "LPIPS": 0.150}, favorite=1)
    _sr_eval(light, "LightSR-Mobile", "light_edsr", "Set14", 7.1, 140.8,
             {"PSNR": 29.85, "SSIM": 0.8455, "LPIPS": 0.198})

    if with_local_assets:
        row = db.get_run("train", sr_train["restormer"])
        folder = _make_local_asset(
            results_root, "SuperResolution", "SSL2SL", row, 300000, 6,
            curve_metrics=(("PSNR", 32.41), ("SSIM", 0.8993)),
        )
        db.update_run("train", sr_train["restormer"], {
            **row,
            "result_path": folder, "config_yaml": C.SAMPLE_CONFIG_YML,
        })

    # == Task: Denoising ======================================================
    dn = db.add_task("Denoising", "Denoising")
    n2n = db.add_work(dn, "N2N-Base", "Noise2Noise baseline")
    real_noise = db.add_work(dn, "RealNoise-SIDD", "Real (non-synthetic) noise, high sigma regime")

    db.add_dataset(n2n, "SIDD", path="/mnt/data/SIDD/train", sample_count=320,
                    image_size="varies", extension="png", created_at=_ts(18))
    db.add_dataset(n2n, "DND", path="/mnt/data/DND/train", sample_count=50,
                    image_size="varies", extension="png", created_at=_ts(12))
    db.add_dataset(real_noise, "SIDD", path="/mnt/data/SIDD/train", sample_count=320,
                    image_size="varies", extension="png", created_at=_ts(5))

    dn_train = {
        "nafnet": add_train("Denoising", n2n, "N2N-Base", {
            "server": "Server 1", "model": "NAFNet", "dataset": "SIDD",
            "gpu_indices": "2", "extra_json": {"noise_sigma": "25", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/N2N-Base/nafnet",
            "status": C.STATUS_DONE, "started_at": _ts(10), "duration_sec": 6 * 3600 + 12 * 60,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "1e-3", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 31.08, "SSIM": 0.8812},
            "notes": "Noise2Noise baseline, sigma=25.", "favorite": 1, "tags": "baseline",
        }),
        "restormer": add_train("Denoising", n2n, "N2N-Base", {
            "server": "Server 2", "model": "Restormer", "dataset": "SIDD",
            "gpu_indices": "0,1", "extra_json": {"noise_sigma": "25", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/N2N-Base/restormer",
            "status": C.STATUS_DONE, "started_at": _ts(8), "duration_sec": 14 * 3600,
            "epochs": "300000", "batch_size": "8", "crop_size": "128", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 31.45, "SSIM": 0.8850},
            "notes": "+0.37dB over NAFNet, ~2.3x slower to train.",
        }),
        "scunet": add_train("Denoising", n2n, "N2N-Base", {
            "server": "Server 3", "model": "SCUNet", "dataset": "SIDD",
            "gpu_indices": "1", "extra_json": {"noise_sigma": "50", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/N2N-Base/scunet_s50",
            "status": C.STATUS_DONE, "started_at": _ts(6), "duration_sec": 9 * 3600 + 40 * 60,
            "epochs": "250000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 29.02, "SSIM": 0.8420},
            "notes": "High-noise setting (sigma=50), expected lower PSNR.",
        }),
        "uformer": add_train("Denoising", n2n, "N2N-Base", {
            "server": "Server 4", "model": "Uformer", "dataset": "SIDD",
            "gpu_indices": "0", "extra_json": {"noise_sigma": "15", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/N2N-Base/uformer_s15",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.6), "duration_sec": None,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "1e-3", "optimizer": "AdamW",
            "metrics_json": {}, "notes": "Low-noise setting, running.",
        }),
        "nafnet_dnd": add_train("Denoising", n2n, "N2N-Base", {
            "server": "Server 1", "model": "NAFNet", "dataset": "DND",
            "gpu_indices": "3", "extra_json": {"noise_sigma": "25", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/DND/train", "result_path": "/mnt/exp/N2N-Base/nafnet_dnd",
            "status": C.STATUS_FAILED, "started_at": _ts(1), "duration_sec": 18 * 60,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "1e-3", "optimizer": "AdamW",
            "metrics_json": {}, "notes": "Crashed after epoch 3.",
            "failure_reason": "Dataloader crashed after epoch 3 (corrupt .tif in shard 07).",
        }),

        "real_restormer": add_train("Denoising", real_noise, "RealNoise-SIDD", {
            "server": "Server 2", "model": "Restormer", "dataset": "SIDD",
            "gpu_indices": "2,3", "extra_json": {"noise_sigma": "50", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/RealNoise-SIDD/restormer",
            "status": C.STATUS_DONE, "started_at": _ts(5), "duration_sec": 16 * 3600,
            "epochs": "300000", "batch_size": "8", "crop_size": "128", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 27.55, "SSIM": 0.7920},
            "notes": "Real (camera) noise, high-sigma regime.",
        }),
        "real_nafnet": add_train("Denoising", real_noise, "RealNoise-SIDD", {
            "server": "Server 3", "model": "NAFNet", "dataset": "SIDD",
            "gpu_indices": "0", "extra_json": {"noise_sigma": "50", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/RealNoise-SIDD/nafnet",
            "status": C.STATUS_DONE, "started_at": _ts(4), "duration_sec": 6 * 3600 + 30 * 60,
            "epochs": "200000", "batch_size": "16", "crop_size": "128", "lr": "1e-3", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 27.88, "SSIM": 0.7965},
            "notes": "Best real-noise result so far.", "favorite": 1,
        }),
        "real_scunet": add_train("Denoising", real_noise, "RealNoise-SIDD", {
            "server": "Server 4", "model": "SCUNet", "dataset": "SIDD",
            "gpu_indices": "1", "extra_json": {"noise_sigma": "50", "algo": "noise2noise"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/RealNoise-SIDD/scunet",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "250000", "batch_size": "16", "crop_size": "128", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {}, "notes": "Queued.",
        }),
    }

    def _dn_eval(work_id: int, work_name: str, source_key: str, bench: str, sigma: str,
                 latency: float, metrics: dict[str, float], favorite: int = 0) -> None:
        source = dn_train[source_key]
        src_row = db.get_run("train", source)
        add_eval("Denoising", work_id, work_name, {
            "server": src_row["server"], "model": src_row["model"],
            "checkpoint_path": f"{src_row['result_path']}/models/net_g_latest.pth",
            "checkpoint_epoch": "latest", "source_train_run_id": source,
            "gpu_indices": "1", "extra_json": {"noise_sigma": sigma, "algo": "noise2noise"},
            "dataset": bench, "dataset_path": f"/mnt/data/benchmark/{bench}",
            "result_path": f"{src_row['result_path']}/results/{bench}",
            "device": "cuda:1", "input_size": "1x321x481", "latency_ms": latency,
            "throughput_fps": round(1000 / latency, 1), "status": C.STATUS_DONE, "started_at": _ts(1),
            "duration_sec": 45, "metrics_json": metrics, "notes": f"sigma={sigma} setting.",
            "favorite": favorite,
        })

    _dn_eval(n2n, "N2N-Base", "nafnet", "BSD68", "25", 12.4, {"PSNR": 31.08, "SSIM": 0.8812}, favorite=1)
    _dn_eval(n2n, "N2N-Base", "nafnet", "Kodak24", "25", 15.1, {"PSNR": 32.90, "SSIM": 0.9005})
    _dn_eval(n2n, "N2N-Base", "restormer", "BSD68", "25", 38.7, {"PSNR": 31.44, "SSIM": 0.8848})
    _dn_eval(n2n, "N2N-Base", "scunet", "DND", "50", 22.0, {"PSNR": 28.60, "SSIM": 0.8210})
    _dn_eval(real_noise, "RealNoise-SIDD", "real_restormer", "BSD68", "50", 39.5, {"PSNR": 27.50, "SSIM": 0.7902})
    _dn_eval(real_noise, "RealNoise-SIDD", "real_nafnet", "BSD68", "50", 12.9, {"PSNR": 27.85, "SSIM": 0.7958}, favorite=1)
    _dn_eval(real_noise, "RealNoise-SIDD", "real_nafnet", "Kodak24", "50", 14.2, {"PSNR": 28.60, "SSIM": 0.8090})

    if with_local_assets:
        row = db.get_run("train", dn_train["nafnet"])
        folder = _make_local_asset(
            results_root, "Denoising", "N2N-Base", row, 200000, 10,
            curve_metrics=(("PSNR", 31.08), ("SSIM", 0.8812)),
        )
        db.update_run("train", dn_train["nafnet"], {
            **row, "result_path": folder,
            "notes": (row.get("notes") or "") + "\nLocal copy synced for review.",
        })

    # == Task: Clustering ======================================================
    cl = db.add_task("Clustering", "Unsupervised Clustering")
    dc2 = db.add_work(cl, "DeepClusterV2", "Second round of unsupervised clustering baselines")
    db.add_dataset(dc2, "CIFAR-10", path="/mnt/data/cifar10", sample_count=60000, extension="png", created_at=_ts(9))
    db.add_dataset(dc2, "STL-10", path="/mnt/data/stl10", sample_count=13000, extension="png", created_at=_ts(9))

    cl_train = {
        "deepcluster": add_train("Clustering", dc2, "DeepClusterV2", {
            "server": "Server 1", "model": "DeepCluster", "dataset": "CIFAR-10", "gpu_indices": "0",
            "dataset_path": "/mnt/data/cifar10", "result_path": "/mnt/exp/DeepClusterV2/deepcluster",
            "status": C.STATUS_DONE, "started_at": _ts(7), "duration_sec": 4 * 3600 + 20 * 60,
            "epochs": "200", "batch_size": "256", "lr": "5e-4", "optimizer": "SGD",
            "metrics_json": {"NMI": 0.812, "ARI": 0.734, "ACC": 88.9},
            "notes": "Baseline.", "favorite": 1, "tags": "baseline",
        }),
        "scan": add_train("Clustering", dc2, "DeepClusterV2", {
            "server": "Server 2", "model": "SCAN", "dataset": "CIFAR-10", "gpu_indices": "0,1",
            "dataset_path": "/mnt/data/cifar10", "result_path": "/mnt/exp/DeepClusterV2/scan",
            "status": C.STATUS_DONE, "started_at": _ts(5), "duration_sec": 6 * 3600,
            "epochs": "200", "batch_size": "128", "lr": "1e-4", "optimizer": "Adam",
            "metrics_json": {"NMI": 0.845, "ARI": 0.781, "ACC": 91.2},
            "notes": "Best on CIFAR-10 so far.",
        }),
        "swav": add_train("Clustering", dc2, "DeepClusterV2", {
            "server": "Server 3", "model": "SwAV", "dataset": "STL-10", "gpu_indices": "0,1,2,3",
            "dataset_path": "/mnt/data/stl10", "result_path": "/mnt/exp/DeepClusterV2/swav",
            "status": C.STATUS_DONE, "started_at": _ts(4), "duration_sec": 9 * 3600 + 15 * 60,
            "epochs": "400", "batch_size": "256", "lr": "3e-4", "optimizer": "Lion",
            "metrics_json": {"NMI": 0.790, "ARI": 0.705, "ACC": 85.4},
            "notes": "Harder dataset (fewer labels available for eval).",
        }),
        "scan_i50": add_train("Clustering", dc2, "DeepClusterV2", {
            "server": "Server 4", "model": "SCAN", "dataset": "ImageNet-50", "gpu_indices": "0,1",
            "dataset_path": "/mnt/data/imagenet50", "result_path": "/mnt/exp/DeepClusterV2/scan_i50",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.5), "duration_sec": None,
            "epochs": "200", "batch_size": "128", "lr": "1e-4", "optimizer": "Adam",
            "metrics_json": {}, "notes": "Scaling SCAN up to ImageNet-50.",
        }),
    }

    def _cl_eval(source_key: str, bench: str, metrics: dict[str, float], favorite: int = 0) -> None:
        source = cl_train[source_key]
        src_row = db.get_run("train", source)
        add_eval("Clustering", dc2, "DeepClusterV2", {
            "server": src_row["server"], "model": src_row["model"],
            "checkpoint_path": f"{src_row['result_path']}/checkpoint.pth",
            "checkpoint_epoch": src_row["epochs"], "source_train_run_id": source,
            "gpu_indices": "0", "dataset": bench, "dataset_path": f"/mnt/data/{bench.lower()}",
            "result_path": f"{src_row['result_path']}/results/{bench}",
            "device": "cuda:0", "status": C.STATUS_DONE, "started_at": _ts(2), "duration_sec": 320,
            "metrics_json": metrics, "notes": f"{bench} test split.", "favorite": favorite,
        })

    _cl_eval("scan", "CIFAR-10-test", {"NMI": 0.848, "ARI": 0.784, "ACC": 91.6}, favorite=1)
    _cl_eval("swav", "STL-10-test", {"NMI": 0.793, "ARI": 0.710, "ACC": 85.9})
    _cl_eval("deepcluster", "CIFAR-10-test", {"NMI": 0.809, "ARI": 0.729, "ACC": 88.5})

    # == Task: Classification =================================================
    cls = db.add_task("Classification", "Image Classification")
    imagenet_base = db.add_work(cls, "ImageNet-Baseline", "Standard ImageNet-1k training baselines")
    distill = db.add_work(cls, "Distillation-Study", "Knowledge distillation ablations")

    db.add_dataset(imagenet_base, "ImageNet-1k", path="/mnt/data/imagenet1k", sample_count=1_281_167,
                    extension="jpeg", created_at=_ts(22))
    db.add_dataset(distill, "CIFAR-100", path="/mnt/data/cifar100", sample_count=50000,
                    extension="png", created_at=_ts(6))
    db.add_dataset(distill, "Food-101", path="/mnt/data/food101", sample_count=75750,
                    extension="jpeg", created_at=_ts(6))

    cls_train = {
        "resnet50": add_train("Classification", imagenet_base, "ImageNet-Baseline", {
            "server": "Server 1", "model": "ResNet-50", "dataset": "ImageNet-1k", "gpu_indices": "0,1,2,3",
            "dataset_path": "/mnt/data/imagenet1k", "result_path": "/mnt/exp/ImageNet-Baseline/resnet50",
            "status": C.STATUS_DONE, "started_at": _ts(16), "duration_sec": 20 * 3600,
            "epochs": "90", "batch_size": "256", "lr": "0.1", "optimizer": "SGD",
            "metrics_json": {"Top-1": 76.8, "Top-5": 93.4}, "notes": "Standard 90-epoch recipe.",
            "tags": "baseline",
        }),
        "vit": add_train("Classification", imagenet_base, "ImageNet-Baseline", {
            "server": "Server 2", "model": "ViT-B/16", "dataset": "ImageNet-1k", "gpu_indices": "0,1,2,3",
            "dataset_path": "/mnt/data/imagenet1k", "result_path": "/mnt/exp/ImageNet-Baseline/vit_b16",
            "status": C.STATUS_DONE, "started_at": _ts(13), "duration_sec": 30 * 3600,
            "epochs": "300", "batch_size": "512", "lr": "3e-3", "optimizer": "AdamW",
            "metrics_json": {"Top-1": 79.5, "Top-5": 94.8}, "notes": "Best model, becomes distillation teacher.",
            "favorite": 1,
        }),
        "convnext": add_train("Classification", imagenet_base, "ImageNet-Baseline", {
            "server": "Server 3", "model": "ConvNeXt-T", "dataset": "ImageNet-1k", "gpu_indices": "0,1",
            "dataset_path": "/mnt/data/imagenet1k", "result_path": "/mnt/exp/ImageNet-Baseline/convnext_t",
            "status": C.STATUS_DONE, "started_at": _ts(10), "duration_sec": 24 * 3600,
            "epochs": "300", "batch_size": "256", "lr": "4e-3", "optimizer": "AdamW",
            "metrics_json": {"Top-1": 78.2, "Top-5": 94.1}, "notes": "Between ResNet and ViT.",
        }),
        "swin": add_train("Classification", imagenet_base, "ImageNet-Baseline", {
            "server": "Server 4", "model": "Swin-T", "dataset": "ImageNet-1k", "gpu_indices": "0,1",
            "dataset_path": "/mnt/data/imagenet1k", "result_path": "/mnt/exp/ImageNet-Baseline/swin_t",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.4), "duration_sec": None,
            "epochs": "300", "batch_size": "256", "lr": "4e-3", "optimizer": "AdamW",
            "metrics_json": {}, "notes": "Running.",
        }),

        "distill_student": add_train("Classification", distill, "Distillation-Study", {
            "server": "Server 1", "model": "ResNet-50", "dataset": "CIFAR-100", "gpu_indices": "1",
            "dataset_path": "/mnt/data/cifar100", "result_path": "/mnt/exp/Distillation-Study/resnet50_distilled",
            "status": C.STATUS_DONE, "started_at": _ts(3), "duration_sec": 5 * 3600 + 40 * 60,
            "epochs": "200", "batch_size": "128", "lr": "0.05", "optimizer": "SGD",
            "metrics_json": {"Top-1": 81.2, "Top-5": 96.5},
            "notes": "Student, distilled from ViT-B/16 teacher.", "tags": "distillation,ablation",
        }),
        "distill_control": add_train("Classification", distill, "Distillation-Study", {
            "server": "Server 2", "model": "ResNet-50", "dataset": "CIFAR-100", "gpu_indices": "1",
            "dataset_path": "/mnt/data/cifar100", "result_path": "/mnt/exp/Distillation-Study/resnet50_control",
            "status": C.STATUS_DONE, "started_at": _ts(3), "duration_sec": 5 * 3600 + 20 * 60,
            "epochs": "200", "batch_size": "128", "lr": "0.05", "optimizer": "SGD",
            "metrics_json": {"Top-1": 78.9, "Top-5": 95.1}, "notes": "No-distillation control.", "tags": "ablation",
        }),
        "distill_food": add_train("Classification", distill, "Distillation-Study", {
            "server": "Server 3", "model": "ResNet-50", "dataset": "Food-101", "gpu_indices": "2",
            "dataset_path": "/mnt/data/food101", "result_path": "/mnt/exp/Distillation-Study/resnet50_food101",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "150", "batch_size": "128", "lr": "0.05", "optimizer": "SGD",
            "metrics_json": {}, "notes": "Queued - transfer study.",
        }),
    }

    def _cls_eval(work_id: int, work_name: str, source_key: str, bench: str,
                  metrics: dict[str, float], tags: str = "", favorite: int = 0) -> None:
        source = cls_train[source_key]
        src_row = db.get_run("train", source)
        add_eval("Classification", work_id, work_name, {
            "server": src_row["server"], "model": src_row["model"],
            "checkpoint_path": f"{src_row['result_path']}/checkpoint_best.pth",
            "checkpoint_epoch": src_row["epochs"], "source_train_run_id": source,
            "gpu_indices": "0", "dataset": bench, "dataset_path": f"/mnt/data/{bench.lower().replace(' ', '_')}",
            "result_path": f"{src_row['result_path']}/results/{bench}",
            "device": "cuda:0", "status": C.STATUS_DONE, "started_at": _ts(1), "duration_sec": 210,
            "metrics_json": metrics, "notes": f"{bench} evaluation.", "tags": tags, "favorite": favorite,
        })

    _cls_eval(imagenet_base, "ImageNet-Baseline", "resnet50", "ImageNet-1k-val",
              {"Top-1": 76.8, "Top-5": 93.4}, favorite=1)
    _cls_eval(imagenet_base, "ImageNet-Baseline", "vit", "ImageNet-1k-val", {"Top-1": 79.5, "Top-5": 94.8})
    _cls_eval(imagenet_base, "ImageNet-Baseline", "convnext", "ImageNet-1k-val", {"Top-1": 78.2, "Top-5": 94.1})
    _cls_eval(distill, "Distillation-Study", "distill_student", "CIFAR-100-test",
              {"Top-1": 81.0, "Top-5": 96.3}, tags="distillation", favorite=1)
    _cls_eval(distill, "Distillation-Study", "distill_control", "CIFAR-100-test",
              {"Top-1": 78.7, "Top-5": 95.0}, tags="ablation")

    if with_local_assets:
        row = db.get_run("train", cls_train["resnet50"])
        _make_local_asset(
            results_root, "Classification", "ImageNet-Baseline", row, 90, 16,
            curve_metrics=(("Top-1", 76.8), ("Top-5", 93.4)),
        )

    # == 몇 개는 실제로 중복/수정해서 History 탭에 다른 종류의 이벤트도 남긴다 ==========
    dup_id = db.duplicate_run("train", sr_train["mambair"])  # -> queued 복제본 (History: duplicated)
    if dup_id:
        count += 1
    db.update_run("train", dn_train["uformer"], {
        **db.get_run("train", dn_train["uformer"]),
        "notes": "Low-noise setting, running.\nUpdate: passed 100k iter without divergence.",
    })  # History: updated
    db.toggle_favorite("train", cl_train["swav"])
    db.toggle_favorite("train", cl_train["swav"])  # 한 번 켰다 끄기 - toggle_favorite 양방향 확인

    return count
