"""처음 실행 시 UI 를 바로 확인할 수 있는 예시 실험 데이터."""
from __future__ import annotations

from datetime import datetime, timedelta

from . import constants as C
from .db import Database


def _ts(days_ago: float) -> str:
    return (datetime.now() - timedelta(days=days_ago)).replace(microsecond=0).isoformat(sep=" ")


def populate(db: Database) -> int:
    """샘플 Train / Inference 기록을 넣고 추가한 건수를 돌려준다."""
    sr = db.add_task("SR", "Super Resolution")
    dn = db.add_task("DN", "Denoising")
    ssl2sl = db.add_work(sr, "SSL2SL", "Self-supervised -> Supervised 전이 실험")
    x4 = db.add_work(sr, "BSR-x4", "Blind SR x4 베이스라인")
    n2n = db.add_work(dn, "N2N-Base", "Noise2Noise 기반 베이스라인")

    train_rows = [
        {
            "work_id": ssl2sl, "server": "Server 1", "model": "Restormer", "dataset": "DIV2K",
            "gpu_indices": "0,1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/SSL2SL/restormer_x4",
            "status": C.STATUS_DONE, "started_at": _ts(6), "duration_sec": 19 * 3600 + 42 * 60,
            "epochs": "300000 iter", "batch_size": "8", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 32.41, "SSIM": 0.8993, "LPIPS": 0.121},
            "exec_command": C.SAMPLE_TRAIN_CMD, "config_yaml": C.SAMPLE_CONFIG_YML,
            "notes": "베이스라인. val_freq 5000 기준 최고 성능 iter 285000.",
        },
        {
            "work_id": ssl2sl, "server": "Server 2", "model": "SwinIR", "dataset": "DIV2K+Flickr2K",
            "gpu_indices": "0,1,2,3", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/swinir_x4",
            "status": C.STATUS_DONE, "started_at": _ts(4), "duration_sec": 27 * 3600,
            "epochs": "500000 iter", "batch_size": "16", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 32.72, "SSIM": 0.9031, "LPIPS": 0.118},
            "exec_command": C.SAMPLE_TRAIN_CMD.replace("restormer_x4", "swinir_x4"),
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "SwinIR"),
            "notes": "DF2K 로 데이터 확장. Restormer 대비 +0.31dB.",
        },
        {
            "work_id": ssl2sl, "server": "Server 3", "model": "MambaIR", "dataset": "DF2K",
            "gpu_indices": "0,1", "extra_json": {"scale": "x4"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/SSL2SL/mambair_x4",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.35), "duration_sec": None,
            "epochs": "400000 iter", "batch_size": "8", "lr": "3e-4", "optimizer": "AdamW",
            "metrics_json": {"PSNR": 32.55, "SSIM": 0.9012},
            "exec_command": C.SAMPLE_TRAIN_CMD.replace("restormer_x4", "mambair_x4"),
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "MambaIR"),
            "notes": "진행 중. 220k iter 시점 중간 지표.",
        },
        {
            "work_id": x4, "server": "Server 4", "model": "HAT", "dataset": "DF2K",
            "gpu_indices": "0,1,2,3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DF2K/train", "result_path": "/mnt/exp/BSR-x4/hat",
            "status": C.STATUS_FAILED, "started_at": _ts(2), "duration_sec": 41 * 60,
            "epochs": "800000 iter", "batch_size": "32", "lr": "1e-4", "optimizer": "AdamW",
            "metrics_json": {},
            "exec_command": C.SAMPLE_TRAIN_CMD.replace("restormer_x4", "hat_x4"),
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "HAT"),
            "notes": "CUDA OOM (batch 32). batch 16 으로 재시도 필요.",
        },
        {
            "work_id": n2n, "server": "Server 1", "model": "NAFNet", "dataset": "SIDD",
            "gpu_indices": "2", "extra_json": {"noise_sigma": "25"},
            "dataset_path": "/mnt/data/SIDD/train", "result_path": "/mnt/exp/N2N-Base/nafnet",
            "status": C.STATUS_QUEUED, "started_at": "", "duration_sec": None,
            "epochs": "200000 iter", "batch_size": "16", "lr": "1e-3", "optimizer": "AdamW",
            "metrics_json": {},
            "exec_command": "python train.py -opt options/train/DN/nafnet_sidd.yml",
            "config_yaml": "", "notes": "Server 1 대기열.",
        },
    ]

    train_rows.append(
        {
            "work_id": x4, "server": "Server 3", "model": "EDSR", "dataset": "DIV2K",
            "gpu_indices": "2,3", "extra_json": {"scale": "x2"},
            "dataset_path": "/mnt/data/DIV2K/train", "result_path": "/mnt/exp/BSR-x4/edsr_x2",
            "status": C.STATUS_RUNNING, "started_at": _ts(0.1), "duration_sec": None,
            "epochs": "300000 iter", "batch_size": "16", "lr": "2e-4", "optimizer": "Adam",
            "metrics_json": {"PSNR": 34.02},
            "exec_command": "CUDA_VISIBLE_DEVICES=2,3 python train.py -opt options/train/BSR/edsr_x2.yml",
            "config_yaml": C.SAMPLE_CONFIG_YML.replace("Restormer", "EDSR"),
            "notes": "Server 3 에서 MambaIR 과 동시 진행 (GPU 2,3).",
        }
    )

    infer_rows = [
        {
            "work_id": ssl2sl, "server": "Server 1", "model": "Restormer",
            "checkpoint_path": "/mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth",
            "gpu_indices": "0", "extra_json": {"scale": "x4"},
            "dataset": "Set5", "dataset_path": "/mnt/data/benchmark/Set5/LR",
            "result_path": "/mnt/exp/SSL2SL/restormer_x4/results/Set5",
            "device": "cuda:0", "input_size": "3x256x256", "latency_ms": 41.7,
            "throughput_fps": 23.98, "status": C.STATUS_DONE, "started_at": _ts(3),
            "duration_sec": 96, "metrics_json": {"PSNR": 32.41, "SSIM": 0.8993, "LPIPS": 0.121},
            "exec_command": C.SAMPLE_INFER_CMD, "config_yaml": "",
            "notes": "Set5 벤치마크.",
        },
        {
            "work_id": ssl2sl, "server": "Server 2", "model": "SwinIR",
            "checkpoint_path": "/mnt/exp/SSL2SL/swinir_x4/models/net_g_500000.pth",
            "gpu_indices": "0", "extra_json": {"scale": "x4"},
            "dataset": "Urban100", "dataset_path": "/mnt/data/benchmark/Urban100/LR",
            "result_path": "/mnt/exp/SSL2SL/swinir_x4/results/Urban100",
            "device": "cuda:0", "input_size": "3x256x256", "latency_ms": 88.2,
            "throughput_fps": 11.34, "status": C.STATUS_DONE, "started_at": _ts(2),
            "duration_sec": 640, "metrics_json": {"PSNR": 27.05, "SSIM": 0.8142, "LPIPS": 0.163},
            "exec_command": C.SAMPLE_INFER_CMD.replace("restormer_x4", "swinir_x4").replace("Set5", "Urban100"),
            "config_yaml": "", "notes": "속도는 Restormer 대비 2배 느림.",
        },
        {
            "work_id": n2n, "server": "Server 4", "model": "NAFNet",
            "checkpoint_path": "/mnt/exp/N2N-Base/nafnet/models/net_g_latest.pth",
            "gpu_indices": "1", "extra_json": {"noise_sigma": "25"},
            "dataset": "BSD68", "dataset_path": "/mnt/data/benchmark/BSD68",
            "result_path": "/mnt/exp/N2N-Base/nafnet/results/BSD68",
            "device": "cuda:1", "input_size": "1x321x481", "latency_ms": 12.4,
            "throughput_fps": 80.6, "status": C.STATUS_DONE, "started_at": _ts(1),
            "duration_sec": 45, "metrics_json": {"PSNR": 31.08, "SSIM": 0.8812},
            "exec_command": "python inference.py --ckpt /mnt/exp/N2N-Base/nafnet/models/net_g_latest.pth "
                            "--input /mnt/data/benchmark/BSD68 --output /mnt/exp/N2N-Base/nafnet/results/BSD68",
            "config_yaml": "", "notes": "sigma=25 설정.",
        },
    ]

    count = 0
    for row in train_rows:
        db.insert_run("train", row)
        count += 1
    for row in infer_rows:
        db.insert_run("inference", row)
        count += 1
    return count
