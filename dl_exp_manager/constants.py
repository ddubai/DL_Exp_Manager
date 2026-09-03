"""앱 전역 상수 / 기본 프리셋."""
from __future__ import annotations

# --- 서버 -------------------------------------------------------------------
DEFAULT_SERVERS = [
    ("Server 1", "192.168.0.101", "RTX 4090 x2"),
    ("Server 2", "192.168.0.102", "RTX 4090 x2"),
    ("Server 3", "192.168.0.103", "A6000 x4"),
    ("Server 4", "192.168.0.104", "A100 x4"),
]

# --- Level 1: DL Task -------------------------------------------------------
DEFAULT_TASKS = [
    ("SR", "Super Resolution"),
    ("DN", "Denoising"),
    ("Clustering", "Unsupervised Clustering"),
    ("Classification", "Image Classification"),
]

# --- Level 2: Work ID (샘플) ------------------------------------------------
DEFAULT_WORKS = {
    "SR": [("SSL2SL", "Self-supervised -> Supervised 전이 실험")],
    "DN": [("N2N-Base", "Noise2Noise 기반 베이스라인")],
}

# --- 콤보박스 프리셋 (setEditable(True) 이므로 직접 입력도 가능) ------------
MODEL_PRESETS = [
    "Restormer",
    "SwinIR",
    "MambaIR",
    "NAFNet",
    "HAT",
    "EDSR",
    "RCAN",
    "Uformer",
    "SCUNet",
    "ResNet-50",
    "ViT-B/16",
]

DATASET_PRESETS = [
    "DIV2K",
    "Flickr2K",
    "SIDD",
    "DND",
    "Set5",
    "Set14",
    "Urban100",
    "BSD68",
    "ImageNet-1k",
]

OPTIMIZER_PRESETS = ["AdamW", "Adam", "SGD", "Lion", "RMSprop"]
DEVICE_PRESETS = ["cuda:0", "cuda:1", "cuda:0,1", "cpu", "mps"]

# 자주 쓰는 메트릭 키 (메트릭은 JSON 으로 자유롭게 확장 가능)
TRAIN_METRIC_PRESETS = ["PSNR", "SSIM", "LPIPS", "Loss", "Accuracy", "mIoU", "NMI", "ARI"]
INFER_METRIC_PRESETS = ["PSNR", "SSIM", "LPIPS", "NIQE", "Accuracy", "Top-5", "FID"]

# --- 실행 상태 --------------------------------------------------------------
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_LIST = [STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]

STATUS_COLORS = {
    STATUS_QUEUED: "#9aa0a6",
    STATUS_RUNNING: "#1a73e8",
    STATUS_DONE: "#188038",
    STATUS_FAILED: "#d93025",
}

STATUS_ICONS = {
    STATUS_QUEUED: "⏳",
    STATUS_RUNNING: "🔵",
    STATUS_DONE: "🟢",
    STATUS_FAILED: "🔴",
}

SAMPLE_CONFIG_YML = """\
# config.yml
name: SSL2SL_Restormer_x4
model:
  type: Restormer
  dim: 48
  num_blocks: [4, 6, 6, 8]
datasets:
  train:
    name: DIV2K
    dataroot: /mnt/data/DIV2K/train
    gt_size: 128
    batch_size: 8
  val:
    name: Set5
    dataroot: /mnt/data/benchmark/Set5
train:
  total_iter: 300000
  optim:
    type: AdamW
    lr: !!float 3e-4
    weight_decay: 0.01
  scheduler:
    type: CosineAnnealingLR
    T_max: 300000
val:
  val_freq: 5000
  metrics: [PSNR, SSIM]
"""

SAMPLE_TRAIN_CMD = (
    "CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 "
    "train.py -opt options/train/SSL2SL/restormer_x4.yml --launcher pytorch"
)

SAMPLE_INFER_CMD = (
    "CUDA_VISIBLE_DEVICES=0 python inference.py "
    "--ckpt /mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth "
    "--input /mnt/data/benchmark/Set5/LR --output /mnt/exp/SSL2SL/restormer_x4/results/Set5"
)
