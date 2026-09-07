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
# "Super-Resolution" 은 하이픈이 있다 - config/task-defs/Super-Resolution.yaml 의 Task
# 이름과 정확히 일치해야 그 파일의 옵션·컬럼·명령어 템플릿을 찾아 쓴다(db.py 의
# _RENAMED_TASKS_V10 이 예전 DB 의 "SuperResolution" 도 여기로 옮겨 준다).
DEFAULT_TASKS = [
    ("Super-Resolution", "Super Resolution"),
    ("Denoising", "Denoising"),
    ("Clustering", "Unsupervised Clustering"),
    ("Classification", "Image Classification"),
]

# --- Level 2: Work ID (샘플) ------------------------------------------------
DEFAULT_WORKS = {
    "Super-Resolution": [("SSL2SL", "Self-supervised -> Supervised transfer experiment")],
    "Denoising": [("N2N-Base", "Noise2Noise baseline")],
}

# 참고: 콤보박스 선택지(model / dataset / optimizer / device …)와 Task 별 지표·컬럼은
# 이 파일이 아니라 `config/options.yaml` 에서 관리한다 (dl_exp_manager/config_store.py).
# 상태 색과 폰트는 `dl_exp_manager/theme/tokens.py` 에 있다.

# options.yaml 이 없을 때의 폴백 지표 프리셋
TRAIN_METRIC_PRESETS = ["PSNR", "SSIM", "LPIPS", "Loss", "Accuracy", "mIoU", "NMI", "ARI"]
EVAL_METRIC_PRESETS = ["PSNR", "SSIM", "LPIPS", "NIQE", "Accuracy", "Top-5", "FID"]

# --- 실행 상태 --------------------------------------------------------------
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_LIST = [STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED]

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

SAMPLE_EVAL_CMD = (
    "CUDA_VISIBLE_DEVICES=0 python evaluate.py "
    "--ckpt /mnt/exp/SSL2SL/restormer_x4/models/net_g_300000.pth "
    "--input /mnt/data/benchmark/Set5/LR --output /mnt/exp/SSL2SL/restormer_x4/results/Set5"
)
