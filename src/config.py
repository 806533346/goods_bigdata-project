"""
COMP5434 Cloud — 集中配置文件。

所有配置项都可以通过环境变量覆盖，方便在本地 / 云端 / CI 切换：
  - 本地开发: 使用默认值
  - 云端部署: export BATCH_SIZE=128
  - Kaggle 实验: export SUBSET_SIZE=500000

配置优先级: 环境变量 > cloud_config.yaml > 代码默认值
"""
import os
import yaml
from pathlib import Path
from typing import Optional

# ── Base paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CONFIGS_DIR = BASE_DIR / "configs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Load cloud config YAML ───────────────────────────────────────────────
_config_path = CONFIGS_DIR / "cloud_config.yaml"
_config = {}
if _config_path.exists():
    with open(_config_path) as f:
        _config = yaml.safe_load(f) or {}

# ── Environment-aware config getter ──────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

# ── Cloud provider ───────────────────────────────────────────────────────
CLOUD_PROVIDER = _env("CLOUD_PROVIDER", "aliyun")
CLOUD_REGION   = _env("CLOUD_REGION", "cn-hangzhou")

# ── Object Storage ───────────────────────────────────────────────────────
OSS_ENDPOINT        = _env("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET          = _env("OSS_BUCKET", "comp5434-bucket")
OSS_ACCESS_KEY_ID   = _env("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = _env("OSS_ACCESS_KEY_SECRET", "")

# S3-compatible alternative
S3_ENDPOINT = _env("S3_ENDPOINT", "")
S3_BUCKET   = _env("S3_BUCKET", OSS_BUCKET)

# Container registry
ACR_ENDPOINT  = _env("ACR_ENDPOINT", "registry.cn-hangzhou.aliyuncs.com")
ACR_NAMESPACE = _env("ACR_NAMESPACE", "comp5434")

# ── Data files (local cache / cloud paths) ───────────────────────────────
TRAIN_CSV    = _env("TRAIN_CSV", str(DATA_DIR / "train.csv"))
TEST_CSV     = _env("TEST_CSV", str(DATA_DIR / "test.csv"))
PRODINFO_CSV = _env("PRODINFO_CSV", str(DATA_DIR / "prodInfo.csv"))

# Cloud storage keys
OSS_TRAIN_CSV    = "raw/train.csv"
OSS_TEST_CSV     = "raw/test.csv"
OSS_PRODINFO_CSV = "raw/prodInfo.csv"
OSS_USER_STATS    = "features/user_stats.parquet"
OSS_PROD_STATS    = "features/prod_stats.parquet"
OSS_PARENT_STATS  = "features/parent_stats.parquet"
OSS_GLOBAL_AVG    = "features/global_avg.npy"
OSS_CHECKPOINTS   = "checkpoints/roberta-base"
OSS_SUBMISSIONS   = "submissions"

# ── Spark configuration ──────────────────────────────────────────────────
SPARK_APP_NAME   = _env("SPARK_APP_NAME", "COMP5434_RatingPrediction_Cloud")
SPARK_MASTER     = _env("SPARK_MASTER", "local[*]")  # "spark://spark-master:7077" for cluster
SPARK_DRIVER_MEM = _env("SPARK_DRIVER_MEMORY", "8g")
SPARK_EXEC_MEM   = _env("SPARK_EXECUTOR_MEMORY", "24g")
SPARK_EXEC_CORES = int(_env("SPARK_EXECUTOR_CORES", "8"))
SPARK_SHUFFLE    = int(_env("SPARK_SHUFFLE_PARTITIONS", "64"))
SPARK_MAX_RESULT = _env("SPARK_MAX_RESULT_SIZE", "4g")

# ── DDP configuration ────────────────────────────────────────────────────
DDP_BACKEND        = _env("DDP_BACKEND", "nccl")
MASTER_ADDR        = _env("MASTER_ADDR", "localhost")
MASTER_PORT        = _env("MASTER_PORT", "29500")
NPROC_PER_NODE     = int(_env("NPROC_PER_NODE", "1"))
NNODES             = int(_env("NNODES", "1"))

# ── Model hyperparameters ────────────────────────────────────────────────
MODEL_NAME    = _env("MODEL_NAME", "roberta-base")
MAX_LENGTH    = int(_env("MAX_LENGTH", "192"))
BATCH_SIZE    = int(_env("BATCH_SIZE", "32"))
GRAD_ACCUM    = int(_env("GRAD_ACCUM", "1"))
EPOCHS        = int(_env("EPOCHS", "1"))
LR            = float(_env("LEARNING_RATE", "2e-5"))
WEIGHT_DECAY  = float(_env("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO  = float(_env("WARMUP_RATIO", "0.1"))
MAX_GRAD_NORM = float(_env("MAX_GRAD_NORM", "1.0"))
VAL_SIZE      = int(_env("VAL_SIZE", "50000"))
SUBSET_SIZE   = int(_env("SUBSET_SIZE")) if _env("SUBSET_SIZE") else None  # None = full

# ── Output files ─────────────────────────────────────────────────────────
SUBMISSION_CSV = str(OUTPUT_DIR / "submission.csv")
METRICS_JSON   = str(OUTPUT_DIR / "metrics.json")
CHECKPOINT_PT  = str(DATA_DIR / "roberta_base_finetuned.pt")
TRAIN_LOG_JSON = str(DATA_DIR / "roberta_base_train_log.json")
TRAIN_EMB_NPY  = str(DATA_DIR / "train_roberta_ft.npy")
TEST_EMB_NPY   = str(DATA_DIR / "test_roberta_ft.npy")

# ── Reproducibility ──────────────────────────────────────────────────────
RANDOM_SEED = int(_env("RANDOM_SEED", "42"))

# ── HuggingFace ──────────────────────────────────────────────────────────
HF_ENDPOINT = _env("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# ── Compression ──────────────────────────────────────────────────────────
COMPRESSION = _env("COMPRESSION", "snappy")  # for Parquet

# ── Worker count for DataLoaders ─────────────────────────────────────────
NUM_WORKERS = int(_env("NUM_WORKERS", "4"))
