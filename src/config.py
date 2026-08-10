"""Central configuration for the COMP5434 project."""
import os

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# ── Data files ───────────────────────────────────────────────────────────
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
PRODINFO_CSV = os.path.join(DATA_DIR, "prodInfo.csv")

# ── Output files ─────────────────────────────────────────────────────────
SUBMISSION_CSV = os.path.join(OUTPUT_DIR, "submission.csv")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")

# ── Spark features (parquet) ─────────────────────────────────────────────
USER_STATS_PARQUET = os.path.join(DATA_DIR, "user_stats.parquet")
PROD_STATS_PARQUET = os.path.join(DATA_DIR, "prod_stats.parquet")
PARENT_STATS_PARQUET = os.path.join(DATA_DIR, "parent_stats.parquet")
GLOBAL_AVG_NPY = os.path.join(DATA_DIR, "global_avg.npy")

# ── RoBERTa model ────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_LENGTH = 192
BATCH_SIZE = 32
GRAD_ACCUM = 1
EPOCHS = 1
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SUBSET_SIZE = 3007439  # full training data
VAL_SIZE = 50000

# ── Saved model artifacts ────────────────────────────────────────────────
ROBERTA_WEIGHTS = os.path.join(DATA_DIR, "roberta_base_finetuned.pt")
ROBERTA_TRAIN_LOG = os.path.join(DATA_DIR, "roberta_base_train_log.json")
TRAIN_EMB_NPY = os.path.join(DATA_DIR, "train_roberta_ft.npy")
TEST_EMB_NPY = os.path.join(DATA_DIR, "test_roberta_ft.npy")

# ── Spark config ─────────────────────────────────────────────────────────
SPARK_APP_NAME = "COMP5434_RatingPrediction"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "16g"
SPARK_EXECUTOR_MEMORY = "16g"

# ── Reproducibility ──────────────────────────────────────────────────────
RANDOM_SEED = 42

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
