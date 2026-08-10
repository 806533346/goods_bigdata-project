"""Main pipeline orchestrator for the COMP5434 project.

Steps:
1. Spark feature engineering (distributed LOO statistics)
2. RoBERTa-base end-to-end fine-tuning
3. Embedding extraction (optional, for MLP)
4. MLP training on embeddings + Spark features (optional)
5. Generate final submission

Usage:
    export PYTHONPATH=src
    python3 src/pipeline.py --data-dir data [--skip-spark] [--skip-finetune] [--skip-mlp]
"""
import os
import sys
import time
import json
import argparse

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, METRICS_JSON, SUBMISSION_CSV
from hardware import get_system_info


def step_spark_features():
    """Step 1: Spark distributed feature engineering."""
    print("\n" + "=" * 60)
    print("STEP 1: Spark Feature Engineering")
    print("=" * 60)
    from spark_features import run_spark_features
    return run_spark_features()


def step_finetune_roberta():
    """Step 2: RoBERTa-base end-to-end fine-tuning."""
    print("\n" + "=" * 60)
    print("STEP 2: RoBERTa-base Fine-tuning")
    print("=" * 60)
    from finetune_roberta import run_finetune
    return run_finetune()


def step_extract_embeddings():
    """Step 3: Extract embeddings from fine-tuned model."""
    print("\n" + "=" * 60)
    print("STEP 3: Embedding Extraction")
    print("=" * 60)
    from extract_embeddings import run_extract_embeddings
    return run_extract_embeddings()


def step_train_mlp():
    """Step 4: MLP training on embeddings + Spark features."""
    print("\n" + "=" * 60)
    print("STEP 4: MLP Training")
    print("=" * 60)
    from train_mlp import run_train_mlp
    return run_train_mlp()


def main():
    parser = argparse.ArgumentParser(description="COMP5434 Project Pipeline")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--skip-spark", action="store_true", help="Skip Spark feature engineering")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip RoBERTa fine-tuning")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--skip-mlp", action="store_true", help="Skip MLP training")
    parser.add_argument("--mlp-only", action="store_true", help="Only run MLP (requires cached files)")
    args = parser.parse_args()

    total_start = time.time()

    # Hardware info
    hw_info = get_system_info()
    print("=" * 60)
    print("COMP5434 Project: Review Rating Prediction")
    print("=" * 60)
    print(f"Hardware: {hw_info['cpu']['cpu_model']}")
    if hw_info["gpu"].get("gpu_available", False) or "gpu_name" in hw_info.get("gpu", {}):
        print(f"GPU: {hw_info['gpu'].get('gpu_name', 'N/A')}")

    timings = {}
    results = {}

    # Step 1: Spark features
    if not args.skip_spark and not args.mlp_only:
        t = step_spark_features()
        timings["spark_features"] = t

    # Step 2: RoBERTa fine-tuning (this is the main model)
    if not args.skip_finetune and not args.mlp_only:
        val_rmse, test_pred = step_finetune_roberta()
        timings["finetune"] = "see roberta_base_train_log.json"
        results["finetune_val_rmse"] = val_rmse

    # Step 3: Extract embeddings (optional, for MLP)
    if not args.skip_embeddings and not args.skip_mlp and not args.mlp_only:
        step_extract_embeddings()

    # Step 4: MLP training (optional alternative)
    if not args.skip_mlp:
        mlp_rmse = step_train_mlp()
        results["mlp_val_rmse"] = mlp_rmse

    # Save final metrics
    total_time = time.time() - total_start
    metrics = {
        "hardware_info": hw_info,
        "timings": timings,
        "results": results,
        "total_time_seconds": total_time,
        "total_time_hours": round(total_time / 3600, 2),
        "submission_file": SUBMISSION_CSV,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"Submission: {SUBMISSION_CSV}")
    print(f"Metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()
