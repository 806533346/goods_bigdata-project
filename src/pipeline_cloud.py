"""
COMP5434 云端训练流水线 — Prefect 编排版。

用 Prefect 替代 shell 脚本的好处:
  - 自动重试: Spot 实例中断后自动重跑失败步骤
  - 任务缓存: 输入没变就跳过，避免重复计算
  - Web 监控: localhost:4200 实时看进度
  - 资源标签: CPU 和 GPU 任务可以跑在不同机器上

也支持不用 Prefect 的 fallback 模式:
  直接 python pipeline_cloud.py → 顺序执行所有步骤

Usage:
    # 本地运行 (不需要 Prefect):
    python pipeline_cloud.py

    # 带 UI 监控:
    prefect server start &           # 开一个终端
    python pipeline_cloud.py --serve  # 开另一个终端

    # 云端部署:
    python pipeline_cloud.py --deploy
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prefect 是可选依赖，没装也能跑
try:
    from prefect import flow, task, get_run_logger
    from prefect.artifacts import create_table_artifact
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


# ── Task Definitions ─────────────────────────────────────────────────────

if PREFECT_AVAILABLE:
    spark_task = task(
        name="spark-feature-engineering",
        retries=2,
        retry_delay_seconds=30,
        tags=["spark", "cpu"],
        log_prints=True,
    )
    finetune_task = task(
        name="roberta-ddp-finetune",
        retries=1,
        retry_delay_seconds=60,
        tags=["gpu", "pytorch", "ddp"],
        log_prints=True,
    )
    embeddings_task = task(
        name="extract-embeddings",
        retries=1,
        tags=["gpu", "pytorch"],
        log_prints=True,
    )
    mlp_task = task(
        name="train-mlp",
        retries=2,
        tags=["gpu", "pytorch", "mlp"],
        log_prints=True,
    )
    submit_task = task(
        name="generate-submission",
        tags=["io"],
        log_prints=True,
    )
else:
    # Fallback decorators for running without Prefect
    def _noop_decorator(**kwargs):
        def wrapper(fn):
            return fn
        return wrapper
    spark_task = _noop_decorator()
    finetune_task = _noop_decorator()
    embeddings_task = _noop_decorator()
    mlp_task = _noop_decorator()
    submit_task = _noop_decorator()


# ── Task Implementations ─────────────────────────────────────────────────

@spark_task
def run_spark_step(use_cloud: bool = False) -> dict:
    """Step 1: Distributed Spark feature engineering."""
    logger = get_run_logger() if PREFECT_AVAILABLE else None

    from spark_features_cloud import run_spark_features
    t0 = time.time()
    elapsed = run_spark_features(use_cloud=use_cloud)
    total = time.time() - t0

    result = {"step": "spark_features", "duration_s": round(total, 1)}
    if logger:
        logger.info(f"Spark features completed in {total:.0f}s")
    return result


@finetune_task
def run_finetune_step(local_mode: bool = False) -> dict:
    """Step 2: RoBERTa-base DDP fine-tuning."""
    logger = get_run_logger() if PREFECT_AVAILABLE else None

    from finetune_roberta_ddp import run_finetune
    val_rmse = run_finetune(local_mode=local_mode)

    result = {"step": "roberta_finetune", "val_rmse": round(val_rmse, 6)}
    if logger:
        logger.info(f"RoBERTa fine-tuning: Val RMSE = {val_rmse:.4f}")
    return result


@embeddings_task
def run_embeddings_step() -> dict:
    """Step 3 (optional): Extract embeddings for MLP path."""
    logger = get_run_logger() if PREFECT_AVAILABLE else None

    from extract_embeddings import run_extract_embeddings
    t0 = time.time()
    run_extract_embeddings()
    elapsed = time.time() - t0

    result = {"step": "embedding_extraction", "duration_s": round(elapsed, 1)}
    if logger:
        logger.info(f"Embeddings extracted in {elapsed:.0f}s")
    return result


@mlp_task
def run_mlp_step() -> dict:
    """Step 4 (optional): Train MLP on embeddings + Spark features."""
    logger = get_run_logger() if PREFECT_AVAILABLE else None

    from train_mlp import run_train_mlp
    val_rmse = run_train_mlp()

    result = {"step": "mlp_training", "val_rmse": round(val_rmse, 6)}
    if logger:
        logger.info(f"MLP training: Val RMSE = {val_rmse:.4f}")
    return result


# ── Flow Definitions ─────────────────────────────────────────────────────

def _get_flow_decorator():
    """Create Prefect flow decorator if available, otherwise noop."""
    if PREFECT_AVAILABLE:
        return flow(
            name="comp5434-pipeline",
            description="Cloud pipeline: Spark → RoBERTa DDP → (MLP) → Submission",
            version="2.0.0",
            log_prints=True,
        )
    return _noop_decorator()


@_get_flow_decorator()
def main_pipeline(
    use_cloud: bool = False,
    run_mlp: bool = False,
    local_mode: bool = False,
):
    """
    COMP5434 端到端训练流水线。

    四步流程:
      Phase 1 → Spark 分布式特征工程 (CPU 集群)
      Phase 2 → RoBERTa DDP 多节点训练 (GPU 集群)
      Phase 3 → 可选: 嵌入提取 + MLP 训练
      Phase 4 → 生成 submission.csv + 保存指标

    Args:
        use_cloud: 从 OSS/S3 读写数据 (本地则用 data/ 目录)
        run_mlp: 是否跑 MLP 路径
        local_mode: 强制单 GPU 模式
    """
    from config import SUBMISSION_CSV, METRICS_JSON, OUTPUT_DIR
    from hardware import get_system_info

    total_start = time.time()
    results = {}

    # Hardware info
    hw = get_system_info()
    print("=" * 60)
    print("COMP5434 Cloud Pipeline")
    print("=" * 60)
    print(f"Platform: {hw['platform']} | Python: {hw['python_version']}")
    print(f"CPU: {hw['cpu']['cpu_model']} ({hw['cpu']['cpu_cores']} cores)")
    gpu = hw["gpu"]
    if gpu["gpu_available"]:
        print(f"GPU: {gpu['gpu_count']}x {gpu['gpus'][0]['name']} ({gpu['gpus'][0]['vram_gb']} GB)")
    if hw["is_distributed"]:
        print(f"DDP: world_size={hw['world_size']}, rank={hw['local_rank']}")
    print("=" * 60)

    # ── Step 1: Spark Features ──
    print("\n>>> Step 1: Spark Feature Engineering")
    spark_result = run_spark_step(use_cloud=use_cloud)
    results["spark"] = spark_result

    # ── Step 2: RoBERTa DDP Fine-tuning ──
    print("\n>>> Step 2: RoBERTa DDP Fine-Tuning")
    ft_result = run_finetune_step(local_mode=local_mode)
    results["finetune"] = ft_result

    # ── Optional MLP path ──
    if run_mlp:
        print("\n>>> Step 3: Embedding Extraction")
        emb_result = run_embeddings_step()
        results["embeddings"] = emb_result

        print("\n>>> Step 4: MLP Training")
        mlp_result = run_mlp_step()
        results["mlp"] = mlp_result

    # ── Save Metrics ──
    total_time = time.time() - total_start
    metrics = {
        "hardware_info": hw,
        "results": results,
        "total_time_s": round(total_time, 1),
        "total_time_h": round(total_time / 3600, 2),
        "submission_file": SUBMISSION_CSV,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    if "finetune" in results:
        print(f"RoBERTa Val RMSE: {results['finetune'].get('val_rmse', 'N/A')}")
    if "mlp" in results:
        print(f"MLP Val RMSE:     {results['mlp'].get('val_rmse', 'N/A')}")
    print(f"Total time:       {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Metrics:          {METRICS_JSON}")
    print(f"Submission:       {SUBMISSION_CSV}")

    # Prefect artifact (if available)
    if PREFECT_AVAILABLE:
        try:
            create_table_artifact(
                key="pipeline-results",
                table=[
                    {"step": k, "val_rmse": v.get("val_rmse", "N/A"), "duration_s": v.get("duration_s", "N/A")}
                    for k, v in results.items()
                ],
                description="COMP5434 pipeline results",
            )
        except Exception:
            pass

    return results


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP5434 Cloud Pipeline")
    parser.add_argument("--cloud", action="store_true",
                        help="Use cloud object storage for data I/O")
    parser.add_argument("--mlp", action="store_true",
                        help="Also run MLP path (embeddings + MLP)")
    parser.add_argument("--local", action="store_true",
                        help="Force single-GPU mode (no DDP)")
    parser.add_argument("--serve", action="store_true",
                        help="Register flow with Prefect server")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy flow to Prefect Cloud/server")
    args = parser.parse_args()

    if not PREFECT_AVAILABLE and (args.serve or args.deploy):
        print("Prefect not installed. Install with: pip install prefect>=3.0.0")
        sys.exit(1)

    if args.serve and PREFECT_AVAILABLE:
        # Register as a served deployment (runs on trigger or schedule)
        main_pipeline.serve(
            name="comp5434-deployment",
            parameters={
                "use_cloud": args.cloud,
                "run_mlp": args.mlp,
                "local_mode": args.local,
            },
        )
    elif args.deploy and PREFECT_AVAILABLE:
        # Deploy to Prefect Cloud
        from prefect.client.schemas.schedules import CronSchedule
        main_pipeline.serve(
            name="comp5434-cloud-deployment",
            cron="0 2 * * *",  # Nightly at 2am
            parameters={
                "use_cloud": True,
                "run_mlp": False,
                "local_mode": False,
            },
        )
        print("Deployment registered. Start workers to execute.")
    else:
        # Run locally immediately
        main_pipeline(
            use_cloud=args.cloud,
            run_mlp=args.mlp,
            local_mode=args.local,
        )
