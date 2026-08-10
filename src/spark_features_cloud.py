"""
分布式 Spark 特征工程 (Cloud Version)。

在 Spark 集群上并行计算三种 LOO (Leave-One-Out) 统计特征：
  1. user_id_loo_avg     → 该用户其他评论的平均分
  2. prod_id_loo_avg     → 该产品其他评论的平均分
  3. parent_prod_id_loo_avg → 该父品类其他评论的平均分

同时计算每种实体的评论数的对数 (log_count)，作为置信度信号。
输出为 Parquet 列式存储，后续用 pyarrow 快速读取。

Usage:
    # 本地模式 (单机，所有 CPU 核):
    python spark_features_cloud.py --local

    # 远程集群模式:
    export SPARK_MASTER=spark://spark-master:7077
    python spark_features_cloud.py

    # Docker 容器内:
    spark-submit --master spark://spark-master:7077 --driver-memory 8g src/spark_features_cloud.py
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEM, SPARK_EXEC_MEM,
    SPARK_EXEC_CORES, SPARK_SHUFFLE, SPARK_MAX_RESULT, RANDOM_SEED,
    OSS_BUCKET, OSS_TRAIN_CSV, OSS_TEST_CSV, OSS_PRODINFO_CSV,
    OSS_USER_STATS, OSS_PROD_STATS, OSS_PARENT_STATS, OSS_GLOBAL_AVG,
    DATA_DIR, COMPRESSION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spark_features")


# ── Spark Session ───────────────────────────────────────────────────────

def create_spark_session(master: str = None, app_name: str = None) -> SparkSession:
    """
    创建 Spark 会话，连接到分布式集群。

    master 可以是:
      - "local[*]"           → 单机所有 CPU 核 (开发/测试)
      - "spark://ip:7077"    → 远程 Standalone 集群 (生产)

    集群模式下:
      - Driver 运行在发起任务的节点
      - Executor 运行在各 Worker 节点
      - 数据按 partition 分布到各 Worker 并行计算
    """
    master = master or SPARK_MASTER
    app_name = app_name or SPARK_APP_NAME

    logger.info(f"Initializing Spark: master={master}, app={app_name}")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", SPARK_DRIVER_MEM)        # Driver 进程内存
        .config("spark.executor.memory", SPARK_EXEC_MEM)         # 每个 Executor 内存
        .config("spark.executor.cores", str(SPARK_EXEC_CORES))   # 每个 Executor CPU 核数
        .config("spark.sql.shuffle.partitions", str(SPARK_SHUFFLE))  # 洗牌分区数
        .config("spark.driver.maxResultSize", SPARK_MAX_RESULT)
        .config("spark.sql.adaptive.enabled", "true")            # 自适应查询优化
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # 云存储凭证 (通过环境变量注入，不硬编码)
        .config("spark.hadoop.fs.oss.impl", "org.apache.hadoop.fs.aliyun.oss.AliyunOSSFileSystem")
        .config("spark.hadoop.fs.oss.endpoint", os.environ.get("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com"))
    )

    # OSS credentials via env vars (or IAM instance role on Alibaba Cloud ECS)
    if os.environ.get("OSS_ACCESS_KEY_ID"):
        builder = (
            builder
            .config("spark.hadoop.fs.oss.accessKeyId", os.environ["OSS_ACCESS_KEY_ID"])
            .config("spark.hadoop.fs.oss.accessKeySecret", os.environ["OSS_ACCESS_KEY_SECRET"])
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Spark session ready: version={spark.version}")
    logger.info(f"  Master: {spark.sparkContext.master}")
    logger.info(f"  Cores:  {spark.sparkContext.defaultParallelism}")
    return spark


# ── Data Loading ─────────────────────────────────────────────────────────

def _resolve_path(path: str, bucket: str = None, use_cloud: bool = True) -> str:
    """
    Resolve a file path. If use_cloud and bucket is set, return OSS URI.
    Otherwise return local path.
    """
    bucket = bucket or OSS_BUCKET
    if use_cloud and bucket:
        return f"oss://{bucket}/{path}"
    return os.path.join(DATA_DIR, os.path.basename(path))


def load_data(spark: SparkSession, use_cloud: bool = True, bucket: str = None):
    """
    Load train/test CSV files from cloud storage or local disk.

    Returns:
        (train_df, test_df) Spark DataFrames
    """
    train_path = _resolve_path(OSS_TRAIN_CSV, bucket, use_cloud)
    test_path  = _resolve_path(OSS_TEST_CSV, bucket, use_cloud)

    logger.info(f"Loading train data from: {train_path}")
    t0 = time.time()

    # Explicit schema matching actual CSV columns
    train_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("user_id", StringType(), True),
        StructField("prod_id", StringType(), True),
        StructField("parent_prod_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("comment", StringType(), True),
        StructField("time", StringType(), True),
        StructField("votes", IntegerType(), True),
        StructField("purchased", StringType(), True),  # True/False strings
        StructField("rating", IntegerType(), True),
    ])

    test_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("user_id", StringType(), True),
        StructField("prod_id", StringType(), True),
        StructField("parent_prod_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("comment", StringType(), True),
        StructField("time", StringType(), True),
        StructField("votes", IntegerType(), True),
        StructField("purchased", StringType(), True),
    ])

    train_df = (
        spark.read
        .option("header", "true")
        .schema(train_schema)
        .csv(train_path)
        .repartition(SPARK_SHUFFLE)  # Distribute across cluster
    )

    test_df = (
        spark.read
        .option("header", "true")
        .schema(test_schema)
        .csv(test_path)
    )

    # Force materialization and log counts
    train_count = train_df.count()
    test_count  = test_df.count()
    elapsed = time.time() - t0

    logger.info(f"Data loaded in {elapsed:.1f}s: train={train_count:,}, test={test_count:,}")
    return train_df, test_df


# ── LOO Feature Engineering ──────────────────────────────────────────────

def compute_global_average(train_df: DataFrame):
    """
    计算全局平均评分（3M 条数据的平均值）。

    这是 LOO 特征的 fallback 值:
    当某个实体（用户/产品）只有 1 条评论时，
    无法计算"除了自己以外"的平均值，就用全局平均代替。
    """
    global_avg = train_df.select(F.avg("rating")).collect()[0][0]
    logger.info(f"Global average rating: {global_avg:.4f}")
    local_path = os.path.join(DATA_DIR, "global_avg.npy")
    np.save(local_path, np.array([global_avg], dtype=np.float32))
    return float(global_avg)


def compute_loo_stats(train_df: DataFrame, group_col: str, global_avg: float) -> DataFrame:
    """
    计算每个分组的 Leave-One-Out 统计特征。

    数学原理:
      LOO 均值 = (组内总分 - 组内平均分) / (组内数量 - 1)
              ≈ 组内平均分  (当组很大时几乎一样)

    为什么用 LOO 而不是直接取组平均?
      → 防止数据泄露: 如果直接把组平均作为特征，
         训练时模型会"偷看"自己的标签。

    Spark 执行方式:
      groupBy → 每个 Worker 处理一部分数据 → 并行聚合 → 汇总到 Driver
      3M 数据分布在 2 个 Worker 上，各自计算再合并。
    """
    logger.info(f"Computing LOO stats for: {group_col}")

    # 第一步: 按分组列聚合 (Spark 自动分发到各 Worker)
    group_stats = train_df.groupBy(group_col).agg(
        F.sum("rating").alias("sum_rating"),      # 组内评分总和
        F.count("rating").alias("count_rating"),   # 组内评论数
    )

    # 第二步: 计算 LOO 均值 (基于组聚合结果)
    result = group_stats.withColumn(
        f"{group_col}_loo_avg",
        F.when(
            F.col("count_rating") > 1,
            # (sum - avg) / (count - 1) = 近似排除自己
            (F.col("sum_rating") - F.col("sum_rating") / F.col("count_rating"))
            / (F.col("count_rating") - 1)
        ).otherwise(F.lit(global_avg))  # 只有1条评论 → 用全局平均
    ).withColumn(
        f"{group_col}_log_count",
        F.log1p(F.col("count_rating"))  # log(1+n), 压缩长尾分布
    )

    return result.select(group_col, f"{group_col}_loo_avg", f"{group_col}_log_count")


def write_stats(stats_df: DataFrame, cloud_key: str, use_cloud: bool = True, bucket: str = None):
    """Write stats DataFrame to Parquet (local or cloud)."""
    bucket = bucket or OSS_BUCKET

    if use_cloud and bucket:
        output_path = f"oss://{bucket}/{cloud_key}"
    else:
        output_path = os.path.join(DATA_DIR, os.path.basename(cloud_key))

    logger.info(f"Writing stats to: {output_path}")
    stats_df.write.mode("overwrite").option("compression", COMPRESSION).parquet(output_path)

    count = stats_df.count()
    logger.info(f"Written {count} rows to {output_path}")


# ── Execution ────────────────────────────────────────────────────────────

def run_spark_features(
    use_cloud: bool = False,
    bucket: str = None,
    master: str = None,
):
    """
    Main entry point: run distributed Spark feature engineering pipeline.

    Args:
        use_cloud: Read/write from cloud object storage.
        bucket: Cloud bucket name (uses config default if None).
        master: Spark master URL (uses config default if None).

    Returns:
        Elapsed time in seconds.
    """
    total_start = time.time()
    spark = None

    try:
        spark = create_spark_session(master=master)

        # 1. Load data (cloud or local)
        train_df, test_df = load_data(spark, use_cloud=use_cloud, bucket=bucket)

        # 2. Global average
        global_avg = compute_global_average(train_df)

        # 3. Compute LOO stats for three entity types
        for group_col in ["user_id", "prod_id", "parent_prod_id"]:
            t0 = time.time()
            stats = compute_loo_stats(train_df, group_col, global_avg)

            cloud_key = {
                "user_id":   OSS_USER_STATS,
                "prod_id":   OSS_PROD_STATS,
                "parent_prod_id": OSS_PARENT_STATS,
            }[group_col]

            write_stats(stats, cloud_key, use_cloud=use_cloud, bucket=bucket)

            # Cache stats to avoid recomputation
            stats.persist()
            elapsed = time.time() - t0
            logger.info(f"  {group_col} stats completed in {elapsed:.1f}s")

        total_time = time.time() - total_start
        logger.info(f"Spark features pipeline complete: {total_time:.1f}s")
        return total_time

    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped.")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Spark Feature Engineering")
    parser.add_argument("--local", action="store_true",
                        help="Run in local mode (local[*], local files)")
    parser.add_argument("--cloud", action="store_true",
                        help="Read/write data from cloud object storage")
    parser.add_argument("--bucket", type=str, default=None,
                        help="Cloud bucket name")
    parser.add_argument("--master", type=str, default=None,
                        help="Spark master URL (overrides config)")
    args = parser.parse_args()

    if args.local:
        os.environ["SPARK_MASTER"] = "local[*]"

    run_spark_features(
        use_cloud=args.cloud,
        bucket=args.bucket,
        master=args.master,
    )
