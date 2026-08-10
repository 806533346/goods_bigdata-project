"""PySpark distributed feature engineering for review rating prediction.

This module uses Apache Spark to:
1. Read train/test CSV files in distributed manner
2. Compute leave-one-out (LOO) statistical features using Window functions
3. Aggregate user/product/parent_product rating statistics
4. Save features as parquet files for downstream model training
"""
import os
import time
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from config import (
    SPARK_APP_NAME, SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY,
    TRAIN_CSV, TEST_CSV, PRODINFO_CSV,
    USER_STATS_PARQUET, PROD_STATS_PARQUET, PARENT_STATS_PARQUET, GLOBAL_AVG_NPY,
    RANDOM_SEED,
)


def create_spark_session():
    """Create a configured Spark session."""
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark):
    """Load train and test CSV files."""
    train_df = (
        spark.read.csv(TRAIN_CSV, header=True, inferSchema=True)
        .repartition(32)
    )
    test_df = (
        spark.read.csv(TEST_CSV, header=True, inferSchema=True)
        .repartition(32)
    )
    print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")
    return train_df, test_df


def compute_global_avg(train_df):
    """Compute global average rating."""
    global_avg = train_df.select(F.avg("rating")).collect()[0][0]
    np.save(GLOBAL_AVG_NPY, np.float32(global_avg))
    print(f"Global average rating: {global_avg:.4f}")
    return float(global_avg)


def compute_loo_stats(df, group_col, output_path, global_avg):
    """Compute leave-one-out statistics for a grouping column.

    For each row, computes:
    - loo_avg: average rating excluding the current row
    - count: total count for the group
    - log_count: log(1 + count) for feature scaling
    """
    # Aggregate per group
    agg = df.groupBy(group_col).agg(
        F.sum("rating").alias("sum_rating"),
        F.count("rating").alias("count_rating"),
        F.avg("rating").alias("avg_rating"),
    )

    # LOO: (sum - current) / (count - 1) when count > 1, else global_avg
    result = agg.withColumn(
        "loo_avg",
        F.when(F.col("count_rating") > 1,
               (F.col("sum_rating") - F.col("avg_rating")) / (F.col("count_rating") - 1))
        .otherwise(F.lit(global_avg))
    ).withColumn(
        "log_count",
        F.log1p(F.col("count_rating"))
    ).select(
        group_col,
        F.col("loo_avg").alias(f"{group_col}_loo_avg"),
        F.col("count_rating").alias(f"{group_col}_count"),
        F.col("log_count").alias(f"{group_col}_log_count"),
    )

    result.write.mode("overwrite").parquet(output_path)
    print(f"Saved {group_col} stats to {output_path}")
    return result


def run_spark_features():
    """Main entry point for Spark feature engineering."""
    start = time.time()
    print("=" * 60)
    print("Spark Feature Engineering")
    print("=" * 60)

    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    # Load data
    train_df, test_df = load_data(spark)

    # Compute global average (from training data only)
    global_avg = compute_global_avg(train_df)

    # Compute LOO statistics for user_id, prod_id, parent_prod_id
    compute_loo_stats(train_df, "user_id", USER_STATS_PARQUET, global_avg)
    compute_loo_stats(train_df, "prod_id", PROD_STATS_PARQUET, global_avg)
    compute_loo_stats(train_df, "parent_prod_id", PARENT_STATS_PARQUET, global_avg)

    elapsed = time.time() - start
    print(f"\nSpark feature engineering done in {elapsed:.1f}s")

    spark.stop()
    return elapsed


if __name__ == "__main__":
    run_spark_features()
