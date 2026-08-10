"""Feature engineering using PySpark (big data parallel processing framework).

This script uses Spark's distributed processing to:
1. Read large CSV files (3M+ rows) in parallel
2. Compute LOO statistical features using Window functions
3. Compute text statistics
4. Save feature lookup tables for downstream MLP training
"""
import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (StructType, StructField, StringType,
                                IntegerType, LongType, BooleanType)

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
OUTPUT_PATH = os.path.join(DATA_PATH, "spark_features")

# Explicit schemas (avoid inferSchema issues with boolean columns)
TRAIN_SCHEMA = StructType([
    StructField("id", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("prod_id", StringType(), True),
    StructField("parent_prod_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("comment", StringType(), True),
    StructField("time", LongType(), True),
    StructField("votes", IntegerType(), True),
    StructField("purchased", BooleanType(), True),
    StructField("rating", IntegerType(), True),
])

TEST_SCHEMA = StructType([
    StructField("id", LongType(), True),
    StructField("user_id", StringType(), True),
    StructField("prod_id", StringType(), True),
    StructField("parent_prod_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("comment", StringType(), True),
    StructField("time", LongType(), True),
    StructField("votes", IntegerType(), True),
    StructField("purchased", BooleanType(), True),
])


def main():
    # === Create Spark Session ===
    spark = SparkSession.builder \
        .appName("RatingPrediction_FeatureEngineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("=== PySpark Feature Engineering (Big Data Framework) ===")
    print(f"Spark version: {spark.version}")
    print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")

    # === 1. Distributed CSV Reading ===
    print("\n[1/4] Reading CSV files with Spark (distributed)...")
    train_df = spark.read.csv(
        os.path.join(DATA_PATH, "train.csv"),
        header=True, schema=TRAIN_SCHEMA
    )
    test_df = spark.read.csv(
        os.path.join(DATA_PATH, "test.csv"),
        header=True, schema=TEST_SCHEMA
    )

    # Cache for reuse
    train_df.cache()
    test_df.cache()

    n_train = train_df.count()
    n_test = test_df.count()
    print(f"  Train: {n_train} rows, Test: {n_test} rows")
    print(f"  Train partitions: {train_df.rdd.getNumPartitions()}")

    # === 2. Text Preprocessing (Spark distributed) ===
    print("\n[2/4] Text preprocessing (title + [SEP] + comment)...")
    train_df = train_df.withColumn("text",
        F.concat(
            F.coalesce(F.col("title"), F.lit("")),
            F.lit(" [SEP] "),
            F.coalesce(F.col("comment"), F.lit(""))
        )
    )
    test_df = test_df.withColumn("text",
        F.concat(
            F.coalesce(F.col("title"), F.lit("")),
            F.lit(" [SEP] "),
            F.coalesce(F.col("comment"), F.lit(""))
        )
    )

    # Text length statistics (Spark distributed aggregation)
    text_stats = train_df.select(
        F.mean(F.length("text")).alias("avg_text_len"),
        F.min(F.length("text")).alias("min_text_len"),
        F.max(F.length("text")).alias("max_text_len"),
    ).collect()[0]
    print(f"  Text length - avg: {text_stats['avg_text_len']:.0f}, "
          f"min: {text_stats['min_text_len']}, max: {text_stats['max_text_len']}")

    # === 3. Statistical Feature Computation (Spark Window Functions) ===
    print("\n[3/4] Computing LOO statistical features with Spark Window functions...")

    # Global average rating
    global_avg = train_df.select(F.mean("rating")).collect()[0][0]
    print(f"  Global average rating: {global_avg:.4f}")

    # --- Training set: LOO features (Leave-One-Out) ---
    # User-level stats using Window partitionBy
    w_user = Window.partitionBy("user_id")
    train_df = train_df.withColumn("user_sum", F.sum("rating").over(w_user))
    train_df = train_df.withColumn("user_cnt", F.count("rating").over(w_user))
    train_df = train_df.withColumn("user_loo_avg",
        F.when(F.col("user_cnt") > 1,
               (F.col("user_sum") - F.col("rating")) / (F.col("user_cnt") - 1))
        .otherwise(F.lit(float(global_avg)))
    )

    # Product-level stats
    w_prod = Window.partitionBy("prod_id")
    train_df = train_df.withColumn("prod_sum", F.sum("rating").over(w_prod))
    train_df = train_df.withColumn("prod_cnt", F.count("rating").over(w_prod))
    train_df = train_df.withColumn("prod_loo_avg",
        F.when(F.col("prod_cnt") > 1,
               (F.col("prod_sum") - F.col("rating")) / (F.col("prod_cnt") - 1))
        .otherwise(F.lit(float(global_avg)))
    )

    # Parent product-level stats
    w_parent = Window.partitionBy("parent_prod_id")
    train_df = train_df.withColumn("parent_sum", F.sum("rating").over(w_parent))
    train_df = train_df.withColumn("parent_cnt", F.count("rating").over(w_parent))
    train_df = train_df.withColumn("parent_loo_avg",
        F.when(F.col("parent_cnt") > 1,
               (F.col("parent_sum") - F.col("rating")) / (F.col("parent_cnt") - 1))
        .otherwise(F.lit(float(global_avg)))
    )

    # Log counts
    train_df = train_df.withColumn("user_log_cnt", F.log1p(F.col("user_cnt")))
    train_df = train_df.withColumn("prod_log_cnt", F.log1p(F.col("prod_cnt")))
    train_df = train_df.withColumn("parent_log_cnt", F.log1p(F.col("parent_cnt")))

    # Fill nulls
    train_df = train_df.na.fill({
        "user_loo_avg": float(global_avg),
        "prod_loo_avg": float(global_avg),
        "parent_loo_avg": float(global_avg),
    })

    # --- Compute aggregate stats for test set (using ALL training data) ---
    print("  Computing aggregate stats for test set...")
    user_agg = train_df.groupBy("user_id").agg(
        F.mean("rating").alias("user_avg"),
        F.count(F.lit(1)).alias("user_cnt")
    )
    prod_agg = train_df.groupBy("prod_id").agg(
        F.mean("rating").alias("prod_avg"),
        F.count(F.lit(1)).alias("prod_cnt")
    )
    parent_agg = train_df.groupBy("parent_prod_id").agg(
        F.mean("rating").alias("parent_avg"),
        F.count(F.lit(1)).alias("parent_cnt")
    )

    # Join stats to test set
    test_df = test_df.join(user_agg, on="user_id", how="left")
    test_df = test_df.join(prod_agg, on="prod_id", how="left")
    test_df = test_df.join(parent_agg, on="parent_prod_id", how="left")

    test_df = test_df.na.fill({
        "user_avg": float(global_avg), "prod_avg": float(global_avg), "parent_avg": float(global_avg),
        "user_cnt": 0, "prod_cnt": 0, "parent_cnt": 0,
    })

    test_df = test_df.withColumn("user_log_cnt", F.log1p(F.col("user_cnt")))
    test_df = test_df.withColumn("prod_log_cnt", F.log1p(F.col("prod_cnt")))
    test_df = test_df.withColumn("parent_log_cnt", F.log1p(F.col("parent_cnt")))

    # === 4. Save Feature Lookup Tables (Parquet) ===
    print("\n[4/4] Saving feature lookup tables to parquet...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save user/prod/parent stats as lookup tables (for MLP to build features with correct order)
    user_stats = train_df.groupBy("user_id").agg(
        F.sum("rating").alias("rating_sum"),
        F.count("rating").alias("rating_count"),
        F.mean("rating").alias("rating_mean"),
    )
    prod_stats = train_df.groupBy("prod_id").agg(
        F.sum("rating").alias("rating_sum"),
        F.count("rating").alias("rating_count"),
        F.mean("rating").alias("rating_mean"),
    )
    parent_stats = train_df.groupBy("parent_prod_id").agg(
        F.sum("rating").alias("rating_sum"),
        F.count("rating").alias("rating_count"),
        F.mean("rating").alias("rating_mean"),
    )

    user_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "user_stats.parquet"))
    prod_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "prod_stats.parquet"))
    parent_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "parent_stats.parquet"))

    # Also save test set aggregate stats
    test_user_stats = test_df.select("user_id", "user_avg", "user_cnt").distinct()
    test_prod_stats = test_df.select("prod_id", "prod_avg", "prod_cnt").distinct()
    test_parent_stats = test_df.select("parent_prod_id", "parent_avg", "parent_cnt").distinct()

    test_user_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "test_user_stats.parquet"))
    test_prod_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "test_prod_stats.parquet"))
    test_parent_stats.write.mode("overwrite").parquet(os.path.join(OUTPUT_PATH, "test_parent_stats.parquet"))

    # Save global average
    np.save(os.path.join(OUTPUT_PATH, "global_avg.npy"), np.array([global_avg], dtype=np.float32))

    # === Summary Statistics (Spark distributed) ===
    print("\n=== Spark Summary Statistics ===")
    rating_dist = train_df.groupBy("rating").count().orderBy("rating").collect()
    print("Rating distribution:")
    for row in rating_dist:
        print(f"  {row['rating']}: {row['count']} ({row['count']/n_train*100:.1f}%)")

    user_count = train_df.select("user_id").distinct().count()
    prod_count = train_df.select("prod_id").distinct().count()
    print(f"\nUnique users: {user_count}")
    print(f"Unique products: {prod_count}")
    print(f"Avg reviews per user: {n_train/user_count:.1f}")
    print(f"Avg reviews per product: {n_train/prod_count:.1f}")

    print(f"\n=== Output Files (in {OUTPUT_PATH}) ===")
    print("  user_stats.parquet      - User-level rating stats (sum, count, mean)")
    print("  prod_stats.parquet      - Product-level rating stats")
    print("  parent_stats.parquet    - Parent product-level rating stats")
    print("  test_user_stats.parquet - Test set user stats (from training data)")
    print("  test_prod_stats.parquet - Test set product stats")
    print("  test_parent_stats.parquet - Test set parent product stats")
    print("  global_avg.npy          - Global average rating")

    # Cleanup
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    print("\nPySpark feature engineering complete!")


if __name__ == "__main__":
    main()
