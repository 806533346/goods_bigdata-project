#!/bin/bash
# Run on GPU-0 (ssh root@<GPU-0-公网IP>)
# Requires: Spark Master + 2 Workers already running
cd /app
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PYTHONPATH=/app/src

echo "Running distributed Spark feature engineering..."
T0=$(date +%s)

python3 -c "
import time, json
from spark_features_cloud import create_spark_session, load_data, compute_global_average, compute_loo_stats, write_stats

t0 = time.time()
spark = create_spark_session(master='spark://<控制节点内网IP>:7077')
train_df, test_df = load_data(spark, use_cloud=False)

global_avg = compute_global_average(train_df)

for col in ['user_id', 'prod_id', 'parent_prod_id']:
    stats = compute_loo_stats(train_df, col, global_avg)
    write_stats(stats, f'data/{col}_stats.parquet', use_cloud=False)
    print(f'{col} done: {stats.count()} rows')

elapsed = time.time() - t0
spark.stop()
with open('/app/output/phase1_timing.json', 'w') as f:
    json.dump({'phase': 'spark_features_distributed', 'time_s': round(elapsed,1), 'workers': 2, 'master': 'spark://<控制节点内网IP>:7077'}, f)
print(f'Phase 1 complete: {elapsed:.0f}s')
"

echo ""
echo "Syncing features to GPU-1..."
scp -o StrictHostKeyChecking=no -r /app/data/user_stats.parquet root@<GPU-1-内网IP>:/app/data/ 2>/dev/null
scp -o StrictHostKeyChecking=no -r /app/data/prod_stats.parquet root@<GPU-1-内网IP>:/app/data/ 2>/dev/null
scp -o StrictHostKeyChecking=no -r /app/data/parent_stats.parquet root@<GPU-1-内网IP>:/app/data/ 2>/dev/null
scp /app/data/global_avg.npy root@<GPU-1-内网IP>:/app/data/ 2>/dev/null
echo "Synced ✓"

T1=$(date +%s)
echo "Total: $((T1 - T0))s"
