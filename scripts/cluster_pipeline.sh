#!/bin/bash
# ============================================================
# COMP5434 — 3-Node GPU Cluster Pipeline
#
# Run this ON YOUR LOCAL MACHINE to orchestrate all 3 nodes.
# Prerequisites: SSH key already set up on all 3 machines.
#
# Usage:
#   bash scripts/cluster_pipeline.sh
# ============================================================
set -euo pipefail

# Allow errors in Phase 1 startup (don't exit on first failure)
set +e

CTRL="root@<控制节点公网IP>"
GPU0="root@<GPU-0-公网IP>"
GPU1="root@<GPU-1-公网IP>"

MASTER_IP="<控制节点内网IP>"
GPU0_IP="<GPU-0-内网IP>"
GPU1_IP="<GPU-1-内网IP>"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
METRICS_FILE="/tmp/comp5434_metrics_${TIMESTAMP}.json"

echo "============================================================"
echo "COMP5434 3-Node GPU Cluster Pipeline"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo ""
echo "Architecture:"
echo "  Control (${MASTER_IP}) → Spark Master + Prefect"
echo "  GPU-0   (${GPU0_IP}) → Spark Worker + DDP rank 0"
echo "  GPU-1   (${GPU1_IP}) → Spark Worker + DDP rank 1"
echo "============================================================"

# ── Phase 0: Verify all nodes ───────────────────────────────────────────
echo ""
echo ">>> Phase 0: Verifying nodes..."

check_node() {
    local name=$1 ip=$2
    echo -n "  ${name} (${ip}): "
    ssh "root@${ip}" "hostname; python3 -c 'import torch; print(f\"PyTorch:{torch.__version__}\"); print(f\"CUDA:{torch.cuda.is_available()}\")' 2>/dev/null" | head -2 | tr '\n' ' '
    echo ""
}

check_node "Control" "${CTRL#*@}" || true
check_node "GPU-0"   "${GPU0#*@}" || true
check_node "GPU-1"   "${GPU1#*@}" || true

echo "  All nodes accessible."

# ── Phase 1: Distributed Spark Feature Engineering ──────────────────────
echo ""
echo "============================================================"
echo "Phase 1: Distributed Spark Feature Engineering"
echo "============================================================"

echo ""
echo ">>> Step 1.1: Starting Spark Master on Control Node..."
ssh ${CTRL} "
pkill -f 'spark.deploy.master' 2>/dev/null || true
sleep 1
nohup /usr/local/bin/spark-class \
  org.apache.spark.deploy.master.Master \
  --host ${MASTER_IP} --port 7077 --webui-port 8080 \
  > /app/output/spark_master.log 2>&1 &
echo \"Spark Master starting...\"
sleep 3
curl -s http://${MASTER_IP}:8080 | grep -q ALIVE && echo 'Master ALIVE ✓' || echo 'Master still starting...'
"
echo "  Spark Master: http://${CTRL#*@}:8080"

echo ""
echo ">>> Step 1.2: Starting Spark Workers on GPU nodes..."
for NODE in "${GPU0}" "${GPU1}"; do
    NODE_IP=$(ssh ${NODE} "hostname -I | awk '{print \$1}'")
    ssh ${NODE} "
    pkill -f 'spark.deploy.worker' 2>/dev/null || true
    sleep 1
    nohup /usr/local/bin/spark-class \
      org.apache.spark.deploy.worker.Worker \
      spark://${MASTER_IP}:7077 \
      --host ${NODE_IP} --cores 4 --memory 16g \
      > /app/output/spark_worker.log 2>&1 &
    sleep 2
    echo \"Spark Worker started on ${NODE_IP} ✓\"
    "
done
echo "  Both workers should now appear at http://${CTRL#*@}:8080"

echo ""
echo ">>> Step 1.3: Running Spark Feature Engineering on GPU-0..."
PHASE1_START=$(date +%s)
ssh ${GPU0} "
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PYTHONPATH=/app/src
cd /app
python3 -c \"
import time, json
from spark_features_cloud import create_spark_session, load_data, compute_global_average, compute_loo_stats, write_stats

t0 = time.time()
spark = create_spark_session(master='spark://${MASTER_IP}:7077')
train_df, test_df = load_data(spark, use_cloud=False)

global_avg = compute_global_average(train_df)

for col in ['user_id', 'prod_id', 'parent_prod_id']:
    stats = compute_loo_stats(train_df, col, global_avg)
    write_stats(stats, f'data/{col}_stats.parquet', use_cloud=False)
    print(f'{col} done: {stats.count()} rows')

elapsed = time.time() - t0
spark.stop()
with open('/app/output/phase1_timing.json', 'w') as f:
    json.dump({'phase': 'spark_features', 'time_s': round(elapsed,1), 'workers': 2, 'master': 'spark://${MASTER_IP}:7077'}, f)
print(f'Phase 1 complete: {elapsed:.0f}s')
\" 2>&1
"
PHASE1_END=$(date +%s)
PHASE1_TIME=$((PHASE1_END - PHASE1_START))

echo ""
echo ">>> Step 1.4: Syncing features to GPU-1..."
ssh ${GPU1} "mkdir -p /app/data && scp -o StrictHostKeyChecking=no root@${GPU0_IP}:/app/data/*.parquet /app/data/ 2>/dev/null"
ssh ${GPU1} "scp root@${GPU0_IP}:/app/data/global_avg.npy /app/data/ 2>/dev/null"
echo "  Features synced ✓"

# ── Phase 2: Multi-node DDP Training ────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 2: Multi-node DDP Training (2×A10)"
echo "============================================================"
echo "Starting on GPU-0 (rank 0) + GPU-1 (rank 1)..."

PHASE2_START=$(date +%s)

# Start both ranks in parallel
echo ""
echo "Launching DDP training on both GPUs..."

# Upload cluster env config to both GPUs
scp /home/nmxc/project_code/big_data_trea_cloud/configs/cluster_env.sh ${GPU0}:/app/configs/ 2>/dev/null
scp /home/nmxc/project_code/big_data_trea_cloud/configs/cluster_env.sh ${GPU1}:/app/configs/ 2>/dev/null

ssh ${GPU1} "
cd /app
source /app/configs/cluster_env.sh
echo \"GPU-1: BATCH=\${BATCH_SIZE}, LR=\${LEARNING_RATE}\"

nohup torchrun \
    --nproc_per_node=\${NPROC_PER_NODE} \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=${GPU0_IP} \
    --master_port=\${MASTER_PORT} \
    src/finetune_roberta_ddp.py \
    > /app/output/train_rank1.log 2>&1 &
echo \"GPU-1 (rank 1) PID: \$!\"
" &

sleep 2

ssh ${GPU0} "
cd /app
source /app/configs/cluster_env.sh
echo \"GPU-0: BATCH=\${BATCH_SIZE}, LR=\${LEARNING_RATE}, Effective Batch=\$((BATCH_SIZE * 2))\"

torchrun \
    --nproc_per_node=\${NPROC_PER_NODE} \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=${GPU0_IP} \
    --master_port=\${MASTER_PORT} \
    src/finetune_roberta_ddp.py \
    2>&1 | tee /app/output/train_rank0.log
"
PHASE2_END=$(date +%s)
PHASE2_TIME=$((PHASE2_END - PHASE2_START))

# ── Phase 3: Results Collection ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 3: Results Collection"
echo "============================================================"

echo ">>> Step 3.1: Collecting training results..."
ssh ${GPU0} "
echo 'Train Log:' && cat /app/data/roberta_base_train_log.json 2>/dev/null | python3 -m json.tool | head -30
echo ''
echo 'Prediction stats:' && tail -20 /app/output/train_rank0.log | grep -E 'Prediction|Submission|RMSE|time'
"

echo ""
echo ">>> Step 3.2: Downloading submission..."
scp ${GPU0}:/app/output/submission.csv /tmp/submission_${TIMESTAMP}.csv 2>/dev/null && \
  echo "  Downloaded: /tmp/submission_${TIMESTAMP}.csv" || \
  echo "  Not ready yet - check manually"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Pipeline Complete! Summary:"
echo "============================================================"
echo "Phase 1 (Spark):   ${PHASE1_TIME}s"
echo "Phase 2 (Training): ${PHASE2_TIME}s"
echo "Total:             $((PHASE1_TIME + PHASE2_TIME))s"

cat << 'EOF' > ${METRICS_FILE}
{
    "pipeline": "3-node GPU cluster",
    "timestamp": "${TIMESTAMP}",
    "architecture": {
        "control": "Spark Master (4vCPU/16GB)",
        "gpu0": "Spark Worker + DDP rank 0 (32vCPU/128GB/1xA10)",
        "gpu1": "Spark Worker + DDP rank 1 (32vCPU/128GB/1xA10)"
    },
    "phase1_spark": {
        "time_s": ${PHASE1_TIME},
        "workers": 2
    },
    "phase2_training": {
        "time_s": ${PHASE2_TIME},
        "nnodes": 2,
        "nproc_per_node": 1,
        "total_effective_batch": 64
    }
}
EOF
echo ""
echo "Metrics saved: ${METRICS_FILE}"
