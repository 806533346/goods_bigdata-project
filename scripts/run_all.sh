#!/bin/bash
# ============================================================
# COMP5434 — One-Click Full Pipeline
# Run:  bash scripts/run_all.sh
# ============================================================
set -e

CTRL="root@<控制节点公网IP>"
GPU0="root@<GPU-0-公网IP>"
GPU1="root@<GPU-1-公网IP>"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
START_ALL=$(date +%s)

echo "============================================================"
echo "COMP5434 Full Pipeline — Distributed Spark + DDP Training"
echo "============================================================"

# ── Step 1: Sync all scripts ───────────────────────────────────────────
echo ""
echo ">>> Syncing scripts to all nodes..."
ssh ${CTRL} "mkdir -p /app/scripts /app/configs"
ssh ${GPU0} "mkdir -p /app/scripts /app/configs"
ssh ${GPU1} "mkdir -p /app/scripts /app/configs"

for NODE in "${CTRL}" "${GPU0}" "${GPU1}"; do
    scp -q scripts/start_spark_master.sh ${NODE}:/app/scripts/ 2>/dev/null || true
    scp -q scripts/start_spark_worker.sh ${NODE}:/app/scripts/ 2>/dev/null || true
    scp -q scripts/run_spark_features.sh ${NODE}:/app/scripts/ 2>/dev/null || true
    scp -q scripts/launch_rank0.sh ${NODE}:/app/scripts/ 2>/dev/null || true
    scp -q scripts/launch_rank1.sh ${NODE}:/app/scripts/ 2>/dev/null || true
    scp -q configs/cluster_env.sh ${NODE}:/app/configs/ 2>/dev/null || true
done
echo "  ✓"

# ── Step 2: Start Spark Cluster ────────────────────────────────────────
echo ""
echo ">>> Starting Spark Master..."
ssh ${CTRL} "bash /app/scripts/start_spark_master.sh"
sleep 3

echo ""
echo ">>> Starting Spark Workers..."
ssh ${GPU0} "bash /app/scripts/start_spark_worker.sh" &
sleep 1
ssh ${GPU1} "bash /app/scripts/start_spark_worker.sh" &
sleep 3
echo "  Workers: check http://<控制节点公网IP>:8080"

# ── Step 3: Run Spark Features ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 1: Distributed Spark Feature Engineering"
echo "============================================================"
PHASE1_START=$(date +%s)
ssh ${GPU0} "bash /app/scripts/run_spark_features.sh"
PHASE1_END=$(date +%s)
PHASE1_TIME=$((PHASE1_END - PHASE1_START))
echo "  Phase 1 complete: ${PHASE1_TIME}s"

# ── Step 4: Launch DDP Training ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 2: Multi-node DDP Training (2×A10, batch=96)"
echo "============================================================"
PHASE2_START=$(date +%s)

# Start rank 1 first (waits for master)
echo "Launching GPU-1 (rank 1)..."
ssh ${GPU1} "bash /app/scripts/launch_rank1.sh" &
sleep 3

echo "Launching GPU-0 (rank 0)..."
ssh ${GPU0} "bash /app/scripts/launch_rank0.sh"
PHASE2_END=$(date +%s)
PHASE2_TIME=$((PHASE2_END - PHASE2_START))

# ── Step 5: Collect Results ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Results"
echo "============================================================"

echo ""
echo ">>> Training log:"
ssh ${GPU0} "cat /app/data/roberta_base_train_log.json 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40" || echo "  (check /app/data/roberta_base_train_log.json on GPU-0)"

echo ""
echo ">>> Submission file:"
ssh ${GPU0} "ls -lh /app/output/submission.csv 2>/dev/null && head -3 /app/output/submission.csv" || echo "  Not found"

echo ""
echo ">>> Download to local:"
echo "    scp ${GPU0}:/app/output/submission.csv ./submission_${TIMESTAMP}.csv"
echo "    scp ${GPU0}:/app/data/roberta_base_train_log.json ./train_log_${TIMESTAMP}.json"

TOTAL_TIME=$((PHASE1_TIME + PHASE2_TIME))
echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Phase 1 (Spark features):  ${PHASE1_TIME}s"
echo "Phase 2 (DDP training):   ${PHASE2_TIME}s"
echo "Total time:               ${TOTAL_TIME}s ($((TOTAL_TIME/60))min)"
echo "============================================================"
