#!/bin/bash
# ============================================================
# COMP5434 GPU Cluster — One-Click Deployment
# Deploys Spark on control node + 8×A10 DDP training on GPU node.
#
# Usage:
#   chmod +x scripts/deploy_cluster.sh
#   bash scripts/deploy_cluster.sh --key ~/comp5434-key.pem
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source configs/cluster_inventory.sh

# ── Parse args ───────────────────────────────────────────────────────────
SSH_KEY="${SSH_KEY:-}"
UPLOAD_DATA=false
SKIP_SPARK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --key) SSH_KEY="$2"; shift 2 ;;
        --upload-data) UPLOAD_DATA=true; shift ;;
        --skip-spark) SKIP_SPARK=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SSH_CTRL="ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${SSH_USER}@${CTRL_PUBLIC_IP}"
SSH_GPU="ssh -i $SSH_KEY -o StrictHostKeyChecking=no ${SSH_USER}@${GPU_PUBLIC_IP}"
SCP_CTRL="scp -i $SSH_KEY -o StrictHostKeyChecking=no"
SCP_GPU="scp -i $SSH_KEY -o StrictHostKeyChecking=no"

echo "============================================================"
echo "COMP5434 GPU Cluster Deployment"
echo "============================================================"
echo "Control: ${CTRL_PUBLIC_IP} (${CTRL_PRIVATE_IP})"
echo "GPU:     ${GPU_PUBLIC_IP} (${GPU_PRIVATE_IP})"
echo "DDP:     ${NPROC_PER_NODE} GPUs"
echo "============================================================"

# ── Step 0: Verify connectivity ─────────────────────────────────────────
echo ""
echo ">>> Step 0: Checking connectivity..."

$SSH_CTRL "hostname && nproc" || { echo "ERROR: Cannot reach control node"; exit 1; }
echo "  Control node OK"

$SSH_GPU "hostname && nvidia-smi -L | wc -l" || { echo "ERROR: Cannot reach GPU node"; exit 1; }
echo "  GPU node OK"
$SSH_GPU "nvidia-smi -L"
$SSH_GPU "nvidia-smi --query-gpu=memory.total --format=csv,noheader"

# ── Step 1: Install Docker on both nodes ─────────────────────────────────
echo ""
echo ">>> Step 1: Installing Docker..."

install_docker() {
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker && systemctl start docker
    docker --version
}

echo "  Installing on control node..."
$SSH_CTRL "$(declare -f install_docker); install_docker" &
CTRL_DOCKER_PID=$!

echo "  Installing on GPU node..."
$SSH_GPU "$(declare -f install_docker); install_docker" &
GPU_DOCKER_PID=$!

wait $CTRL_DOCKER_PID $GPU_DOCKER_PID
echo "  Docker installed on both nodes."

# ── Step 2: Install NVIDIA Container Toolkit on GPU node ─────────────────
echo ""
echo ">>> Step 2: Installing NVIDIA Container Toolkit on GPU node..."

$SSH_GPU << 'INSTALL_NVIDIA'
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo "  NVIDIA Container Toolkit installed."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
INSTALL_NVIDIA

echo "  GPU Docker OK."

# ── Step 3: Build and push Docker images ─────────────────────────────────
echo ""
echo ">>> Step 3: Building Docker images..."

docker build -t comp5434/spark-runner -f docker/Dockerfile.spark .
docker build -t comp5434/torch-trainer -f docker/Dockerfile.torch .

echo "  Images built."

# ── Step 4: Transfer images to cloud nodes ───────────────────────────────
echo ""
echo ">>> Step 4: Transferring Docker images..."

echo "  Saving images..."
docker save comp5434/spark-runner -o /tmp/spark-runner.tar
docker save comp5434/torch-trainer -o /tmp/torch-trainer.tar

echo "  Uploading to control node..."
$SCP_CTRL /tmp/spark-runner.tar ${SSH_USER}@${CTRL_PUBLIC_IP}:/tmp/
$SSH_CTRL "docker load -i /tmp/spark-runner.tar && rm /tmp/spark-runner.tar"

echo "  Uploading to GPU node..."
$SCP_GPU /tmp/torch-trainer.tar ${SSH_USER}@${GPU_PUBLIC_IP}:/tmp/
$SSH_GPU "docker load -i /tmp/torch-trainer.tar && rm /tmp/torch-trainer.tar"

rm /tmp/spark-runner.tar /tmp/torch-trainer.tar
echo "  Images transferred."

# ── Step 5: Upload source code ───────────────────────────────────────────
echo ""
echo ">>> Step 5: Uploading source code..."

$SSH_CTRL "mkdir -p /app/{src,configs,scripts,data,output}"
$SSH_GPU  "mkdir -p /app/{src,configs,scripts,data,output}"

# Control node gets Spark code
$SCP_CTRL src/config.py ${SSH_USER}@${CTRL_PUBLIC_IP}:/app/src/
$SCP_CTRL src/hardware.py ${SSH_USER}@${CTRL_PUBLIC_IP}:/app/src/
$SCP_CTRL src/spark_features_cloud.py ${SSH_USER}@${CTRL_PUBLIC_IP}:/app/src/
$SCP_CTRL configs/cloud_config.yaml ${SSH_USER}@${CTRL_PUBLIC_IP}:/app/configs/
$SCP_CTRL configs/cluster_inventory.sh ${SSH_USER}@${CTRL_PUBLIC_IP}:/app/configs/

# GPU node gets training code
$SCP_GPU src/config.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/hardware.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/cloud_io.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/finetune_roberta_ddp.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/extract_embeddings.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/train_mlp.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU src/pipeline_cloud.py ${SSH_USER}@${GPU_PUBLIC_IP}:/app/src/
$SCP_GPU configs/cloud_config.yaml ${SSH_USER}@${GPU_PUBLIC_IP}:/app/configs/
$SCP_GPU configs/cluster_inventory.sh ${SSH_USER}@${GPU_PUBLIC_IP}:/app/configs/
$SCP_GPU scripts/launch_ddp.sh ${SSH_USER}@${GPU_PUBLIC_IP}:/app/scripts/

$SSH_GPU "chmod +x /app/scripts/*.sh"
echo "  Source code uploaded."

# ── Step 6: Upload data ──────────────────────────────────────────────────
if [ "$UPLOAD_DATA" = true ]; then
    echo ""
    echo ">>> Step 6: Uploading training data..."

    DATA_DIR="$PROJECT_DIR/../big_data_trea/data"

    if [ -f "$DATA_DIR/train.csv" ]; then
        echo "  Uploading train.csv (this may take a few minutes)..."
        $SSH_GPU "mkdir -p /app/data"
        $SCP_GPU "$DATA_DIR/train.csv" ${SSH_USER}@${GPU_PUBLIC_IP}:/app/data/
        $SCP_GPU "$DATA_DIR/test.csv" ${SSH_USER}@${GPU_PUBLIC_IP}:/app/data/
        $SCP_GPU "$DATA_DIR/prodInfo.csv" ${SSH_USER}@${GPU_PUBLIC_IP}:/app/data/
        echo "  Data uploaded."
    else
        echo "  WARNING: Data not found at $DATA_DIR"
        echo "  Upload data manually or use --skip-spark"
    fi
fi

# ── Step 7: Run Spark feature engineering on control node ────────────────
if [ "$SKIP_SPARK" = false ]; then
    echo ""
    echo ">>> Step 7: Running Spark feature engineering..."
    $SSH_CTRL << 'RUN_SPARK'
cd /app
docker run --rm \
    -v /app/data:/app/data \
    -v /app/output:/app/output \
    -e PYTHONPATH=/app/src \
    comp5434/spark-runner \
    spark-submit --master "local[4]" --driver-memory 8g \
    /app/src/spark_features_cloud.py --local
echo "Spark features complete."
RUN_SPARK

    # Pull results from control to GPU
    echo "  Syncing features to GPU node..."
    $SSH_CTRL "cd /app && tar czf /tmp/features.tar.gz data/*.parquet data/global_avg.npy"
    $SCP_CTRL ${SSH_USER}@${CTRL_PUBLIC_IP}:/tmp/features.tar.gz /tmp/
    $SCP_GPU /tmp/features.tar.gz ${SSH_USER}@${GPU_PUBLIC_IP}:/tmp/
    $SSH_GPU "cd /app && tar xzf /tmp/features.tar.gz && rm /tmp/features.tar.gz"
    $SSH_CTRL "rm /tmp/features.tar.gz"
    rm /tmp/features.tar.gz
    echo "  Features synced."
fi

# ── Step 8: Launch DDP Training on GPU node ──────────────────────────────
echo ""
echo ">>> Step 8: Launching 8×A10 DDP Training..."

$SSH_GPU << 'RUN_DDP'
cd /app
export PYTHONPATH=/app/src
export HF_ENDPOINT=https://hf-mirror.com
export TRAIN_CSV=/app/data/train.csv
export TEST_CSV=/app/data/test.csv
export PRODINFO_CSV=/app/data/prodInfo.csv
export DATA_DIR=/app/data

# Log GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Launch 8-GPU DDP training
docker run --rm --gpus all \
    --network host \
    -v /app/data:/app/data \
    -v /app/output:/app/output \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -e PYTHONPATH=/app/src \
    -e TRAIN_CSV=/app/data/train.csv \
    -e TEST_CSV=/app/data/test.csv \
    -e PRODINFO_CSV=/app/data/prodInfo.csv \
    -e DATA_DIR=/app/data \
    -e NPROC_PER_NODE=8 \
    -e MASTER_ADDR=localhost \
    -e MASTER_PORT=29500 \
    comp5434/torch-trainer \
    torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=29500 \
    /app/src/finetune_roberta_ddp.py

echo "Training complete!"
RUN_DDP

# ── Done ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Deployment & Training Complete!"
echo "============================================================"
echo ""
echo "Results on GPU node:"
$SSH_GPU "ls -lh /app/output/"
echo ""
echo "Download submission:"
echo "  scp -i $SSH_KEY ${SSH_USER}@${GPU_PUBLIC_IP}:/app/output/submission.csv ./"
echo ""
echo "Spark UI:  http://${CTRL_PUBLIC_IP}:8080"
