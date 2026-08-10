#!/bin/bash
# ============================================================
# COMP5434 Cloud - Deployment Script
# Provisions cloud VMs, pushes Docker images, starts cluster.
#
# Prerequisites:
#   - Cloud CLI installed (aliyun CLI / aws CLI)
#   - Docker images built: docker build -f docker/Dockerfile.spark -t spark-runner .
#   - Cloud credentials configured
#
# Usage:
#   bash scripts/deploy_cloud.sh                     # interactive
#   bash scripts/deploy_cloud.sh --provider aliyun    # Alibaba Cloud
#   bash scripts/deploy_cloud.sh --provider aws       # AWS
#   bash scripts/deploy_cloud.sh --single-vm          # cheapest: one multi-GPU VM
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source configs/cloud_config.yaml 2>/dev/null || true

# ── Configuration ────────────────────────────────────────────────────────
PROVIDER="${CLOUD_PROVIDER:-aliyun}"
REGION="${CLOUD_REGION:-cn-hangzhou}"
SINGLE_VM=false

# Instance types
declare -A SPARK_MASTER_TYPE
SPARK_MASTER_TYPE[aliyun]="ecs.g7.xlarge"       # 4vCPU 16GB
SPARK_MASTER_TYPE[aws]="m6i.xlarge"

declare -A SPARK_WORKER_TYPE
SPARK_WORKER_TYPE[aliyun]="ecs.g7.2xlarge"      # 8vCPU 32GB
SPARK_WORKER_TYPE[aws]="r6i.2xlarge"

declare -A GPU_TYPE
GPU_TYPE[aliyun]="ecs.gn6v-c8g1.4xlarge"        # 4x T4 16GB
GPU_TYPE[aws]="g4dn.12xlarge"

declare -A IMAGE_REGISTRY
IMAGE_REGISTRY[aliyun]="registry.cn-hangzhou.aliyuncs.com/comp5434"
IMAGE_REGISTRY[aws]="${AWS_ACCOUNT_ID:-}.dkr.ecr.${REGION}.amazonaws.com/comp5434"

# ── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider) PROVIDER="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --single-vm) SINGLE_VM=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

REGISTRY="${IMAGE_REGISTRY[$PROVIDER]}"
SPARK_IMAGE="${REGISTRY}/spark-runner:latest"
TORCH_IMAGE="${REGISTRY}/torch-trainer:latest"

echo "============================================================"
echo "COMP5434 Cloud Deployment"
echo "============================================================"
echo "Provider:  $PROVIDER"
echo "Region:    $REGION"
echo "Mode:      $([ "$SINGLE_VM" = true ] && echo 'Single VM' || echo 'Cluster')"
echo "Registry:  $REGISTRY"
echo "============================================================"

# ── Step 1: Build Docker Images ──────────────────────────────────────────
echo ""
echo ">>> Step 1: Building Docker images..."

docker build -t spark-runner -f docker/Dockerfile.spark .
docker build -t torch-trainer -f docker/Dockerfile.torch .

echo "Images built successfully."

# ── Step 2: Push to Container Registry ───────────────────────────────────
echo ""
echo ">>> Step 2: Pushing images to registry..."

docker tag spark-runner "$SPARK_IMAGE"
docker tag torch-trainer "$TORCH_IMAGE"

case $PROVIDER in
    aliyun)
        echo "Logging into Alibaba Cloud ACR..."
        docker login --username="${ACR_USERNAME:-}" --password="${ACR_PASSWORD:-}" \
            "$(echo "$REGISTRY" | cut -d/ -f1)"
        ;;
    aws)
        echo "Logging into AWS ECR..."
        aws ecr get-login-password --region "$REGION" | \
            docker login --username AWS --password-stdin "$REGISTRY"
        ;;
esac

docker push "$SPARK_IMAGE" &
docker push "$TORCH_IMAGE" &
wait

echo "Images pushed."

# ── Step 3: Provision Cloud Resources ────────────────────────────────────
echo ""
echo ">>> Step 3: Provisioning cloud resources..."

if [ "$SINGLE_VM" = true ]; then
    # Single multi-GPU VM for everything (cheapest)
    echo "Provisioning single GPU VM: ${GPU_TYPE[$PROVIDER]}"
    echo ""
    echo "  Manual steps (run in cloud console or CLI):"
    echo "  1. Create 1x ${GPU_TYPE[$PROVIDER]} with GPU driver + Docker"
    echo "  2. SSH in and run:"
    echo "     docker pull $TORCH_IMAGE"
    echo "     docker pull $SPARK_IMAGE"
    echo "  3. Run pipeline:"
    echo "     docker run --gpus all -v data:/app/data $TORCH_IMAGE \\"
    echo "       python3 src/pipeline_cloud.py --cloud"
    echo ""
else
    # Distributed cluster
    echo "Provisioning distributed cluster:"
    echo "  - 1x ${SPARK_MASTER_TYPE[$PROVIDER]} (Spark master)"
    echo "  - 2x ${SPARK_WORKER_TYPE[$PROVIDER]} (Spark workers)"
    echo "  - 1x ${GPU_TYPE[$PROVIDER]} (GPU trainer)"

    echo ""
    echo "  On Spark master:"
    echo "    docker compose -f docker/docker-compose.spark.yml up -d spark-master"
    echo ""
    echo "  On each Spark worker:"
    echo "    export SPARK_MASTER_IP=<master-private-ip>"
    echo "    docker compose -f docker/docker-compose.spark.yml up -d spark-worker"
    echo ""
    echo "  On GPU node:"
    echo "    docker run --gpus all -v data:/app/data \\"
    echo "      -e SPARK_MASTER=spark://<master-ip>:7077 \\"
    echo "      $TORCH_IMAGE bash scripts/launch_ddp.sh"
fi

# ── Step 4: Upload Data to Object Storage ────────────────────────────────
echo ""
echo ">>> Step 4: Upload data to cloud storage..."
echo "  Run the following to upload data:"
echo "  python3 -c \""
echo "    from src.cloud_io import upload_to_cloud"
echo "    upload_to_cloud('data/train.csv', 'raw/train.csv')"
echo "    upload_to_cloud('data/test.csv', 'raw/test.csv')"
echo "    upload_to_cloud('data/prodInfo.csv', 'raw/prodInfo.csv')"
echo "  \""

echo ""
echo "============================================================"
echo "Deployment instructions ready!"
echo "============================================================"
echo ""
echo "To monitor Spark:  http://<spark-master-ip>:8080"
echo "To monitor Prefect: http://<orchestrator-ip>:4200"
