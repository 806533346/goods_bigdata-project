#!/bin/bash
# ============================================================
# COMP5434 Cloud - DDP Launch Script
# Launches multi-GPU distributed training via torchrun.
#
# Usage:
#   bash scripts/launch_ddp.sh                  # auto-detect GPU count
#   bash scripts/launch_ddp.sh --nnodes 2       # multi-node
#   bash scripts/launch_ddp.sh --local           # single GPU fallback
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
NPROC=${NPROC_PER_NODE:-}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
LOCAL_MODE=false
EXTRA_ARGS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc) NPROC="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --local) LOCAL_MODE=true; shift ;;
        --cloud) EXTRA_ARGS+=("--cloud"); shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Auto-detect GPU count
if [ -z "$NPROC" ]; then
    if command -v nvidia-smi &>/dev/null; then
        NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NPROC=1
    fi
fi

echo "============================================================"
echo "COMP5434 DDP Training Launcher"
echo "============================================================"
echo "GPUs:          $NPROC"
echo "Nodes:         $NNODES"
echo "Node rank:     $NODE_RANK"
echo "Master:        $MASTER_ADDR:$MASTER_PORT"
echo "Mode:          $([ "$LOCAL_MODE" = true ] && echo 'Single GPU' || echo 'DDP')"
echo "============================================================"

# Set HF mirror for China
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="src:$PYTHONPATH"

if [ "$LOCAL_MODE" = true ]; then
    # Single GPU (no DDP overhead)
    echo "Running in single-GPU mode..."
    exec python3 src/finetune_roberta_ddp.py --local "${EXTRA_ARGS[@]}"
else
    # DDP via torchrun
    echo "Launching torchrun with $NPROC GPUs..."
    exec torchrun \
        --nproc_per_node="$NPROC" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        src/finetune_roberta_ddp.py "${EXTRA_ARGS[@]}"
fi
