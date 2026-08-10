#!/bin/bash
# Run on GPU-0 (DDP Master)
cd /app
source /app/configs/cluster_env.sh
echo "GPU-0 Rank 0: BATCH=$BATCH_SIZE, Effective=$((BATCH_SIZE*2))"

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=10.0.1.160 \
    --master_port=29501 \
    src/finetune_roberta_ddp.py \
    2>&1 | tee /app/output/train_rank0.log
