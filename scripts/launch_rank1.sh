#!/bin/bash
# Run on GPU-1 (DDP Worker)
cd /app
source /app/configs/cluster_env.sh
echo "GPU-1 Rank 1: BATCH=$BATCH_SIZE"

nohup torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=10.0.1.160 \
    --master_port=29501 \
    src/finetune_roberta_ddp.py \
    > /app/output/train_rank1.log 2>&1 &

echo "Rank 1 PID: $!"
