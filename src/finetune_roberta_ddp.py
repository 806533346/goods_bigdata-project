"""
RoBERTa-base End-to-End Fine-Tuning with DistributedDataParallel (DDP).

Supports:
  - Single GPU (--local)
  - Multi-GPU DDP via torchrun (the default)
  - BF16 mixed precision
  - Gradient accumulation for larger effective batch sizes
  - Checkpoint save/load to cloud object storage
  - Automatic linear LR scaling with world_size

Usage:
    # Single GPU (local dev):
    python finetune_roberta_ddp.py --local

    # Multi-GPU DDP (launched by torchrun):
    torchrun --nproc_per_node=4 --master_port=29500 finetune_roberta_ddp.py

    # Multi-node DDP:
    torchrun --nproc_per_node=4 --nnodes=2 \\
        --master_addr=<head-node-ip> --master_port=29500 \\
        finetune_roberta_ddp.py
"""
import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, GRAD_ACCUM, EPOCHS,
    LR, WEIGHT_DECAY, WARMUP_RATIO, MAX_GRAD_NORM,
    VAL_SIZE, SUBSET_SIZE, RANDOM_SEED,
    DDP_BACKEND, MASTER_ADDR, MASTER_PORT, NUM_WORKERS,
    DATA_DIR, OUTPUT_DIR, CHECKPOINT_PT, TRAIN_LOG_JSON,
    TRAIN_CSV, TEST_CSV, PRODINFO_CSV, SUBMISSION_CSV,
)
from cloud_io import upload_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_ddp")


# ── DDP Setup ────────────────────────────────────────────────────────────

def is_distributed() -> bool:
    """Check if running under torchrun (RANK env var is set)."""
    return "RANK" in os.environ


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp():
    """
    初始化分布式训练进程组。

    torchrun 自动设置环境变量:
      MASTER_ADDR, MASTER_PORT → 主节点地址
      RANK → 全局进程序号 (0 ~ world_size-1)
      LOCAL_RANK → 本机 GPU 序号
      WORLD_SIZE → 总进程数

    NCCL 是 NVIDIA 的 GPU 间通信库，做 All-Reduce 梯度同步最快。
    """
    if not is_distributed():
        return 0, 1, 0

    rank     = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    # 每个进程绑定自己的 GPU
    torch.cuda.set_device(local_rank)

    # 建立进程组，之后所有进程可通过 NCCL 通信
    dist.init_process_group(
        backend=DDP_BACKEND,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    if is_main_process():
        logger.info(f"DDP initialized: world_size={world_size}, backend={DDP_BACKEND}")

    # 等所有进程都初始化完毕再继续
    dist.barrier()
    return rank, world_size, local_rank


def cleanup_ddp():
    if is_distributed():
        dist.destroy_process_group()


# ── Dataset ──────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
    """Dataset for review rating prediction with RoBERTa tokenization."""

    def __init__(self, texts: list, ratings: list, tokenizer, max_length: int = MAX_LENGTH):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = float(self.ratings[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rating": torch.tensor(rating, dtype=torch.float32),
        }


# ── Model ────────────────────────────────────────────────────────────────

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token dimension, weighted by attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class RatingRegressor(nn.Module):
    """
    评分预测模型: RoBERTa 编码器 + 均值池化 + 回归头。

    输入:  title + [SEP] + comment   (tokenized, max_length=192)
    中间:  RoBERTa-base 12层 Transformer → 768维向量
    输出:  1 个浮点数 (1-5 分预测)

    为什么不直接用 CLS token?
      CLS 是给分类任务设计的，mean pooling 能更好地利用
      所有 token 的信息，对回归任务效果更好。
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.head(self.dropout(pooled))


# ── Evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, world_size: int = 1):
    """Compute RMSE on validation set."""
    model.eval()
    total_se = 0.0
    total_n  = 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        ratings        = batch["rating"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            preds = model(input_ids, attention_mask).squeeze(-1)

        preds = preds.clamp(1.0, 5.0)
        total_se += ((preds - ratings) ** 2).sum().item()
        total_n  += ratings.size(0)

    # Aggregate across DDP processes
    if world_size > 1:
        se_tensor = torch.tensor([total_se], device=device)
        n_tensor  = torch.tensor([total_n], device=device)
        dist.all_reduce(se_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_tensor,  op=dist.ReduceOp.SUM)
        total_se = se_tensor.item()
        total_n  = n_tensor.item()

    rmse = (total_se / total_n) ** 0.5 if total_n > 0 else float("inf")
    model.train()
    return rmse


# ── Training Loop ────────────────────────────────────────────────────────

def train_epoch_chunk(
    model, dataloader, optimizer, scheduler, device,
    epoch: int, world_size: int,
    start_step: int, end_step: int,
    grad_accum: int = GRAD_ACCUM,
    max_grad_norm: float = MAX_GRAD_NORM,
):
    """Train a chunk of steps with gradient accumulation and BF16 mixed precision."""
    model.train()
    total_loss = 0.0
    steps = 0
    accum_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(dataloader):
        # Skip to start_step
        actual_step = i // grad_accum
        if actual_step < start_step:
            continue
        if actual_step >= end_step:
            break

        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        ratings        = batch["rating"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            preds = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.functional.mse_loss(preds, ratings)

        loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss
            steps += 1

            if steps % 1000 == 0 and is_main_process():
                avg_loss = total_loss / steps
                elapsed = time.time() - t0
                eta = (elapsed / steps) * (end_step - start_step - steps)
                logger.info(
                    f"Step {start_step + steps:6d}/{(end_step - start_step):6d} | "
                    f"Loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )

            accum_loss = 0.0

    return total_loss / max(steps, 1)


# ── Old train_epoch kept for reference ──

def train_epoch(
    model, dataloader, optimizer, scheduler, device,
    epoch: int, world_size: int, grad_accum: int = GRAD_ACCUM,
    max_grad_norm: float = MAX_GRAD_NORM,
):
    """Train one full epoch (kept for backward compatibility)."""
    model.train()
    total_loss = 0.0
    steps = 0
    accum_loss = 0.0
    t0 = time.time()

    for i, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        ratings        = batch["rating"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            preds = model(input_ids, attention_mask).squeeze(-1)
            loss = nn.functional.mse_loss(preds, ratings)

        loss = loss / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss
            steps += 1

            if steps % 1000 == 0 and is_main_process():
                avg_loss = total_loss / steps
                elapsed = time.time() - t0
                eta = (elapsed / steps) * (len(dataloader) / grad_accum - steps)
                logger.info(
                    f"Epoch {epoch:2d} | Step {steps:6d}/{len(dataloader)//grad_accum:6d} | "
                    f"Loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )

            accum_loss = 0.0

    return total_loss / max(steps, 1)


# ── Main Entry Point ─────────────────────────────────────────────────────

def run_finetune(local_mode: bool = False):
    """
    Run RoBERTa fine-tuning (single-GPU or DDP).

    Args:
        local_mode: If True, force single-GPU mode without DDP.
    """
    # ── DDP Setup ──
    if local_mode:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        logger.info(f"Device: {device}, World size: {world_size}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Load Data ──
    if is_main_process():
        logger.info("Loading training data...")

    df = pd.read_csv(TRAIN_CSV)
    n_total = len(df)

    # Optional subset for quick experiments
    if SUBSET_SIZE and SUBSET_SIZE < n_total:
        df = df.sample(n=SUBSET_SIZE, random_state=RANDOM_SEED)
        n_total = len(df)

    # Train/val split (deterministic)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n_total)
    val_idx  = indices[:VAL_SIZE]
    train_idx = indices[VAL_SIZE:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    # Prepare texts: title + [SEP] + comment
    train_texts = (df_train["title"].fillna("") + " [SEP] " + df_train["comment"].fillna("")).tolist()
    val_texts   = (df_val["title"].fillna("") + " [SEP] " + df_val["comment"].fillna("")).tolist()
    train_ratings = df_train["rating"].tolist()
    val_ratings   = df_val["rating"].tolist()

    if is_main_process():
        logger.info(f"Data loaded: train={len(train_texts):,}, val={len(val_texts):,}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ReviewDataset(train_texts, train_ratings, tokenizer, MAX_LENGTH)
    val_ds   = ReviewDataset(val_texts, val_ratings, tokenizer, MAX_LENGTH)

    # ── DataLoaders with DistributedSampler ──
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE * 2, sampler=val_sampler,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

    # ── Model ──
    model = RatingRegressor(MODEL_NAME).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ── Optimizer (线性 LR 缩放) ──
    # DDP 下有效 batch = 单卡 batch × GPU 数量 × 梯度累积步数
    # 根据"线性缩放规则": 有效 batch 翻倍 → LR 也应翻倍
    # 例如: 单卡 BATCH=128, 2 GPUs → effective=256, LR × 2
    effective_batch = BATCH_SIZE * world_size * GRAD_ACCUM
    scaled_lr = LR * world_size

    # DDP 包装后 model 的属性都加了 "module." 前缀，取 .module 得到原始模型
    raw_model = model.module if world_size > 1 else model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=scaled_lr,
        weight_decay=WEIGHT_DECAY,      # 只衰减权重，不衰减 bias
    )

    # OneCycleLR: 先线形预热 → 达到峰值 → 余弦衰减到 0
    # 比普通的线性衰减收敛更快、泛化更好
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=scaled_lr,
        total_steps=total_steps,
        pct_start=WARMUP_RATIO,          # 前 10% 步数为预热
        anneal_strategy="cos",           # 余弦曲线衰减
    )

    if is_main_process():
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"LR scaled: {LR:.2e} → {scaled_lr:.2e} (×{world_size})")
        logger.info(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # ── Training Loop ──
    best_val_rmse = float("inf")
    train_log = {
        "model": MODEL_NAME,
        "device": str(device),
        "world_size": world_size,
        "effective_batch_size": effective_batch,
        "base_lr": LR,
        "scaled_lr": scaled_lr,
        "total_steps": total_steps,
        "epochs": [],
    }
    total_start = time.time()

    # Early stopping settings
    VAL_EVERY_N_STEPS = 2000  # Validate every N steps
    PATIENCE = 3              # Stop if no improvement for N validations
    no_improve_count = 0

    for epoch in range(1, EPOCHS + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # Split training into chunks with intermediate validation
        total_batches = len(train_loader) // GRAD_ACCUM
        chunk_start = 0
        best_step = 0

        while chunk_start < total_batches:
            chunk_end = min(chunk_start + VAL_EVERY_N_STEPS, total_batches)

            t0 = time.time()
            train_loss_chunk = train_epoch_chunk(
                model, train_loader, optimizer, scheduler, device,
                epoch=epoch, world_size=world_size,
                start_step=chunk_start, end_step=chunk_end,
                grad_accum=GRAD_ACCUM, max_grad_norm=MAX_GRAD_NORM,
            )
            chunk_start = chunk_end
            train_time_so_far = time.time() - total_start

            # Intermediate validation
            val_rmse = evaluate(model, val_loader, device, world_size)

            if is_main_process():
                logger.info(
                    f"Epoch {epoch} | Step {chunk_end}/{total_batches} | "
                    f"Train Loss: {train_loss_chunk:.4f} | Val RMSE: {val_rmse:.4f} | "
                    f"Time: {train_time_so_far:.0f}s"
                )

                # Save best checkpoint
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_step = chunk_end
                    no_improve_count = 0
                    ckpt = raw_model.state_dict()
                    torch.save(ckpt, CHECKPOINT_PT)
                    logger.info(f"✅ Best checkpoint (step={best_step}, RMSE={best_val_rmse:.4f})")
                else:
                    no_improve_count += 1
                    logger.info(f"No improvement ×{no_improve_count} (best={best_val_rmse:.4f})")

                # Early stopping
                if no_improve_count >= PATIENCE:
                    logger.info(f"⏹ Early stop at step {chunk_end} (patience={PATIENCE})")
                    break

        # Save per-epoch summary
        train_time = time.time() - total_start
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss_chunk, 6),
            "val_rmse": round(best_val_rmse, 6),
            "train_time_s": round(train_time, 1),
            "best_step": best_step,
        }
        train_log["epochs"].append(epoch_info)

        if no_improve_count >= PATIENCE:
            break

        # Synchronize
        if world_size > 1:
            dist.barrier()
            best_rmse_tensor = torch.tensor([best_val_rmse], device=device)
            dist.broadcast(best_rmse_tensor, src=0)
            best_val_rmse = best_rmse_tensor.item()

    total_time = time.time() - total_start

    # ── Save Training Log (with hardware info) ──
    train_log["best_val_rmse"] = round(best_val_rmse, 6)
    train_log["total_time_s"] = round(total_time, 1)
    train_log["total_time_h"] = round(total_time / 3600, 2)

    if is_main_process():
        # Collect hardware info
        from hardware import get_system_info
        hw = get_system_info()
        train_log["hardware"] = hw

        os.makedirs(os.path.dirname(TRAIN_LOG_JSON), exist_ok=True)
        with open(TRAIN_LOG_JSON, "w") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)
        logger.info(f"Training log saved: {TRAIN_LOG_JSON}")
        logger.info(f"Best Val RMSE: {best_val_rmse:.4f}")
        logger.info(f"Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")

    # ── Test Set Prediction ──
    if is_main_process():
        logger.info("Generating test predictions...")
        raw_model.eval()

        # Reload best checkpoint
        state = torch.load(CHECKPOINT_PT, map_location=device, weights_only=True)
        raw_model.load_state_dict(state)

        # Load test data
        df_test = pd.read_csv(TEST_CSV)
        test_texts = (df_test["title"].fillna("") + " [SEP] " + df_test["comment"].fillna("")).tolist()
        test_ds = ReviewDataset(test_texts, [0.0] * len(test_texts), tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    preds = raw_model(input_ids, attn_mask).squeeze(-1)
                predictions.append(preds.clamp(1.0, 5.0).float().cpu().numpy())

        predictions = np.concatenate(predictions)
        submission = pd.DataFrame({"id": range(len(predictions)), "rating": predictions})
        submission.to_csv(SUBMISSION_CSV, index=False)
        logger.info(f"Submission saved: {SUBMISSION_CSV} ({len(predictions)} rows)")
        logger.info(f"Prediction stats: mean={predictions.mean():.3f}, std={predictions.std():.3f}")

        train_log["prediction_stats"] = {
            "mean": round(float(predictions.mean()), 4),
            "std": round(float(predictions.std()), 4),
            "min": round(float(predictions.min()), 4),
            "max": round(float(predictions.max()), 4),
        }
        with open(TRAIN_LOG_JSON, "w") as f:
            json.dump(train_log, f, indent=2, ensure_ascii=False)

    cleanup_ddp()
    return best_val_rmse


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa DDP Fine-Tuning")
    parser.add_argument("--local", action="store_true",
                        help="Force single-GPU mode (no DDP)")
    args = parser.parse_args()

    run_finetune(local_mode=args.local)
