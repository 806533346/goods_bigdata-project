"""Fine-tune RoBERTa-base end-to-end for rating prediction.

This module performs end-to-end fine-tuning of RoBERTa-base on the full
training dataset. The model backbone and regression head are jointly
optimized using MSE loss with BF16 mixed precision.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.amp import autocast as amp_autocast

from config import (
    TRAIN_CSV, TEST_CSV, ROBERTA_WEIGHTS, ROBERTA_TRAIN_LOG,
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR,
    WEIGHT_DECAY, WARMUP_RATIO, SUBSET_SIZE, VAL_SIZE, RANDOM_SEED,
)
from hardware import get_system_info


class ReviewDataset(Dataset):
    """Dataset for review text + rating pairs."""

    def __init__(self, texts, ratings, tokenizer, max_length):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rating": torch.tensor(self.ratings[idx], dtype=torch.float32),
        }


class RatingRegressor(nn.Module):
    """RoBERTa backbone + mean pooling + linear regression head."""

    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        x = self.dropout(pooled)
        return self.head(x).squeeze(-1)


def evaluate(model, dataloader, device):
    """Compute RMSE on validation data."""
    model.eval()
    total_se = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ratings = batch["rating"].to(device)
            with amp_autocast("cuda", dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
            preds = torch.clamp(preds.float(), 1.0, 5.0)
            total_se += ((preds - ratings) ** 2).sum().item()
            n += len(ratings)
    return np.sqrt(total_se / n)


def run_finetune():
    """Main fine-tuning entry point."""
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hardware info
    device_info = get_system_info()
    print("=== Device Info ===")
    for k, v in device_info.items():
        print(f"  {k}: {v}")
    print()

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Combine title + comment
    for df in [train_df, test_df]:
        df["text"] = df["title"].fillna("") + " [SEP] " + df["comment"].fillna("")

    # Sample subset
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(train_df))
    val_idx = idx[:VAL_SIZE]
    train_idx = idx[VAL_SIZE : VAL_SIZE + SUBSET_SIZE]

    train_texts = train_df["text"].iloc[train_idx].tolist()
    train_ratings = train_df["rating"].iloc[train_idx].values.astype(np.float32)
    val_texts = train_df["text"].iloc[val_idx].tolist()
    val_ratings = train_df["rating"].iloc[val_idx].values.astype(np.float32)
    test_texts = test_df["text"].tolist()

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_ds = ReviewDataset(train_texts, train_ratings, tokenizer, MAX_LENGTH)
    val_ds = ReviewDataset(val_texts, val_ratings, tokenizer, MAX_LENGTH)
    test_ds = ReviewDataset(test_texts, np.zeros(len(test_texts), dtype=np.float32), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print(f"Loading {MODEL_NAME}...")
    model = RatingRegressor(MODEL_NAME).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # Training params
    train_params = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "subset_size": len(train_texts),
        "val_size": VAL_SIZE,
        "total_params_M": round(total_params / 1e6, 1),
        "trainable_params_M": round(trainable_params / 1e6, 1),
        "precision": "bf16",
        "optimizer": "AdamW",
        "scheduler": "linear_warmup",
        "loss_function": "MSELoss",
        "grad_clip": 1.0,
    }

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n=== Fine-tuning {MODEL_NAME} ===")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"Steps/epoch: {len(train_loader) // GRAD_ACCUM}, Total steps: {total_steps}")
    print(f"LR: {LR}, Warmup: {warmup_steps} steps, Precision: BF16")

    best_val_rmse = float("inf")
    best_state = None
    epoch_logs = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ratings = batch["rating"].to(device)

            with amp_autocast("cuda", dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
                loss = nn.functional.mse_loss(preds, ratings) / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 1000 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (step + 1) * len(train_loader)
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/(step+1):.4f} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        # Evaluate
        val_rmse = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} done | Val RMSE: {val_rmse:.4f} | Time: {epoch_time:.0f}s")

        epoch_logs.append({
            "epoch": epoch + 1,
            "avg_loss": total_loss / len(train_loader),
            "val_rmse": val_rmse,
            "time_seconds": epoch_time,
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  * New best! Saving checkpoint...")
            torch.save({
                "model_state_dict": best_state,
                "model_name": MODEL_NAME,
                "best_val_rmse": best_val_rmse,
                "train_params": train_params,
                "epoch_logs": epoch_logs,
            }, ROBERTA_WEIGHTS)
            print(f"  Checkpoint saved to {ROBERTA_WEIGHTS}")

        torch.cuda.empty_cache()

    # Predict test
    print("\n=== Predicting test ===")
    model.load_state_dict(best_state)
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with amp_autocast("cuda", dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
            preds = torch.clamp(preds.float(), 1.0, 5.0)
            test_preds.append(preds.cpu().numpy())

    test_pred = np.concatenate(test_preds)

    total_time = time.time() - start

    # Save training log
    log = {
        "device_info": device_info,
        "train_params": train_params,
        "epoch_logs": epoch_logs,
        "best_val_rmse": best_val_rmse,
        "total_time_seconds": total_time,
        "total_time_hours": round(total_time / 3600, 2),
        "prediction_stats": {
            "min": float(test_pred.min()),
            "max": float(test_pred.max()),
            "mean": float(test_pred.mean()),
        },
    }
    with open(ROBERTA_TRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")

    return best_val_rmse, test_pred


if __name__ == "__main__":
    run_finetune()
