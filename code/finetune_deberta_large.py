"""Fine-tune DeBERTa-v3-large end-to-end for rating prediction."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.amp import autocast as amp_autocast

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
OUTPUT_PATH = os.path.join(DATA_PATH, "submission_finetune.csv")
MODEL_NAME = "roberta-base"
MAX_LENGTH = 192
BATCH_SIZE = 32
GRAD_ACCUM = 1  # effective batch size = 32
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SUBSET_SIZE = 500000  # 500K examples for training
VAL_SIZE = 50000      # 50K for validation


class ReviewDataset(Dataset):
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
    def __init__(self, model_name):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.deberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_emb = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        x = self.dropout(pooled)
        return self.head(x).squeeze(-1)


def evaluate(model, dataloader, device):
    model.eval()
    total_se = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ratings = batch["rating"].to(device)
            with amp_autocast('cuda', dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
            preds = torch.clamp(preds.float(), 1.0, 5.0)
            total_se += ((preds - ratings) ** 2).sum().item()
            n += len(ratings)
    return np.sqrt(total_se / n)


def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Combine title + comment
    for df in [train_df, test_df]:
        df["text"] = df["title"].fillna("") + " [SEP] " + df["comment"].fillna("")

    # Sample subset
    np.random.seed(42)
    idx = np.random.permutation(len(train_df))
    val_idx = idx[:VAL_SIZE]
    train_idx = idx[VAL_SIZE:VAL_SIZE + SUBSET_SIZE]

    train_texts = train_df["text"].iloc[train_idx].tolist()
    train_ratings = train_df["rating"].iloc[train_idx].values.astype(np.float32)
    val_texts = train_df["text"].iloc[val_idx].tolist()
    val_ratings = train_df["rating"].iloc[val_idx].values.astype(np.float32)
    test_texts = test_df["text"].tolist()

    print(f"Train subset: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_ds = ReviewDataset(train_texts, train_ratings, tokenizer, MAX_LENGTH)
    val_ds = ReviewDataset(val_texts, val_ratings, tokenizer, MAX_LENGTH)
    test_ds = ReviewDataset(test_texts, np.zeros(len(test_texts), dtype=np.float32), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    print("Loading DeBERTa-v3-base...")
    model = RatingRegressor(MODEL_NAME).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n=== Fine-tuning RoBERTa-base ===")
    print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"Steps/epoch: {len(train_loader) // GRAD_ACCUM}, Total steps: {total_steps}")
    print(f"LR: {LR}, Warmup: {warmup_steps} steps")

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ratings = batch["rating"].to(device)

            with amp_autocast('cuda', dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
                loss = nn.functional.mse_loss(preds, ratings) / GRAD_ACCUM

            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 500 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (step + 1) * len(train_loader)
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        # Evaluate
        val_rmse = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} done | Val RMSE: {val_rmse:.4f} | Time: {epoch_time:.0f}s")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  * New best!")

    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")

    # Predict test
    print("\n=== Predicting test ===")
    model.load_state_dict(best_state)
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with amp_autocast('cuda', dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
            preds = torch.clamp(preds.float(), 1.0, 5.0)
            test_preds.append(preds.cpu().numpy())

    test_pred = np.concatenate(test_preds)
    submission = pd.DataFrame({"id": test_df["id"].values, "rating": test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved to {OUTPUT_PATH}")
    print(f"Prediction stats: min={test_pred.min():.2f}, max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")
    print(f"\nTotal time: {time.time() - start:.1f}s ({(time.time() - start)/3600:.1f}h)")


if __name__ == "__main__":
    main()
