"""Extract embeddings from fine-tuned RoBERTa for MLP training."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast as amp_autocast

from config import (
    TRAIN_CSV, TEST_CSV, ROBERTA_WEIGHTS, TRAIN_EMB_NPY, TEST_EMB_NPY,
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, RANDOM_SEED,
)


class TextDataset(Dataset):
    """Dataset for text only (no ratings needed for embedding extraction)."""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
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
        }


class EmbeddingExtractor(nn.Module):
    """RoBERTa backbone with mean pooling for embedding extraction."""

    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return pooled


def extract_embeddings(texts, model, tokenizer, device, batch_size=128):
    """Extract 768-dim embeddings for a list of texts."""
    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_embeddings = []
    model.eval()
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with amp_autocast("cuda", dtype=torch.bfloat16):
                emb = model(input_ids, attention_mask)
            all_embeddings.append(emb.float().cpu().numpy())

            if (step + 1) % 500 == 0:
                elapsed = time.time() - start
                eta = elapsed / (step + 1) * len(loader)
                print(f"  Step {step+1}/{len(loader)} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    embeddings = np.concatenate(all_embeddings).astype(np.float16)
    elapsed = time.time() - start
    print(f"  Done: {embeddings.shape} in {elapsed:.0f}s")
    return embeddings


def run_extract_embeddings():
    """Main entry point for embedding extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    for df in [train_df, test_df]:
        df["text"] = df["title"].fillna("") + " [SEP] " + df["comment"].fillna("")

    # Load fine-tuned model
    print(f"Loading fine-tuned {MODEL_NAME}...")
    model = EmbeddingExtractor(MODEL_NAME).to(device)

    checkpoint = torch.load(ROBERTA_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"  Loaded checkpoint (Val RMSE: {checkpoint['best_val_rmse']:.4f})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Extract train embeddings
    print(f"\nExtracting train embeddings ({len(train_df)} texts)...")
    train_emb = extract_embeddings(train_df["text"].tolist(), model, tokenizer, device)
    np.save(TRAIN_EMB_NPY, train_emb)
    print(f"  Saved to {TRAIN_EMB_NPY} ({train_emb.nbytes / 1e9:.2f} GB)")

    # Extract test embeddings
    print(f"\nExtracting test embeddings ({len(test_df)} texts)...")
    test_emb = extract_embeddings(test_df["text"].tolist(), model, tokenizer, device)
    np.save(TEST_EMB_NPY, test_emb)
    print(f"  Saved to {TEST_EMB_NPY} ({test_emb.nbytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    run_extract_embeddings()
