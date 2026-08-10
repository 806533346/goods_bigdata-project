"""Extract embeddings from fine-tuned RoBERTa-base model."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast as amp_autocast

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
CHECKPOINT_PATH = os.path.join(DATA_PATH, "roberta_base_finetuned.pt")
EMBED_PATH = os.path.join(DATA_PATH, "embeddings")
TRAIN_EMBED_SAVE = os.path.join(EMBED_PATH, "train_roberta_ft.npy")
TEST_EMBED_SAVE = os.path.join(EMBED_PATH, "test_roberta_ft.npy")
MODEL_NAME = "roberta-base"
MAX_LENGTH = 192
BATCH_SIZE = 256  # large batch since no gradients needed


class TextDataset(Dataset):
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


def extract_embeddings(model, dataloader, n_samples, device):
    """Extract mean-pooled embeddings (768-dim) from RoBERTa backbone."""
    embeddings = np.zeros((n_samples, 768), dtype=np.float16)
    idx = 0
    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with amp_autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                token_emb = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            emb_np = pooled.float().cpu().numpy()
            n = len(emb_np)
            embeddings[idx:idx + n] = emb_np.astype(np.float16)
            idx += n

            if idx % (BATCH_SIZE * 50) == 0:
                elapsed = time.time() - start
                speed = idx / elapsed
                eta = (n_samples - idx) / speed
                print(f"  {idx}/{n_samples} ({idx/n_samples*100:.1f}%) | {speed:.0f} samples/s | ETA {eta:.0f}s")

    return embeddings


def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(EMBED_PATH, exist_ok=True)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    for df in [train_df, test_df]:
        df["text"] = df["title"].fillna("") + " [SEP] " + df["comment"].fillna("")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Build model and load fine-tuned weights
    print(f"Loading {MODEL_NAME} backbone...")
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)

    print(f"Loading fine-tuned checkpoint from {CHECKPOINT_PATH} ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    # Checkpoint stores full RatingRegressor state_dict with "backbone." prefix
    state_dict = checkpoint["model_state_dict"]
    backbone_state = {k.replace("backbone.", "", 1): v for k, v in state_dict.items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_state)
    print(f"Loaded! Previous best Val RMSE: {checkpoint.get('best_val_rmse', 'N/A')}")
    backbone.eval()

    # Extract train embeddings
    print(f"\n=== Extracting train embeddings ({len(train_df)} samples) ===")
    train_texts = train_df["text"].tolist()
    train_ds = TextDataset(train_texts, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    train_emb = extract_embeddings(backbone, train_loader, len(train_texts), device)
    print(f"Train embeddings shape: {train_emb.shape}, dtype: {train_emb.dtype}")
    np.save(TRAIN_EMBED_SAVE, train_emb)
    print(f"Saved to {TRAIN_EMBED_SAVE} ({train_emb.nbytes / 1e9:.2f} GB)")

    # Extract test embeddings
    print(f"\n=== Extracting test embeddings ({len(test_df)} samples) ===")
    test_texts = test_df["text"].tolist()
    test_ds = TextDataset(test_texts, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    test_emb = extract_embeddings(backbone, test_loader, len(test_texts), device)
    print(f"Test embeddings shape: {test_emb.shape}, dtype: {test_emb.dtype}")
    np.save(TEST_EMBED_SAVE, test_emb)
    print(f"Saved to {TEST_EMBED_SAVE} ({test_emb.nbytes / 1e6:.2f} MB)")

    total = time.time() - start
    print(f"\n=== Summary ===")
    print(f"Total time: {total:.1f}s ({total/60:.1f} min)")
    print(f"Train embeddings: {train_emb.shape} -> {TRAIN_EMBED_SAVE}")
    print(f"Test embeddings:  {test_emb.shape} -> {TEST_EMBED_SAVE}")


if __name__ == "__main__":
    main()
