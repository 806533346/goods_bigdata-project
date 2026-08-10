import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
EMBED_PATH = os.path.join(DATA_PATH, "embeddings")

MODEL_NAME = "microsoft/deberta-v3-base"
BATCH_SIZE = 256  # increased - RTX 4060 has enough VRAM
MAX_LENGTH = 192  # 95% of texts fit within this length


def extract_embeddings(texts, model, tokenizer, device):
    """Extract embeddings using mean pooling over token embeddings."""
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        if (i // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * len(texts)
            print(f"  Batch {i // BATCH_SIZE}/{total_batches} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    return np.vstack(all_embeddings)


def get_embeddings_with_cache(texts, cache_path, model, tokenizer, device):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting DeBERTa-v3 embeddings ({len(texts)} texts)...")
    t0 = time.time()
    embeddings = extract_embeddings(texts, model, tokenizer, device)
    print(f"Done in {time.time() - t0:.1f}s, shape: {embeddings.shape}")

    np.save(cache_path, embeddings)
    print(f"Cached to {cache_path}")
    return embeddings


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), usecols=["title", "comment"])
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), usecols=["id", "title", "comment"])
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Combine title + comment
    train_texts = (train_df["title"].fillna("") + " [SEP] " + train_df["comment"].fillna("")).tolist()
    test_texts = (test_df["title"].fillna("") + " [SEP] " + test_df["comment"].fillna("")).tolist()

    # Load model
    print("Loading DeBERTa-v3-base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    # FP16 for speed and memory
    model = model.half()
    print(f"Model loaded. Hidden dim: {model.config.hidden_size}")

    os.makedirs(EMBED_PATH, exist_ok=True)

    # Extract embeddings
    train_emb = get_embeddings_with_cache(
        train_texts,
        os.path.join(EMBED_PATH, "train_deberta.npy"),
        model, tokenizer, device,
    )
    test_emb = get_embeddings_with_cache(
        test_texts,
        os.path.join(EMBED_PATH, "test_deberta.npy"),
        model, tokenizer, device,
    )

    print(f"\nTrain embeddings: {train_emb.shape}, dtype: {train_emb.dtype}")
    print(f"Test embeddings: {test_emb.shape}, dtype: {test_emb.dtype}")
    print(f"Total time: {time.time() - start_time:.1f}s")
