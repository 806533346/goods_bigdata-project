import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
OUTPUT_PATH = "/home/nmxc/project_code/big_data_trea/data/submission_bert.csv"
EMBED_PATH = "/home/nmxc/project_code/big_data_trea/data/embeddings"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 256
MAX_LENGTH = 256


def load_data():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Combine title + comment as BERT input
    for df in [train_df, test_df]:
        df["text"] = df["title"].fillna("") + " [SEP] " + df["comment"].fillna("")

    return train_df, test_df


def extract_bert_embeddings(texts, model, tokenizer, device):
    """Extract BERT embeddings using mean pooling over token embeddings."""
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
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask  # (batch, hidden)

        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  Batch {i // BATCH_SIZE}/{total_batches}")

    return np.vstack(all_embeddings)


def get_embeddings_with_cache(texts, cache_path, model, tokenizer, device):
    """Extract embeddings with disk caching."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting BERT embeddings ({len(texts)} texts)...")
    t0 = time.time()
    embeddings = extract_bert_embeddings(texts, model, tokenizer, device)
    print(f"Done in {time.time() - t0:.1f}s, shape: {embeddings.shape}")

    np.save(cache_path, embeddings)
    print(f"Cached to {cache_path}")
    return embeddings


def build_features(train_df, test_df, model, tokenizer, device):
    """Build feature matrix: BERT embeddings + raw numeric features (no stats)."""
    os.makedirs(EMBED_PATH, exist_ok=True)

    train_emb = get_embeddings_with_cache(
        train_df["text"].tolist(),
        os.path.join(EMBED_PATH, "train_emb.npy"),
        model,
        tokenizer,
        device,
    )
    test_emb = get_embeddings_with_cache(
        test_df["text"].tolist(),
        os.path.join(EMBED_PATH, "test_emb.npy"),
        model,
        tokenizer,
        device,
    )

    # Raw numeric features (NOT statistical features)
    train_numeric = np.column_stack([
        (train_df["purchased"] == True).astype(np.float32).values,
        train_df["votes"].fillna(0).astype(np.float32).values,
    ])
    test_numeric = np.column_stack([
        (test_df["purchased"] == True).astype(np.float32).values,
        test_df["votes"].fillna(0).astype(np.float32).values,
    ])

    X_train = np.hstack([train_emb, train_numeric])
    X_test = np.hstack([test_emb, test_numeric])
    y_train = train_df["rating"].values.astype(np.float32)

    return X_train, y_train, X_test


def train_and_validate(X_train, y_train, X_test, test_ids):
    """Train Ridge regression with validation, then retrain on full data."""
    print("\n=== Validation (80/20 split) ===")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Try different alpha values
    best_rmse = float("inf")
    best_alpha = 1.0
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        reg = Ridge(alpha=alpha)
        reg.fit(X_tr, y_tr)
        val_pred = np.clip(reg.predict(X_val), 1, 5)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  alpha={alpha:>5}, RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}, Validation RMSE: {best_rmse:.4f}")

    print("\n=== Training final model on ALL data ===")
    final_reg = Ridge(alpha=best_alpha)
    final_reg.fit(X_train, y_train)

    # Train RMSE
    train_pred = np.clip(final_reg.predict(X_train), 1, 5)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"Train RMSE: {train_rmse:.4f}")

    # Predict test
    test_pred = np.clip(final_reg.predict(X_test), 1, 5)
    submission = pd.DataFrame({"id": test_ids, "rating": test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")
    print(f"Prediction stats: min={test_pred.min():.2f}, max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")

    return best_rmse


def main():
    start_time = time.time()

    # Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device).half()  # FP16 for speed
    model.eval()
    print(f"Model loaded (FP16)")

    # Load data
    train_df, test_df = load_data()

    # Build features
    X_train, y_train, X_test = build_features(train_df, test_df, model, tokenizer, device)
    print(f"\nFeature matrix: train={X_train.shape}, test={X_test.shape}")

    # Train and predict
    val_rmse = train_and_validate(X_train, y_train, X_test, test_df["id"].values)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Validation RMSE: {val_rmse:.4f}")


if __name__ == "__main__":
    main()
