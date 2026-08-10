import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
EMBED_PATH = os.path.join(DATA_PATH, "embeddings")
OUTPUT_PATH = os.path.join(DATA_PATH, "submission_bert.csv")


def main():
    start = time.time()

    # Load cached embeddings
    print("Loading cached BERT embeddings...")
    X_train_emb = np.load(os.path.join(EMBED_PATH, "train_emb.npy"))
    X_test_emb = np.load(os.path.join(EMBED_PATH, "test_emb.npy"))
    print(f"Train embeddings: {X_train_emb.shape}, Test embeddings: {X_test_emb.shape}")

    # Load labels and IDs (lightweight)
    print("Loading labels and IDs...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), usecols=["rating", "purchased", "votes"])
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), usecols=["id", "purchased", "votes"])

    y_train = train_df["rating"].values.astype(np.float32)

    # Add raw numeric features (NOT statistical features)
    train_numeric = np.column_stack([
        (train_df["purchased"] == True).astype(np.float32).values,
        train_df["votes"].fillna(0).astype(np.float32).values,
    ])
    test_numeric = np.column_stack([
        (test_df["purchased"] == True).astype(np.float32).values,
        test_df["votes"].fillna(0).astype(np.float32).values,
    ])

    # Free pandas memory
    del train_df
    import gc; gc.collect()

    X_train = np.hstack([X_train_emb, train_numeric])
    X_test = np.hstack([X_test_emb, test_numeric])
    del X_train_emb, X_test_emb, train_numeric, test_numeric
    gc.collect()

    print(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

    # Validation split (index-based, no copy)
    print("\n=== Validation (80/20 split) ===")
    np.random.seed(42)
    n = len(X_train)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    best_rmse = float("inf")
    best_alpha = 1.0
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        reg = Ridge(alpha=alpha)
        reg.fit(X_train[train_idx], y_train[train_idx])
        val_pred = np.clip(reg.predict(X_train[val_idx]), 1, 5)
        rmse = np.sqrt(mean_squared_error(y_train[val_idx], val_pred))
        print(f"  alpha={alpha:>5}, RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha}, Validation RMSE: {best_rmse:.4f}")

    # Train final model on ALL data
    print("\n=== Training final model on ALL data ===")
    final_reg = Ridge(alpha=best_alpha)
    final_reg.fit(X_train, y_train)

    train_pred = np.clip(final_reg.predict(X_train), 1, 5)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"Train RMSE: {train_rmse:.4f}")

    # Predict
    test_pred = np.clip(final_reg.predict(X_test), 1, 5)
    submission = pd.DataFrame({"id": test_df["id"].values, "rating": test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")
    print(f"Prediction stats: min={test_pred.min():.2f}, max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")

    print(f"\nTotal time: {time.time() - start:.1f}s")
    print(f"Validation RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()
