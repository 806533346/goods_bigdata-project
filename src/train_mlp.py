"""MLP regression head on RoBERTa embeddings + Spark statistical features."""
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_CSV, TEST_CSV, TRAIN_EMB_NPY, TEST_EMB_NPY,
    USER_STATS_PARQUET, PROD_STATS_PARQUET, PARENT_STATS_PARQUET, GLOBAL_AVG_NPY,
    OUTPUT_DIR, METRICS_JSON, SUBMISSION_CSV, RANDOM_SEED,
)


# ── MLP Architecture ─────────────────────────────────────────────────────
class MLPRegressor(nn.Module):
    """768-dim embedding + 8-dim stats → 2048 → 1024 → 512 → 1"""

    def __init__(self, input_dim=776):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_spark_features(train_df, test_df):
    """Load Spark-computed LOO statistical features."""
    import pyarrow.parquet as pq

    global_avg = float(np.load(GLOBAL_AVG_NPY))

    user_stats = pq.read_table(USER_STATS_PARQUET).to_pandas()
    prod_stats = pq.read_table(PROD_STATS_PARQUET).to_pandas()
    parent_stats = pq.read_table(PARENT_STATS_PARQUET).to_pandas()

    # Merge with train/test
    for df_name, df in [("train", train_df), ("test", test_df)]:
        df = df.merge(user_stats, on="user_id", how="left")
        df = df.merge(prod_stats, on="prod_id", how="left")
        df = df.merge(parent_stats, on="parent_prod_id", how="left")

        # Fill missing with global average
        df["user_id_loo_avg"] = df["user_id_loo_avg"].fillna(global_avg)
        df["prod_id_loo_avg"] = df["prod_id_loo_avg"].fillna(global_avg)
        df["parent_prod_id_loo_avg"] = df["parent_prod_id_loo_avg"].fillna(global_avg)
        df["user_id_log_count"] = df["user_id_log_count"].fillna(0)
        df["prod_id_log_count"] = df["prod_id_log_count"].fillna(0)
        df["parent_prod_id_log_count"] = df["parent_prod_id_log_count"].fillna(0)

        if df_name == "train":
            train_df = df
        else:
            test_df = df

    return train_df, test_df


def prepare_features(train_df, test_df, train_emb, test_emb):
    """Combine embeddings + statistical features."""
    # 8 statistical features
    stat_cols = [
        "user_id_loo_avg", "user_id_log_count",
        "prod_id_loo_avg", "prod_id_log_count",
        "parent_prod_id_loo_avg", "parent_prod_id_log_count",
        "purchased", "votes",
    ]

    train_stats = train_df[stat_cols].values.astype(np.float32)
    test_stats = test_df[stat_cols].values.astype(np.float32)

    # Normalize votes
    train_stats[:, 7] = np.log1p(np.clip(train_stats[:, 7], 0, None))
    test_stats[:, 7] = np.log1p(np.clip(test_stats[:, 7], 0, None))

    # Scale stat features
    scaler = StandardScaler()
    train_stats[:, [1, 3, 5, 6, 7]] = scaler.fit_transform(train_stats[:, [1, 3, 5, 6, 7]])
    test_stats[:, [1, 3, 5, 6, 7]] = scaler.transform(test_stats[:, [1, 3, 5, 6, 7]])

    # Concatenate: 768 (embedding) + 8 (stats) = 776
    X_train = np.concatenate([train_emb, train_stats], axis=1).astype(np.float32)
    X_test = np.concatenate([test_emb, test_stats], axis=1).astype(np.float32)

    y_train = train_df["rating"].values.astype(np.float32)

    return X_train, y_train, X_test


def train_mlp(X_train, y_train, X_val, y_val, input_dim=776):
    """Train MLP with early stopping."""
    torch.manual_seed(RANDOM_SEED)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    EARLY_STOP_PATIENCE = 20

    for epoch in range(100):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = []
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                val_preds.append(preds)
            val_preds = np.concatenate(val_preds)
            val_preds = np.clip(val_preds, 1.0, 5.0)
            val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Val RMSE: {val_rmse:.4f} | Best: {best_val_rmse:.4f}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_rmse


def run_train_mlp():
    """Main MLP training entry point."""
    start = time.time()
    print("=" * 60)
    print("MLP Training on RoBERTa Embeddings + Spark Features")
    print("=" * 60)

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Load Spark features
    print("Loading Spark features...")
    train_df, test_df = load_spark_features(train_df, test_df)

    # Load embeddings
    print("Loading embeddings...")
    train_emb = np.load(TRAIN_EMB_NPY)
    test_emb = np.load(TEST_EMB_NPY)
    print(f"  Train emb: {train_emb.shape}, Test emb: {test_emb.shape}")

    # Prepare features
    print("Preparing features...")
    X_train, y_train, X_test = prepare_features(train_df, test_df, train_emb, test_emb)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Train/val split
    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(X_train))
    val_size = 50000
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Train MLP
    print(f"\nTraining MLP ({X_tr.shape[1]} dim input)...")
    model, best_val_rmse = train_mlp(X_tr, y_tr, X_val, y_val, input_dim=X_train.shape[1])
    print(f"Best Val RMSE: {best_val_rmse:.4f}")

    # Predict test
    print("\nPredicting test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    test_pred = np.clip(test_pred, 1.0, 5.0)

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"].values, "rating": test_pred})
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission saved to {SUBMISSION_CSV}")

    # Save metrics
    elapsed = time.time() - start
    metrics = {
        "best_val_rmse": float(best_val_rmse),
        "total_time_seconds": elapsed,
        "prediction_stats": {
            "min": float(test_pred.min()),
            "max": float(test_pred.max()),
            "mean": float(test_pred.mean()),
        },
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s")
    return best_val_rmse


if __name__ == "__main__":
    run_train_mlp()
