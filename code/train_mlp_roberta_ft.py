"""Train MLP on fine-tuned RoBERTa-base embeddings + statistical features."""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "/home/nmxc/project_code/big_data_trea/data"
EMBED_PATH = os.path.join(DATA_PATH, "embeddings")
OUTPUT_PATH = os.path.join(DATA_PATH, "submission_roberta_mlp.csv")

HIDDEN1 = 2048
HIDDEN2 = 1024
HIDDEN3 = 512
DROPOUT = 0.1
BATCH_SIZE = 1024
EPOCHS = 200
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 30


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.BatchNorm1d(HIDDEN3),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN3, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx]


def evaluate(model, X, y, device, batch_size=2048):
    model.eval()
    total_se = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
            yb = torch.from_numpy(y[i:i+batch_size]).to(device)
            pred = torch.clamp(model(xb), 1.0, 5.0)
            total_se += ((pred - yb) ** 2).sum().item()
            n += len(yb)
    return np.sqrt(total_se / n)


def compute_loo_stats(df, col, rating_col="rating"):
    group_sum = df.groupby(col)[rating_col].transform("sum")
    group_count = df.groupby(col)[rating_col].transform("count")
    loo_sum = group_sum - df[rating_col]
    loo_count = group_count - 1
    loo_avg = loo_sum / loo_count.where(loo_count > 0, np.nan)
    log_count = np.log1p(group_count)
    return loo_avg.values, log_count.values


def compute_test_stats(test_df, train_df, col, global_avg):
    stats = train_df.groupby(col)["rating"].agg(["mean", "count"])
    mean_dict = stats["mean"].to_dict()
    count_dict = stats["count"].to_dict()
    avg = test_df[col].map(mean_dict).fillna(global_avg).values.astype(np.float16)
    cnt = np.log1p(test_df[col].map(count_dict).fillna(0).values).astype(np.float16)
    return avg, cnt


def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load cached fine-tuned RoBERTa embeddings
    print("Loading fine-tuned RoBERTa-base embeddings (float16)...")
    X_train_emb = np.load(os.path.join(EMBED_PATH, "train_roberta_ft.npy")).astype(np.float16)
    X_test_emb = np.load(os.path.join(EMBED_PATH, "test_roberta_ft.npy")).astype(np.float16)
    print(f"Train emb: {X_train_emb.shape}, Test emb: {X_test_emb.shape}")

    train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"), usecols=["rating"])
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"), usecols=["id"])

    y_train = train_df["rating"].values.astype(np.float32)

    np.random.seed(42)
    n = len(train_df)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    # Only embeddings, no statistical features
    X_tr = X_train_emb[train_idx].copy()
    X_val = X_train_emb[val_idx].copy()
    X_test = X_test_emb.copy()

    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]

    # Free memory
    del train_df, X_train_emb, X_test_emb
    import gc; gc.collect()

    print(f"Feature matrix: train={X_tr.shape}({X_tr.dtype}), val={X_val.shape}, test={X_test.shape}")

    train_ds = NumpyDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    input_dim = X_tr.shape[1]
    model = MLPRegressor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training MLP on fine-tuned RoBERTa embeddings ({EPOCHS} epochs, early stop={EARLY_STOP_PATIENCE}) ===")
    print(f"Architecture: {input_dim} -> {HIDDEN1} -> {HIDDEN2} -> {HIDDEN3} -> 1")
    print(f"Features: {input_dim} RoBERTa-ft embeddings only (no statistical features)")
    print(f"MLP params: {total_params/1e6:.2f}M")

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_rmse = evaluate(model, X_val, y_val, device)
        train_rmse = evaluate(model, X_tr, y_tr, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        print(f"  Epoch {epoch+1:>3}/{EPOCHS} | Train: {train_rmse:.4f} | Val: {val_rmse:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}{marker}")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")

    # Predict test
    print("\n=== Predicting test with best validation model ===")
    model.load_state_dict(best_state)
    model.eval()

    del X_tr, y_tr, X_val, y_val, train_ds, train_loader, best_state
    gc.collect()

    test_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 2048):
            xb = torch.from_numpy(X_test[i:i+2048].astype(np.float32)).to(device)
            pred = torch.clamp(model(xb), 1.0, 5.0)
            test_preds.append(pred.cpu().numpy())
    test_pred = np.concatenate(test_preds)

    submission = pd.DataFrame({"id": test_df["id"].values, "rating": test_pred})
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")
    print(f"Prediction stats: min={test_pred.min():.2f}, max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")


if __name__ == "__main__":
    main()
