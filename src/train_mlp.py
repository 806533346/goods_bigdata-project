"""
MLP 训练 — RoBERTa 嵌入 + Spark 统计特征 → 评分预测。

这是端到端微调的替代方案，分为两步:
  第一步 (extract_embeddings.py): RoBERTa → 768 维文本嵌入
  第二步 (本文件):         768维嵌入 + 8维统计特征 → MLP → 评分

架构:
  输入: 776 维 (768 RoBERTa + 8 Spark 统计特征)
  隐藏层: 2048 → 1024 → 512 (每层后跟 BatchNorm + GELU + Dropout)
  输出: 1 维 (评分)

效果不如端到端微调 (RMSE 0.65 vs 0.54)，但训练快 100 倍。
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, OUTPUT_DIR, RANDOM_SEED,
    TRAIN_CSV, TEST_CSV, TRAIN_EMB_NPY, TEST_EMB_NPY,
    SUBMISSION_CSV, METRICS_JSON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_mlp")


class MLPRegressor(nn.Module):
    """
    MLP 回归器 — 776 维输入，4 层全连接，1 维输出。

    BatchNorm: 稳定训练
    GELU: 比 ReLU 更平滑，深层网络效果更好
    Dropout(0.3): 防止过拟合 (每层随机丢 30% 神经元)
    """

    def __init__(self, input_dim: int = 776, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2048),   # 776 → 2048
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),         # 2048 → 1024
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),          # 1024 → 512
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),             # 512 → 1 (评分)
        )

    def forward(self, x):
        return self.layers(x)


def load_spark_features() -> pd.DataFrame:
    """
    加载 Spark 生成的 Parquet 统计特征。

    三个统计表:
      user_stats     → 用户级 LOO 平均分 + 评论数对数
      prod_stats     → 产品级 LOO 平均分 + 评论数对数
      parent_stats   → 父产品级 LOO 平均分 + 评论数对数

    Parquet 用 pyarrow 读取，不依赖 Spark，内存占用极小。
    """
    import pyarrow.parquet as pq

    user_stats   = pq.read_table(os.path.join(DATA_DIR, "user_stats.parquet")).to_pandas()
    prod_stats   = pq.read_table(os.path.join(DATA_DIR, "prod_stats.parquet")).to_pandas()
    parent_stats = pq.read_table(os.path.join(DATA_DIR, "parent_stats.parquet")).to_pandas()

    return user_stats.set_index("user_id"), prod_stats.set_index("prod_id"), parent_stats.set_index("parent_prod_id")


def prepare_features(df: pd.DataFrame, embeddings: np.ndarray,
                     user_stats, prod_stats, parent_stats) -> np.ndarray:
    """
    合并 RoBERTa 嵌入和 Spark 统计特征，生成 776 维输入。

    特征组成:
      768 维 → RoBERTa 文本嵌入 (语义信息)
      + 8 维 → Spark 统计特征 (用户/产品/父产品的 LOO 均值 + 对数评论数 + purchased + votes)

    特征缩放:
      LOO 均值 (索引 0,2,4) → 不缩放 (本身在 1-5 之间)
      对数统计 (索引 1,3,5) → StandardScaler 标准化
      purchased, votes (索引 6,7) → StandardScaler 标准化
    """
    global_avg = np.load(os.path.join(DATA_DIR, "global_avg.npy")).item()

    features = []

    # 3 LOO averages (indices 0, 2, 4)
    for col, stats_df, key in [
        ("user_id", user_stats, "user_id_loo_avg"),
        ("prod_id", prod_stats, "prod_id_loo_avg"),
        ("parent_prod_id", parent_stats, "parent_prod_id_loo_avg"),
    ]:
        mapped = df[col].map(stats_df[key]) if key in stats_df.columns else None
        if mapped is None:
            mapped = pd.Series([global_avg] * len(df))
        features.append(mapped.fillna(global_avg).values.reshape(-1, 1))

    # 3 log counts (indices 1, 3, 5)
    for col, stats_df, key in [
        ("user_id", user_stats, "user_id_log_count"),
        ("prod_id", prod_stats, "prod_id_log_count"),
        ("parent_prod_id", parent_stats, "parent_prod_id_log_count"),
    ]:
        mapped = df[col].map(stats_df[key]) if key in stats_df.columns else None
        if mapped is None:
            mapped = pd.Series([0.0] * len(df))
        features.append(mapped.fillna(0.0).values.reshape(-1, 1))

    # purchased, votes
    features.append(df["purchased"].fillna(0).values.reshape(-1, 1))
    votes = np.log1p(np.clip(df["votes"].fillna(0).values, 0, None))
    features.append(votes.reshape(-1, 1))

    stats = np.concatenate(features, axis=1).astype(np.float32)  # (N, 8)

    # Scale log-transformed columns (indices 1, 3, 5, 6, 7)
    scaler = StandardScaler()
    stats[:, [1, 3, 5, 6, 7]] = scaler.fit_transform(stats[:, [1, 3, 5, 6, 7]])

    # Concatenate embeddings + stats
    return np.concatenate([embeddings.astype(np.float32), stats], axis=1)


def run_train_mlp():
    """
    MLP 完整训练流程。

    相比 RoBERTa 端到端微调的优势:
      1. 训练极快: 几百万参数的 MLP vs 1.25 亿参数的 RoBERTa
      2. 冻结嵌入: RoBERTa 权重不变，只训练 MLP
      3. 适用场景: 快速实验、超参搜索

    劣势:
      文本理解能力受损: 不能端到端优化语义表示
      → Kaggle RMSE 约 0.65 (端到端为 0.54)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 加载数据: CSV + 预提取的嵌入 + Spark 统计特征
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    train_emb = np.load(TRAIN_EMB_NPY).astype(np.float32)
    test_emb  = np.load(TEST_EMB_NPY).astype(np.float32)

    # 加载 Spark 生成的统计特征
    user_stats, prod_stats, parent_stats = load_spark_features()

    # 合并所有特征: 768 嵌入 + 8 统计 = 776 维
    X_train = prepare_features(df_train, train_emb, user_stats, prod_stats, parent_stats)
    y_train = df_train["rating"].values.astype(np.float32).reshape(-1, 1)
    X_test  = prepare_features(df_test, test_emb, user_stats, prod_stats, parent_stats)

    # 预留 5 万条件验证集
    n_val = 50000
    indices = np.random.RandomState(RANDOM_SEED).permutation(len(X_train))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    X_tr, X_va = X_train[train_idx], X_train[val_idx]
    y_tr, y_va = y_train[train_idx], y_train[val_idx]

    # DataLoader: 训练 batch=512，验证 batch=1024
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=512, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=1024, shuffle=False, pin_memory=True,
    )

    # 模型: 4 层 MLP
    model = MLPRegressor(input_dim=X_train.shape[1]).to(device)
    # AdamW: 优化的权重衰减实现
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # ReduceLROnPlateau: 验证不改善时 LR 减半，耐心 5 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    patience_counter = 0
    max_patience = 20

    logger.info(f"Training MLP: {len(train_idx):,} train, {len(val_idx):,} val")

    for epoch in range(1, 201):  # max 200 epochs
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)

        # Validation
        model.eval()
        val_se = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b).clamp(1, 5)
                val_se += ((preds - y_b) ** 2).sum().item()

        train_rmse = (train_loss / len(train_idx)) ** 0.5
        val_rmse   = (val_se / len(val_idx)) ** 0.5

        scheduler.step(val_rmse)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "mlp_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Predict
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "mlp_best.pt"), weights_only=True))
    model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test)),
        batch_size=1024, shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for (X_b,) in test_loader:
            preds.append(model(X_b.to(device)).clamp(1, 5).cpu().numpy())
    predictions = np.concatenate(preds).flatten()

    # Submission
    submission = pd.DataFrame({
        "id": range(len(predictions)),
        "rating": predictions,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission.to_csv(SUBMISSION_CSV, index=False)
    logger.info(f"Submission saved: {SUBMISSION_CSV}")

    return best_val_rmse
