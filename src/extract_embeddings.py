"""
从微调好的 RoBERTa 模型中提取 768 维文本嵌入。

用途: 将文本转为固定大小的向量，供下游 MLP 模型使用。
     EmbeddingExtractor 只取 RoBERTa 的编码器部分，去掉了回归头，
     输出的是纯粹的语义向量 (768 维)，而非评分预测。

相比端到端微调 (finetune_roberta_ddp.py):
  端到端: 文本 → RoBERTa → 评分 (一步到位, 效果最好)
  嵌入+MLP: 文本 → RoBERTa → 768维嵌入 → MLP → 评分 (可选路径)

支持从本地或云端加载 checkpoint。
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, NUM_WORKERS,
    DATA_DIR, TRAIN_CSV, TEST_CSV, CHECKPOINT_PT,
    TRAIN_EMB_NPY, TEST_EMB_NPY,
)
from finetune_roberta_ddp import mean_pooling   # 复用相同的均值池化函数
from cloud_io import download_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_embeddings")


class TextDataset(Dataset):
    """
    文本数据集，不含评分标签。

    和训练用的 ReviewDataset 不同之处:
      ReviewDataset 返回 (input_ids, attention_mask, rating)
      TextDataset   只返回 (input_ids, attention_mask)
    """

    def __init__(self, texts: list, tokenizer, max_length: int = MAX_LENGTH):
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


class EmbeddingExtractor(torch.nn.Module):
    """
    RoBERTa 编码器 (不含回归头)。

    只做: 文本 → 12层 Transformer → mean pooling → 768维向量
    不做: 评分预测 (Dropout + Linear head 被去掉)

    加载 checkpoint 时用 strict=False:
      只加载 backbone 的权重，跳过头部和 dropout 层
    """

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # 取所有 token 的最后一层隐状态，用 attention_mask 加权平均
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return mean_pooling(outputs.last_hidden_state, attention_mask)


def extract_embeddings(
    dataset: Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
    desc: str = "Extracting",
) -> np.ndarray:
    """
    批量提取文本嵌入向量。

    返回 float16 格式，节省磁盘空间:
      float32: 3M × 768 × 4 bytes = 9.2 GB
      float16: 3M × 768 × 2 bytes = 4.6 GB (省一半)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    model.eval()
    all_embeddings = []

    with torch.no_grad():  # 不计算梯度，加速推理
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            # BF16 推理，和训练时保持一致
            with autocast("cuda", dtype=torch.bfloat16):
                emb = model(input_ids, attn_mask)

            # BF16 → float32 → numpy，因为 numpy 不支持 BF16
            all_embeddings.append(emb.float().cpu().numpy())

            if (i + 1) % 500 == 0:
                logger.info(f"  {desc}: {i+1}/{len(loader)} batches")

    # 最终存为 float16，省磁盘空间
    result = np.concatenate(all_embeddings, axis=0).astype(np.float16)
    logger.info(f"  {desc} complete: shape={result.shape}, dtype={result.dtype}")
    return result


def run_extract_embeddings(checkpoint_path: Optional[str] = None):
    """
    加载微调好的模型，提取训练集和测试集的嵌入。

    流程:
      1. 从本地/云端加载 checkpoint
      2. 去掉回归头，只保留 RoBERTa backbone
      3. 对 3M 训练数据和 10K 测试数据提取 768 维向量
      4. 保存为 .npy 文件供 MLP 训练使用
    """
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 加载 checkpoint (优先本地，找不到则从云端下载)
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PT
    if not os.path.exists(checkpoint_path):
        logger.info("Local checkpoint not found, trying cloud...")
        download_checkpoint(version="best")
        checkpoint_path = CHECKPOINT_PT

    # 加载数据
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # 拼接标题和评论 (和训练时完全一致)
    train_texts = (df_train["title"].fillna("") + " [SEP] " + df_train["comment"].fillna("")).tolist()
    test_texts  = (df_test["title"].fillna("") + " [SEP] " + df_test["comment"].fillna("")).tolist()

    train_ds = TextDataset(train_texts, tokenizer)
    test_ds  = TextDataset(test_texts, tokenizer)

    # 加载模型 — 只取 backbone 权重
    model = EmbeddingExtractor(MODEL_NAME)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # 过滤: 只保留 backbone. 开头的参数, 去掉 head. 和 dropout.
    state = {k: v for k, v in state.items() if k.startswith("backbone")}
    model.load_state_dict(state, strict=False)  # strict=False 允许部分加载
    model = model.to(device)

    # 提取嵌入
    t0 = time.time()
    train_emb = extract_embeddings(train_ds, model, device, batch_size=128, desc="Train")
    test_emb  = extract_embeddings(test_ds, model, device, batch_size=128, desc="Test")

    np.save(TRAIN_EMB_NPY, train_emb)
    np.save(TEST_EMB_NPY, test_emb)

    elapsed = time.time() - t0
    logger.info(f"Embeddings saved in {elapsed:.0f}s")
    logger.info(f"  Train: {TRAIN_EMB_NPY} ({os.path.getsize(TRAIN_EMB_NPY)/1e9:.1f} GB)")
    logger.info(f"  Test:  {TEST_EMB_NPY} ({os.path.getsize(TEST_EMB_NPY)/1e6:.1f} MB)")
