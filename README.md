# Kaggle COMP5434 Big Data Project: Review Rating Prediction

预测电商平台评论评分（1-5分）的大数据分析项目。使用 **Apache Spark** 进行分布式数据预处理与特征工程，结合 **RoBERTa-base** 深度学习模型进行端到端微调，实现高精度的评分预测。

## 目录

- [项目概述](#项目概述)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [数据处理](#数据处理)
- [模型训练](#模型训练)
- [实验结果](#实验结果)
- [可复现性说明](#可复现性说明)
- [Kaggle提交](#kaggle提交)

---

## 项目概述

### 任务定义

本项目旨在根据用户评论的文本内容（标题+正文）和元数据（用户ID、产品ID、购买状态等），预测评论的评分（1-5分）。这是一个**回归问题**，评价指标为**均方根误差（RMSE）**。

### 核心挑战

1. **大数据处理**：训练集包含 300 万+ 条评论记录，需要使用分布式计算框架
2. **文本理解**：评论文本长度不一，需要捕捉语义、情感等深层信息
3. **特征工程**：需防止数据泄露（Leave-One-Out编码）
4. **计算效率**：在有限的 GPU 显存（8GB）下训练 1.25 亿参数的模型

### 解决方案

采用 **PySpark + PyTorch** 的混合架构：
- **PySpark**：分布式读取 CSV、计算 LOO 统计特征、聚合用户/产品级统计
- **RoBERTa-base**：端到端微调，将文本转为 768 维语义嵌入并预测评分
- **BF16 混合精度**：加速训练并节省显存

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据层 (300万条记录)                       │
│  train.csv (3,007,439)  +  test.csv (10,000)  +  prodInfo.csv   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: PySpark 分布式特征工程                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • 分布式 CSV 读取 (32 分区)                              │    │
│  │ • Window 函数计算 LOO 统计特征                           │    │
│  │ • groupBy 聚合: user_id / prod_id / parent_prod_id      │    │
│  │ • 输出: Parquet 格式统计表                               │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 2: RoBERTa-base 端到端微调                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  文本: title + [SEP] + comment                          │    │
│  │    ↓ Tokenizer (max_length=192)                         │    │
│  │  RoBERTa-base (12层 Transformer, 125M 参数)             │    │
│  │    ↓ Mean Pooling (768维)                               │    │
│  │  Dropout(0.1) → Linear(768→1)                           │    │
│  │    ↓ MSELoss + 反向传播 (BF16 混合精度)                  │    │
│  │  输出: 模型权重 + 预测评分                                │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 3: 生成提交文件                                 │
│  submission.csv (id, rating) — 提交至 Kaggle                     │
└─────────────────────────────────────────────────────────────────┘
```

### 大数据框架使用说明

本项目在以下环节使用了 **Apache Spark (PySpark)** 大数据并行/分布式处理框架：

| 环节 | Spark 操作 | 说明 |
|------|-----------|------|
| 数据读取 | `spark.read.csv().repartition(32)` | 将 300 万条数据分布到 32 个分区并行读取 |
| 统计聚合 | `df.groupBy().agg()` | 分布式计算用户/产品/父产品的评分统计 |
| LOO 编码 | `Window.partitionBy()` | 使用窗口函数计算留一法均值，防止数据泄露 |
| 特征输出 | `df.write.parquet()` | 以列式存储格式保存，便于后续高效读取 |

---

## 项目结构

```text
big_data_trea/
├── data/                          # 数据目录
│   ├── train.csv                  # 训练集 (3,007,439 条)
│   ├── test.csv                   # 测试集 (10,000 条)
│   ├── prodInfo.csv               # 产品元数据 (259,791 条)
│   ├── user_stats.parquet         # Spark输出的用户统计特征
│   ├── prod_stats.parquet         # Spark输出的产品统计特征
│   ├── parent_stats.parquet       # Spark输出的父产品统计特征
│   ├── global_avg.npy             # 全局平均评分
│   ├── roberta_base_finetuned.pt  # 微调后的模型权重
│   └── roberta_base_train_log.json # 训练日志
│
├── output/                        # 输出目录
│   ├── submission.csv             # Kaggle提交文件
│   └── metrics.json               # 训练指标和硬件信息
│
├── scripts/                       # 数据脚本
│   ├── download_data.sh           # 从Kaggle下载数据
│   └── import_data.sh             # 从本地zip导入数据
│
├── src/                           # 源代码
│   ├── config.py                  # 集中配置（路径、超参数）
│   ├── hardware.py                # 硬件信息收集
│   ├── spark_features.py          # PySpark分布式特征工程
│   ├── finetune_roberta.py        # RoBERTa-base端到端微调
│   ├── extract_embeddings.py      # 嵌入提取（可选）
│   ├── train_mlp.py               # MLP训练（可选替代方案）
│   └── pipeline.py                # 主流程编排器
│
├── code/                          # 早期实验代码（备份）
├── requirements.txt               # Python依赖
├── run.sh                         # 一键运行脚本
├── report.md                      # 英文项目报告
├── report_cn.md                   # 中文项目报告
└── README.md                      # 本文档
```

---

## 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | 8核 | 24核+ |
| 内存 | 16 GB | 32 GB+ |
| GPU | 8 GB VRAM | 16 GB+ VRAM |
| 存储 | 20 GB | 50 GB+ SSD |

### 软件要求

| 软件 | 版本 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| Python | 3.10+ |
| Java | 8+ (JDK 21 兼容) |
| PyTorch | 2.6.0+ (CUDA 12.4) |
| PySpark | 4.0.0+ |
| Transformers | 4.40.0+ |

### Python 依赖

详见 [requirements.txt](requirements.txt)：

```text
pyspark>=4.0.0       # 大数据分布式处理框架
torch>=2.6.0          # 深度学习框架
transformers>=4.40.0  # 预训练模型库
pandas>=2.0.0         # 数据处理
numpy>=1.24.0         # 数值计算
scikit-learn>=1.3.0   # 机器学习工具
pyarrow>=14.0.0       # Parquet文件支持
py-cpuinfo>=9.0.0     # CPU信息收集
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd /path/to/big_data_trea

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

**方式 A：从 Kaggle 下载（推荐）**

```bash
# 先配置 Kaggle API Token
# 将 kaggle.json 放到 ~/.kaggle/
bash scripts/download_data.sh
```

**方式 B：从本地导入**

```bash
# 将 comp-5434-2526-sem-3-project.zip 下载到 ~/Downloads/
bash scripts/import_data.sh
```

### 3. 运行完整 Pipeline

```bash
# 一键运行（推荐）
bash run.sh
```

或分步运行：

```bash
export PYTHONPATH=src

# Step 1: Spark 分布式特征工程 (~5分钟)
python3 src/spark_features.py

# Step 2: RoBERTa-base 端到端微调 (~6.5小时)
python3 src/finetune_roberta.py

# Step 3: 嵌入提取（可选，用于MLP方案）
python3 src/extract_embeddings.py

# Step 4: MLP 训练（可选替代方案）
python3 src/train_mlp.py
```

### 4. 通过 Pipeline 编排器运行

```bash
export PYTHONPATH=src

# 完整运行
python3 src/pipeline.py --data-dir data

# 跳过某些步骤
python3 src/pipeline.py --skip-spark --skip-mlp
```

---

## 数据处理

### 数据集概览

| 数据集 | 记录数 | 主要字段 |
|--------|--------|---------|
| train.csv | 3,007,439 | id, title, comment, user_id, prod_id, rating |
| test.csv | 10,000 | id, title, comment, user_id, prod_id |
| prodInfo.csv | 259,791 | parent_prod_id, main_category, price |

### 评分分布

| 评分 | 占比 | 说明 |
|------|------|------|
| 1分 | ~8% | 较少 |
| 2分 | ~4% | 较少 |
| 3分 | ~7% | 中等 |
| 4分 | ~20% | 较多 |
| 5分 | ~61% | 占主导 |

评分分布严重偏向5星，全局平均评分约 3.94。

### Spark 特征工程

使用 PySpark 的 **Window 函数** 计算留一法（LOO）统计特征，防止数据泄露：

```python
# 用户级LOO特征：排除当前评论后的平均评分
w_user = Window.partitionBy("user_id")
user_stats = train_df.groupBy("user_id").agg(
    F.sum("rating").alias("sum_rating"),
    F.count("rating").alias("count_rating"),
).withColumn(
    "loo_avg",
    F.when(F.col("count_rating") > 1,
           (F.col("sum_rating") - F.col("rating")) / (F.col("count_rating") - 1))
    .otherwise(F.lit(global_avg))
)
```

**输出的统计特征**：

| 特征 | 说明 |
|------|------|
| user_id_loo_avg | 用户平均评分（排除当前评论） |
| user_id_log_count | 用户评论数的对数 |
| prod_id_loo_avg | 产品平均评分（排除当前评论） |
| prod_id_log_count | 产品评论数的对数 |
| parent_prod_id_loo_avg | 父产品平均评分（排除当前评论） |
| parent_prod_id_log_count | 父产品评论数的对数 |

---

## 模型训练

### RoBERTa-base 端到端微调（主模型）

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | roberta-base | 125M 参数 |
| 最大序列长度 | 192 | 截断长文本 |
| 批大小 | 32 | 受8GB显存限制 |
| 训练轮数 | 1 | 全量数据 |
| 学习率 | 2e-5 | AdamW 优化器 |
| 权重衰减 | 0.01 | |
| 预热比例 | 0.1 | 线性预热调度 |
| 精度 | BF16 | 混合精度训练 |
| 梯度裁剪 | 1.0 | 防止梯度爆炸 |
| 训练数据 | 2,957,439 条 | 全量（留5万做验证） |

### 模型架构

```python
class RatingRegressor(nn.Module):
    def __init__(self, model_name="roberta-base"):
        self.backbone = AutoModel.from_pretrained(model_name)  # 12层Transformer
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, 1)  # 回归头

    def forward(self, input_ids, attention_mask):
        # 1. RoBERTa 编码
        outputs = self.backbone(input_ids, attention_mask)
        # 2. 均值池化（按 attention_mask 加权平均）
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        # 3. 回归预测
        return self.head(self.dropout(pooled))
```

### 训练流程

1. **数据加载**：PySpark 读取 CSV → 随机划分训练/验证集
2. **文本处理**：`title + [SEP] + comment` → Tokenizer → input_ids
3. **前向传播**：RoBERTa 编码 → 均值池化 → 回归头 → 预测评分
4. **损失计算**：MSE Loss（预测评分 vs 真实评分）
5. **反向传播**：BF16 混合精度 → 梯度裁剪 → AdamW 更新
6. **验证评估**：每个 epoch 后在验证集上计算 RMSE
7. **检查点保存**：验证 RMSE 提升时保存模型权重

---

## 实验结果

### 成绩进展

| 方案 | 训练数据 | Kaggle RMSE | 说明 |
|------|---------|-------------|------|
| GBT + 统计特征（数据泄露） | 全量 | 1.556 | 初始方案 |
| DistilBERT + 岭回归 | 50万 | 0.860 | |
| DistilBERT + MLP + 统计特征 | 50万 | 0.732 | |
| DeBERTa-v3 + MLP + 统计特征 | 50万 | 0.670 | |
| RoBERTa-base 端到端微调 | 50万 | 0.541 | 权重未保存 |
| RoBERTa-base 端到端微调（重训） | 50万 | 0.618 | 权重已保存 |
| **RoBERTa-base 端到端微调** | **全量** | **待测** | **最终方案** |

### 消融研究

| 实验 | Val RMSE | Kaggle RMSE | 关键发现 |
|------|---------|-------------|---------|
| RoBERTa-base 端到端（完整） | 1.14 | 0.541 | 整体最佳 |
| RoBERTa-base + MLP（含统计特征） | 1.114 | 0.648 | MLP不如端到端 |
| RoBERTa-base + MLP（无统计特征） | 1.115 | 0.651 | 统计特征略有帮助 |
| DeBERTa-v3 + MLP | 1.125 | 0.670 | DeBERTa在BF16下不稳定 |

### 关键发现

1. **端到端微调优于固定嵌入+MLP**：0.541 vs 0.648
2. **BF16 稳定性至关重要**：RoBERTa 的绝对位置编码在 BF16 下稳定，DeBERTa 的相对位置编码会产生 NaN
3. **全量数据训练**：使用全部 300 万条数据训练，充分体现大数据价值
4. **Spark 高效处理大数据**：PySpark 分布式处理 300 万条记录用于特征工程

---

## 可复现性说明

### 随机种子

- `np.random.seed(42)`：数据划分
- `torch.manual_seed(42)`：模型初始化

### CUDA 非确定性

由于 CUDA 的某些操作（如 cuDNN 的 atomicAdd）具有非确定性，相同代码和种子在不同运行中可能产生略有不同的结果。这是深度学习的常见现象。

如需完全确定性，可添加：

```python
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
```

但这会增加训练时间约 20-30%。

### 硬件信息

训练硬件信息会自动保存到 `output/metrics.json`：

```json
{
  "hardware_info": {
    "cpu": {"cpu_model": "13th Gen Intel Core i9-13900HX", "cpu_cores": 24},
    "gpu": {"gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU", "total_vram_gb": 8.19}
  }
}
```

---

## Kaggle 提交

### 提交文件格式

`output/submission.csv`：

```csv
id,rating
0,4.32
1,1.85
2,5.00
...
```

### 提交方式

1. **通过 Kaggle CLI**：

```bash
kaggle competitions submit -c comp-5434-2526-sem-3-project \
    -f output/submission.csv \
    -m "RoBERTa-base fine-tuned on full data"
```

2. **通过网页提交**：

访问 [Kaggle 竞赛页面](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd)，上传 `output/submission.csv`。

---

## 参考文献

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*.
3. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *ICLR*.
4. Zaharia, M., et al. (2010). "Apache Spark: A Unified Engine for Big Data Processing." *Communications of the ACM*.
5. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
6. Hugging Face Transformers. https://huggingface.co/docs/transformers

---

## 许可证

本项目仅用于 COMP5434 课程作业目的。
