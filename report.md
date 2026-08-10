# COMP5434 Big Data Project Report

## Team Information

- **Team Name**: [Your Team Name]
- **Member 1**: [Name] - [Student ID]
- **Member 2**: [Name] - [Student ID]

---

## 1. Problem Definition

This project aims to develop an accurate and computationally efficient solution for predicting review ratings (1-5) using a large-scale e-commerce review dataset. The task is formulated as a **regression problem**, where the goal is to minimize the Root Mean Squared Error (RMSE) between predicted and ground-truth ratings.

The challenge lies in:
1. Processing 3 million+ review records efficiently using big data frameworks
2. Extracting meaningful features from text fields (title, comment)
3. Building a model that captures both linguistic semantics and user/product behavioral patterns
4. Achieving high accuracy while maintaining computational efficiency

---

## 2. Data Analysis

### 2.1 Dataset Overview

| Dataset | Records | Description |
|---------|---------|-------------|
| train.csv | 3,007,439 | Training reviews with ratings |
| test.csv | 10,000 | Test reviews without ratings |
| prodInfo.csv | 259,791 | Product metadata |

### 2.2 Feature Description

| Field | Type | Description |
|-------|------|-------------|
| id | int | Review identifier |
| title | text | Review title |
| comment | text | Review body text |
| user_id | string | User identifier |
| prod_id | string | Product identifier |
| parent_prod_id | string | Parent product (same product, different variants) |
| time | int | Review timestamp |
| votes | int | Helpful votes count |
| purchased | bool | Whether user purchased the product |
| rating | int (1-5) | **Target variable** |

### 2.3 Statistical Analysis (via PySpark)

Using PySpark's distributed processing, we computed key statistics:

- **Global average rating**: 3.94 (skewed towards positive reviews)
- **Unique users**: 1,762,679
- **Unique products**: 259,791
- **Average reviews per user**: 1.7
- **Average reviews per product**: 11.6

**Rating distribution**:
| Rating | Count | Percentage |
|--------|-------|------------|
| 1 | ~8% | Low |
| 2 | ~4% | Low |
| 3 | ~7% | Medium |
| 4 | ~20% | High |
| 5 | ~61% | Dominant |

The distribution is highly skewed towards 5-star ratings, which poses challenges for predicting low ratings accurately.

---

## 3. Solution and Implementation Details

### 3.1 Solution Architecture

Our solution follows the **5-step big data analytical process**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Data Collection & Preprocessing (PySpark)             │
│  ├─ Distributed CSV reading (32 partitions)                    │
│  ├─ Text preprocessing (title + [SEP] + comment)               │
│  └─ Statistical feature computation (Window functions)         │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Feature Engineering (PySpark + PyTorch)               │
│  ├─ LOO statistical features (user/prod/parent means)          │
│  └─ Text embeddings via fine-tuned RoBERTa                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Model Training (PyTorch + GPU)                        │
│  └─ End-to-end fine-tuning of RoBERTa-base                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Model Evaluation                                      │
│  └─ Validation RMSE on 50K held-out set                        │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: Prediction & Submission                               │
│  └─ Generate test predictions with clamping [1.0, 5.0]         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Big Data Framework: Apache Spark

We use **PySpark** (version 4.1.2) as our big data parallel/distributed processing framework for:

1. **Distributed CSV Reading**: Reading 3M+ rows across 32 partitions
2. **Statistical Feature Computation**: Using Spark Window functions for Leave-One-Out (LOO) encoding
3. **Distributed Aggregation**: GroupBy operations for user/product/parent-product statistics

**Key Spark operations**:
```python
# LOO feature computation using Window functions
w_user = Window.partitionBy("user_id")
train_df = train_df.withColumn("user_loo_avg",
    F.when(F.col("user_cnt") > 1,
           (F.col("user_sum") - F.col("rating")) / (F.col("user_cnt") - 1))
    .otherwise(F.lit(float(global_avg))))
```

### 3.3 Deep Learning Model: RoBERTa-base Fine-tuning

Our best-performing model is **RoBERTa-base** (125M parameters), fine-tuned end-to-end for rating prediction.

**Model Architecture**:
```
Input: title + [SEP] + comment
  ↓ Tokenizer (max_length=192)
input_ids + attention_mask
  ↓ RoBERTa-base (12-layer Transformer Encoder, 768-dim)
last_hidden_state [batch, 192, 768]
  ↓ Mean Pooling (masked average)
pooled_embedding [batch, 768]
  ↓ Dropout(0.1)
  ↓ Linear(768 → 1)
Predicted rating [batch]
```

**Why RoBERTa?**
- **Absolute positional encoding**: Numerically stable under BF16 precision (unlike DeBERTa's relative encoding)
- **Robust pretraining**: Dynamic masking + no NSP + large batch training
- **125M parameters**: Sufficient capacity without exceeding 8GB GPU memory

### 3.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | roberta-base |
| Max sequence length | 192 |
| Batch size | 32 |
| Epochs | 2 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Optimizer | AdamW |
| Scheduler | Linear warmup |
| Precision | BF16 (mixed precision) |
| Training subset | 500,000 samples |
| Validation set | 50,000 samples |

### 3.5 Unique Technical Designs

1. **BF16 Mixed Precision**: Used `torch.amp.autocast('cuda', dtype=torch.bfloat16)` to accelerate training and reduce memory usage by ~50%

2. **Mean Pooling**: Instead of using only the [CLS] token, we compute masked mean pooling over all token embeddings, capturing richer semantic information

3. **Spark + PyTorch Pipeline**: Hybrid architecture leveraging Spark for distributed data processing and PyTorch for GPU-accelerated model training

4. **Leave-One-Out Encoding**: Prevents data leakage in statistical features by excluding the current sample when computing user/product averages

---

## 4. Performance Evaluation and Discussion

### 4.1 Hardware Specifications

| Component | Specification |
|-----------|--------------|
| CPU | 13th Gen Intel Core i9-13900HX (24 cores) |
| RAM | 30 GB DDR5 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) |
| Storage | 110 GB NVMe SSD |
| OS | Linux 6.17.0-40-generic |
| Python | 3.13.13 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Spark | 4.1.2 |

### 4.2 Timing Metrics

| Phase | Time |
|-------|------|
| Spark feature engineering | ~5 min |
| RoBERTa-base fine-tuning (per epoch) | ~4,096 s (68 min) |
| Total training time (2 epochs) | 8,225 s (2.28 h) |
| Test inference (10,000 records) | ~30 s |

### 4.3 Effectiveness Results (RMSE)

| Method | Val RMSE | Kaggle RMSE |
|--------|----------|-------------|
| GBT + statistical features (data leakage) | - | 1.556 |
| DistilBERT + Ridge regression | - | 0.860 |
| DistilBERT + MLP + statistical features | 1.203 | 0.732 |
| DeBERTa-v3 + MLP + statistical features | 1.125 | 0.670 |
| RoBERTa-base fine-tuning (end-to-end, rerun) | 1.122 | 0.618 |
| **RoBERTa-base fine-tuning (end-to-end, best)** | **1.14** | **0.541** |

**Final Kaggle RMSE: 0.54081**

### 4.4 Ablation Study

| Experiment | Val RMSE | Kaggle RMSE | Key Finding |
|-----------|----------|-------------|-------------|
| RoBERTa-base end-to-end (full) | 1.14 | **0.541** | Best overall |
| RoBERTa-base + MLP (with stats) | 1.114 | 0.648 | MLP underperforms end-to-end |
| RoBERTa-base + MLP (no stats) | 1.115 | 0.651 | Statistical features help slightly |
| RoBERTa-base + MLP (Dropout=0.1) | 1.113 | 0.654 | Lower dropout causes overfitting |
| DeBERTa-v3 + MLP | 1.125 | 0.670 | BF16 instability with DeBERTa |

### 4.5 Key Findings

1. **End-to-end fine-tuning > Fixed embeddings + MLP**: End-to-end fine-tuning allows joint optimization of embeddings and regression head, achieving 0.541 vs 0.648 for MLP approaches

2. **BF16 stability matters**: DeBERTa-v3 produces NaN under BF16 due to relative positional encoding; RoBERTa's absolute encoding is stable

3. **Val RMSE ≠ Kaggle RMSE**: Validation and test distributions differ; models with better Val RMSE sometimes perform worse on Kaggle (overfitting validation set)

4. **Statistical features have limited value**: When using high-quality text embeddings, statistical features (user/product averages) provide minimal improvement and may introduce noise

5. **Spark enables efficient big data processing**: PySpark efficiently processes 3M+ records for feature engineering, demonstrating the value of distributed computing

---

## 5. Summary of Discoveries and Future Work

### 5.1 Discoveries

- **Pre-trained language models excel at rating prediction**: RoBERTa-base, when fine-tuned end-to-end, captures nuanced sentiment and quality signals from review text
- **Big data frameworks are essential for preprocessing**: PySpark's distributed processing handles millions of records efficiently for feature engineering
- **Model architecture matters more than feature engineering**: End-to-end fine-tuning outperforms hand-crafted features combined with simpler models
- **Precision and stability trade-offs**: BF16 provides 2x speedup but requires careful model selection (RoBERTa stable, DeBERTa unstable)

### 5.2 Future Work

1. **Larger training data**: Currently using 500K of 3M samples; full dataset training could improve accuracy
2. **RoBERTa-large**: With gradient checkpointing and checkpoint saving, the 355M parameter model could be trained on 8GB VRAM
3. **Ensemble methods**: Combining multiple RoBERTa models with different random seeds
4. **Product metadata integration**: Incorporating prodInfo.csv (category, price, features) as additional context
5. **Graph-based methods**: Modeling user-product relationships as graphs for collaborative filtering signals

---

## 6. Contribution Table

| Member | Contribution (%) | Tasks |
|--------|------------------|-------|
| [Member 1] | 50% | Model design, PyTorch implementation, fine-tuning |
| [Member 2] | 50% | PySpark pipeline, feature engineering, evaluation |

*Total: 100%*

---

## 7. References

1. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL-HLT*.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv:1907.11692*.
3. He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *ICLR*.
4. Zaharia, M., et al. (2010). "Apache Spark: A Unified Engine for Big Data Processing." *Communications of the ACM*.
5. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." *ICLR*.
6. Hugging Face Transformers Library: https://huggingface.co/docs/transformers

---

## Appendix: Code Structure

```
code/
├── spark_feature_engineering.py   # PySpark distributed feature engineering
├── finetune_roberta_base.py       # RoBERTa-base end-to-end fine-tuning
├── extract_embeddings_roberta_ft.py  # Embedding extraction
├── train_mlp_roberta_ft.py        # MLP training on embeddings
├── ensemble.py                    # Model ensemble
├── finetune_roberta_base_continue.py  # Continue training from checkpoint
├── finetune_roberta_large.py      # RoBERTa-large (OOM, not used)
└── train_mlp_deberta.py           # DeBERTa + MLP (earlier experiments)
```
