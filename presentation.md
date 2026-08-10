---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
header: 'COMP5434 Big Data Project'
footer: 'Review Rating Prediction | 2026'
---

<!-- _backgroundColor: #1a1a2e -->
<!-- _color: white -->
<!-- _paginate: false -->

# COMP5434 Big Data Project
## Review Rating Prediction

**Team**: [待填写]

**Date**: July 2026

---

<!-- _backgroundColor: #16213e -->
<!-- _color: white -->

# Outline

1. Problem Definition
2. Data Analysis
3. Solution Architecture
4. Implementation
5. Performance Evaluation
6. Summary & Future Work

---

<!-- _backgroundColor: #0f3460 -->
<!-- _color: white -->

# 1. Problem Definition

## Task
- **Goal**: Predict review ratings (1.0-5.0) from text and metadata
- **Type**: Regression problem
- **Metric**: RMSE (Root Mean Square Error)

## Data Scale
| Dataset | Records |
|---------|---------|
| Training | 3,007,439 |
| Test | 10,000 |

## Challenges
- Large-scale data (3M records) → requires big data framework
- Complex text features → traditional TF-IDF insufficient
- Rating imbalance (4-5 stars dominate 80%+)

---

# 2. Data Analysis

## Rating Distribution
```
5.0 ████████████████████████████ 60%
4.0 ████████████               20%
3.0 ████                        7%
2.0 ███                         5%
1.0 ██████                      8%
```

## User/Product Statistics (PySpark)
| Statistic | Value |
|-----------|-------|
| Unique users | 1,762,679 |
| Unique products | 259,791 |
| Avg reviews/user | 1.7 |
| Avg reviews/product | 11.6 |
| Global avg rating | 3.9428 |

---

<!-- _backgroundColor: #533483 -->
<!-- _color: white -->

# 3. Solution Architecture

## Cloud Distributed Cluster

```
┌─────────────────────────────────────────────────┐
│              阿里云 VPC                           │
│                                                  │
│  ┌──────────┐                                   │
│  │ Control   │  Spark Master :7077              │
│  │ Node      │                                   │
│  └────┬─────┘                                   │
│       │                                          │
│  ┌────┴────┐                                    │
│  ▼         ▼                                    │
│ ┌────┐  ┌────┐   Phase 1: Spark Feature Eng     │
│ │GPU-0│◄─►│GPU-1│   Phase 2: DDP Training       │
│ │A10  │  │A10  │                                │
│ └────┘  └────┘                                  │
│ 32core   32core   Total: 68 vCPU, 272GB RAM    │
│ 128GB    128GB    2× NVIDIA A10 (24GB each)    │
└─────────────────────────────────────────────────┘
```

---

# 3.1 Two-Phase Computation

| Phase | Framework | Parallelism | Nodes | Time |
|-------|-----------|-------------|-------|------|
| **Feature Engineering** | Apache Spark 4.1.2 | Master + 2 Workers, 8 cores | 3 | **23.3s** |
| **Model Training** | PyTorch DDP + NCCL | 2 nodes × 1 GPU | 2 | **1.44h** |

## Pipeline Flow
```
Phase 0: Node validation (10s)
    ↓
Phase 1: Spark feature engineering (23.3s)
    → LOO statistics, Parquet output
    ↓
Phase 2: DDP training (1.44h)
    → RoBERTa-base fine-tuning, 11,553 steps
    ↓
Phase 3: Test prediction (10s)
    → submission.csv
```

---

# 3.2 Spark Feature Engineering

## Leave-One-Out (LOO) Statistics
- Prevent label leakage by excluding current row
- Computed using Spark Window functions

```python
# LOO mean: exclude current row
w = Window.partitionBy("user_id").rowsBetween(
    Window.unboundedPreceding, Window.unboundedFollowing
)
user_loo_mean = (F.sum("rating").over(w) - F.col("rating")) / 
                (F.count("rating").over(w) - 1)
```

## Output Features
| Feature | Records | Description |
|---------|---------|-------------|
| user_stats | 1,762,679 | User-level LOO mean & count |
| prod_stats | 259,791 | Product-level LOO mean & count |
| parent_stats | 213,571 | Parent product LOO mean & count |

---

# 3.3 RoBERTa-base End-to-End Fine-tuning

## Model Architecture
```
Input: title + [SEP] + comment
    ↓ Tokenizer (max_length=192)
input_ids + attention_mask
    ↓ RoBERTa-base (12-layer Transformer, 125M params)
last_hidden_state [batch, 192, 768]
    ↓ Mean Pooling (masked average)
pooled_embedding [batch, 768]
    ↓ Dropout(0.1) → Linear(768→1)
Predicted rating [batch]
    ↓ MSELoss → Backprop (all 124.6M params)
```

## Key Design
- **End-to-end**: BERT parameters + regression head jointly optimized
- **Mean pooling**: More stable than [CLS] token
- **BF16**: 2x speedup without quality loss

---

# 4. Training Configuration

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Checkpoint** | roberta-base (125M) |
| **Max length** | 192 tokens |
| **Batch size (per GPU)** | 128 |
| **Effective batch** | 256 (128 × 2) |
| **Optimizer** | AdamW (wd=0.01) |
| **Learning rate** | 1.2e-4 (linear scaled) |
| **Iterations** | 11,553 steps |
| **Epochs** | 1 (full data) |
| **Precision** | BF16 |
| **Scheduler** | OneCycleLR (warmup 10%) |
| **Seed** | 42 |
| **Distributed backend** | NCCL |

---

# 4.1 Training Process

## Loss Convergence
```
Step  5,000 (43%): Loss = 1.3788
Step  7,000 (61%): Loss = 1.3504  ↓
Step  9,000 (78%): Loss = 1.3304  ↓
Step 11,000 (95%): Loss = 1.3172  ↓
Step 11,553 (100%): Loss = 1.3136  ↓
```

> Loss steadily decreased (1.38 → 1.31), no overfitting.

## GPU Utilization
| Metric | Value |
|--------|-------|
| GPU utilization | ~95% |
| Peak VRAM | 12.4 GB / 24 GB (51.7%) |
| Peak temperature | 72°C |
| NCCL overhead | ~5% |

---

<!-- _backgroundColor: #e94560 -->
<!-- _color: white -->

# 5. Performance Evaluation

## Final Results

| Method | Training Data | Training Time | Val. RMSE | Kaggle RMSE |
|--------|--------------|---------------|-----------|-------------|
| DistilBERT + Ridge | 500k | 30 min | -- | 0.860 |
| DistilBERT + MLP | 500k | 40 min | 1.203 | 0.73249 |
| DeBERTa-v3 + MLP | 500k | 55 min | 1.125 | 0.67042 |
| RoBERTa (subset) | 500k | 2.28 h | 1.140 | 0.54081 |
| **RoBERTa (full, local)** | **2.95M** | **6.9 h** | **1.1305** | **0.51904** ⭐ |
| RoBERTa (full, cloud DDP) | 2.95M | 1.44 h | 1.1369 | 0.53943 |

---

# 5.1 Key Findings

## 1. Big Data Value
- Full data (2.95M) vs subset (500k): **0.51904 vs 0.54081**
- Data quantity > epoch count for deep learning

## 2. Distributed Training Acceleration
- Cloud 2×A10 DDP: **1.44h** vs Local: **6.9h** → **4.8x speedup**
- Spark feature engineering: 23.3s for 3M records

## 3. Data Scalability
| Data Size | Local GPU | Cloud DDP | Speedup |
|-----------|-----------|-----------|---------|
| 500k | 2.3h | 0.5h | 4.6x |
| 2.95M | 6.9h | 1.44h | 4.8x |
| 10M (est.) | 24h | 5h | 4.8x |
| 50M (est.) | 120h ❌ | 25h ✅ | 4.8x |

> Distributed advantage grows with data size

---

# 5.2 Ablation Study

| Comparison | Option A | Option B | Conclusion |
|------------|----------|----------|------------|
| **Data volume** | 500k × 2ep (0.54081) | Full × 1ep (0.51904) | Full data better |
| **End-to-end vs MLP** | E2E (0.61814) | MLP (0.64759) | E2E better |
| **Statistics features** | With (0.64759) | Without (0.65071) | Slight help |
| **Model scale** | DistilBERT 66M (0.732) | RoBERTa 125M (0.519) | Larger better |
| **Hardware** | Local RTX4060 (0.51904) | Cloud A10×2 (0.53943) | Local slightly better |

## Key Insights
- **Full data training** is the most important factor
- **End-to-end fine-tuning** outperforms fixed embedding + MLP
- **Distributed training** provides scalability for larger datasets

---

<!-- _backgroundColor: #2d3436 -->
<!-- _color: white -->

# 6. Summary

## Achievements
- ✅ Built **Spark + DDP** distributed cluster (3 nodes, 2× A10)
- ✅ Processed **3M records** with PySpark feature engineering
- ✅ Achieved **Kaggle RMSE 0.51904** (from 1.556 baseline)
- ✅ Demonstrated **4.8x speedup** with distributed training
- ✅ Verified **scalability** for larger datasets

## Technology Stack
| Layer | Technology |
|-------|-----------|
| Big Data Processing | Apache Spark 4.1.2 |
| Deep Learning | PyTorch 2.6.0 + DDP |
| Pre-trained Model | RoBERTa-base (125M) |
| Distributed Training | NCCL |
| Cloud Platform | Alibaba Cloud ECS |

---

# 6.1 Future Work

## Optimization Directions

1. **Multi-epoch training**
   - Full data × 2-3 epochs may further reduce RMSE

2. **Larger models**
   - RoBERTa-large (355M params) with A10 24GB

3. **More GPUs**
   - 4-8 node DDP for larger batch and faster training

4. **Model ensemble**
   - Combine multiple RoBERTa predictions

5. **Auto-tuning**
   - Ray Tune / Optuna for hyperparameter search

---

<!-- _backgroundColor: #1a1a2e -->
<!-- _color: white -->
<!-- _paginate: false -->

# Thank You

## Q&A

**Team**: [待填写]

**Final Kaggle RMSE**: 0.51904

**Repository**: big_data_trea

---
