# COMP5434 GPU 集群 — 配置汇总

## 集群节点

| 角色 | 实例类型 | CPU 核数 | 内存 | GPU | 内网 IP | 公网 IP |
|------|---------|---------|------|-----|---------|---------|
| 控制节点 (Spark Master) | ecs.g7.xlarge | 4 vCPU | 16 GB | — | 10.0.1.159 | <控制节点公网IP> |
| GPU 节点 0 (DDP Rank 0) | ecs.gn7i-c32g1.8xlarge | 32 vCPU | 128 GB | 1× A10 24GB | 10.0.1.160 | <GPU-0-公网IP> |
| GPU 节点 1 (DDP Rank 1) | ecs.gn7i-c32g1.8xlarge | 32 vCPU | 128 GB | 1× A10 24GB | 10.0.1.161 | <GPU-1-公网IP> |

| 汇总 | 值 |
|------|-----|
| 总节点数 | 3 |
| 总 vCPU | 68 |
| 总内存 | 272 GB |
| 总 GPU | 2× NVIDIA A10 |

---

## Phase 1: Spark 分布式特征工程

| 配置项 | 值 | 说明 |
|------|-----|------|
| **Number of Servers** | **3** | 1 Master + 2 Workers |
| **Executor Cores** | **8** | 2 Workers × 4 cores each |
| Executor Memory | 16 GB × 2 | 每 Worker 分配 16 GB |
| Driver Memory | 8 GB | Master 节点 |
| Shuffle Partitions | 64 | 并行度和网络通信粒度 |
| 数据量 | 3,007,439 条 | |
| 处理耗时 | 23.3 秒 | |

### Spark 集群连接信息

```
Master URL:    spark://10.0.1.159:7077
Web UI:        http://<控制节点公网IP>:8080
Worker 0:      http://10.0.1.160:8081
Worker 1:      http://10.0.1.161:8081
```

---

## Phase 2: 多节点 DDP 训练

| 配置项 | 值 | 说明 |
|------|-----|------|
| **Number of Servers** | **2** | GPU-0 + GPU-1 |
| **GPUs per Server** | **1** | |
| **World Size** | **2** | 2 节点 × 1 GPU |
| **DataLoader Workers** | **16** | 每个节点 16 线程加载数据 |
| Distributed Backend | NCCL | GPU 间梯度同步 |
| 有效 Batch Size | 256 | 128 × 2 GPUs |
| 混合精度 | BFloat16 | 省显存，加速训练 |

### 模型超参

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | RoBERTa-base | 125M 参数 |
| 最大序列长度 | 192 | title + [SEP] + comment |
| Batch Size (per GPU) | 128 | |
| 有效 Batch Size | 256 | 128 × 2 |
| 学习率 (Base) | 6.0e-5 | |
| 学习率 (Scaled) | 1.2e-4 | ×2 线性缩放 |
| Warmup | 10% | 前 1,155 步线性预热 |
| 调度器 | OneCycleLR | cosine 衰减 |
| 优化器 | AdamW | weight_decay=0.01 |
| 梯度裁剪 | 1.0 | |
| Epochs | 1 | 全量 2.95M 数据 |
| 总步数 | 11,553 | |
| 训练耗时 | 1.4 小时 | |

### DDP 连接信息

```
Master:         10.0.1.160:29505  (NCCL)
Process 0:      GPU-0, rank=0
Process 1:      GPU-1, rank=1
```

---

## 硬件详情

### GPU

```
型号:        NVIDIA A10
CUDA:        12.4
PyTorch:     2.6.0+cu124
计算能力:    8.6
显存:        22 GB (nvidia-smi 报告 24 GB)
```

### CPU

```
型号:        Intel Xeon Platinum 8369B @ 2.90GHz
架构:        x86_64
每节点核数:  32
```

---

## 软件版本

| 软件 | 版本 |
|------|------|
| Apache Spark (PySpark) | 4.1.2 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.12.1 |
| Python | 3.10.12 |
| OS | Ubuntu 22.04 LTS |
| Java | OpenJDK 21 |

---

## 训练结果

| 指标 | 值 |
|------|-----|
| Val RMSE | 1.1369 |
| Train Loss | 1.3136 |
| 预测均值 ± 标准差 | 4.20 ± 0.88 |
| GPU 峰值显存 | 12.4 GB |
| GPU 峰值温度 | 72°C |
| 模型大小 | 476 MB |
