# COMP5434 Cloud — 云端分布式训练集群

基于原 [`big_data_trea`](../big_data_trea/) 项目的云端并行计算版本。将单机 Spark + 单 GPU 升级为**分布式 Spark 集群 + 多节点 DDP 训练**。

## 实际部署架构

```
阿里云 VPC (10.0.0.0/16)
├── 控制节点 ecs.g7.xlarge       4vCPU/16GB     Spark Master
├── GPU-0   ecs.gn7i-c32g1.8xlarge  32vCPU/128GB/1×A10   Spark Worker + DDP Rank 0
└── GPU-1   ecs.gn7i-c32g1.8xlarge  32vCPU/128GB/1×A10   Spark Worker + DDP Rank 1
```

## 训练结果

| 指标 | 值 |
|------|-----|
| **Val RMSE** | **1.1369** |
| 训练时间 | 1.44 小时 |
| Spark 特征 | 23.3 秒 (分布式, 2 Workers) |
| 模型 | RoBERTa-base (125M) |
| 有效 Batch | 256 (128×2 GPUs) |
| 学习率 | 6.0e-5 |
| GPU 显存 | 12.4 GB / 24 GB |

> 📄 完整报告: [`output/training_report.md`](output/training_report.md)

## 快速开始

### 前提

```bash
# 1. 配置 SSH 免密登录
ssh-copy-id root@<控制节点IP>
ssh-copy-id root@<GPU-0-IP>
ssh-copy-id root@<GPU-1-IP>
```

### 一键运行

```bash
bash scripts/run_all.sh
```

自动完成: Spark 集群 → 分布式特征 → 2节点 DDP 训练 → 收集结果

### 手动运行

```bash
# 1. 启动 Spark Master (控制节点)
ssh root@<控制节点IP> 'bash /app/scripts/start_spark_master.sh'

# 2. 启动 Spark Workers (两个 GPU 节点)
ssh root@<GPU-0-IP> 'bash /app/scripts/start_spark_worker.sh'
ssh root@<GPU-1-IP> 'bash /app/scripts/start_spark_worker.sh'

# 3. 运行分布式特征工程
ssh root@<GPU-0-IP> 'bash /app/scripts/run_spark_features.sh'

# 4. 启动 DDP 训练 (先 GPU-1 后 GPU-0)
ssh root@<GPU-1-IP> 'bash /app/scripts/launch_rank1.sh'
ssh root@<GPU-0-IP> 'bash /app/scripts/launch_rank0.sh'

# 5. 监控
ssh root@<GPU-0-IP> 'tail -f /app/output/train_rank0.log'
```

## 目录结构

```text
big_data_trea_cloud/
├── src/                           # 核心代码
│   ├── config.py                  # 集中配置 (环境变量驱动)
│   ├── cloud_io.py                # 云端 I/O (OSS/S3)
│   ├── hardware.py                # 硬件信息收集
│   ├── spark_features_cloud.py    # 分布式 Spark 特征工程
│   ├── finetune_roberta_ddp.py    # RoBERTa DDP 多节点训练 ★
│   ├── extract_embeddings.py      # 嵌入提取
│   ├── train_mlp.py               # MLP 训练
│   └── pipeline_cloud.py          # Prefect 编排
├── scripts/                       # 运维脚本
│   ├── run_all.sh                 # 一键运行 ★
│   ├── start_spark_master.sh      # 启动 Spark Master
│   ├── start_spark_worker.sh      # 启动 Spark Worker
│   ├── run_spark_features.sh      # 运行特征工程
│   ├── launch_rank0.sh            # DDP Rank 0 启动
│   └── launch_rank1.sh            # DDP Rank 1 启动
├── configs/
│   ├── cloud_config.yaml          # 云端配置
│   └── cluster_env.sh             # 训练超参 ★
├── docker/
│   ├── Dockerfile.spark           # Spark 镜像
│   ├── Dockerfile.torch           # PyTorch GPU 镜像
│   └── docker-compose.yml         # 本地开发
├── output/                        # 训练结果 ★
│   ├── submission_cloud.csv       # Kaggle 提交
│   ├── roberta_base_finetuned.pt  # 模型权重 (476MB)
│   ├── roberta_base_train_log.json # 训练日志
│   └── training_report.md         # 完整报告
└── README.md
```

## 推荐配置 (验证过)

| 参数 | 值 | 说明 |
|------|-----|------|
| BATCH_SIZE | 128 | 每 GPU |
| LEARNING_RATE | 6.0e-5 | 保守，避免崩塌 |
| NUM_WORKERS | 16 | CPU 核数的一半 |
| MASTER_PORT | 29505 | DDP 通信端口 |

> ⚠️ 不要超过 LR=1.2e-4，否则模型会崩塌（所有预测 = 4.53）


