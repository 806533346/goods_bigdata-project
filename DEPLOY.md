# COMP5434 云端集群部署指南

## 前提准备

### 1. 阿里云账号
- 实名认证（GPU 实例需要）
- 开通 ECS、VPC、安全组服务
- 申请 GPU 实例配额（配额中心搜 `gn7i`）

### 2. 本地环境
```bash
# 安装 SSH 客户端 (Linux/Mac 自带, Windows 用 Git Bash)
ssh -V
```

---

## 第一步：购买云服务器

### 控制节点（1 台）
```
计费方式:     抢占式实例
实例规格:     ecs.g7.xlarge (4vCPU, 16GB)
镜像:         Ubuntu 22.04 LTS
系统盘:       40GB ESSD
网络:         新建 VPC, 交换机 10.0.1.0/24
公网 IP:      分配
安全组:       新建, 入方向放行 22, 7077, 8080, 29500-29600
登录方式:     SSH 密钥 (下载 .pem 文件)
实例名称:     comp5434-control
```

### GPU 节点（2 台）
```
计费方式:     抢占式实例
实例规格:     ecs.gn7i-c32g1.8xlarge (32vCPU, 128GB, 1×A10 24GB)
镜像:         Ubuntu 22.04 + 勾选自动安装 GPU 驱动 (CUDA 12.4)
系统盘:       100GB ESSD
网络:         选已有 VPC (和控制器同一个)
公网 IP:      分配 (第一台), 不分配 (第二台)
安全组:       选已有
登录方式:     同一 SSH 密钥
实例中断模式:  停机不收费
自动恢复:     勾选
实例名称:     comp5434-gpu-0, comp5434-gpu-1
```

### 安全组入方向规则

| 端口 | 来源 | 用途 |
|------|------|------|
| 22 | 0.0.0.0/0 | SSH |
| 7077 | 10.0.0.0/16 | Spark Master RPC |
| 8080-8081 | 0.0.0.0/0 | Spark Web UI |
| 29500-29600 | 10.0.0.0/16 | DDP 训练通信 |

---

## 第二步：配置 SSH 免密登录

```bash
# 把密钥复制到所有节点 (每台需输入一次密码)
ssh-copy-id -i ~/你的密钥.pem root@<控制节点公网IP>
ssh-copy-id -i ~/你的密钥.pem root@<GPU-0-公网IP>
ssh-copy-id -i ~/你的密钥.pem root@<GPU-1-公网IP>

# 测试免密连接
ssh root@<控制节点公网IP> "hostname"
ssh root@<GPU-0-公网IP> "nvidia-smi -L"
ssh root@<GPU-1-公网IP> "nvidia-smi -L"
```

---

## 第三步：上传代码和数据

```bash
cd /path/to/big_data_cloud

# 在所有节点上创建目录
for IP in <控制节点IP> <GPU-0-IP> <GPU-1-IP>; do
  ssh root@$IP "mkdir -p /app/{src,configs,scripts,data,output}"
done

# 上传源代码 (所有节点)
for IP in <控制节点IP> <GPU-0-IP> <GPU-1-IP>; do
  scp src/*.py root@$IP:/app/src/
  scp configs/*.yaml configs/*.sh root@$IP:/app/configs/
  scp scripts/*.sh root@$IP:/app/scripts/
done

# 上传训练数据 (两个 GPU 节点)
for IP in <GPU-0-IP> <GPU-1-IP>; do
  scp data/train.csv data/test.csv data/prodInfo.csv root@$IP:/app/data/
done
```

---

## 第四步：安装环境

```bash
# 控制节点 (Spark)
ssh root@<控制节点IP> "
apt-get update && apt-get install -y openjdk-21-jdk-headless
pip3 install pyspark pyarrow numpy pandas -i https://mirrors.aliyun.com/pypi/simple/
"

# GPU 节点 ×2 (PyTorch + Spark)
ssh root@<GPU-0-IP> "
apt-get update && apt-get install -y openjdk-21-jdk-headless
pip3 install torch transformers pyspark pyarrow numpy pandas scikit-learn tqdm -i https://mirrors.aliyun.com/pypi/simple/
"
ssh root@<GPU-1-IP> "
apt-get update && apt-get install -y openjdk-21-jdk-headless
pip3 install torch transformers pyspark pyarrow numpy pandas scikit-learn tqdm -i https://mirrors.aliyun.com/pypi/simple/
"
```

---

## 第五步：运行训练

### 方式一：一键运行（推荐）

```bash
# 修改 scripts/run_all.sh 顶部 IP 为实际值
vim scripts/run_all.sh

# 运行
bash scripts/run_all.sh
```

### 方式二：手动逐步运行

```bash
# 1. 启动 Spark Master (控制节点)
ssh root@<控制节点IP> 'bash /app/scripts/start_spark_master.sh'

# 2. 启动 Spark Workers (两个 GPU 节点)
ssh root@<GPU-0-IP> 'bash /app/scripts/start_spark_worker.sh'
ssh root@<GPU-1-IP> 'bash /app/scripts/start_spark_worker.sh'

# 3. 验证: 浏览器打开 http://<控制节点公网IP>:8080 看到 2 个 Workers

# 4. 运行分布式特征工程
ssh root@<GPU-0-IP> 'bash /app/scripts/run_spark_features.sh'

# 5. 启动 DDP 训练 (开两个终端窗口)
# 窗口 A — 先启动 GPU-1
ssh root@<GPU-1-IP> 'bash /app/scripts/launch_rank1.sh'

# 窗口 B — 紧接着启动 GPU-0
ssh root@<GPU-0-IP> 'bash /app/scripts/launch_rank0.sh'

# 6. 监控训练
ssh root@<GPU-0-IP> 'tail -f /app/output/train_rank0.log'
```

---

## 第六步：下载结果

```bash
scp root@<GPU-0-IP>:/app/output/submission.csv ./output/
scp root@<GPU-0-IP>:/app/data/roberta_base_finetuned.pt ./output/
scp root@<GPU-0-IP>:/app/data/roberta_base_train_log.json ./output/
```

---

## 销毁资源

```bash
bash scripts/teardown_cloud.sh
```

或在阿里云控制台手动释放所有 ECS 实例。

---

## 常见问题

| 问题 | 解决 |
|------|------|
| Spark Master 连不上 | 检查安全组 7077 端口、确认 JAVA_HOME 已设 |
| DDP 报 Connection refused | 先启动 GPU-1，等 3 秒再启动 GPU-0 |
| 模型崩塌 (全预测 4.53) | LR 太高，cluster_env.sh 中用 6e-5 |
| CUDA 版本不匹配 | GPU 节点镜像选"自动安装 CUDA 12.4" |
| pip 下载慢 | 加 `-i https://mirrors.aliyun.com/pypi/simple/` |
