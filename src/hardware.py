"""
硬件信息收集 — 自动记录训练所用的 CPU/GPU 配置。

用在训练结束后，将硬件信息写入 train_log.json，
方便复现实验和对比不同云实例的性能。

示例输出:
  {
    "cpu": {"cpu_model": "Intel Xeon Platinum 8369B", "cpu_cores": 32},
    "gpu": {"gpu_count": 1, "gpus": [{"name": "NVIDIA A10", "vram_gb": 22.07}]},
    "cuda_version": "12.4",
    "pytorch_version": "2.6.0+cu124"
  }
"""
import os
import json
import platform
import multiprocessing

import torch


def get_cpu_info() -> dict:
    """Collect CPU information."""
    try:
        import cpuinfo
        cpu_model = cpuinfo.get_cpu_info()["brand_raw"]
    except Exception:
        cpu_model = platform.processor() or "Unknown"

    return {
        "cpu_model": cpu_model,
        "cpu_cores": multiprocessing.cpu_count(),
        "cpu_architecture": platform.machine(),
    }


def get_gpu_info() -> dict:
    """Collect GPU information from all available devices."""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "gpu_count": 0}

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_gb": round(props.total_memory / (1024 ** 3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
        })

    return {
        "gpu_available": True,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }


def get_system_info() -> dict:
    """Collect full system information for cloud VMs."""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        # Cloud environment detection
        "is_distributed": "RANK" in os.environ,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def save_hardware_info(output_path: str):
    """Save hardware info to JSON file."""
    info = get_system_info()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info
