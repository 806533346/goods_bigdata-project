"""Hardware information collection for the project report."""
import platform
import cpuinfo
import multiprocessing
import json
import os


def get_cpu_info():
    """Get CPU model and thread count."""
    info = cpuinfo.get_cpu_info()
    return {
        "cpu_model": info.get("brand_raw", "Unknown"),
        "cpu_cores": multiprocessing.cpu_count(),
        "cpu_arch": platform.machine(),
    }


def get_gpu_info():
    """Get GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "total_vram_gb": round(props.total_memory / 1e9, 2),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            }
    except ImportError:
        pass
    return {"gpu_available": False}


def get_system_info():
    """Collect all hardware/system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
    }


def save_hardware_info(output_path):
    """Save hardware info to JSON."""
    info = get_system_info()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


if __name__ == "__main__":
    info = get_system_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
