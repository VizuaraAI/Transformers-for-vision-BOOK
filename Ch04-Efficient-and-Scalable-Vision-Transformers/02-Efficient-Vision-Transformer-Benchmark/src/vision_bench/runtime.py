"""Device selection, seeding, environment capture, and synchronization."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any


def select_device(requested: str = "auto") -> Any:
    """Select CUDA, then Apple MPS, then CPU unless explicitly requested."""

    import torch

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but this PyTorch build cannot use it")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without promising cross-device identity."""

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def synchronize(device: Any) -> None:
    """Wait for queued accelerator work before reading wall-clock time."""

    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: Any) -> None:
    """Reset accelerator peak-memory tracking when supported."""

    import torch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_mb(device: Any) -> float | None:
    """Return peak CUDA memory or current MPS tensor memory in MiB."""

    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    return None


def environment_info(device: Any) -> dict[str, Any]:
    """Capture enough runtime metadata to interpret and reproduce a result."""

    import timm
    import torch
    import torchvision

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device_type": device.type,
        "pid": os.getpid(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device_name": properties.name,
                "device_memory_mb": properties.total_memory / (1024**2),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    elif device.type == "mps":
        info["device_name"] = "Apple Metal Performance Shaders"
    else:
        info["device_name"] = platform.processor() or "CPU"
    return info
