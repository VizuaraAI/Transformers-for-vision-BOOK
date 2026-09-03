"""Environment diagnostics and a synthetic model-contract smoke test."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vision_bench.metrics import averaged_logits
from vision_bench.models import MODEL_SPECS, create_model, parameter_count
from vision_bench.runtime import select_device, synchronize


def doctor_report(project_root: Path) -> dict[str, Any]:
    """Return dependency, accelerator, and disk diagnostics."""

    import torch

    distributions = ["torch", "torchvision", "timm", "modal", "numpy", "pandas"]
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    disk = shutil.disk_usage(project_root)
    return {
        "python": sys.version.split()[0],
        "python_supported": sys.version_info[:2] == (3, 12),
        "platform": platform.platform(),
        "project_root": str(project_root.resolve()),
        "dependencies": versions,
        "accelerators": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
        "disk_free_gb": disk.free / (1024**3),
    }


def synthetic_smoke_test(
    aliases: Iterable[str], device_name: str = "auto", input_size: int = 224
) -> list[dict[str, Any]]:
    """Instantiate models without downloads and validate one inference pass each."""

    import torch

    device = select_device(device_name)
    results: list[dict[str, Any]] = []
    for alias in aliases:
        if alias not in MODEL_SPECS:
            raise ValueError(f"Unknown smoke-test model: {alias}")
        model = create_model(alias, num_classes=100, pretrained=False).to(device).eval()
        sample = torch.zeros(1, 3, input_size, input_size, device=device)
        synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = averaged_logits(model(sample))
        synchronize(device)
        if tuple(logits.shape) != (1, 100):
            raise AssertionError(f"{alias} produced {tuple(logits.shape)}, expected (1, 100)")
        if alias == "deit_distilled" and hasattr(model, "set_distilled_training"):
            model.train()
            model.set_distilled_training(True)
            with torch.no_grad():
                two_heads = model(sample)
            if not isinstance(two_heads, tuple) or len(two_heads) != 2:
                raise AssertionError("Distilled DeiT did not expose two heads in training mode")
        results.append(
            {
                "model": alias,
                "parameters": parameter_count(model),
                "output_shape": list(logits.shape),
                "forward_seconds": time.perf_counter() - started,
                "device": device.type,
                "status": "ok",
            }
        )
        del model, sample
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results
