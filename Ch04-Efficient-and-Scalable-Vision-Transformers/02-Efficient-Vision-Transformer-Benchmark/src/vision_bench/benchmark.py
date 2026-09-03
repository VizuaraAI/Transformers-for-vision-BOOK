"""Repeatable inference profiling for trained chapter models."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from vision_bench.checkpointing import atomic_json, load_checkpoint
from vision_bench.config import BenchmarkConfig, ProjectConfig, RunConfig
from vision_bench.engine import run_directory
from vision_bench.models import create_model, get_model_spec, parameter_count
from vision_bench.runtime import (
    environment_info,
    peak_memory_mb,
    reset_peak_memory,
    select_device,
    synchronize,
)


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without an extra dependency."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(latencies_ms: Iterable[float], batch_size: int) -> dict[str, float]:
    """Summarize synchronized per-batch latency samples."""

    values = [float(value) for value in latencies_ms]
    if not values:
        raise ValueError("At least one latency sample is required")
    mean_ms = statistics.fmean(values)
    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": statistics.median(values),
        "latency_p90_ms": _percentile(values, 0.90),
        "latency_std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "throughput_images_per_second": batch_size * 1000.0 / mean_ms,
    }


def count_macs(model: Any, input_size: int, device: Any) -> dict[str, Any]:
    """Count multiply-accumulate operations for one image with fvcore.

    fvcore treats one fused multiply-add as one operation. Unsupported operators
    are retained in the result so readers can judge the estimate.
    """

    import torch
    from fvcore.nn import FlopCountAnalysis

    sample = torch.zeros(1, 3, input_size, input_size, device=device)
    try:
        analysis = FlopCountAnalysis(model, sample)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = int(analysis.total())
        unsupported = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
        return {
            "measured_macs": total,
            "measured_gmacs": total / 1e9,
            "unsupported_operators": unsupported,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on third-party tracing support
        return {
            "measured_macs": None,
            "measured_gmacs": None,
            "unsupported_operators": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _precision_context(device: Any, precision: str) -> Any:
    import torch

    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _timed_forward(model: Any, sample: Any, device: Any) -> float:
    """Run one forward pass and return synchronized milliseconds."""

    import torch

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(sample)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    synchronize(device)
    started = time.perf_counter()
    model(sample)
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def benchmark_model(
    model: Any,
    alias: str,
    config: BenchmarkConfig,
    input_size: int,
    device: Any,
) -> list[dict[str, Any]]:
    """Benchmark every configured precision and batch size for one model."""

    import torch

    model.eval()
    if hasattr(model, "set_distilled_training"):
        model.set_distilled_training(False)
    records: list[dict[str, Any]] = []
    for precision in config.precisions:
        if precision == "fp16" and device.type != "cuda":
            for batch_size in config.batch_sizes:
                records.append(
                    {
                        "model": alias,
                        "precision": precision,
                        "batch_size": batch_size,
                        "status": "skipped",
                        "reason": "fp16 reference timing is enabled only on CUDA",
                    }
                )
            continue
        for batch_size in config.batch_sizes:
            sample = torch.randn(batch_size, 3, input_size, input_size, device=device)
            reset_peak_memory(device)
            latencies: list[float] = []
            with torch.inference_mode(), _precision_context(device, precision):
                for _ in range(config.warmup_iterations):
                    model(sample)
                synchronize(device)
                for _ in range(config.repeats):
                    for _ in range(config.timed_iterations):
                        latencies.append(_timed_forward(model, sample, device))
            record: dict[str, Any] = {
                "model": alias,
                "precision": precision,
                "batch_size": batch_size,
                "status": "complete",
                "warmup_iterations": config.warmup_iterations,
                "timed_iterations": config.timed_iterations,
                "repeats": config.repeats,
                "samples": len(latencies),
                "peak_memory_mb": peak_memory_mb(device),
            }
            record.update(summarize_latencies(latencies, batch_size))
            records.append(record)
            del sample
    return records


def _representative_run(project: ProjectConfig, alias: str) -> RunConfig:
    candidates = [run for run in project.runs if run.model == alias]
    if not candidates:
        raise ValueError(f"Preset has no run for model '{alias}'")
    return min(
        candidates,
        key=lambda run: (run.mode != "standard", run.seed != 42, run.seed),
    )


def benchmark_suite(
    project: ProjectConfig,
    artifact_root: Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Profile one representative trained checkpoint per architecture."""

    import torch

    device = select_device(device_name)
    aliases = list(dict.fromkeys(run.model for run in project.runs))
    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for alias in aliases:
        run = _representative_run(project, alias)
        directory = run_directory(artifact_root, project, run)
        checkpoint_path = directory / "best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing trained checkpoint for {run.key}: {checkpoint_path}. "
                "Run the training suite before benchmarking."
            )
        model = create_model(alias, num_classes=100, pretrained=False)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint.get("config_fingerprint") != project.fingerprint:
            raise ValueError(f"Checkpoint fingerprint mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        spec = get_model_spec(alias)
        macs = count_macs(model, project.data.input_size, device)
        profiles.append(
            {
                "model": alias,
                "display_name": spec.display_name,
                "checkpoint": spec.checkpoint,
                "representative_run": run.key,
                "parameters": parameter_count(model),
                "expected_parameters_m": spec.expected_params_m,
                "expected_gmacs_from_model_card": spec.expected_gmacs,
                **macs,
            }
        )
        records.extend(
            benchmark_model(model, alias, project.benchmark, project.data.input_size, device)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "preset": project.name,
        "tutorial_only": project.tutorial_only,
        "config_fingerprint": project.fingerprint,
        "protocol": {
            "input_size": project.data.input_size,
            "synchronization": "CUDA events per iteration; synchronized wall clock otherwise",
            "includes_data_loading": False,
            "weights": "best validation checkpoint; one representative per architecture",
        },
        "environment": environment_info(device),
        "profiles": profiles,
        "measurements": records,
    }
    output_path = artifact_root / project.name / "benchmark.json"
    atomic_json(output_path, payload)
    return payload
