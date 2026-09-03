"""Atomic checkpoints, JSON artifacts, and random-state restoration."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, cast


def _cpu_byte_tensor(value: Any, name: str) -> Any:
    """Return a contiguous CPU byte tensor suitable for PyTorch RNG APIs."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _normalize_checkpoint_rng_tensors(checkpoint: dict[str, Any]) -> None:
    """Keep device-independent RNG metadata on CPU after checkpoint loading.

    ``torch.load(..., map_location='cuda')`` maps every tensor, including CPU
    RNG and data-loader generator states, onto CUDA. PyTorch's CPU RNG and
    ``torch.Generator`` restoration APIs require CPU byte tensors, so normalize
    those small metadata tensors without changing model or optimizer tensors.
    """

    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        if "torch" in rng_state:
            rng_state["torch"] = _cpu_byte_tensor(rng_state["torch"], "torch RNG state")
        if "cuda" in rng_state:
            rng_state["cuda"] = [
                _cpu_byte_tensor(item, f"CUDA RNG state {index}")
                for index, item in enumerate(rng_state["cuda"])
            ]
        if "mps" in rng_state:
            rng_state["mps"] = _cpu_byte_tensor(rng_state["mps"], "MPS RNG state")
    if "data_generator_state" in checkpoint:
        checkpoint["data_generator_state"] = _cpu_byte_tensor(
            checkpoint["data_generator_state"], "data generator state"
        )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file to avoid partial artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one machine-readable metric record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file, ignoring blank lines."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically replace a JSON Lines file with the supplied records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU, and available accelerator RNG states."""

    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state created by :func:`capture_rng_state`."""

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch"], "torch RNG state"))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_tensor(item, f"CUDA RNG state {index}")
                for index, item in enumerate(state["cuda"])
            ]
        )
    if "mps" in state and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(_cpu_byte_tensor(state["mps"], "MPS RNG state"))


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically save a trusted, resumable PyTorch checkpoint."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, map_location: str | Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint produced locally by this project."""

    import torch

    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint does not contain a mapping: {path}")
    _normalize_checkpoint_rng_tensors(value)
    return cast(dict[str, Any], value)
