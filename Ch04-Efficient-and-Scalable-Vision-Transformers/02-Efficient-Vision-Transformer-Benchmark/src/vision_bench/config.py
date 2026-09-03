"""Validated experiment configuration for all execution presets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

RunMode = Literal["standard", "hard_kd"]


@dataclass(frozen=True)
class DataConfig:
    """Dataset locations, split size, and input-pipeline settings."""

    root: str = "data"
    split_seed: int = 2027
    train_per_class: int = 450
    val_per_class: int = 50
    test_limit: int | None = None
    input_size: int = 224
    num_workers: int = 8

    def validate(self) -> None:
        if self.train_per_class < 1 or self.val_per_class < 1:
            raise ValueError("train_per_class and val_per_class must both be positive")
        if self.train_per_class + self.val_per_class > 500:
            raise ValueError("CIFAR-100 has only 500 official training images per class")
        if self.input_size <= 0 or self.num_workers < 0:
            raise ValueError("input_size must be positive and num_workers cannot be negative")
        if self.test_limit is not None and not 100 <= self.test_limit <= 10_000:
            raise ValueError("test_limit must be between 100 and 10,000, or null")


@dataclass(frozen=True)
class TrainingConfig:
    """Shared fine-tuning hyperparameters."""

    epochs: int = 20
    batch_size: int = 64
    backbone_lr: float = 5e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    amp: bool = True
    random_erasing: float = 0.25

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0 <= self.random_erasing <= 1:
            raise ValueError("random_erasing must be in [0, 1]")
        if min(self.backbone_lr, self.head_lr, self.min_lr, self.gradient_clip) <= 0:
            raise ValueError("learning rates and gradient_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        invalid_warmup = self.warmup_epochs < 0 or self.warmup_epochs >= self.epochs
        single_epoch_without_warmup = self.epochs == 1 and self.warmup_epochs == 0
        if invalid_warmup and not single_epoch_without_warmup:
            raise ValueError("warmup_epochs must be smaller than epochs")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inference timing protocol."""

    precisions: tuple[str, ...] = ("fp32", "fp16")
    batch_sizes: tuple[int, ...] = (1, 64)
    warmup_iterations: int = 50
    timed_iterations: int = 200
    repeats: int = 5

    def validate(self) -> None:
        if not self.precisions or any(value not in {"fp32", "fp16"} for value in self.precisions):
            raise ValueError("precisions must contain fp32 and/or fp16")
        if not self.batch_sizes or any(value < 1 for value in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if min(self.warmup_iterations, self.timed_iterations, self.repeats) < 1:
            raise ValueError("benchmark iteration counts must be positive")


@dataclass(frozen=True)
class RunConfig:
    """One model, training mode, and seed."""

    model: str
    mode: RunMode
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}-{self.mode}-seed{self.seed}"


@dataclass(frozen=True)
class ProjectConfig:
    """Fully resolved project preset."""

    name: str
    tutorial_only: bool
    description: str
    pretrained: bool
    data: DataConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    runs: tuple[RunConfig, ...]

    def validate(self) -> None:
        from vision_bench.models import MODEL_SPECS

        self.data.validate()
        self.training.validate()
        self.benchmark.validate()
        if not self.runs:
            raise ValueError("A preset must define at least one run")
        invalid_models = sorted({run.model for run in self.runs} - set(MODEL_SPECS))
        if invalid_models:
            raise ValueError(f"Unknown model alias(es): {', '.join(invalid_models)}")
        if len({run.key for run in self.runs}) != len(self.runs):
            raise ValueError("Every model/mode/seed run must be unique")
        if any(run.mode == "hard_kd" and run.model != "deit_distilled" for run in self.runs):
            raise ValueError("hard_kd is supported only for deit_distilled")
        if any(run.mode == "hard_kd" for run in self.runs):
            teacher_exists = any(
                run.model == "convnext" and run.mode == "standard" for run in self.runs
            )
            if not teacher_exists:
                raise ValueError("A hard_kd preset must train the convnext teacher first")
            teacher_position = next(
                index
                for index, run in enumerate(self.runs)
                if run.model == "convnext" and run.mode == "standard"
            )
            first_kd_position = next(
                index for index, run in enumerate(self.runs) if run.mode == "hard_kd"
            )
            if teacher_position > first_kd_position:
                raise ValueError("The convnext teacher must appear before hard_kd runs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _reject_unknown(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} field(s): {', '.join(unknown)}")


def _resolve_preset(value: str | Path, project_root: Path | None = None) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    name = candidate.stem
    roots = [project_root] if project_root else []
    roots.extend([Path.cwd(), Path(__file__).resolve().parents[2]])
    for root in roots:
        if root is not None:
            path = root / "configs" / f"{name}.yaml"
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(
        f"Could not find preset '{value}'. Run from the project root or pass a YAML path."
    )


def load_project_config(value: str | Path, project_root: Path | None = None) -> ProjectConfig:
    """Load, validate, and expand a smoke, quick, or full YAML preset."""

    path = _resolve_preset(value, project_root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Preset must be a YAML mapping: {path}")
    _reject_unknown(
        raw,
        {
            "name",
            "tutorial_only",
            "description",
            "pretrained",
            "data",
            "training",
            "benchmark",
            "runs",
        },
        "top-level",
    )
    data_raw = dict(raw.get("data", {}))
    train_raw = dict(raw.get("training", {}))
    benchmark_raw = dict(raw.get("benchmark", {}))
    _reject_unknown(data_raw, set(DataConfig.__dataclass_fields__), "data")
    _reject_unknown(train_raw, set(TrainingConfig.__dataclass_fields__), "training")
    _reject_unknown(benchmark_raw, set(BenchmarkConfig.__dataclass_fields__), "benchmark")
    benchmark_raw["precisions"] = tuple(benchmark_raw.get("precisions", ("fp32", "fp16")))
    benchmark_raw["batch_sizes"] = tuple(benchmark_raw.get("batch_sizes", (1, 64)))

    expanded_runs: list[RunConfig] = []
    for index, run_raw in enumerate(raw.get("runs", [])):
        if not isinstance(run_raw, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        _reject_unknown(run_raw, {"model", "mode", "seeds"}, f"runs[{index}]")
        seeds = run_raw.get("seeds", [])
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"runs[{index}].seeds must be a non-empty list")
        mode_value = str(run_raw.get("mode", "standard"))
        if mode_value not in {"standard", "hard_kd"}:
            raise ValueError(f"runs[{index}].mode must be standard or hard_kd")
        mode = cast(RunMode, mode_value)
        for seed in seeds:
            expanded_runs.append(
                RunConfig(
                    model=str(run_raw["model"]),
                    mode=mode,
                    seed=int(seed),
                )
            )

    config = ProjectConfig(
        name=str(raw["name"]),
        tutorial_only=bool(raw["tutorial_only"]),
        description=str(raw["description"]),
        pretrained=bool(raw["pretrained"]),
        data=DataConfig(**data_raw),
        training=TrainingConfig(**train_raw),
        benchmark=BenchmarkConfig(**benchmark_raw),
        runs=tuple(expanded_runs),
    )
    config.validate()
    return config
