"""CIFAR-100 splitting, transforms, and data-loader construction."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vision_bench.config import DataConfig, TrainingConfig


@dataclass(frozen=True)
class SplitManifest:
    """Persistent, auditable indices for train, validation, and test."""

    dataset: str
    split_seed: int
    train_per_class: int
    val_per_class: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stratified_indices(
    targets: Sequence[int],
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose deterministic, non-overlapping per-class train/validation indices."""

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    if len(by_class) != 100:
        raise ValueError(f"Expected 100 CIFAR classes, found {len(by_class)}")

    train: list[int] = []
    val: list[int] = []
    rng = random.Random(seed)
    for class_id in sorted(by_class):
        indices = list(by_class[class_id])
        if len(indices) < train_per_class + val_per_class:
            raise ValueError(
                f"Class {class_id} has {len(indices)} items; need {train_per_class + val_per_class}"
            )
        rng.shuffle(indices)
        train.extend(indices[:train_per_class])
        val.extend(indices[train_per_class : train_per_class + val_per_class])
    return tuple(sorted(train)), tuple(sorted(val))


def stratified_limit(targets: Sequence[int], limit: int | None, seed: int) -> tuple[int, ...]:
    """Return all indices or a balanced deterministic subset for tutorial presets."""

    if limit is None or limit >= len(targets):
        return tuple(range(len(targets)))
    if limit < 100:
        raise ValueError("test_limit must be at least 100 so every class is represented")
    per_class, remainder = divmod(limit, 100)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    rng = random.Random(seed + 1)
    selected: list[int] = []
    for class_id in sorted(by_class):
        indices = list(by_class[class_id])
        rng.shuffle(indices)
        count = per_class + (1 if class_id < remainder else 0)
        selected.extend(indices[:count])
    return tuple(sorted(selected))


def _manifest_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def create_manifest(
    train_targets: Sequence[int], test_targets: Sequence[int], config: DataConfig
) -> SplitManifest:
    """Create a manifest from dataset targets without reading image pixels."""

    train, val = stratified_indices(
        train_targets,
        config.train_per_class,
        config.val_per_class,
        config.split_seed,
    )
    test = stratified_limit(test_targets, config.test_limit, config.split_seed)
    payload: dict[str, Any] = {
        "dataset": "CIFAR-100",
        "split_seed": config.split_seed,
        "train_per_class": config.train_per_class,
        "val_per_class": config.val_per_class,
        "train_indices": train,
        "val_indices": val,
        "test_indices": test,
    }
    return SplitManifest(**payload, checksum=_manifest_checksum(payload))


def manifest_path(data_root: Path, config: DataConfig) -> Path:
    """Return the unique manifest path for a preset's split settings."""

    name = (
        f"cifar100-seed{config.split_seed}-train{config.train_per_class}"
        f"-val{config.val_per_class}-test{config.test_limit or 'all'}.json"
    )
    return data_root / "splits" / name


def write_manifest(path: Path, manifest: SplitManifest) -> None:
    """Atomically save a split manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> SplitManifest:
    """Load and checksum-validate a split manifest."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    checksum = raw.pop("checksum")
    expected = _manifest_checksum(raw)
    if checksum != expected:
        raise ValueError(f"Split manifest checksum mismatch: {path}")
    for name in ("train_indices", "val_indices", "test_indices"):
        raw[name] = tuple(int(value) for value in raw[name])
    return SplitManifest(**raw, checksum=checksum)


def prepare_cifar100(config: DataConfig, project_root: Path) -> tuple[Path, SplitManifest]:
    """Download CIFAR-100 if needed and create or verify its split manifest."""

    from torchvision.datasets import CIFAR100

    data_root = (project_root / config.root).resolve()
    official_train = CIFAR100(data_root, train=True, download=True)
    official_test = CIFAR100(data_root, train=False, download=True)
    path = manifest_path(data_root, config)
    if path.exists():
        manifest = load_manifest(path)
    else:
        manifest = create_manifest(official_train.targets, official_test.targets, config)
        write_manifest(path, manifest)
    return data_root, manifest


def build_transforms(
    mean: tuple[float, ...],
    std: tuple[float, ...],
    input_size: int,
    random_erasing: float,
) -> tuple[Any, Any]:
    """Build shared geometry with checkpoint-native normalization."""

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.Resize((input_size, input_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=random_erasing, value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _seed_worker(worker_id: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    data_root: Path,
    manifest: SplitManifest,
    data_config: DataConfig,
    training_config: TrainingConfig,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    seed: int,
    pin_memory: bool,
) -> tuple[Any, Any, Any]:
    """Construct model-specific loaders over the shared split indices."""

    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR100

    train_transform, eval_transform = build_transforms(
        mean,
        std,
        data_config.input_size,
        training_config.random_erasing,
    )
    train_base = CIFAR100(data_root, train=True, transform=train_transform, download=False)
    eval_train_base = CIFAR100(data_root, train=True, transform=eval_transform, download=False)
    test_base = CIFAR100(data_root, train=False, transform=eval_transform, download=False)
    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": training_config.batch_size,
        "num_workers": data_config.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        # Recreate and seed workers at each epoch boundary. This makes the
        # checkpointed loader generator sufficient to resume data order and
        # augmentation streams in the same environment.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        Subset(train_base, manifest.train_indices),
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        Subset(eval_train_base, manifest.val_indices),
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        Subset(test_base, manifest.test_indices),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, test_loader
