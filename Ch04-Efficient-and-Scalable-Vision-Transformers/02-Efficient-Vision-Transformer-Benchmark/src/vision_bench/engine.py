"""Fine-tuning, evaluation, distillation, resumption, and suite orchestration."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from vision_bench.checkpointing import (
    append_jsonl,
    atomic_json,
    capture_rng_state,
    load_checkpoint,
    load_jsonl,
    restore_rng_state,
    save_checkpoint,
    write_jsonl,
)
from vision_bench.config import ProjectConfig, RunConfig
from vision_bench.data import build_loaders, prepare_cifar100
from vision_bench.distillation import HardTokenDistillationLoss, freeze_teacher
from vision_bench.metrics import AverageMeter, averaged_logits, topk_correct
from vision_bench.models import (
    classifier_parameter_ids,
    create_model,
    get_model_spec,
    parameter_count,
    preprocessing_config,
)
from vision_bench.runtime import (
    environment_info,
    peak_memory_mb,
    reset_peak_memory,
    seed_everything,
    select_device,
    synchronize,
)

CheckpointCallback = Callable[[], None]


@dataclass(frozen=True)
class EpochResult:
    """Metrics produced by one complete pass over a loader."""

    loss: float
    top1: float
    top5: float
    samples: int
    duration_seconds: float
    images_per_second: float
    peak_memory_mb: float | None
    classification_loss: float | None = None
    distillation_loss: float | None = None


def _optimizer_groups(model: Any, config: Any) -> list[dict[str, Any]]:
    classifier_ids = classifier_parameter_ids(model)
    no_decay_names: set[str] = set()
    no_decay_keywords: set[str] = set()
    name_provider = getattr(model, "no_weight_decay", None)
    keyword_provider = getattr(model, "no_weight_decay_keywords", None)
    if callable(name_provider):
        no_decay_names.update(str(value) for value in name_provider())
    if callable(keyword_provider):
        no_decay_keywords.update(str(value) for value in keyword_provider())
    groups: dict[tuple[bool, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = id(parameter) in classifier_ids
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or name in no_decay_names
            or any(keyword in name for keyword in no_decay_keywords)
        )
        key = (is_head, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": config.head_lr if is_head else config.backbone_lr,
                "weight_decay": 0.0 if no_decay else config.weight_decay,
                "group_name": f"{'head' if is_head else 'backbone'}-{'no_decay' if no_decay else 'decay'}",
            }
        groups[key]["params"].append(parameter)
    return list(groups.values())


def create_optimizer(model: Any, config: Any) -> Any:
    """Create AdamW with separate head/backbone learning rates and no-decay groups."""

    import torch

    return torch.optim.AdamW(_optimizer_groups(model, config))


def create_scheduler(optimizer: Any, config: Any, steps_per_epoch: int) -> Any:
    """Create per-step linear warmup followed by cosine decay."""

    import torch

    total_steps = max(1, config.epochs * steps_per_epoch)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    lambdas: list[Callable[[int], float]] = []
    for group in optimizer.param_groups:
        initial_lr = float(group["lr"])
        minimum_factor = min(1.0, config.min_lr / initial_lr)

        def schedule(
            step: int,
            *,
            base_minimum: float = minimum_factor,
            warmup: int = warmup_steps,
            total: int = total_steps,
        ) -> float:
            if warmup and step < warmup:
                return max(1e-8, (step + 1) / warmup)
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return base_minimum + (1 - base_minimum) * cosine

        lambdas.append(schedule)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def _autocast(device: Any, enabled: bool) -> Any:
    import torch

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(enabled: bool) -> Any:
    import torch

    return torch.amp.GradScaler("cuda", enabled=enabled)


def _step_optimizer_and_scheduler(scaler: Any, optimizer: Any, scheduler: Any) -> bool:
    """Advance the scheduler only when GradScaler performed an optimizer update.

    ``GradScaler.step`` skips ``optimizer.step`` when it detects non-finite
    gradients and lowers its scale during ``update``. Advancing the scheduler
    on that skipped batch would desynchronize the learning-rate schedule from
    the number of parameter updates and triggers PyTorch's scheduler-order
    warning when it happens on the first batch.
    """

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer_stepped = float(scaler.get_scale()) >= scale_before
    if optimizer_stepped:
        scheduler.step()
    return optimizer_stepped


def train_one_epoch(
    model: Any,
    loader: Iterable[Any],
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    device: Any,
    config: Any,
    distillation_loss: HardTokenDistillationLoss | None = None,
) -> EpochResult:
    """Train exactly one epoch and return sample-weighted metrics."""

    import torch

    model.train()
    if hasattr(model, "set_distilled_training"):
        model.set_distilled_training(distillation_loss is not None)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_meter = AverageMeter()
    class_loss_meter = AverageMeter()
    distill_loss_meter = AverageMeter()
    correct1 = 0
    correct5 = 0
    samples = 0
    amp_enabled = bool(config.amp and device.type == "cuda")
    reset_peak_memory(device)
    synchronize(device)
    started = time.perf_counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp_enabled):
            output = model(images)
            if distillation_loss is None:
                loss = criterion(averaged_logits(output), targets)
                parts: dict[str, float] = {}
            else:
                loss, parts = distillation_loss(images, output, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        _step_optimizer_and_scheduler(scaler, optimizer, scheduler)

        batch_size = int(targets.shape[0])
        logits = averaged_logits(output).detach()
        top1, top5 = topk_correct(logits, targets)
        loss_meter.update(float(loss.detach().item()), batch_size)
        if parts:
            class_loss_meter.update(parts["classification_loss"], batch_size)
            distill_loss_meter.update(parts["distillation_loss"], batch_size)
        correct1 += top1
        correct5 += top5
        samples += batch_size

    synchronize(device)
    duration = time.perf_counter() - started
    return EpochResult(
        loss=loss_meter.average,
        top1=100.0 * correct1 / samples,
        top5=100.0 * correct5 / samples,
        samples=samples,
        duration_seconds=duration,
        images_per_second=samples / duration,
        peak_memory_mb=peak_memory_mb(device),
        classification_loss=class_loss_meter.average if class_loss_meter.count else None,
        distillation_loss=distill_loss_meter.average if distill_loss_meter.count else None,
    )


def evaluate(
    model: Any,
    loader: Iterable[Any],
    device: Any,
    *,
    return_predictions: bool = False,
) -> tuple[EpochResult, dict[str, Any] | None]:
    """Evaluate with ordinary cross-entropy and optionally return CPU predictions."""

    import torch

    model.eval()
    if hasattr(model, "set_distilled_training"):
        model.set_distilled_training(False)
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    correct1 = 0
    correct5 = 0
    samples = 0
    all_predictions: list[Any] = []
    all_targets: list[Any] = []
    reset_peak_memory(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            output = model(images)
            logits = averaged_logits(output)
            loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            top1, top5 = topk_correct(logits, targets)
            loss_meter.update(float(loss.item()), batch_size)
            correct1 += top1
            correct5 += top5
            samples += batch_size
            if return_predictions:
                all_predictions.append(logits.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
    synchronize(device)
    duration = time.perf_counter() - started
    result = EpochResult(
        loss=loss_meter.average,
        top1=100.0 * correct1 / samples,
        top5=100.0 * correct5 / samples,
        samples=samples,
        duration_seconds=duration,
        images_per_second=samples / duration,
        peak_memory_mb=peak_memory_mb(device),
    )
    payload = None
    if return_predictions:
        payload = {
            "predictions": torch.cat(all_predictions).numpy(),
            "targets": torch.cat(all_targets).numpy(),
        }
    return result, payload


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_top1: float,
    best_val_loss: float,
    best_epoch: int,
    cumulative_seconds: float,
    data_generator: Any,
    config_fingerprint: str,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_top1": best_top1,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "cumulative_seconds": cumulative_seconds,
        "data_generator_state": data_generator.get_state(),
        "config_fingerprint": config_fingerprint,
        "rng_state": capture_rng_state(),
    }


def _load_teacher(checkpoint_path: Path, device: Any) -> Any:
    teacher = create_model("convnext", num_classes=100, pretrained=False)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.to(device)
    return freeze_teacher(teacher)


def run_directory(artifact_root: Path, project: ProjectConfig, run: RunConfig) -> Path:
    """Return an immutable, configuration-qualified artifact directory."""

    return artifact_root / project.name / f"{run.key}-{project.fingerprint[:10]}"


def run_experiment(
    project: ProjectConfig,
    run: RunConfig,
    project_root: Path,
    artifact_root: Path,
    *,
    device_name: str = "auto",
    teacher_checkpoint: Path | None = None,
    resume: bool = True,
    on_checkpoint: CheckpointCallback | None = None,
) -> dict[str, Any]:
    """Execute or resume one configured fine-tuning experiment."""

    import numpy as np
    import torch

    get_model_spec(run.model)
    device = select_device(device_name)
    seed_everything(run.seed)
    output_dir = run_directory(artifact_root, project, run)
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        return load_json(complete_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output_dir / "resolved_config.json",
        {
            "project": project.to_dict(),
            "run": asdict(run),
            "fingerprint": project.fingerprint,
        },
    )

    data_root, manifest = prepare_cifar100(project.data, project_root)
    model = create_model(run.model, num_classes=100, pretrained=project.pretrained)
    model.to(device)
    preprocess = preprocessing_config(model)
    train_loader, val_loader, test_loader = build_loaders(
        data_root,
        manifest,
        project.data,
        project.training,
        preprocess["mean"],
        preprocess["std"],
        run.seed,
        pin_memory=device.type == "cuda",
    )
    data_generator = train_loader.generator
    if data_generator is None:
        raise RuntimeError("Training loader must expose its dedicated random generator")

    teacher = None
    kd_loss = None
    if run.mode == "hard_kd":
        if teacher_checkpoint is None or not teacher_checkpoint.exists():
            raise FileNotFoundError(
                "Hard distillation needs the trained ConvNeXt best.pt checkpoint. "
                "Run the suite (which orders dependencies) or pass --teacher-checkpoint."
            )
        teacher = _load_teacher(teacher_checkpoint, device)
        teacher_preprocess = preprocessing_config(teacher)
        if (
            teacher_preprocess["mean"] != preprocess["mean"]
            or teacher_preprocess["std"] != preprocess["std"]
        ):
            raise ValueError(
                "Teacher and student normalization differ; add an explicit input adapter"
            )
        kd_loss = HardTokenDistillationLoss(
            teacher,
            label_smoothing=project.training.label_smoothing,
            alpha=0.5,
        )

    optimizer = create_optimizer(model, project.training)
    scheduler = create_scheduler(optimizer, project.training, len(train_loader))
    amp_enabled = bool(project.training.amp and device.type == "cuda")
    scaler = _make_scaler(amp_enabled)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    start_epoch = 0
    best_top1 = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    cumulative_seconds = 0.0
    metrics_path = output_dir / "metrics.jsonl"

    if resume and latest_path.exists():
        checkpoint = load_checkpoint(latest_path, map_location=device)
        if checkpoint["config_fingerprint"] != project.fingerprint:
            raise ValueError("Refusing to resume: checkpoint and preset fingerprints differ")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint["rng_state"])
        if "data_generator_state" in checkpoint:
            data_generator.set_state(checkpoint["data_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint["best_top1"])
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        cumulative_seconds = float(checkpoint.get("cumulative_seconds", 0.0))
        best_checkpoint_epoch = -1
        if best_path.exists():
            best_checkpoint_epoch = int(load_checkpoint(best_path, map_location="cpu")["epoch"])
        if best_checkpoint_epoch != best_epoch:
            if int(checkpoint["epoch"]) != best_epoch:
                raise RuntimeError("Best and latest checkpoints cannot be reconciled safely")
            # A crash can occur after latest.pt is committed but before the
            # matching best.pt write. In exactly that case, latest contains
            # the validation-best model and repairs the pair losslessly.
            save_checkpoint(best_path, checkpoint)
        records = [
            record
            for record in load_jsonl(metrics_path)
            if int(record.get("epoch", 0)) <= start_epoch
        ]
        write_jsonl(metrics_path, records)
    else:
        # This configuration-qualified directory belongs to the new run. Clear
        # stale rows so a deliberate non-resume cannot duplicate epochs.
        write_jsonl(metrics_path, [])

    for epoch in range(start_epoch, project.training.epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            project.training,
            kd_loss,
        )
        val_result, _ = evaluate(model, val_loader, device)
        cumulative_seconds += train_result.duration_seconds + val_result.duration_seconds
        learning_rates = {
            group["group_name"]: float(group["lr"]) for group in optimizer.param_groups
        }
        record = {
            "epoch": epoch + 1,
            "train": asdict(train_result),
            "validation": asdict(val_result),
            "learning_rates": learning_rates,
            "cumulative_seconds": cumulative_seconds,
        }
        append_jsonl(metrics_path, record)
        improved = val_result.top1 > best_top1 or (
            math.isclose(val_result.top1, best_top1) and val_result.loss < best_val_loss
        )
        if improved:
            best_top1 = val_result.top1
            best_val_loss = val_result.loss
            best_epoch = epoch
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_top1,
            best_val_loss,
            best_epoch,
            cumulative_seconds,
            data_generator,
            project.fingerprint,
        )
        save_checkpoint(latest_path, payload)
        if improved:
            save_checkpoint(best_path, payload)
        if on_checkpoint is not None:
            on_checkpoint()
        print(
            f"[{run.key}] epoch {epoch + 1:02d}/{project.training.epochs}: "
            f"train loss={train_result.loss:.4f}, val top1={val_result.top1:.2f}%"
        )

    best_checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    test_result, predictions = evaluate(model, test_loader, device, return_predictions=True)
    if predictions is not None:
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            **predictions,
            indices=np.asarray(manifest.test_indices, dtype=np.int64),
        )
    summary = {
        "status": "complete",
        "run_id": output_dir.name,
        "run_key": run.key,
        "model": run.model,
        "display_name": get_model_spec(run.model).display_name,
        "checkpoint": get_model_spec(run.model).checkpoint,
        "mode": run.mode,
        "seed": run.seed,
        "tutorial_only": project.tutorial_only,
        "config_fingerprint": project.fingerprint,
        "best_validation_top1": best_top1,
        "best_validation_loss": best_val_loss,
        "best_epoch": int(best_checkpoint["epoch"]) + 1,
        "test": asdict(test_result),
        "parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "environment": environment_info(device),
        "split_checksum": manifest.checksum,
    }
    atomic_json(complete_path, summary)
    if on_checkpoint is not None:
        on_checkpoint()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""

    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def run_suite(
    project: ProjectConfig,
    project_root: Path,
    artifact_root: Path,
    *,
    device_name: str = "auto",
    resume: bool = True,
    on_checkpoint: CheckpointCallback | None = None,
) -> list[dict[str, Any]]:
    """Run a preset sequentially, resolving the teacher before KD students."""

    results: list[dict[str, Any]] = []
    teacher_best: Path | None = None
    for run in project.runs:
        result = run_experiment(
            project,
            run,
            project_root,
            artifact_root,
            device_name=device_name,
            teacher_checkpoint=teacher_best,
            resume=resume,
            on_checkpoint=on_checkpoint,
        )
        results.append(result)
        if run.model == "convnext" and run.mode == "standard":
            teacher_best = run_directory(artifact_root, project, run) / "best.pt"
    return results
