"""Exact model registry and model-related helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """One pinned pretrained architecture used by the chapter project."""

    alias: str
    display_name: str
    checkpoint: str
    family: str
    expected_params_m: float
    expected_gmacs: float
    distilled: bool = False
    role: str = "candidate"


MODEL_SPECS: dict[str, ModelSpec] = {
    "vit": ModelSpec(
        "vit",
        "ViT-S/16",
        "vit_small_patch16_224.augreg_in1k",
        "plain global-attention transformer",
        22.1,
        4.3,
    ),
    "deit": ModelSpec(
        "deit",
        "DeiT-S/16",
        "deit_small_patch16_224.fb_in1k",
        "data-efficient ViT recipe",
        22.1,
        4.6,
    ),
    "deit_distilled": ModelSpec(
        "deit_distilled",
        "Distilled DeiT-S/16",
        "deit_small_distilled_patch16_224.fb_in1k",
        "DeiT with class and distillation tokens",
        22.4,
        4.6,
        distilled=True,
    ),
    "swin": ModelSpec(
        "swin",
        "Swin-T",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "hierarchical shifted-window transformer",
        28.3,
        4.5,
    ),
    "efficientformer": ModelSpec(
        "efficientformer",
        "EfficientFormer-L1",
        "efficientformer_l1.snap_dist_in1k",
        "latency-oriented compact transformer",
        12.3,
        1.3,
    ),
    "convnext": ModelSpec(
        "convnext",
        "ConvNeXt-T",
        "convnext_tiny.fb_in1k",
        "modern convolutional network",
        28.6,
        4.5,
        role="teacher/reference",
    ),
}


def get_model_spec(alias: str) -> ModelSpec:
    """Return a model spec or an actionable error listing valid aliases."""

    try:
        return MODEL_SPECS[alias]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{alias}'. Choose one of: {valid}") from exc


def create_model(alias: str, num_classes: int = 100, pretrained: bool = True) -> Any:
    """Instantiate a pinned timm model with a task-specific classifier."""

    import timm

    spec = get_model_spec(alias)
    return timm.create_model(spec.checkpoint, pretrained=pretrained, num_classes=num_classes)


def classifier_parameter_ids(model: Any) -> set[int]:
    """Collect parameter identities belonging to one or more classifier heads."""

    classifier = model.get_classifier()
    modules = classifier if isinstance(classifier, (tuple, list)) else (classifier,)
    return {id(parameter) for module in modules for parameter in module.parameters()}


def parameter_count(model: Any, trainable_only: bool = False) -> int:
    """Count model parameters without relying on model-card metadata."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def preprocessing_config(model: Any) -> dict[str, Any]:
    """Return the checkpoint-native input size, interpolation, mean, and std."""

    cfg = model.pretrained_cfg
    return {
        "input_size": tuple(cfg.get("input_size", (3, 224, 224))),
        "interpolation": cfg.get("interpolation", "bicubic"),
        "mean": tuple(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": tuple(cfg.get("std", (0.229, 0.224, 0.225))),
    }
