"""Transparent DeiT hard token-distillation loss."""

from __future__ import annotations

from typing import Any


class HardTokenDistillationLoss:
    """Combine ground-truth class loss with a frozen teacher's hard target.

    The class-token head learns from labels. The distillation-token head learns
    from the teacher's argmax prediction. The teacher is always evaluated
    without gradient tracking.
    """

    def __init__(self, teacher: Any, label_smoothing: float = 0.1, alpha: float = 0.5):
        import torch

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.teacher = teacher
        self.alpha = alpha
        self.class_criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.distillation_criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, inputs: Any, outputs: Any, labels: Any) -> tuple[Any, dict[str, float]]:
        import torch

        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError(
                "Hard token distillation requires separate (class_head, distillation_head) logits"
            )
        class_logits, distillation_logits = outputs
        classification_loss = self.class_criterion(class_logits, labels)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            teacher_targets = teacher_logits.argmax(dim=1)
        distillation_loss = self.distillation_criterion(distillation_logits, teacher_targets)
        combined = (1 - self.alpha) * classification_loss + self.alpha * distillation_loss
        parts = {
            "classification_loss": float(classification_loss.detach().item()),
            "distillation_loss": float(distillation_loss.detach().item()),
        }
        return combined, parts


def freeze_teacher(teacher: Any) -> Any:
    """Put a teacher in evaluation mode and permanently disable its gradients."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher
