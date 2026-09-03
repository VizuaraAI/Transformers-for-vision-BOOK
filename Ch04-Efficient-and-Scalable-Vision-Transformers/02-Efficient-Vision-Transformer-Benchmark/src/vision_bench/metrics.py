"""Small, dependency-light metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AverageMeter:
    """Track a sample-weighted mean."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, amount: int = 1) -> None:
        self.total += float(value) * amount
        self.count += amount

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def topk_correct(logits: Any, targets: Any, topk: tuple[int, ...] = (1, 5)) -> list[int]:
    """Return correct prediction counts for each requested k."""

    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1, largest=True, sorted=True).indices
    matches = predictions.eq(targets.view(-1, 1))
    return [int(matches[:, : min(k, max_k)].any(dim=1).sum().item()) for k in topk]


def averaged_logits(output: Any) -> Any:
    """Convert a standard tensor or two DeiT heads into inference logits."""

    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(f"Expected two DeiT heads, received {len(output)} outputs")
        return (output[0] + output[1]) / 2
    return output
