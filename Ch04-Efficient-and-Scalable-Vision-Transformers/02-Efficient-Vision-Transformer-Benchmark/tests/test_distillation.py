import pytest
import torch

from vision_bench.distillation import HardTokenDistillationLoss, freeze_teacher


class TinyTeacher(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def test_hard_distillation_combines_losses_and_blocks_teacher_gradients() -> None:
    teacher = TinyTeacher()
    teacher.linear.weight.data.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]]))
    freeze_teacher(teacher)
    loss_function = HardTokenDistillationLoss(teacher, label_smoothing=0.0, alpha=0.5)
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([2, 2])
    class_logits = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], requires_grad=True)
    distill_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], requires_grad=True)

    loss, parts = loss_function(inputs, (class_logits, distill_logits), labels)
    expected = 0.5 * torch.nn.functional.cross_entropy(
        class_logits, labels
    ) + 0.5 * torch.nn.functional.cross_entropy(distill_logits, torch.tensor([0, 1]))
    assert loss.item() == pytest.approx(expected.item())
    assert set(parts) == {"classification_loss", "distillation_loss"}
    loss.backward()
    assert class_logits.grad is not None
    assert distill_logits.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_hard_distillation_requires_two_student_heads() -> None:
    loss_function = HardTokenDistillationLoss(TinyTeacher())
    with pytest.raises(ValueError, match="separate"):
        loss_function(torch.ones(1, 2), torch.ones(1, 3), torch.zeros(1, dtype=torch.long))
