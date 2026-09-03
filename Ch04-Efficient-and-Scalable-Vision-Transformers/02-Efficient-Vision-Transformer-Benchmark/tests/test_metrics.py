import pytest
import torch

from vision_bench.metrics import AverageMeter, averaged_logits, topk_correct


def test_average_meter_is_sample_weighted() -> None:
    meter = AverageMeter()
    meter.update(2.0, 3)
    meter.update(5.0, 1)
    assert meter.average == pytest.approx(2.75)


def test_topk_and_two_head_average() -> None:
    first = torch.tensor([[5.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    second = torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 5.0]])
    logits = averaged_logits((first, second))
    assert torch.equal(logits, (first + second) / 2)
    assert topk_correct(logits, torch.tensor([0, 1]), topk=(1, 2)) == [1, 2]


def test_averaged_logits_rejects_unexpected_tuple() -> None:
    with pytest.raises(ValueError, match="Expected two"):
        averaged_logits((torch.ones(1), torch.ones(1), torch.ones(1)))
