import torch
from torch.utils.data import DataLoader, TensorDataset

from vision_bench.config import TrainingConfig
from vision_bench.engine import (
    _step_optimizer_and_scheduler,
    create_optimizer,
    create_scheduler,
    evaluate,
    train_one_epoch,
)


class TinyImageClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 8))
        self.head = torch.nn.Linear(8, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.backbone(inputs)))

    def get_classifier(self) -> torch.nn.Module:
        return self.head


class FakeScaler:
    def __init__(self, *, scale_before: float, scale_after: float) -> None:
        self.scale = scale_before
        self.scale_after = scale_after

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: object) -> None:
        del optimizer

    def update(self) -> None:
        self.scale = self.scale_after


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_scheduler_does_not_advance_when_amp_skips_optimizer_step() -> None:
    scaler = FakeScaler(scale_before=65536.0, scale_after=32768.0)
    scheduler = CountingScheduler()

    optimizer_stepped = _step_optimizer_and_scheduler(scaler, object(), scheduler)

    assert optimizer_stepped is False
    assert scheduler.steps == 0


def test_scheduler_advances_after_successful_amp_optimizer_step() -> None:
    scaler = FakeScaler(scale_before=32768.0, scale_after=32768.0)
    scheduler = CountingScheduler()

    optimizer_stepped = _step_optimizer_and_scheduler(scaler, object(), scheduler)

    assert optimizer_stepped is True
    assert scheduler.steps == 1


def test_one_epoch_training_and_evaluation_on_cpu() -> None:
    torch.manual_seed(4)
    images = torch.randn(6, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    model = TinyImageClassifier()
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        warmup_epochs=0,
        amp=False,
        label_smoothing=0.0,
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    before = model.head.weight.detach().clone()
    train_result = train_one_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        config,
    )
    evaluation, predictions = evaluate(model, loader, torch.device("cpu"), return_predictions=True)

    assert train_result.samples == 6
    assert train_result.loss > 0
    assert 0 <= train_result.top1 <= 100
    assert not torch.equal(before, model.head.weight)
    assert evaluation.samples == 6
    assert predictions is not None
    assert predictions["predictions"].shape == (6,)
