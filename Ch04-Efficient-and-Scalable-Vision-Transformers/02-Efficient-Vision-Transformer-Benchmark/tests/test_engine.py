from dataclasses import replace

import pytest
import torch
from conftest import PROJECT_ROOT

from vision_bench.config import TrainingConfig, load_project_config
from vision_bench.engine import create_optimizer, create_scheduler, run_directory


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.token = torch.nn.Parameter(torch.ones(1, 1, 4))
        self.head = torch.nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs))

    def get_classifier(self) -> torch.nn.Module:
        return self.head

    def no_weight_decay(self) -> set[str]:
        return {"token"}


def test_optimizer_uses_head_and_backbone_learning_rates() -> None:
    model = TinyClassifier()
    config = TrainingConfig(backbone_lr=1e-4, head_lr=1e-3)
    optimizer = create_optimizer(model, config)
    by_name = {group["group_name"]: group for group in optimizer.param_groups}
    assert by_name["backbone-decay"]["lr"] == pytest.approx(1e-4)
    assert by_name["head-decay"]["lr"] == pytest.approx(1e-3)
    assert by_name["backbone-no_decay"]["weight_decay"] == 0
    assert by_name["head-no_decay"]["weight_decay"] == 0
    no_decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0
        for parameter in group["params"]
    }
    assert id(model.token) in no_decay_ids


def test_scheduler_warms_then_decays() -> None:
    model = TinyClassifier()
    config = TrainingConfig(epochs=4, warmup_epochs=1, backbone_lr=1e-4, head_lr=1e-3)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=2)
    learning_rates = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] > 0
    assert max(learning_rates[:2]) >= learning_rates[0]
    assert learning_rates[-1] < max(learning_rates)


def test_run_directory_is_configuration_qualified(tmp_path) -> None:
    project = load_project_config("quick", PROJECT_ROOT)
    run = project.runs[0]
    first = run_directory(tmp_path, project, run)
    changed = replace(project, training=replace(project.training, batch_size=32))
    second = run_directory(tmp_path, changed, run)
    assert first != second
    assert run.key in first.name
