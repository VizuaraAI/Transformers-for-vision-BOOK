from dataclasses import replace

import pytest
from conftest import PROJECT_ROOT

from vision_bench.config import RunConfig, load_project_config


@pytest.mark.parametrize(
    ("preset", "expected_runs", "tutorial_only"),
    [("smoke", 7, True), ("quick", 6, True), ("full", 13, False)],
)
def test_presets_expand_and_validate(preset: str, expected_runs: int, tutorial_only: bool) -> None:
    project = load_project_config(preset, PROJECT_ROOT)
    assert len(project.runs) == expected_runs
    assert project.tutorial_only is tutorial_only
    assert len(project.fingerprint) == 64


def test_full_distillation_seeds_are_paired() -> None:
    project = load_project_config("full", PROJECT_ROOT)
    standard = {
        run.seed for run in project.runs if run.model == "deit_distilled" and run.mode == "standard"
    }
    distilled = {
        run.seed for run in project.runs if run.model == "deit_distilled" and run.mode == "hard_kd"
    }
    assert standard == distilled == {42, 43, 44}


def test_fingerprint_changes_with_protocol() -> None:
    project = load_project_config("quick", PROJECT_ROOT)
    changed = replace(project, training=replace(project.training, batch_size=32))
    assert changed.fingerprint != project.fingerprint


def test_duplicate_run_is_rejected() -> None:
    project = load_project_config("quick", PROJECT_ROOT)
    invalid = replace(project, runs=(*project.runs, project.runs[0]))
    with pytest.raises(ValueError, match="must be unique"):
        invalid.validate()


def test_unknown_model_is_rejected() -> None:
    project = load_project_config("quick", PROJECT_ROOT)
    invalid = replace(project, runs=(RunConfig("not-a-model", "standard", 42),))
    with pytest.raises(ValueError, match="Unknown model"):
        invalid.validate()
