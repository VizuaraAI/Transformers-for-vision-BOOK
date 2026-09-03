import json

from vision_bench.checkpointing import atomic_json, write_jsonl
from vision_bench.config import (
    BenchmarkConfig,
    DataConfig,
    ProjectConfig,
    RunConfig,
    TrainingConfig,
)
from vision_bench.engine import run_directory
from vision_bench.reporting import (
    aggregate_results,
    generate_report,
    markdown_table,
    paired_distillation_gains,
)


def _project() -> ProjectConfig:
    project = ProjectConfig(
        name="test",
        tutorial_only=True,
        description="test fixture",
        pretrained=False,
        data=DataConfig(train_per_class=1, val_per_class=1, test_limit=100),
        training=TrainingConfig(epochs=1, warmup_epochs=0),
        benchmark=BenchmarkConfig(
            precisions=("fp32",),
            batch_sizes=(1,),
            warmup_iterations=1,
            timed_iterations=1,
            repeats=1,
        ),
        runs=(RunConfig("vit", "standard", 42),),
    )
    project.validate()
    return project


def test_aggregation_and_paired_gain() -> None:
    rows = [
        {
            "model": "deit_distilled",
            "display_name": "D",
            "mode": "standard",
            "seed": 42,
            "test_top1": 70.0,
            "test_top5": 90.0,
            "best_validation_top1": 69.0,
            "parameters_m": 22.0,
            "training_seconds": 10.0,
            "training_peak_memory_mb": 100.0,
        },
        {
            "model": "deit_distilled",
            "display_name": "D",
            "mode": "hard_kd",
            "seed": 42,
            "test_top1": 72.0,
            "test_top5": 91.0,
            "best_validation_top1": 71.0,
            "parameters_m": 22.0,
            "training_seconds": 12.0,
            "training_peak_memory_mb": 110.0,
        },
    ]
    assert len(aggregate_results(rows)) == 2
    gains = paired_distillation_gains(rows)
    assert gains[0]["gain_percentage_points"] == 2.0


def test_generate_report_from_artifacts(tmp_path) -> None:
    project = _project()
    run = project.runs[0]
    directory = run_directory(tmp_path, project, run)
    summary = {
        "display_name": "ViT-S/16",
        "best_epoch": 1,
        "best_validation_top1": 50.0,
        "test": {"top1": 49.0, "top5": 80.0},
        "parameters": 1_000_000,
    }
    atomic_json(directory / "complete.json", summary)
    write_jsonl(
        directory / "metrics.jsonl",
        [
            {
                "epoch": 1,
                "train": {"loss": 2.0, "top1": 30.0, "peak_memory_mb": 100.0},
                "validation": {"loss": 1.5, "top1": 50.0},
                "cumulative_seconds": 10.0,
            }
        ],
    )
    report_path = generate_report(project, tmp_path)
    assert report_path.exists()
    assert "Tutorial-only result" in report_path.read_text(encoding="utf-8")
    report_summary = json.loads(
        (report_path.parent / "report_summary.json").read_text(encoding="utf-8")
    )
    assert report_summary["completed_runs"] == 1


def test_markdown_table_escapes_pipes() -> None:
    rendered = markdown_table([{"value": "a|b"}], [("value", "Value")])
    assert "a\\|b" in rendered
