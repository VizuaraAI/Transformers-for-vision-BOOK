"""Artifact collection, tables, figures, and a reproducible Markdown report."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from vision_bench.checkpointing import atomic_json, load_jsonl
from vision_bench.config import ProjectConfig
from vision_bench.engine import load_json, run_directory


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.2f}"
    return str(value)


def markdown_table(rows: Iterable[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Render a compact GitHub-flavored Markdown table."""

    materialized = list(rows)
    if not materialized:
        return "_No rows available._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in materialized:
        cells = [_format_value(row.get(key)).replace("|", "\\|") for key, _ in columns]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def collect_run_results(
    project: ProjectConfig, artifact_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Collect completed run summaries and per-epoch histories."""

    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in project.runs:
        directory = run_directory(artifact_root, project, run)
        complete_path = directory / "complete.json"
        if not complete_path.exists():
            missing.append(run.key)
            continue
        summary = load_json(complete_path)
        summary_fingerprint = summary.get("config_fingerprint")
        if summary_fingerprint is not None and summary_fingerprint != project.fingerprint:
            raise ValueError(f"Run summary fingerprint mismatch: {complete_path}")
        metrics = load_jsonl(directory / "metrics.jsonl")
        cumulative_seconds = float(metrics[-1]["cumulative_seconds"]) if metrics else None
        train_peaks = [
            record.get("train", {}).get("peak_memory_mb")
            for record in metrics
            if record.get("train", {}).get("peak_memory_mb") is not None
        ]
        row = {
            "run_key": run.key,
            "model": run.model,
            "display_name": summary["display_name"],
            "mode": run.mode,
            "seed": run.seed,
            "best_epoch": summary["best_epoch"],
            "best_validation_top1": summary["best_validation_top1"],
            "test_top1": summary["test"]["top1"],
            "test_top5": summary["test"]["top5"],
            "parameters": summary["parameters"],
            "parameters_m": summary["parameters"] / 1e6,
            "training_seconds": cumulative_seconds,
            "training_peak_memory_mb": max(train_peaks) if train_peaks else None,
            "run_directory": str(directory),
        }
        rows.append(row)
        for record in metrics:
            histories.append(
                {
                    "run_key": run.key,
                    "model": run.model,
                    "display_name": summary["display_name"],
                    "mode": run.mode,
                    "seed": run.seed,
                    "epoch": record["epoch"],
                    "train_loss": record["train"]["loss"],
                    "train_top1": record["train"]["top1"],
                    "validation_loss": record["validation"]["loss"],
                    "validation_top1": record["validation"]["top1"],
                    "cumulative_seconds": record["cumulative_seconds"],
                    "classification_loss": record["train"].get("classification_loss"),
                    "distillation_loss": record["train"].get("distillation_loss"),
                }
            )
    return rows, histories, missing


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate test accuracy by architecture and training mode."""

    import pandas as pd

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["model", "display_name", "mode"], sort=False, dropna=False)
    aggregate = grouped.agg(
        seeds=("seed", "count"),
        test_top1_mean=("test_top1", "mean"),
        test_top1_std=("test_top1", "std"),
        test_top5_mean=("test_top5", "mean"),
        best_validation_top1_mean=("best_validation_top1", "mean"),
        parameters_m=("parameters_m", "first"),
        training_seconds_mean=("training_seconds", "mean"),
        training_peak_memory_mb=("training_peak_memory_mb", "mean"),
    ).reset_index()
    return cast(list[dict[str, Any]], aggregate.to_dict(orient="records"))


def paired_distillation_gains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate the isolated hard-KD gain over the same distilled model."""

    standard = {
        int(row["seed"]): row
        for row in rows
        if row["model"] == "deit_distilled" and row["mode"] == "standard"
    }
    distilled = {
        int(row["seed"]): row
        for row in rows
        if row["model"] == "deit_distilled" and row["mode"] == "hard_kd"
    }
    return [
        {
            "seed": seed,
            "standard_top1": standard[seed]["test_top1"],
            "hard_kd_top1": distilled[seed]["test_top1"],
            "gain_percentage_points": distilled[seed]["test_top1"] - standard[seed]["test_top1"],
        }
        for seed in sorted(set(standard) & set(distilled))
    ]


def _load_benchmark(project: ProjectConfig, artifact_root: Path) -> dict[str, Any] | None:
    path = artifact_root / project.name / "benchmark.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    payload = cast(dict[str, Any], value)
    if payload.get("config_fingerprint") != project.fingerprint:
        raise ValueError("Benchmark artifact was produced with a different preset fingerprint")
    return payload


def _reference_throughput(benchmark: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if benchmark is None:
        return {}
    complete = [row for row in benchmark["measurements"] if row["status"] == "complete"]
    selected: dict[str, dict[str, Any]] = {}
    for alias in dict.fromkeys(row["model"] for row in complete):
        candidates = [row for row in complete if row["model"] == alias]
        candidates.sort(
            key=lambda row: (
                row["precision"] != "fp16",
                -int(row["batch_size"]),
            )
        )
        selected[alias] = candidates[0]
    return selected


def _save_figures(
    report_dir: Path,
    histories: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    gains: list[dict[str, Any]],
    throughput: dict[str, dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if histories:
        frame = pd.DataFrame(histories)
        fig, axis = plt.subplots(figsize=(10, 6))
        for run_key, group in frame.groupby("run_key", sort=False):
            axis.plot(group["epoch"], group["validation_top1"], label=run_key, alpha=0.8)
        axis.set(title="Validation accuracy during fine-tuning", xlabel="Epoch", ylabel="Top-1 (%)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = figure_dir / "validation_accuracy.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path.name)

    if gains:
        frame = pd.DataFrame(gains)
        fig, axis = plt.subplots(figsize=(7, 5))
        for _, row in frame.iterrows():
            axis.plot(
                ["Standard fine-tuning", "Hard distillation"],
                [row["standard_top1"], row["hard_kd_top1"]],
                marker="o",
                label=f"seed {int(row['seed'])}",
            )
        axis.set(title="Paired DeiT distillation comparison", ylabel="Test top-1 (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = figure_dir / "distillation_gain.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path.name)

    scatter_rows = []
    for row in aggregate:
        timing = throughput.get(str(row["model"]))
        if timing is not None:
            scatter_rows.append(
                {
                    **row,
                    "throughput": timing["throughput_images_per_second"],
                    "timing_label": f"{timing['precision']}, batch {timing['batch_size']}",
                }
            )
    if scatter_rows:
        fig, axis = plt.subplots(figsize=(9, 6))
        for row in scatter_rows:
            label = f"{row['display_name']} ({row['mode']})"
            axis.scatter(
                row["throughput"],
                row["test_top1_mean"],
                s=max(40, float(row["parameters_m"]) * 8),
                alpha=0.75,
            )
            axis.annotate(
                label,
                (row["throughput"], row["test_top1_mean"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set(
            title="Accuracy-throughput trade-off",
            xlabel="Inference throughput (images/s)",
            ylabel="Mean test top-1 (%)",
        )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = figure_dir / "accuracy_throughput.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path.name)
    return created


def generate_report(project: ProjectConfig, artifact_root: Path) -> Path:
    """Generate CSV data, figures, JSON summary, and a Markdown report."""

    import pandas as pd

    rows, histories, missing = collect_run_results(project, artifact_root)
    if not rows:
        raise FileNotFoundError(
            f"No completed runs found under {artifact_root / project.name}. Train first."
        )
    aggregate = aggregate_results(rows)
    gains = paired_distillation_gains(rows)
    benchmark = _load_benchmark(project, artifact_root)
    throughput = _reference_throughput(benchmark)
    report_dir = artifact_root / project.name / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "run_results.csv", index=False)
    pd.DataFrame(histories).to_csv(report_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(aggregate).to_csv(report_dir / "aggregate_results.csv", index=False)
    pd.DataFrame(gains).to_csv(report_dir / "distillation_gains.csv", index=False)
    if benchmark is not None:
        pd.DataFrame(benchmark["profiles"]).to_csv(report_dir / "model_profiles.csv", index=False)
        pd.DataFrame(benchmark["measurements"]).to_csv(
            report_dir / "inference_benchmark.csv", index=False
        )
    figures = _save_figures(report_dir, histories, aggregate, gains, throughput)

    comparison_rows = []
    for row in aggregate:
        timing = throughput.get(str(row["model"]))
        comparison_rows.append(
            {
                **row,
                "test_top1": (
                    f"{row['test_top1_mean']:.2f} ± {row['test_top1_std']:.2f}"
                    if int(row["seeds"]) > 1
                    else f"{row['test_top1_mean']:.2f}"
                ),
                "throughput": (
                    timing["throughput_images_per_second"] if timing is not None else None
                ),
                "timing": (
                    f"{timing['precision']}, batch {timing['batch_size']}"
                    if timing is not None
                    else None
                ),
            }
        )

    tutorial_warning = (
        "> **Tutorial-only result.** This preset uses fewer images and/or epochs. "
        "Do not present it as the chapter's final comparison.\n\n"
        if project.tutorial_only
        else ""
    )
    missing_text = (
        "\n\nIncomplete configured runs: " + ", ".join(f"`{key}`" for key in missing) + "."
        if missing
        else ""
    )
    gain_text = markdown_table(
        gains,
        [
            ("seed", "Seed"),
            ("standard_top1", "Standard top-1"),
            ("hard_kd_top1", "Hard-KD top-1"),
            ("gain_percentage_points", "Gain (points)"),
        ],
    )
    environment_text = "_Run `vision-bench benchmark` to add hardware measurements._"
    if benchmark is not None:
        environment = benchmark["environment"]
        environment_text = (
            f"Device: **{environment.get('device_name', environment['device_type'])}**; "
            f"PyTorch {environment['torch']}; precision and batch size are shown per row. "
            "Timing excludes data loading and includes synchronized model forward passes."
        )
    figure_text = "\n".join(
        f"![{name.removesuffix('.png').replace('_', ' ').title()}](figures/{name})"
        for name in figures
    )
    report = f"""# {project.name.title()} vision-transformer benchmark

{tutorial_warning}Configuration fingerprint: `{project.fingerprint}`.

## Main comparison

{
        markdown_table(
            comparison_rows,
            [
                ("display_name", "Model"),
                ("mode", "Training"),
                ("seeds", "Seeds"),
                ("test_top1", "Test top-1 (%)"),
                ("parameters_m", "Parameters (M)"),
                ("throughput", "Images/s"),
                ("timing", "Timing setting"),
            ],
        )
    }

A `±` value is the sample standard deviation across seeds; single-seed rows have no
uncertainty estimate. Throughput is an architecture property here, so the same timing is
reused for standard and hard-distilled runs of the same Distilled DeiT model.{missing_text}

## Isolated training-time distillation gain

This paired comparison holds the Distilled DeiT architecture and seed fixed. It isolates
the effect of using the ConvNeXt teacher during fine-tuning; comparing ordinary DeiT with
Distilled DeiT would also change the checkpoint and architecture.

{gain_text}

## Hardware protocol

{environment_text}

The operation counter reports MAC-style fvcore operations (one multiply-add is counted as
one), plus any unsupported operators in `model_profiles.csv`. Peak memory is allocator
memory, not whole-system power or memory use.

## Training behavior and trade-offs

{figure_text or "_No figures were generated._"}

## Reproducibility files

- `run_results.csv`: one row per seed and training mode
- `epoch_metrics.csv`: learning curves and elapsed training time
- `aggregate_results.csv`: grouped accuracy statistics
- `distillation_gains.csv`: paired hard-distillation differences
- `model_profiles.csv` and `inference_benchmark.csv`: created after hardware benchmarking
- each run directory: resolved configuration, checkpoints, predictions, and environment data

Select checkpoints with validation accuracy only. The official CIFAR-100 test split is
evaluated after training and is not used for early stopping or hyperparameter selection.
"""
    report_path = report_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    atomic_json(
        report_dir / "report_summary.json",
        {
            "schema_version": 1,
            "preset": project.name,
            "tutorial_only": project.tutorial_only,
            "config_fingerprint": project.fingerprint,
            "completed_runs": len(rows),
            "missing_runs": missing,
            "aggregate": aggregate,
            "distillation_gains": gains,
            "figures": figures,
        },
    )
    return report_path
