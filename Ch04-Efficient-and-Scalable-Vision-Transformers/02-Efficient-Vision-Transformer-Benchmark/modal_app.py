"""Resumable, bounded-concurrency Modal launcher for the full chapter experiment.

Run from this directory with:

    modal run modal_app.py --preset full --stage all
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "vision-transformer-benchmark"
REMOTE_SOURCE_ROOT = Path("/root/project")
VOLUME_ROOT = Path("/vol")
ARTIFACT_ROOT = VOLUME_ROOT / "artifacts"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_dir("src", remote_path=str(REMOTE_SOURCE_ROOT / "src"), copy=True)
    .add_local_dir("configs", remote_path=str(REMOTE_SOURCE_ROOT / "configs"), copy=True)
    .env(
        {
            "PYTHONPATH": str(REMOTE_SOURCE_ROOT / "src"),
            "HF_HOME": str(VOLUME_ROOT / "cache" / "huggingface"),
            "TORCH_HOME": str(VOLUME_ROOT / "cache" / "torch"),
        }
    )
    .workdir(str(REMOTE_SOURCE_ROOT))
)
retry_policy = modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0)


def _load_remote_config(preset: str) -> Any:
    from vision_bench.config import load_project_config

    return load_project_config(REMOTE_SOURCE_ROOT / "configs" / f"{preset}.yaml")


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=32768,
    volumes={str(VOLUME_ROOT): volume},
    timeout=24 * 60 * 60,
    retries=retry_policy,
    max_containers=4,
    single_use_containers=True,
)
def train_remote(preset: str, model: str, mode: str, seed: int) -> dict[str, Any]:
    """Train one configuration; retries resume from its latest committed epoch."""

    from vision_bench.engine import run_directory, run_experiment

    volume.reload()
    project = _load_remote_config(preset)
    matches = [
        run for run in project.runs if run.model == model and run.mode == mode and run.seed == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique configured run for {model}/{mode}/seed {seed}")
    run = matches[0]
    teacher_checkpoint = None
    if mode == "hard_kd":
        teacher = next(
            candidate
            for candidate in project.runs
            if candidate.model == "convnext" and candidate.mode == "standard"
        )
        teacher_checkpoint = run_directory(ARTIFACT_ROOT, project, teacher) / "best.pt"
    result = run_experiment(
        project,
        run,
        VOLUME_ROOT,
        ARTIFACT_ROOT,
        device_name="cuda",
        teacher_checkpoint=teacher_checkpoint,
        resume=True,
        on_checkpoint=volume.commit,
    )
    volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=8.0,
    memory=32768,
    volumes={str(VOLUME_ROOT): volume},
    timeout=6 * 60 * 60,
    retries=retry_policy,
    single_use_containers=True,
)
def benchmark_remote(preset: str) -> dict[str, Any]:
    """Benchmark every architecture sequentially on one L4 container."""

    from vision_bench.benchmark import benchmark_suite

    volume.reload()
    result = benchmark_suite(_load_remote_config(preset), ARTIFACT_ROOT, device_name="cuda")
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    volumes={str(VOLUME_ROOT): volume},
    timeout=60 * 60,
    retries=retry_policy,
    single_use_containers=True,
)
def report_remote(preset: str) -> str:
    """Create tables and figures on a CPU container."""

    from vision_bench.reporting import generate_report

    volume.reload()
    path = generate_report(_load_remote_config(preset), ARTIFACT_ROOT)
    volume.commit()
    return str(path)


@app.local_entrypoint()
def main(preset: str = "full", stage: str = "all") -> None:
    """Run train, benchmark, report, or all stages."""

    from vision_bench.config import load_project_config

    allowed = {"train", "benchmark", "report", "all"}
    if stage not in allowed:
        raise ValueError(f"stage must be one of: {', '.join(sorted(allowed))}")
    local_project = load_project_config(Path("configs") / f"{preset}.yaml")
    outputs: dict[str, Any] = {}
    if stage in {"train", "all"}:
        teacher = next(
            run for run in local_project.runs if run.model == "convnext" and run.mode == "standard"
        )
        outputs["teacher"] = train_remote.remote(preset, teacher.model, teacher.mode, teacher.seed)
        jobs = [
            (preset, run.model, run.mode, run.seed) for run in local_project.runs if run != teacher
        ]
        outputs["students"] = list(
            train_remote.starmap(jobs, order_outputs=False, return_exceptions=False)
        )
    if stage in {"benchmark", "all"}:
        benchmark = benchmark_remote.remote(preset)
        outputs["benchmark_measurements"] = len(benchmark["measurements"])
    if stage in {"report", "all"}:
        outputs["report"] = report_remote.remote(preset)
    print(json.dumps(outputs, indent=2, sort_keys=True))
