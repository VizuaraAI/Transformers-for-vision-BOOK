"""Command-line interface used by readers, notebooks, and cloud jobs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vision_bench.benchmark import benchmark_suite
from vision_bench.config import ProjectConfig, RunConfig, load_project_config
from vision_bench.data import prepare_cifar100
from vision_bench.doctor import doctor_report, synthetic_smoke_test
from vision_bench.engine import run_directory, run_experiment, run_suite
from vision_bench.models import MODEL_SPECS
from vision_bench.reporting import generate_report


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    project_root = Path(args.project_root).resolve()
    artifact_root = Path(args.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    return project_root, artifact_root.resolve()


def _load(args: argparse.Namespace, project_root: Path) -> ProjectConfig:
    return load_project_config(args.preset, project_root)


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _find_run(project: ProjectConfig, model: str, mode: str, seed: int) -> RunConfig:
    matches = [
        run for run in project.runs if run.model == model and run.mode == mode and run.seed == seed
    ]
    if len(matches) != 1:
        available = ", ".join(run.key for run in project.runs)
        raise ValueError(
            f"Preset '{project.name}' has no unique run for {model}/{mode}/seed {seed}. "
            f"Available: {available}"
        )
    return matches[0]


def _teacher_checkpoint(project: ProjectConfig, artifact_root: Path, run: RunConfig) -> Path | None:
    if run.mode != "hard_kd":
        return None
    teacher = next(
        candidate
        for candidate in project.runs
        if candidate.model == "convnext" and candidate.mode == "standard"
    )
    return run_directory(artifact_root, project, teacher) / "best.pt"


def _add_project_options(parser: argparse.ArgumentParser, *, include_device: bool = False) -> None:
    parser.add_argument("--preset", default="quick", help="Preset name or YAML path")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project directory containing configs/ (default: current directory)",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts",
        help="Artifact directory, relative to project root unless absolute",
    )
    if include_device:
        parser.add_argument(
            "--device", default="auto", help="auto, cpu, mps, cuda, or a device such as cuda:0"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the documented command-line parser."""

    parser = argparse.ArgumentParser(
        prog="vision-bench",
        description="Book-ready ViT, DeiT, Swin, and EfficientFormer comparison",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check Python, packages, devices, and disk"
    )
    doctor_parser.add_argument("--project-root", default=".")

    subparsers.add_parser("list-models", help="Show exact timm checkpoints and model roles")

    config_parser = subparsers.add_parser("show-config", help="Print a resolved preset")
    _add_project_options(config_parser)

    data_parser = subparsers.add_parser("prepare-data", help="Download CIFAR-100 and freeze splits")
    _add_project_options(data_parser)

    smoke_parser = subparsers.add_parser(
        "smoke", help="Run one synthetic forward pass through each architecture"
    )
    smoke_parser.add_argument(
        "--models", default=",".join(MODEL_SPECS), help="Comma-separated model aliases"
    )
    smoke_parser.add_argument("--device", default="auto")
    smoke_parser.add_argument("--input-size", type=int, default=224)

    train_parser = subparsers.add_parser("train", help="Train or resume one configured run")
    _add_project_options(train_parser, include_device=True)
    train_parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    train_parser.add_argument("--mode", default="standard", choices=["standard", "hard_kd"])
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-resume", action="store_true")

    suite_parser = subparsers.add_parser("suite", help="Train or resume every run in a preset")
    _add_project_options(suite_parser, include_device=True)
    suite_parser.add_argument("--no-resume", action="store_true")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Profile trained checkpoints on one device"
    )
    _add_project_options(benchmark_parser, include_device=True)

    report_parser = subparsers.add_parser("report", help="Build CSVs, figures, and report.md")
    _add_project_options(report_parser)

    all_parser = subparsers.add_parser(
        "all", help="Run training, hardware benchmark, and report in sequence"
    )
    _add_project_options(all_parser, include_device=True)
    all_parser.add_argument("--no-resume", action="store_true")
    return parser


def _run_command(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        _json_print(doctor_report(Path(args.project_root)))
        return 0
    if args.command == "list-models":
        _json_print({alias: spec.__dict__ for alias, spec in MODEL_SPECS.items()})
        return 0
    if args.command == "smoke":
        aliases = [value.strip() for value in args.models.split(",") if value.strip()]
        _json_print(synthetic_smoke_test(aliases, args.device, args.input_size))
        return 0

    project_root, artifact_root = _paths(args)
    project = _load(args, project_root)
    if args.command == "show-config":
        _json_print({**project.to_dict(), "fingerprint": project.fingerprint})
        return 0
    if args.command == "prepare-data":
        data_root, manifest = prepare_cifar100(project.data, project_root)
        _json_print(
            {
                "data_root": str(data_root),
                "train_images": len(manifest.train_indices),
                "validation_images": len(manifest.val_indices),
                "test_images": len(manifest.test_indices),
                "split_checksum": manifest.checksum,
            }
        )
        return 0
    if args.command == "train":
        run = _find_run(project, args.model, args.mode, args.seed)
        result = run_experiment(
            project,
            run,
            project_root,
            artifact_root,
            device_name=args.device,
            teacher_checkpoint=_teacher_checkpoint(project, artifact_root, run),
            resume=not args.no_resume,
        )
        _json_print(result)
        return 0
    if args.command == "suite":
        _json_print(
            run_suite(
                project,
                project_root,
                artifact_root,
                device_name=args.device,
                resume=not args.no_resume,
            )
        )
        return 0
    if args.command == "benchmark":
        _json_print(benchmark_suite(project, artifact_root, device_name=args.device))
        return 0
    if args.command == "report":
        print(generate_report(project, artifact_root))
        return 0
    if args.command == "all":
        run_suite(
            project,
            project_root,
            artifact_root,
            device_name=args.device,
            resume=not args.no_resume,
        )
        benchmark_suite(project, artifact_root, device_name=args.device)
        print(generate_report(project, artifact_root))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    return _run_command(build_parser().parse_args(argv))
