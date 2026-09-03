#!/usr/bin/env python3
"""Build the Chapter 4 Hugging Face upload bundle.

The source ``best.pt`` checkpoints contain optimizer and resume state. This exporter validates
their recorded SHA-256 hashes, extracts only ``model_state``, writes safe model-only Safetensors,
and generates a provenance manifest for the book repository.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vision_bench.checkpointing import load_checkpoint
from vision_bench.models import get_model_spec

BOOK_TITLE = "Transformer Vision: From Image, Video to World Models & Robotics"
REPOSITORY_ID = "Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book"
CHAPTER_DIRECTORY = "chapter-04-efficient-and-scalable-vision-transformers"
EXPECTED_FINGERPRINT = "7b87d57ae5546600c460245a81431288db34c3e40a44c2e83da01acc4160d731"
MODEL_DIRECTORIES = {
    "convnext": "convnext-t",
    "vit": "vit-s16",
    "swin": "swin-t",
    "efficientformer": "efficientformer-l1",
    "deit": "deit-s16",
    "deit_distilled": "deit-distilled-s16",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Vision benchmark project root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Completed result bundle; defaults below project-root.",
    )
    parser.add_argument(
        "--cards-root",
        type=Path,
        default=None,
        help="Hugging Face model-card sources; defaults below project-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory to create as the upload root.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    """Read a conventional two-column SHA-256 manifest."""

    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum at {path}:{line_number}")
        digest, relative_path = parts
        checksums[relative_path.lstrip("* ")] = digest
    return checksums


def read_run_rows(path: Path) -> list[dict[str, str]]:
    """Load and validate the completed per-run table."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 13:
        raise ValueError(f"Expected 13 completed runs, found {len(rows)} in {path}")
    required = {
        "run_key",
        "model",
        "display_name",
        "mode",
        "seed",
        "best_epoch",
        "best_validation_top1",
        "test_top1",
        "test_top5",
        "parameters",
        "training_seconds",
        "training_peak_memory_mb",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing run-results columns: {', '.join(sorted(missing))}")
    return rows


def one_source_directory(checkpoint_root: Path, run_key: str) -> Path:
    """Resolve exactly one fingerprinted source directory for a run key."""

    matches = sorted(path for path in checkpoint_root.glob(f"{run_key}-*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint directory for {run_key}, found {matches}")
    return matches[0]


def model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return a contiguous CPU tensor mapping suitable for Safetensors."""

    raw_state = checkpoint.get("model_state")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("Checkpoint has no non-empty model_state mapping")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Invalid model_state entry: {name!r} ({type(value).__name__})")
        state[name] = value.detach().to(device="cpu").contiguous()
    return state


def verify_safetensors(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    """Verify exported keys and tensor shapes without unpickling any data."""

    with safe_open(path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        expected_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Safetensors key mismatch for {path}: missing={missing}, extra={extra}"
            )
        for name, tensor in state.items():
            exported_shape = tuple(handle.get_slice(name).get_shape())
            if exported_shape != tuple(tensor.shape):
                raise ValueError(
                    f"Safetensors shape mismatch for {path}:{name}: "
                    f"{exported_shape} != {tuple(tensor.shape)}"
                )


def copy_supporting_artifacts(
    project_root: Path,
    results_root: Path,
    cards_root: Path,
    output_root: Path,
) -> Path:
    """Copy model cards, configuration, raw tables, and figures into the bundle."""

    chapter_output = output_root / CHAPTER_DIRECTORY
    (chapter_output / "configs").mkdir(parents=True)
    (chapter_output / "results").mkdir(parents=True)
    shutil.copy2(cards_root / "README.md", output_root / "README.md")
    shutil.copy2(cards_root / CHAPTER_DIRECTORY / "README.md", chapter_output / "README.md")
    shutil.copy2(project_root / "configs" / "full.yaml", chapter_output / "configs" / "full.yaml")
    shutil.copy2(results_root / "benchmark.json", chapter_output / "results" / "benchmark.json")
    shutil.copytree(
        results_root / "report",
        chapter_output / "results",
        dirs_exist_ok=True,
    )
    return chapter_output


def huggingface_loading_section() -> str:
    """Return the Hugging Face-specific replacement for the report's loading section."""

    return """## 14. Loading a downloaded checkpoint

The Hugging Face files contain model parameters only and use Safetensors. This example downloads and
restores the validation-selected ViT on CPU:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import timm

repo_id = (
    "Mayank022/"
    "Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book"
)
filename = (
    "chapter-04-efficient-and-scalable-vision-transformers/"
    "checkpoints/vit-s16/standard/seed-42/model.safetensors"
)
weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
state_dict = load_file(weights_path, device="cpu")

model = timm.create_model(
    "vit_small_patch16_224.augreg_in1k",
    pretrained=False,
    num_classes=100,
)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

For `deit-distilled-s16`, disable distilled-training output before ordinary evaluation if the
installed `timm` model exposes the switch:

```python
if hasattr(model, "set_distilled_training"):
    model.set_distilled_training(False)
```

Use `manifest.json` to map each published path to its exact `timm` identifier, training mode, seed,
metrics, and source checksum. Use checkpoint-native preprocessing as shown in this chapter's
README. These model-only exports cannot resume training because optimizer, scheduler, scaler, and
random-number-generator state intentionally remain in the controlled source checkpoints.
"""


def write_adapted_report(source: Path, destination: Path, exported_bytes: int) -> None:
    """Adapt local artifact links and checkpoint instructions for the Hub bundle."""

    text = source.read_text(encoding="utf-8")
    replacements = {
        "modal_full_results_20260816/report/": "results/",
        "modal_full_results_20260816/benchmark.json": "results/benchmark.json",
        "modal_full_results_20260816/checkpoints/SHA256SUMS": "checkpoints/SHA256SUMS",
        "modal_full_results_20260816/checkpoints": "checkpoints",
        "[`modal_full_results_20260816`](modal_full_results_20260816)": ("[`results`](results)"),
        "[`src/vision_bench/models.py`](src/vision_bench/models.py)": (
            "[the chapter model table](README.md#exact-upstream-model-identifiers)"
        ),
        "| [`checkpoints`](checkpoints) | Local validation-selected `best.pt` files |": (
            "| [`checkpoints`](checkpoints) | Exported model-only Safetensors files |"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    source_paragraph = """After all 13 public Hugging Face files passed path, byte-size, and LFS SHA-256 verification, the
local `checkpoints` directory was deleted at the user's request. The
book project workspace now contains no `.pt` or `.safetensors` weight files. The Modal volume remains
the durable source for complete resumable `best.pt` and `latest.pt` run artifacts."""
    exported_gib = exported_bytes / 1024**3
    hub_paragraph = f"""The Hugging Face checkpoint directory contains 13 model-only
`model.safetensors` files totaling approximately {exported_gib:.2f} GiB. Their local source and
temporary export copies were removed after public hash verification. Complete resumable `best.pt`
and `latest.pt` artifacts remain on the Modal volume."""
    if source_paragraph not in text:
        raise ValueError("Could not find the source-checkpoint paragraph in EXPERIMENT_REPORT.md")
    text = text.replace(source_paragraph, hub_paragraph)

    section_pattern = re.compile(
        r"## 14\. Loading a downloaded checkpoint\n.*?(?=\n## 15\. Final takeaway)",
        flags=re.DOTALL,
    )
    text, replacement_count = section_pattern.subn(huggingface_loading_section().rstrip(), text)
    if replacement_count != 1:
        raise ValueError(f"Expected to replace one loading section, replaced {replacement_count}")
    destination.write_text(text, encoding="utf-8")


def export_weights(
    rows: list[dict[str, str]],
    checkpoint_root: Path,
    chapter_output: Path,
    source_checksums: Mapping[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Export every validation-selected model state and return manifest records."""

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    checksum_lines: list[str] = []
    for index, row in enumerate(rows, start=1):
        run_key = row["run_key"]
        source_directory = one_source_directory(checkpoint_root, run_key)
        source_path = source_directory / "best.pt"
        source_relative = str(source_path.relative_to(checkpoint_root))
        expected_source_sha = source_checksums.get(source_relative)
        if expected_source_sha is None:
            raise ValueError(f"No source checksum recorded for {source_relative}")
        actual_source_sha = sha256_file(source_path)
        if actual_source_sha != expected_source_sha:
            raise ValueError(
                f"Source checksum mismatch for {source_relative}: "
                f"{actual_source_sha} != {expected_source_sha}"
            )

        checkpoint = load_checkpoint(source_path, map_location="cpu")
        fingerprint = checkpoint.get("config_fingerprint")
        if fingerprint != EXPECTED_FINGERPRINT:
            raise ValueError(f"Configuration fingerprint mismatch in {source_path}: {fingerprint}")
        recorded_best_epoch = int(row["best_epoch"])
        if int(checkpoint["best_epoch"]) + 1 != recorded_best_epoch:
            raise ValueError(f"Best-epoch mismatch for {run_key}")
        if abs(float(checkpoint["best_top1"]) - float(row["best_validation_top1"])) > 1e-8:
            raise ValueError(f"Best-validation-accuracy mismatch for {run_key}")

        alias = row["model"]
        spec = get_model_spec(alias)
        model_directory = MODEL_DIRECTORIES[alias]
        mode_directory = row["mode"].replace("_", "-")
        seed = int(row["seed"])
        relative_model_path = (
            Path("checkpoints")
            / model_directory
            / mode_directory
            / f"seed-{seed}"
            / "model.safetensors"
        )
        destination = chapter_output / relative_model_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        state = model_state(checkpoint)
        metadata = {
            "book": BOOK_TITLE,
            "chapter": "4",
            "model_alias": alias,
            "display_name": spec.display_name,
            "timm_checkpoint": spec.checkpoint,
            "training_mode": row["mode"],
            "seed": str(seed),
            "num_classes": "100",
            "config_fingerprint": EXPECTED_FINGERPRINT,
            "source_run": run_key,
        }
        temporary = destination.with_suffix(".tmp.safetensors")
        save_file(state, temporary, metadata=metadata)
        temporary.replace(destination)
        verify_safetensors(destination, state)

        exported_sha = sha256_file(destination)
        exported_size = destination.stat().st_size
        total_bytes += exported_size
        checksum_relative = str(relative_model_path.relative_to("checkpoints"))
        checksum_lines.append(f"{exported_sha}  {checksum_relative}")
        entries.append(
            {
                "run_key": run_key,
                "model": alias,
                "display_name": row["display_name"],
                "family": spec.family,
                "training_mode": row["mode"],
                "seed": seed,
                "timm_checkpoint": spec.checkpoint,
                "num_classes": 100,
                "input_size": 224,
                "best_epoch": recorded_best_epoch,
                "best_validation_top1": float(row["best_validation_top1"]),
                "test_top1": float(row["test_top1"]),
                "test_top5": float(row["test_top5"]),
                "parameters": int(row["parameters"]),
                "training_seconds": float(row["training_seconds"]),
                "training_peak_memory_mb": float(row["training_peak_memory_mb"]),
                "model_file": str(Path(CHAPTER_DIRECTORY) / relative_model_path),
                "exported_sha256": exported_sha,
                "exported_bytes": exported_size,
                "source_checkpoint": source_relative,
                "source_checkpoint_sha256": actual_source_sha,
                "source_checkpoint_bytes": source_path.stat().st_size,
            }
        )
        print(
            f"[{index:02d}/{len(rows):02d}] exported {run_key} -> "
            f"{relative_model_path} ({exported_size / 1024**2:.1f} MiB)"
        )
        del state
        del checkpoint
        gc.collect()

    checksum_path = chapter_output / "checkpoints" / "SHA256SUMS"
    checksum_path.write_text("\n".join(sorted(checksum_lines)) + "\n", encoding="utf-8")
    return entries, total_bytes


def main() -> None:
    """Build and validate one complete upload tree."""

    args = parse_args()
    project_root = args.project_root.resolve()
    results_root = (args.results_root or project_root / "modal_full_results_20260816").resolve()
    cards_root = (args.cards_root or project_root / "huggingface").resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output_root}")
    output_root.mkdir(parents=True)

    checkpoint_root = results_root / "checkpoints"
    source_checksums = read_checksum_manifest(checkpoint_root / "SHA256SUMS")
    rows = read_run_rows(results_root / "report" / "run_results.csv")
    chapter_output = copy_supporting_artifacts(
        project_root,
        results_root,
        cards_root,
        output_root,
    )
    entries, exported_bytes = export_weights(
        rows,
        checkpoint_root,
        chapter_output,
        source_checksums,
    )
    manifest = {
        "schema_version": 1,
        "book_title": BOOK_TITLE,
        "repository_id": REPOSITORY_ID,
        "chapter": 4,
        "chapter_title": "Efficient and scalable vision transformers",
        "chapter_directory": CHAPTER_DIRECTORY,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "format": "model-only Safetensors state dictionaries",
        "dataset": "CIFAR-100",
        "config_fingerprint": EXPECTED_FINGERPRINT,
        "torch_version": torch.__version__,
        "safetensors_version": importlib.metadata.version("safetensors"),
        "checkpoint_count": len(entries),
        "total_exported_bytes": exported_bytes,
        "models": entries,
    }
    (chapter_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_adapted_report(
        project_root / "EXPERIMENT_REPORT.md",
        chapter_output / "EXPERIMENT_REPORT.md",
        exported_bytes,
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "repository_id": REPOSITORY_ID,
                "checkpoint_count": len(entries),
                "total_exported_bytes": exported_bytes,
                "total_exported_gib": round(exported_bytes / 1024**3, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
