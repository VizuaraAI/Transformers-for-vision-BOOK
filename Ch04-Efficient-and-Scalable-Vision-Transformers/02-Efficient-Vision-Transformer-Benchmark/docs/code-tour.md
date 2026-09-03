# Code tour

The notebooks and commands are deliberately thin. Experiment logic lives in importable, tested
modules so the book does not teach readers to maintain several drifting copies of a training loop.

## Configuration

`configs/{smoke,quick,full}.yaml` define data, optimization, timing, and run matrices.
`vision_bench.config` rejects unknown fields, validates values and dependency ordering, expands seed
lists into individual runs, and creates the fingerprint used in artifact paths.

Change a preset copy when designing an extension. Do not edit a resolved artifact after training.

## Models

`vision_bench.models` is the single model registry. It maps short aliases to exact `timm` tags and
contains checkpoint metadata used in reports. Classifiers are replaced with 100-class heads through
`timm.create_model(..., num_classes=100)`.

The optimizer asks the model for its classifier module(s), which matters because Distilled DeiT has
two heads. Classifier parameters receive the head learning rate. Biases, one-dimensional parameters,
and token/position terms declared by the model use no weight decay.

## Data

`vision_bench.data` contains pure stratified-index functions, manifest checksum logic, transforms,
worker seeding, and loaders. Separating indices from image access makes the most important leakage
and reproducibility logic easy to unit test.

## Training and evaluation

`vision_bench.engine` owns the optimizer, per-step warmup/cosine schedule, AMP, gradient clipping,
training loop, evaluation loop, checkpoint selection, final test call, and suite ordering.

Metric averages are sample-weighted, so a short final batch does not count as much as a full batch.
Top-5 is safely capped if a future extension has fewer than five classes.

## Distillation

`vision_bench.distillation.HardTokenDistillationLoss` requires an explicit pair of student logits.
It obtains teacher targets inside `torch.no_grad()` and returns both component losses for logging.
`freeze_teacher` disables all teacher gradients in addition to setting evaluation mode.

## Checkpoints

`vision_bench.checkpointing` writes JSON and PyTorch state through temporary files followed by an
atomic rename. A checkpoint includes model, optimizer, scheduler, scaler, epoch, best validation
state, cumulative time, preset fingerprint, Python/NumPy/PyTorch random states, and the dedicated
data-order generator.

## Hardware profiling

`vision_bench.benchmark` counts live parameters, traces MACs, warms each configuration, synchronizes
accelerator timing, and summarizes latency samples. It loads one trained representative per
architecture because weights and distillation mode do not change the execution graph.

## Reports

`vision_bench.reporting` scans configuration-qualified run directories rather than accepting pasted
numbers. It emits raw and aggregate CSV files, paired distillation gains, figures, a Markdown report,
and a machine-readable report summary.

## Commands and cloud execution

`vision_bench.cli` exposes the same package functions to readers. `modal_app.py` supplies
infrastructure—L4 allocation, parallel mapping, retries, and persistent storage—without maintaining
a second training implementation.

## Tests

`tests/` focuses on the contracts most likely to invalidate an experiment: preset expansion, split
balance, checksum detection, loss math, optimizer groups, checkpoint log reconciliation helpers,
benchmark statistics, report aggregation, CLI parsing, and notebook validity. Real `timm` forward
tests are marked `integration` because they are slower.
