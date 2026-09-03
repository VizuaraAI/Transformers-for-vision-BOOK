# Running locally

Run commands from the project directory. Prefix every command with `uv run` unless you activated a
virtual environment manually.

## 1. Inspect before spending compute

```bash
uv run vision-bench doctor
uv run vision-bench list-models
uv run vision-bench show-config --preset quick
uv run vision-bench smoke --models vit,deit_distilled --device auto
```

`show-config` prints the expanded run list and fingerprint. Verify the preset name and
`tutorial_only` flag before continuing.

## 2. Prepare CIFAR-100

```bash
uv run vision-bench prepare-data --preset quick
```

The command downloads CIFAR-100 through torchvision, creates the stratified manifest if it does not
exist, and prints image counts plus the checksum. Running it again verifies and reuses the manifest.

## 3. Train one model

```bash
uv run vision-bench train \
  --preset quick \
  --model vit \
  --mode standard \
  --seed 42 \
  --device cuda
```

Use `--device mps` on a supported Apple machine or `--device cpu` for a very small debugging run.
During training, one concise line is printed per epoch. Detailed machine-readable metrics live in
the run directory.

## 4. Resume safely

Stop a run after an epoch with Ctrl-C, then execute the same command. `latest.pt` restores model,
optimizer, scheduler, gradient scaler, random generators, best validation score, and elapsed time.
Metric rows are reconciled to the committed checkpoint before the next epoch.

Changing the YAML changes its fingerprint and therefore creates a different run directory. A
checkpoint with another fingerprint is rejected. For an incomplete run, `--no-resume` deliberately
starts again and clears stale metric rows. A run with `complete.json` is treated as immutable and
returned immediately; use another artifact root when you intentionally want a second identical run.

## 5. Run the complete preset sequence

```bash
uv run vision-bench suite --preset quick --device cuda
```

The suite runs ConvNeXt first so its best checkpoint exists before hard distillation. Completed runs
are detected through `complete.json` and returned immediately, so rerunning the suite is safe.

To train hard distillation as a single command, first complete the configured ConvNeXt run:

```bash
uv run vision-bench train --preset quick --model convnext --seed 42 --device cuda
uv run vision-bench train \
  --preset quick --model deit_distilled --mode hard_kd --seed 42 --device cuda
```

## 6. Benchmark on one device

```bash
uv run vision-bench benchmark --preset quick --device cuda
```

The command loads one representative best checkpoint per architecture and profiles them sequentially
in the same process. Do not run other GPU workloads at the same time. The full benchmark performs
many synchronized forwards by design; first verify the protocol with quick.

CPU or MPS runs produce FP32 measurements and explicit skipped FP16 rows. They are useful for a
separate local deployment question but are not interchangeable with L4 results.

## 7. Generate the report

```bash
uv run vision-bench report --preset quick
```

Open `artifacts/quick/report/report.md`. A report can be generated from partially completed suites;
missing run keys are printed in the report. Accuracy-throughput figures appear only after the
benchmark artifact exists.

## One-command path

After rehearsing the individual stages:

```bash
uv run vision-bench all --preset full --device cuda
```

This is sequential and can be long. It is intentionally simple for a single workstation. The Modal
launcher parallelizes independent runs after the teacher is ready.

## Custom artifact locations

Every relevant command accepts `--artifact-root`. A relative value is resolved under the project
directory:

```bash
uv run vision-bench suite \
  --preset quick --device cuda --artifact-root /mnt/experiments/vision-bench
```

Data location is a preset field (`data.root`), while run outputs are a CLI concern. Keep large
artifacts outside version control.

## Reading a run directory

- `resolved_config.json`: exact project/run settings and fingerprint
- `metrics.jsonl`: one append-only record per completed epoch
- `latest.pt`: resumable state after the newest epoch
- `best.pt`: validation-selected checkpoint used for test and timing
- `test_predictions.npz`: predicted/true class IDs and official test-set indices
- `complete.json`: final metrics, parameter counts, environment, and split checksum

PyTorch checkpoints should be treated as trusted local artifacts. This project loads optimizer and
random-generator state and therefore uses PyTorch's general checkpoint format, not an untrusted
model-exchange boundary.
