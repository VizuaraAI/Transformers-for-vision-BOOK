# Troubleshooting

## `Could not find preset`

Run from the `vision-transformer-benchmark` directory or pass the YAML path:

```bash
uv run vision-bench show-config --preset configs/quick.yaml
```

If you invoke the installed command elsewhere, add `--project-root /absolute/path/to/the/project`.

## Python version is unsupported

The environment is pinned to Python 3.12. With uv:

```bash
uv python install 3.12
uv sync --extra dev --extra docs
```

Delete no environments unless you know they are disposable; `uv sync` can create the correct local
environment alongside other Python installations.

## CUDA was requested but is unavailable

Run `vision-bench doctor`. If `cuda_available` is false, verify the NVIDIA driver and install the
correct PyTorch build using official platform guidance. Use `--device cpu` only to debug; silently
falling back would make a claimed CUDA benchmark misleading, so an explicit CUDA request fails.

## Out of GPU memory

Do not change only one model's batch size inside a comparison. Copy the preset, reduce the shared
`training.batch_size`, and run every candidate under the new fingerprint. Gradient accumulation is
not implemented because it would complicate training-throughput interpretation; it can be a clearly
documented extension.

For inference, create a preset with smaller shared benchmark batch sizes. Report the new batch size.

## A pretrained checkpoint will not download

Check network access and free disk space. `timm` may use a Hugging Face cache. Modal stores this under
`/vol/cache/huggingface`; local defaults depend on your environment. Rerunning is safe after a
transient download failure.

Do not silently switch to `pretrained: false` for a real comparison. The smoke preset uses random
weights only because its purpose is code-path validation.

## Hard distillation cannot find the teacher

Run ConvNeXt first or use `suite`, which orders dependencies:

```bash
uv run vision-bench train --preset quick --model convnext --seed 42 --device cuda
uv run vision-bench train \
  --preset quick --model deit_distilled --mode hard_kd --seed 42 --device cuda
```

The teacher path includes the preset fingerprint, so a teacher from an incompatible preset is not
accepted accidentally.

## Resume refuses a checkpoint

The fingerprint differs. Use the same YAML that created the run or let the changed configuration
start in its own directory. Never edit a checkpoint's fingerprint to bypass the guard.

## The metric log has fewer rows than expected

Only completed epochs are committed. Resume the run. On restart, rows beyond the latest committed
checkpoint are removed so a crash between metric and checkpoint writes cannot duplicate an epoch.

## FP16 benchmark rows say `skipped`

That is expected on CPU and MPS. The reference FP16 path is CUDA-only. FP32 rows are still produced.

## MAC tracing reports unsupported operators

Open `model_profiles.csv`. The counter retains unsupported operator names and counts instead of
pretending the trace is complete. Compare the measured value with the expected model-card GMACs and
describe the uncertainty. Throughput is still measured from actual forwards.

## Timing is noisy

Close other GPU programs, use the full timing repetitions, ensure all architectures run in one
benchmark command, and avoid thermal/power mode changes. Do not remove synchronization. Re-run the
entire benchmark artifact rather than selectively repeating a slow model.

## Modal Volume artifacts are missing

Confirm the volume name is `vision-transformer-benchmark-data` and paths begin with `/artifacts`.
The remote functions explicitly reload before reading and commit after each epoch/stage. Check the
App logs for a function failure before the final commit. Consult Modal's current Volumes guide if the
CLI syntax has changed.

## The quick ranking looks surprising

Quick deliberately uses 5,000 training examples and three epochs. Its optimization variance can be
larger than real model differences. Use it to debug tables and curves, not to replace the full run.
