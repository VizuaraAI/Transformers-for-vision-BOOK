# Efficient and scalable vision-transformer benchmark

[← Back to the Chapter 4 projects](../Readme.md)

This chapter project turns the ViT/DeiT/Swin discussion into one controlled image-classification
experiment. It fine-tunes pretrained ViT-S, DeiT-S, Swin-T, and EfficientFormer-L1 checkpoints on
the same fixed CIFAR-100 split; runs a paired DeiT hard-distillation experiment with a fine-tuned
ConvNeXt teacher; and then measures accuracy, learning behavior, parameter count, MACs, peak
memory, latency, and throughput. Readers finish with a reproducible accuracy-versus-efficiency
report rather than a collection of incomparable model-card numbers.

The code is organized as a small production-style Python package, but the workflow is deliberately
progressive. A beginner can start with environment and synthetic checks, an intermediate reader can
run the short instructional preset, and a reader with a CUDA GPU can reproduce the complete,
multi-seed benchmark locally or on Modal.

## Completed experiment report

The full NVIDIA L4 experiment has been completed. Read
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for the objective, exact protocol, all measured
results, architecture-by-architecture interpretation, limitations, artifact map, and instructions
for verifying and loading the publicly archived validation-selected weights.

## Hugging Face model archive

The 13 validation-selected inference weights are also stored in the public repository
[`Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book`](https://huggingface.co/Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book),
under `chapter-04-efficient-and-scalable-vision-transformers/`. The Hub copy uses model-only
Safetensors files totaling 1.06 GiB, with a manifest, SHA-256 checksums, configuration, result tables,
figures, and a Hub-adapted experiment report. After remote hash verification, the local weight copies
were deleted; the larger resumable `best.pt` and `latest.pt` artifacts remain on Modal.

The six upstream model-card licenses were checked before release: five are Apache-2.0 and Swin-T is
MIT. Maintained Hub model-card sources are in [`huggingface`](huggingface), and
[`scripts/export_huggingface_bundle.py`](scripts/export_huggingface_bundle.py) rebuilds and validates
the upload bundle without exposing `.env` credentials.

## Contents

- [Completed experiment report](#completed-experiment-report)
- [Hugging Face model archive](#hugging-face-model-archive)
- [Learning outcomes](#learning-outcomes)
- [What is compared](#what-is-compared)
- [Why these variants matter](#why-these-variants-matter)
- [Start here](#start-here)
- [CLI command reference](#cli-command-reference)
- [Presets](#presets)
- [Exact experiment protocol](#exact-experiment-protocol)
- [Training and resuming individual runs](#training-and-resuming-individual-runs)
- [Reproduce the chapter result](#reproduce-the-chapter-result)
- [Inference benchmark protocol](#inference-benchmark-protocol)
- [Results and report generation](#results-and-report-generation)
- [Output layout](#output-layout)
- [Codebase structure](#codebase-structure)
- [Tests and quality checks](#tests-and-quality-checks)
- [Extending the project](#extending-the-project)
- [Common problems](#common-problems)
- [Documentation map](#documentation-map)
- [Reproducibility boundary and limitations](#reproducibility-boundary-and-limitations)
- [Primary references](#primary-references)

## Learning outcomes

This is not intended to be a leaderboard script. It is a guided experiment in which readers learn
how architectural and training choices become measurable engineering trade-offs. By the end of the
project, a reader should be able to:

- explain why global self-attention becomes expensive as the token grid grows;
- distinguish DeiT's data-efficient training recipe from a fundamentally different backbone;
- explain the purpose of Swin's windows, shifted windows, patch merging, and hierarchy;
- implement hard teacher distillation without allowing gradients to enter the teacher;
- design a fixed train/validation/test protocol that avoids accidental test-set tuning;
- count parameters and MAC-style operations from live implementations;
- measure latency, throughput, and accelerator memory with correct synchronization;
- compare learning curves instead of relying only on a final accuracy number;
- recognize why FLOPs/MACs and real throughput can rank models differently; and
- make conditional recommendations for accuracy-first, throughput-first, and memory-constrained
  deployments.

The final project deliverables are:

1. validation-selected checkpoints for every configured run;
2. one machine-readable metric history per run;
3. test predictions tied to official CIFAR-100 indices;
4. same-device inference measurements;
5. raw and aggregate CSV tables;
6. validation, distillation, and accuracy-throughput figures; and
7. a generated Markdown report suitable for adapting into the chapter narrative.

## What is compared

| Alias | Exact `timm` checkpoint | Reference parameters | Reference GMACs | Project role |
| --- | --- | ---: | ---: | --- |
| `vit` | `vit_small_patch16_224.augreg_in1k` | 22.1M | 4.3 | Plain global-attention baseline |
| `deit` | `deit_small_patch16_224.fb_in1k` | 22.1M | 4.6 | Data-efficient ViT training recipe |
| `deit_distilled` | `deit_small_distilled_patch16_224.fb_in1k` | 22.4M | 4.6 | Standard vs active hard distillation |
| `swin` | `swin_tiny_patch4_window7_224.ms_in1k` | 28.3M | 4.5 | Shifted-window hierarchical transformer |
| `efficientformer` | `efficientformer_l1.snap_dist_in1k` | 12.3M | 1.3 | Compact, latency-oriented model |
| `convnext` | `convnext_tiny.fb_in1k` | 28.6M | 4.5 | Teacher and CNN-style reference |

The parameter and GMAC columns are orientation metadata from the model definitions/model cards. The
100-class task heads change the live parameter count slightly, and the project recomputes both live
parameters and trace-based MACs instead of copying these values into the final report.

The practical comparison is intentionally checkpoint-level: architecture, pretraining recipe, and
published weights all matter to a practitioner. The paired Distilled DeiT runs are the narrower
causal comparison because they hold architecture, initialization family, split, and seed fixed and
change only whether the fine-tuning teacher is active.

## Why these variants matter

### ViT: the plain global-attention baseline

ViT converts non-overlapping image patches into tokens and applies global self-attention. If the
number of tokens is `N`, the attention map scales approximately with `N²`. That makes input
resolution and patch size important compute and memory decisions. ViT also begins with fewer built-in
locality and translation priors than a conventional CNN, which helps explain why pretraining data and
regularization are central to its practical training story.

### DeiT: efficiency can come from the recipe

DeiT-S is deliberately close to a small ViT architecture. Its importance is the demonstration that
augmentation, regularization, optimization, and teacher supervision can substantially improve data
efficiency. This means the broad ViT-versus-DeiT row is a practical checkpoint comparison, not proof
that one isolated layer change caused an accuracy difference.

Distilled DeiT adds a learned distillation token and a second classifier head. This project uses that
model twice: once without a live fine-tuning teacher and once with a frozen ConvNeXt teacher. Matched
seeds make this the project's cleanest measurement of active training-time distillation.

### Swin: hierarchy and bounded attention

Swin applies attention inside local windows rather than across the entire token grid. Consecutive
blocks shift the window partition, allowing information to cross earlier window boundaries. Patch
merging reduces spatial resolution and increases channel capacity, producing a hierarchy analogous
to the multi-stage feature maps of a CNN. These are the load-bearing Swin ideas the project is meant
to make concrete.

### EfficientFormer: optimize for the runtime, not only the operation table

EfficientFormer-L1 is the compact candidate. It combines efficient convolution-like processing with
attention in later stages and was designed with observed latency in mind. It lets readers test a
critical systems lesson: fewer parameters or MACs often help, but neither metric completely predicts
latency on a particular runtime and device.

### ConvNeXt: controlled teacher and CNN reference

ConvNeXt-T is trained on the same downstream split and serves two purposes. It supplies hard class
targets for Distilled DeiT, and it provides a modern CNN-style reference. It is not counted as a
fifth transformer candidate, and teacher training cost is reported separately from student inference
cost.

### Variants kept in the paper map

MobileViT, TinyViT, and MaxViT remain important chapter references, but adding all of them to the
main experiment would multiply compute without isolating another clean mechanism:

| Variant | Load-bearing idea | Suggested use here |
| --- | --- | --- |
| MobileViT | CNN locality plus transformer context in mobile blocks | Optional mobile-runtime extension |
| TinyViT | Compact hierarchy plus distillation | Alternative compact-model extension |
| MaxViT | Local block attention plus global grid attention | Higher-compute architecture extension |

The main benchmark therefore keeps one global transformer, one data-efficient ViT family, one
windowed hierarchy, and one compact practical model.

## Start here

Prerequisites are Python 3.12, Git, and roughly 10 GB of free space. A CUDA GPU is strongly
recommended for training; the setup and synthetic checks also work on CPU or Apple MPS.

```bash
cd Ch04-Efficient-and-Scalable-Vision-Transformers/02-Efficient-Vision-Transformer-Benchmark
uv sync --extra dev --extra docs
uv run vision-bench doctor
uv run vision-bench show-config --preset quick
uv run vision-bench smoke --models vit,deit_distilled --device auto
```

`smoke` uses random inputs and does not download weights or CIFAR-100. It validates model creation,
output shapes, and the two-head Distilled DeiT contract. It is a correctness check, not a speed test.

Next, download the dataset and freeze the split:

```bash
uv run vision-bench prepare-data --preset quick
```

Run the three-epoch instructional suite, benchmark its checkpoints, and build a report:

```bash
uv run vision-bench suite --preset quick --device cuda
uv run vision-bench benchmark --preset quick --device cuda
uv run vision-bench report --preset quick
```

The quick preset is labelled `tutorial_only` in every output. Its reduced data and epoch count make
it useful for learning the pipeline, not for drawing final model-ranking conclusions.

## CLI command reference

Every operation is exposed through `vision-bench` or equivalently `python -m vision_bench`:

| Command | Purpose | Downloads or trains? |
| --- | --- | --- |
| `doctor` | Print Python, dependency, disk, CUDA, and MPS diagnostics | No |
| `list-models` | Print exact checkpoints, roles, and reference metadata | No |
| `show-config` | Validate and print an expanded preset plus fingerprint | No |
| `prepare-data` | Download CIFAR-100 and create/verify the split manifest | Downloads data |
| `smoke` | Instantiate models and run synthetic output-contract checks | No pretrained downloads |
| `train` | Train or resume one configured model/mode/seed | Yes |
| `suite` | Train or resume all preset runs in dependency order | Yes |
| `benchmark` | Profile trained best checkpoints on one device | Runs inference |
| `report` | Generate CSVs, figures, JSON, and Markdown | No training |
| `all` | Run `suite`, `benchmark`, and `report` sequentially | Yes |

Most experiment commands accept:

- `--preset quick`, `--preset full`, or a YAML path;
- `--project-root` when the command is launched outside this directory;
- `--artifact-root` to place large outputs on another disk; and
- `--device auto|cpu|mps|cuda|cuda:0` for compute stages.

Inspect exact arguments at any time:

```bash
uv run vision-bench --help
uv run vision-bench train --help
uv run vision-bench benchmark --help
```

## Presets

| Preset | Purpose | Data | Epochs | Reportable? |
| --- | --- | --- | --- | --- |
| `smoke` | End-to-end debugging | 1 train + 1 validation image/class | 1 | No |
| `quick` | Reader walkthrough | 50 train + 10 validation images/class | 3 | No |
| `full` | Chapter experiment | 450 train + 50 validation images/class | 20 | Yes |

All presets use the same code paths. A SHA-256 fingerprint of the fully resolved configuration is
embedded in run directories and checkpoints so incompatible experiments cannot silently resume into
one another.

## Exact experiment protocol

### Dataset and split policy

The downstream task is 100-class image classification on CIFAR-100. The full preset splits the
official 50,000-image training set with seed `2027`:

- 450 images per class for training: 45,000 total;
- 50 images per class for validation: 5,000 total; and
- the untouched official 10,000-image test set for final evaluation.

The integer indices are stored under `data/splits/` in a manifest with a SHA-256 checksum. All
architectures and seeds use exactly the same indices. The validation split selects the best epoch;
the test split is evaluated only after training and never controls learning rate, epoch selection, or
hyperparameters.

Quick and smoke presets create their own balanced manifests. Those presets are useful for learning
and debugging but intentionally leave much of the training data unused.

### Input transforms

Training images pass through this shared geometric and augmentation pipeline:

1. random `32 × 32` crop with four pixels of reflected padding;
2. random horizontal flip;
3. RandAugment with two operations and magnitude 9;
4. bicubic resize to `224 × 224`;
5. tensor conversion and checkpoint-native normalization; and
6. random erasing with probability 0.25 for quick and full runs.

Validation and test images use deterministic bicubic resizing followed by tensor conversion and the
same checkpoint-native normalization. Geometry is shared across candidates, while normalization is
allowed to follow the published checkpoint because changing the expected upstream input distribution
would make transfer less faithful. Hard distillation explicitly refuses to run if the teacher and
student normalization statistics differ.

Upsampling native `32 × 32` CIFAR images to `224 × 224` does not create new visual detail. It is a
practical way to compare pretrained image models with their expected input size, and it must be
reported as a limitation.

### Shared fine-tuning settings

| Setting | Full-preset value |
| --- | --- |
| Epochs | 20 |
| Batch size | 64 |
| Optimizer | AdamW |
| Backbone learning rate | `5e-5` |
| New classifier learning rate | `5e-4` |
| Weight decay | 0.05 |
| No-decay terms | Biases, 1-D parameters, and model-declared token/position terms |
| Schedule | 2-epoch linear warmup followed by cosine decay |
| Minimum learning rate | `1e-6` |
| Label smoothing | 0.1 |
| Gradient clipping | Global norm 1.0 |
| Precision | CUDA automatic mixed precision; FP32 on CPU/MPS |
| Checkpoint selection | Highest validation top-1; validation loss breaks ties |

Every backbone remains trainable. The task-specific classifier receives a larger learning rate
because it is newly initialized for 100 classes. Model-specific learning-rate searches are avoided:
all candidates receive the same downstream optimization opportunity.

### Standard and distilled objectives

Ordinary runs minimize smoothed cross-entropy against the CIFAR-100 label. In a standard Distilled
DeiT control, the model has its two-token architecture but no live teacher; cross-entropy is applied
to the averaged class-head and distillation-head logits.

For active hard distillation, the frozen ConvNeXt teacher predicts an argmax class for the same
augmented image. Distilled DeiT exposes its heads separately:

- the class-token head learns from the true label using smoothed cross-entropy; and
- the distillation-token head learns from the teacher's hard class using ordinary cross-entropy.

The combined loss is:

```text
L = (1 - α) × L_class + α × L_teacher, where α = 0.5
```

Teacher inference runs in evaluation mode and without gradient tracking. At validation and test
time, the two student heads are averaged. The saved metric history exposes `classification_loss` and
`distillation_loss` separately so readers can diagnose both objectives.

### Full run matrix

| Model | Mode | Seeds | Role in the analysis |
| --- | --- | --- | --- |
| ConvNeXt-T | Standard | 42 | Teacher and CNN-style reference |
| ViT-S/16 | Standard | 42 | Plain global-attention baseline |
| Swin-T | Standard | 42 | Windowed hierarchical baseline |
| EfficientFormer-L1 | Standard | 42 | Compact efficiency baseline |
| DeiT-S/16 | Standard | 42, 43, 44 | Data-efficient DeiT checkpoint |
| Distilled DeiT-S | Standard | 42, 43, 44 | Same-student no-live-teacher control |
| Distilled DeiT-S | Hard KD | 42, 43, 44 | Active teacher treatment |

This expands to 13 independently resumable runs. Only the matched three-seed DeiT rows receive a
sample standard deviation. The single-seed rows are practical measurements and should not be
presented as precise estimates of seed variance.

### Metrics collected during training

Every completed epoch records:

- sample-weighted training and validation loss;
- top-1 and top-5 accuracy;
- images processed per second;
- epoch duration and cumulative train-plus-validation time;
- peak framework-allocator memory where supported;
- current learning rates for all optimizer groups; and
- both hard-distillation loss components when applicable.

After validation selects `best.pt`, the project evaluates the official test subset, records top-1
and top-5 accuracy, and saves predicted labels, true labels, and official test-set indices in
`test_predictions.npz`.

## Training and resuming individual runs

Training one model first is the easiest way to understand the pipeline:

```bash
uv run vision-bench train \
  --preset quick \
  --model vit \
  --mode standard \
  --seed 42 \
  --device cuda
```

Replace `cuda` with `mps` or `cpu` for a learning run. An explicit unavailable device raises an
error instead of silently changing the benchmark hardware.

To run the hard-distillation student individually, complete its teacher first:

```bash
uv run vision-bench train \
  --preset quick --model convnext --mode standard --seed 42 --device cuda

uv run vision-bench train \
  --preset quick --model deit_distilled --mode hard_kd --seed 42 --device cuda
```

`latest.pt` stores model, optimizer, scheduler, gradient-scaler, best-validation, random-generator,
and data-order state. Repeating an interrupted command resumes at the next uncommitted epoch. Metric
rows are reconciled to the latest checkpoint, and a partially committed `best.pt`/`latest.pt` pair is
repaired only when that repair is provably lossless.

For an incomplete run, `--no-resume` deliberately restarts that configuration and clears its stale
metric history. A directory containing `complete.json` is immutable; use another `--artifact-root`
when you intentionally want to repeat an identical completed configuration.

Useful single-stage commands are:

```bash
# Train or resume every run in dependency order
uv run vision-bench suite --preset quick --device cuda

# Time trained architectures on the selected device
uv run vision-bench benchmark --preset quick --device cuda

# Generate CSV tables, figures, JSON summary, and Markdown
uv run vision-bench report --preset quick
```

## Reproduce the chapter result

The complete experiment is deliberately separate from the instructional presets. Do not begin it
until the quick suite has produced a valid report on the intended environment.

### Local CUDA execution

On a sufficiently capable NVIDIA GPU, run training, timing, and report generation sequentially:

```bash
uv run vision-bench all --preset full --device cuda
```

This command is resumable. If the workstation is interrupted, execute it again with the same preset,
artifact root, and device. Completed runs return immediately; incomplete runs continue from their
latest committed epoch.

For easier monitoring, the same workflow can be executed as explicit stages:

```bash
uv run vision-bench suite --preset full --device cuda
uv run vision-bench benchmark --preset full --device cuda
uv run vision-bench report --preset full
```

### Modal L4 execution

The cloud launcher uses a persistent Volume, trains the ConvNeXt teacher first, then distributes the
remaining independent runs across at most four L4 containers. Each training container receives eight
CPU cores for image augmentation. All architecture timings are later collected sequentially inside
one L4 container so the reference throughput rows share a hardware context.

Authenticate and rehearse with quick:

```bash
uv run modal setup
uv run modal run modal_app.py --preset quick --stage all
```

For non-interactive authentication, copy [`.env.example`](.env.example) to `.env`, replace its
placeholders, and follow the shell-loading instructions in [the Modal guide](docs/running-modal.md).
The populated `.env` file is ignored by Git.

Then launch the full workflow:

```bash
uv run modal run modal_app.py --preset full --stage all
```

Available cloud stages are `train`, `benchmark`, `report`, and `all`:

```bash
uv run modal run modal_app.py --preset full --stage train
uv run modal run modal_app.py --preset full --stage benchmark
uv run modal run modal_app.py --preset full --stage report
```

Each epoch is committed to the persistent Volume. Transient retries reload the latest checkpoint and
continue. Cloud execution incurs charges; check current pricing and account limits before starting.
Read [the Modal guide](docs/running-modal.md) for cancellation, artifact download, persistence, and
cost-control details.

## Inference benchmark protocol

The hardware benchmark is intentionally separate from test-set evaluation. Accuracy depends on the
trained weights; execution speed primarily depends on the architecture, input shape, precision,
batch size, software stack, and device.

For each unique architecture, `vision-bench benchmark`:

1. loads a representative validation-best checkpoint;
2. verifies that its configuration fingerprint matches the requested preset;
3. counts live model parameters;
4. traces MAC-style operations for one `224 × 224` input with fvcore;
5. creates random tensors directly on the target device;
6. performs untimed warmup forwards;
7. measures synchronized inference-only forwards; and
8. writes the protocol, environment, profiles, and raw summaries to `benchmark.json`.

The full preset measures:

| Dimension | Values |
| --- | --- |
| Precision | FP32 and CUDA FP16 |
| Batch size | 1 and 64 |
| Warmup | 50 forwards per precision/batch configuration |
| Timed work | 200 forwards × 5 repeats |
| Input | In-memory random `N × 3 × 224 × 224` tensor |
| Data loading | Excluded |
| CUDA timing | Per-iteration CUDA events with synchronization |
| CPU/MPS timing | Synchronized wall clock |

Reported timing statistics include mean, median, p90, and standard deviation of batch latency plus
images per second. Throughput is calculated as:

```text
images_per_second = batch_size × 1000 / mean_batch_latency_ms
```

Batch 1 is the more relevant row for interactive request latency. Batch 64 is the more relevant row
for a saturated batch service. Neither row predicts a phone, browser, edge accelerator, or another
GPU without measuring that runtime directly.

FP16 rows are skipped outside CUDA rather than silently substituting a different precision. Peak
memory is framework-allocator memory, not whole-process or total system memory. MPS exposes current
allocated tensor memory rather than CUDA-style peak statistics, and the artifact retains the device
type so those meanings are not mixed.

fvcore counts one multiply-add as one operation in this project, so the report calls the result
MAC-style rather than assuming every paper's FLOP convention is identical. Unsupported traced
operators are stored explicitly. The expected model-card GMAC value is retained as a reasonableness
reference but never silently replaces a failed or incomplete trace.

## Results and report generation

Generate a report after any completed runs:

```bash
uv run vision-bench report --preset full
```

The report can describe a partially completed suite and lists all missing run keys. An
accuracy-throughput figure appears only when `benchmark.json` is available.

The main table contains:

- model and training mode;
- number of completed seeds;
- mean test top-1 accuracy;
- sample standard deviation when more than one seed exists;
- parameter count; and
- reference throughput with its precision and batch size.

Single-seed rows intentionally omit a `± 0` value because zero would falsely imply that seed
uncertainty was measured. For the three-seed DeiT comparisons, the report uses sample standard
deviation rather than presenting a three-sample interval as highly precise.

The isolated fine-tuning distillation effect is computed seed by seed:

```text
hard-KD gain = Distilled DeiT hard-KD top-1 − Distilled DeiT standard top-1
```

This contrast holds the student architecture and seed fixed. Ordinary DeiT versus Distilled DeiT is
also displayed, but that comparison changes the checkpoint family and token/head architecture and
must not be described as a pure teacher-loss ablation.

Use the generated figures to answer different questions:

- `validation_accuracy.png`: how each run learned and whether it plateaued or became unstable;
- `distillation_gain.png`: whether active teacher supervision helped consistently across seeds; and
- `accuracy_throughput.png`: which candidates lie on the measured accuracy/speed frontier.

The repository does not ship invented reference numbers. A reportable table must come from a
completed full preset and one same-device benchmark invocation. The publication checklist is in
[`reference_results/README.md`](reference_results/README.md).

## Output layout

```text
artifacts/
└── full/
    ├── vit-standard-seed42-<fingerprint>/
    │   ├── resolved_config.json
    │   ├── metrics.jsonl
    │   ├── latest.pt
    │   ├── best.pt
    │   ├── test_predictions.npz
    │   └── complete.json
    ├── ... one directory per run ...
    ├── benchmark.json
    └── report/
        ├── report.md
        ├── run_results.csv
        ├── epoch_metrics.csv
        ├── aggregate_results.csv
        ├── inference_benchmark.csv
        └── figures/
```

| Artifact | Meaning |
| --- | --- |
| `resolved_config.json` | Expanded project/run settings and full fingerprint |
| `metrics.jsonl` | One independently readable row per committed epoch |
| `latest.pt` | State-complete resume checkpoint after the newest epoch |
| `best.pt` | Validation-selected model used for final evaluation |
| `test_predictions.npz` | Predictions, targets, and official test indices |
| `complete.json` | Final accuracy, parameter count, environment, and split checksum |
| `benchmark.json` | Same-device model profiles and timing summaries |
| `report/*.csv` | Raw runs, epoch history, aggregates, KD gains, profiles, and timing |
| `report/report_summary.json` | Machine-readable report inventory and grouped results |
| `report/report.md` | Human-readable chapter-project report |

`latest.pt` stores enough state to continue an interrupted run in the same environment. Bit-for-bit
identity across different hardware or software stacks is not promised. `best.pt` is selected using
validation accuracy, with validation loss as the tie-breaker. The official test split is evaluated
only after training.

## Codebase structure

```text
02-Efficient-Vision-Transformer-Benchmark/
├── configs/
│   ├── smoke.yaml
│   ├── quick.yaml
│   └── full.yaml
├── docs/                         # Searchable book-project documentation
├── notebooks/                    # Thin guided notebooks; no duplicate training loop
├── reference_results/            # Publication policy, initially no fabricated numbers
├── src/vision_bench/
│   ├── benchmark.py              # MAC, latency, throughput, and memory protocol
│   ├── checkpointing.py          # Atomic JSON/checkpoint and RNG-state handling
│   ├── cli.py                    # Reader-facing command-line interface
│   ├── config.py                 # YAML validation, seed expansion, fingerprints
│   ├── data.py                   # CIFAR split manifests, transforms, loaders
│   ├── distillation.py           # Hard token-distillation objective
│   ├── doctor.py                 # Environment and synthetic model checks
│   ├── engine.py                 # Optimizer, train/evaluate loops, suite orchestration
│   ├── metrics.py                # Sample-weighted and top-k helpers
│   ├── models.py                 # Exact checkpoint registry and model helpers
│   ├── reporting.py              # CSV, figures, JSON, and Markdown generation
│   └── runtime.py                # Devices, seeding, synchronization, environment metadata
├── tests/                        # Unit and real-model integration contracts
├── modal_app.py                  # Resumable bounded-concurrency L4 workflow
├── mkdocs.yml
├── pyproject.toml
└── uv.lock
```

The package is the single implementation source. CLI commands, notebooks, and Modal functions call
the same modules, which prevents a book notebook, local script, and cloud job from quietly using
different loss functions or transforms.

### Configuration validation

The loader rejects unknown fields, invalid dataset sizes, unsupported modes, duplicate run keys, and
hard-KD presets without an earlier ConvNeXt teacher. YAML seed lists are expanded into individual
`RunConfig` objects. The fingerprint hashes the fully resolved protocol rather than only the filename.

### Checkpoint consistency

JSON and PyTorch checkpoints are written through a temporary file followed by an atomic rename.
Resume restores model, optimizer, scheduler, gradient scaler, Python/NumPy/PyTorch accelerator RNG
states, and the dedicated data-loader generator. Data workers are recreated at epoch boundaries from
the checkpointed stream. If a crash happens between writing `latest.pt` and a newly improved
`best.pt`, the pair is repaired only when `latest.pt` is demonstrably the recorded best epoch.

### Why the notebooks are thin

The notebooks focus on inspection, execution, and interpretation. They import the tested package or
call `python -m vision_bench`; they do not contain a second copy of the data or training pipeline.
This makes them easier for beginners to read and safer for maintainers to update.

Follow them in order:

1. `01_inspect_experiment.ipynb`: inspect models, presets, fingerprints, and the run matrix;
2. `02_run_quick_experiment.ipynb`: opt into training and read epoch metrics; and
3. `03_analyze_results.ipynb`: aggregate completed runs and inspect hardware measurements.

## Tests and quality checks

Install the development extras, then run the complete test suite:

```bash
uv sync --extra dev --extra docs
uv run pytest
```

The tests cover:

- smoke, quick, and full preset expansion;
- configuration errors and fingerprint changes;
- deterministic balanced split construction and checksum corruption detection;
- sample-weighted metrics and two-head logit averaging;
- hard-distillation loss math and frozen-teacher gradients;
- classifier/backbone learning-rate and no-weight-decay groups;
- warmup/cosine schedule behavior;
- a complete synthetic train-and-evaluate epoch;
- synchronized benchmark statistics and non-CUDA FP16 skipping;
- report aggregation, paired distillation gains, and report generation;
- notebook schema validity; and
- real instantiation and parameter-scale checks for all six pinned `timm` models.

Run the slower synthetic forwards separately:

```bash
uv run vision-bench smoke --device auto
```

The smoke command instantiates every architecture without pretrained weights, performs a real
`224 × 224` forward, checks the `[1, 100]` output contract, and verifies that Distilled DeiT can expose
two training heads. Its printed time includes startup and is not a publishable benchmark.

The full local quality gate is:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src/vision_bench modal_app.py
uv run pytest
uv run mkdocs build --strict
uv lock --check
uv build
```

At implementation handoff, these checks passed with 37 tests, all six real model registrations, a
real Distilled DeiT/ConvNeXt hard-KD backward pass, strict documentation, and valid wheel/source
builds. The full CIFAR training suite and billable cloud job are intentionally not bundled as if they
had been run; readers generate those experimental results in their declared environment.

## Extending the project

### Change the shared compute budget

Copy a preset and edit the copy:

```bash
cp configs/quick.yaml configs/my_experiment.yaml
```

Change shared fields such as batch size, epochs, or input size for every model. A changed protocol
receives a new fingerprint and cannot resume into the old checkpoint directory. Avoid giving only
one candidate a larger epoch budget unless the new research question explicitly studies
model-specific tuning.

### Add another model

To add MobileViT, TinyViT, MaxViT, or another supported `timm` checkpoint:

1. add one exact checkpoint tag and metadata entry to `MODEL_SPECS` in `models.py`;
2. add the alias to a copied YAML preset;
3. instantiate it with `pretrained=False` and verify the 100-class output contract;
4. confirm its checkpoint-native normalization and input resolution;
5. run parameter and MAC profiling;
6. add or update an integration test; and
7. document whether it replaces the compact candidate or creates a new comparison axis.

Do not add a model to only the final table. It must pass through the same split, evaluation, artifact,
and timing pipeline.

### Change the dataset

The current engine assumes CIFAR-100 and a 100-class classifier in a few deliberate places. A new
dataset extension should introduce a dataset specification rather than scattering another constant
through the package. Preserve the same principles: a frozen manifest, validation-only checkpoint
selection, sealed test evaluation, class-count-aware metrics, and identical candidate transforms
unless checkpoint-native preprocessing requires an explicit exception.

### Add a new distillation method

Soft distillation would replace the hard argmax target with temperature-scaled teacher
probabilities. Treat it as a new mode, record temperature and loss weight in the preset fingerprint,
retain the standard same-student control, and add tests showing that teacher tensors remain outside
the gradient graph.

### Benchmark another deployment target

Create a separate benchmark artifact for CPU, mobile, browser, ONNX, TensorRT, Core ML, or another
runtime. Do not merge throughput numbers from different devices into one ranking. If preprocessing
or host-to-device transfer is included, state that explicitly because the built-in benchmark isolates
model forward execution.

## Common problems

- **CUDA requested but unavailable:** run `vision-bench doctor`; fix the driver/PyTorch environment
  or choose `mps`/`cpu` explicitly for a non-reference learning run.
- **Out of memory:** reduce the shared batch size in a copied preset and rerun every candidate under
  the new fingerprint.
- **Teacher checkpoint missing:** train the configured ConvNeXt run first or use `suite`, which orders
  dependencies automatically.
- **Checkpoint fingerprint mismatch:** use the original YAML or allow the changed protocol to create
  its own run directory; never edit the hash inside a checkpoint.
- **FP16 timing skipped:** expected on CPU and MPS; CUDA is required for the project's FP16 reference
  path.
- **No report rows:** at least one run must contain `complete.json`; run training before reporting.
- **No accuracy-throughput figure:** create `benchmark.json` with `vision-bench benchmark` first.
- **Noisy timing:** close other accelerator workloads and rerun all candidates together rather than
  selectively retiming one model.

See [the full troubleshooting guide](docs/troubleshooting.md) for download, MAC-tracing, Volume, and
resume details.

## Documentation map

- [Chapter project guide](docs/chapter-project.md): the reader journey and learning checkpoints
- [Setup](docs/setup.md): Python, GPU, installation, and environment verification
- [Project design](docs/project-design.md): model choices, claims, and confounds
- [Experiment protocol](docs/experiment-protocol.md): splits, transforms, losses, and run matrix
- [Run locally](docs/running-locally.md): commands, resumption, and artifacts
- [Run on Modal](docs/running-modal.md): scalable cloud execution and cost safety
- [Interpret results](docs/results.md): metric definitions and defensible conclusions
- [Code tour](docs/code-tour.md): where each concern lives
- [Troubleshooting](docs/troubleshooting.md) and [glossary](docs/glossary.md)

Build a searchable local documentation site with:

```bash
uv run mkdocs serve
```

Launch the guided notebooks with `uv run jupyter lab`; follow them in numeric order.

## Reproducibility boundary and limitations

The project preserves exact presets, split indices, checksums, seeds, dependency versions,
environment metadata, optimizer/scheduler state, data-order state, and run fingerprints. That makes
the experiment auditable and resumable, but it does not make every result universal.

### What this experiment can support

- validation and test accuracy under the frozen CIFAR-100 transfer protocol;
- paired active hard-distillation gain for matched Distilled DeiT seeds;
- parameter counts from the live task-specific implementations;
- trace-based MAC estimates with unsupported operators disclosed;
- training behavior and allocator memory recorded by this code; and
- latency/throughput comparisons within one same-device `benchmark.json` artifact.

### What it cannot support by itself

- a universal ranking across datasets, image resolutions, runtimes, or devices;
- a pure architectural explanation for broad ViT-versus-DeiT checkpoint differences;
- reliable seed uncertainty for the single-seed practical baselines;
- mobile efficiency conclusions from an NVIDIA L4 measurement;
- energy-efficiency conclusions without measuring power; or
- claims about detection and segmentation from a classification experiment.

Important limitations to include in any chapter write-up are:

1. upstream checkpoints were produced with different pretraining recipes;
2. CIFAR-100 images are upsampled from 32 to 224 pixels;
3. only the focused DeiT comparisons receive three seeds;
4. MAC conventions and tracing support vary across tools;
5. allocator memory is not whole-system memory; and
6. throughput is valid only for the recorded device, precision, batch size, input shape, and
   software environment.

Exact seeds and split indices are preserved, but bit-for-bit identity is not promised across CUDA,
MPS, CPU, driver versions, or GPU models. The repository intentionally ships no invented reference
accuracy table; [`reference_results/README.md`](reference_results/README.md) explains how a verified
full run can be published.

## Primary references

- Dosovitskiy et al., [An Image Is Worth 16×16 Words: Transformers for Image Recognition at
  Scale](https://arxiv.org/abs/2010.11929) — ViT.
- Touvron et al., [Training data-efficient image transformers & distillation through
  attention](https://arxiv.org/abs/2012.12877) — DeiT and the distillation token.
- Liu et al., [Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows](https://arxiv.org/abs/2103.14030) — window attention, shifts, and hierarchy.
- Li et al., [EfficientFormer: Vision Transformers at MobileNet
  Speed](https://arxiv.org/abs/2206.01191) — latency-oriented compact design.
- Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) — ConvNeXt teacher/reference.
- Mehta and Rastegari, [MobileViT](https://arxiv.org/abs/2110.02178); Wu et al.,
  [TinyViT](https://arxiv.org/abs/2207.10666); and Tu et al.,
  [MaxViT](https://arxiv.org/abs/2204.01697) — practical variants in the chapter's paper map.

The exact executable checkpoint identifiers are pinned in `src/vision_bench/models.py`; paper-level
architecture names alone are not sufficient to reproduce a checkpoint comparison.
