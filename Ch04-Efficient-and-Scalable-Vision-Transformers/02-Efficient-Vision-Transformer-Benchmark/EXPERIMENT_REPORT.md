# What actually wins under a fixed vision-transformer budget?

## A controlled CIFAR-100 study of ViT, DeiT, Swin, EfficientFormer, and distillation

This report records the completed experiment for the chapter project **Efficient and Scalable
Vision Transformers**. The goal was not to reproduce a collection of headline numbers from model
cards. It was to put several important architecture and training ideas into the same experimental
frame: the same image-classification task, the same train/validation/test policy, the same
fine-tuning budget, and the same inference hardware.

The result is a practical answer rather than a universal leaderboard. On this task, a small plain
ViT was already an excellent L4 throughput baseline; Swin's hierarchy did not make it faster;
EfficientFormer dramatically reduced parameter count and MACs but gave up accuracy; and active
hard distillation produced a small, repeatable accuracy gain at training time without changing
deployment cost. Just as importantly, the benchmark showed why parameter count, theoretical
operations, latency, and throughput must be measured separately.

> **Experiment status:** complete. All 13 training runs reached 20 epochs, all 24 inference cases
> completed, the report artifacts were generated, and all 13 validation-selected model states were
> published to Hugging Face and verified before their local weight copies were removed.

---

## Contents

- [Executive summary](#executive-summary)
- [1. Objective and research questions](#1-objective-and-research-questions)
- [2. Models and why they were selected](#2-models-and-why-they-were-selected)
- [3. Experimental design](#3-experimental-design)
- [4. How the experiment was executed](#4-how-the-experiment-was-executed)
- [5. Accuracy and training results](#5-accuracy-and-training-results)
- [6. What did distillation buy?](#6-what-did-distillation-buy)
- [7. Parameters and operation counts](#7-parameters-and-operation-counts)
- [8. Inference benchmark](#8-inference-benchmark)
- [9. Interpretation by model family](#9-interpretation-by-model-family)
- [10. Decision guide](#10-decision-guide)
- [11. Threats to validity and limitations](#11-threats-to-validity-and-limitations)
- [12. Reproducibility record and artifact map](#12-reproducibility-record-and-artifact-map)
- [13. Reproducing the experiment](#13-reproducing-the-experiment)
- [14. Loading a downloaded checkpoint](#14-loading-a-downloaded-checkpoint)
- [15. Final takeaway](#15-final-takeaway)
- [16. Primary references](#16-primary-references)

## Executive summary

- **Best single-run test accuracy:** hard-distilled DeiT-S/16, seed 42, at **89.83%** top-1.
- **Best multi-seed transformer result:** hard-distilled DeiT-S/16 at
  **89.72% ± 0.11** top-1 across three seeds.
- **Best single-seed reference row:** ConvNeXt-T at **89.78%**, although its one run does not
  provide an uncertainty estimate.
- **Controlled distillation result:** hard KD improved Distilled DeiT by **+0.09, +0.40, and
  +0.30 percentage points** for seeds 42, 43, and 44. The mean paired gain was **+0.26 points**.
- **Fastest FP16 batch-64 result on the NVIDIA L4:** ViT-S/16 at approximately
  **1,995 images/s**. DeiT-S/16 and Distilled DeiT were effectively in the same throughput band.
- **Smallest candidate:** EfficientFormer-L1 at **11.48M parameters** and **1.32 GMACs**. It
  reached **86.83%** top-1 and about **1,961 FP16 images/s** at batch 64.
- **Most expensive model in this training setup:** Swin-T, at about **60.5 minutes** per run and
  **4.84 GB** peak CUDA allocator memory. It also had the lowest measured FP16 throughput.
- **Most useful systems lesson:** MACs did not predict the runtime ordering. EfficientFormer used
  roughly 69% fewer measured MACs than ViT, yet their FP16 batch-64 throughput was nearly the same
  on this GPU.

These findings are specific to this dataset, software stack, and NVIDIA L4. They are evidence for
how to make a deployment decision, not proof that one architecture always dominates another.

## 1. Objective and research questions

The project asks one central question:

> When data, fine-tuning budget, input resolution, evaluation rules, and hardware are held as
> constant as practical, what do ViT, DeiT, Swin, and a compact efficient model trade in accuracy,
> training behavior, model size, operation count, memory, latency, and throughput?

That question is divided into six measurable subquestions:

1. How competitive is a plain global-attention ViT after transfer learning on a modest dataset?
2. Does active teacher distillation improve a distilled DeiT when architecture, seed, and data are
   held fixed?
3. Does Swin's windowed hierarchical design translate into lower memory use or higher throughput in
   this implementation?
4. How much accuracy does a compact EfficientFormer exchange for fewer parameters and MACs?
5. Do MAC counts predict real latency and throughput?
6. Which conclusions survive a multi-seed comparison, and which remain tentative single-seed
   observations?

The project fine-tunes pretrained models. It does **not** train these architectures from scratch.
That choice matches the workflow most readers will use when compute and labeled data are limited.

## 2. Models and why they were selected

| Project alias | Exact `timm` checkpoint | Role in the chapter |
| --- | --- | --- |
| `vit` | `vit_small_patch16_224.augreg_in1k` | Plain global-attention baseline |
| `deit` | `deit_small_patch16_224.fb_in1k` | Data-efficient ViT recipe |
| `deit_distilled` | `deit_small_distilled_patch16_224.fb_in1k` | Controlled standard-versus-hard-KD comparison |
| `swin` | `swin_tiny_patch4_window7_224.ms_in1k` | Shifted-window hierarchical transformer |
| `efficientformer` | `efficientformer_l1.snap_dist_in1k` | Compact, latency-oriented candidate |
| `convnext` | `convnext_tiny.fb_in1k` | CNN-style reference and frozen teacher |

These are checkpoint-level comparisons: an architecture, its upstream training recipe, and its
published weights arrive together in practice. For that reason, a broad ViT-versus-DeiT comparison
cannot isolate one causal mechanism. The narrower Distilled DeiT experiment can: standard and
hard-KD runs use the same student architecture, initialization family, split, and seeds; only the
teacher loss changes.

The model selection also separates **load-bearing ideas** from a larger paper map:

- ViT supplies patch tokens and global self-attention.
- DeiT shows that the training recipe and teacher supervision can matter as much as a new block.
- Swin supplies local windows, shifted-window communication, patch merging, and a feature hierarchy.
- EfficientFormer represents compact, latency-oriented hybrid design.
- ConvNeXt provides a strong modern convolutional reference and a teacher whose inference graph is
  not needed after student training.

MobileViT, TinyViT, and MaxViT remain valuable related work, but adding every family would make this
chapter project slower and less controlled. The selected set is large enough to expose the central
trade-offs while remaining reproducible within a few hours of rented GPU time.

## 3. Experimental design

### 3.1 Dataset and split discipline

The task is CIFAR-100 classification. CIFAR-100 contains 100 classes and 32×32 source images. The
official 50,000-image training split was divided once, with split seed `2027`, into:

- **45,000 training images:** 450 images per class;
- **5,000 validation images:** 50 images per class; and
- **10,000 test images:** the untouched official test split.

Every run used exactly these indices. Checkpoint selection used validation accuracy only, with
validation loss as the tie-breaker. The official test split was evaluated after training and was not
used for early stopping or hyperparameter selection. This separation matters: repeatedly choosing
settings from test accuracy would turn the test set into another validation set.

All inputs were resized to 224×224 to match the pretrained checkpoints. Training augmentation was:

1. random 32×32 crop with four pixels of reflected padding;
2. random horizontal flip;
3. RandAugment with `num_ops=2` and `magnitude=9`;
4. bicubic resize to 224×224;
5. checkpoint-native normalization; and
6. random erasing with probability `0.25`.

Validation and test preprocessing was deterministic: resize followed by the same checkpoint-native
normalization. Resizing a 32×32 dataset to 224×224 is intentionally a transfer-learning exercise;
it should not be confused with evidence from naturally high-resolution data.

### 3.2 Run matrix

The complete preset contains 13 training runs:

| Model and mode | Seeds | Runs | Purpose |
| --- | --- | ---: | --- |
| ConvNeXt-T, standard | 42 | 1 | Teacher and CNN reference |
| ViT-S/16, standard | 42 | 1 | Plain ViT baseline |
| Swin-T, standard | 42 | 1 | Hierarchical/windowed baseline |
| EfficientFormer-L1, standard | 42 | 1 | Compact candidate |
| DeiT-S/16, standard | 42, 43, 44 | 3 | Multi-seed DeiT estimate |
| Distilled DeiT-S/16, standard | 42, 43, 44 | 3 | No active teacher during fine-tuning |
| Distilled DeiT-S/16, hard KD | 42, 43, 44 | 3 | Paired active-distillation treatment |

Only the DeiT family received three seeds because the controlled distillation question is the main
training ablation. The other model rows are useful engineering observations, but one seed cannot
support strong claims about small accuracy differences.

### 3.3 Fine-tuning recipe

All runs used the `full` preset:

| Setting | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 64 |
| Optimizer | AdamW |
| Pretrained backbone learning rate | `5e-5` |
| Classification-head learning rate | `5e-4` |
| Weight decay | `0.05` |
| Schedule | 2-epoch warmup, then cosine decay |
| Minimum learning rate | `1e-6` |
| Label smoothing | `0.1` |
| Gradient clipping | `1.0` |
| Automatic mixed precision | Enabled |
| Input size | 224×224 |

The 10× higher head learning rate lets the new 100-class classifier adapt more quickly while
preserving useful pretrained features. Each checkpoint contains the model, optimizer, scheduler,
random-number-generator, and data-generator state, so an interrupted run can resume without silently
changing its planned data order.

### 3.4 Hard distillation

The ConvNeXt-T teacher was fine-tuned first and then frozen. During a hard-KD student run, the
teacher's predicted class becomes a target for the distillation head. With `alpha = 0.5`, the loss is

\[
\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{class}}
             + \alpha\,\mathcal{L}_{\text{teacher}}.
\]

The teacher is in evaluation mode and receives no gradients. At student evaluation time, the class
and distillation-head predictions are averaged. The teacher adds work only while training; it is not
part of the deployed student.

### 3.5 How to read the metrics

| Metric | Meaning in this report |
| --- | --- |
| Top-1 accuracy | Percentage for which the highest-scoring class is correct |
| Top-5 accuracy | Percentage for which the correct class appears among the five highest scores |
| Best validation accuracy | Selection metric used to choose `best.pt`; not the final test score |
| Parameters | Live parameter count after replacing the classifier with a 100-class head |
| GMACs | Traced multiply-accumulate estimate for one 224×224 image |
| Peak memory | Maximum PyTorch accelerator allocator usage inside the measured operation |
| Batch latency | Synchronized time for one complete model forward pass at the stated batch size |
| Images per second | `batch_size × 1000 / mean_latency_ms`; higher is better |

Accuracy measures predictive quality; parameters approximate storage capacity; MACs approximate
work; and latency and throughput measure the realized implementation. They answer different
questions and should not be collapsed into one vague notion of “efficiency.”

## 4. How the experiment was executed

The final run used Modal with a persistent volume named
`vision-transformer-benchmark-data`. The successful artifacts live remotely under
`/artifacts/full` and have also been copied into this repository where noted below.

| Provenance field | Recorded value |
| --- | --- |
| Completion date | 16 August 2026 |
| Modal app ID | `ap-731KstAj4aSwjYegWlbGfo` |
| State at handoff | Stopped; zero running tasks |
| Training/benchmark accelerator | NVIDIA L4 |
| Persistent volume | `vision-transformer-benchmark-data` |
| Remote result root | `/artifacts/full` |
| Local result root | `modal_full_results_20260816` |

Execution was staged deliberately:

1. Train the ConvNeXt teacher first.
2. Launch the remaining student jobs with bounded concurrency of at most four NVIDIA L4
   containers.
3. Benchmark every architecture sequentially in one L4 container, so devices are not mixed within
   the timing table.
4. Generate CSV tables, figures, and the machine-generated report on a CPU container.

The Modal functions commit the persistent volume after every checkpoint and automatically resume
from the latest committed epoch after a retry or restart. This was important in practice because an
earlier execution stopped between stages; the final invocation reused completed work rather than
starting it again. The completed app has been stopped, so it is no longer consuming compute, while
the volume retains the artifacts.

The completed run is identified by configuration fingerprint:

```text
7b87d57ae5546600c460245a81431288db34c3e40a44c2e83da01acc4160d731
```

That fingerprint appears in the report and checkpoints and prevents results from an incompatible
configuration from being combined accidentally.

### 4.1 Compute accounting

The recorded training durations sum to **27,362.9 seconds**, or approximately **7.60 L4
GPU-hours**, across all 13 runs. Parallel student jobs reduced the successful workflow's elapsed
wall-clock time to roughly **2 hours 4 minutes**; wall time is therefore not the same as total GPU
consumption. Based on the recorded latency means, the configured inference warm-ups and timed
forwards account for about another **18 minutes** of L4 execution, before model loading, MAC tracing,
and container overhead.

These are workload measurements, not a billing statement. Provider charges can also include
container startup, downloads, CPU, memory, retries, and rounding, and the applicable unit price can
change. For planning a repeat, budget approximately eight L4 GPU-hours plus setup and benchmark
overhead, while expecting much less elapsed time when four training containers are available.

## 5. Accuracy and training results

### 5.1 Aggregate comparison

| Model | Mode | Seeds | Test top-1 (%) | Best val top-1 (%) | Params (M) | Mean train time (min) | Peak train memory (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-T | Standard | 1 | **89.78** | 89.86 | 27.90 | 46.49 | 4,245.67 |
| ViT-S/16 | Standard | 1 | 89.22 | 89.60 | 21.70 | 27.89 | 2,563.02 |
| Swin-T | Standard | 1 | 89.54 | 89.54 | 27.60 | 60.52 | 4,843.58 |
| EfficientFormer-L1 | Standard | 1 | 86.83 | 86.86 | **11.48** | 28.35 | 2,633.86 |
| DeiT-S/16 | Standard | 3 | 88.86 ± 0.03 | 89.72 | 21.70 | 27.94 | 2,563.02 |
| Distilled DeiT-S/16 | Standard | 3 | 89.46 ± 0.27 | 90.17 | 21.74 | 28.36 | 2,566.76 |
| Distilled DeiT-S/16 | Hard KD | 3 | **89.72 ± 0.11** | **90.27** | 21.74 | 41.29 | 3,051.61 |

The `±` values are sample standard deviations across seeds. A blank uncertainty estimate for a
single-seed row should not be read as zero variance.

![Validation accuracy curves for all completed runs](modal_full_results_20260816/report/figures/validation_accuracy.png)

Several best checkpoints occurred at epoch 19 or 20, and only ConvNeXt selected substantially
earlier, at epoch 17. The 20-epoch budget therefore appears close to the useful end of this preset,
but the experiment does not prove that every model was fully saturated. Extending training would be
a new experiment and should preserve the same selection discipline.

### 5.2 Every completed run

| Model | Mode | Seed | Best epoch | Best val top-1 | Test top-1 | Test top-5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-T | Standard | 42 | 17 | 89.86 | 89.78 | 98.86 |
| ViT-S/16 | Standard | 42 | 20 | 89.60 | 89.22 | 98.32 |
| Swin-T | Standard | 42 | 20 | 89.54 | 89.54 | 98.65 |
| EfficientFormer-L1 | Standard | 42 | 20 | 86.86 | 86.83 | 98.26 |
| DeiT-S/16 | Standard | 42 | 20 | 89.80 | 88.89 | 98.09 |
| DeiT-S/16 | Standard | 43 | 20 | 89.70 | 88.86 | 98.22 |
| DeiT-S/16 | Standard | 44 | 18 | 89.66 | 88.83 | 98.28 |
| Distilled DeiT-S/16 | Standard | 42 | 19 | 90.04 | 89.74 | 98.25 |
| Distilled DeiT-S/16 | Standard | 43 | 20 | 90.16 | 89.21 | 98.24 |
| Distilled DeiT-S/16 | Standard | 44 | 19 | 90.30 | 89.42 | 98.31 |
| Distilled DeiT-S/16 | Hard KD | 42 | 20 | 90.46 | **89.83** | 98.88 |
| Distilled DeiT-S/16 | Hard KD | 43 | 20 | 90.06 | 89.61 | **98.94** |
| Distilled DeiT-S/16 | Hard KD | 44 | 19 | 90.28 | 89.72 | 98.85 |

The best individual result, 89.83%, came from hard-KD seed 42. ConvNeXt's 89.78% is the strongest
single-seed reference row, while the hard-KD aggregate is the strongest multi-seed transformer
result. Their 0.06-point difference is too small to treat as meaningful without matched repetitions.

## 6. What did distillation buy?

The most defensible causal result in the project is the paired standard-versus-hard-KD comparison:

| Seed | Standard Distilled DeiT | Hard KD | Paired gain |
| ---: | ---: | ---: | ---: |
| 42 | 89.74 | 89.83 | +0.09 points |
| 43 | 89.21 | 89.61 | +0.40 points |
| 44 | 89.42 | 89.72 | +0.30 points |
| **Mean** | **89.46** | **89.72** | **+0.26 points** |

![Paired hard-distillation gains](modal_full_results_20260816/report/figures/distillation_gain.png)

All three paired differences are positive. The treatment also reduced the observed test-accuracy
standard deviation from 0.27 to 0.11 points in these three runs. Three seeds are too few to claim a
general variance reduction, but the direction is worth following up.

The gain was not free during training:

- mean training time rose from **28.36 to 41.29 minutes**, an increase of about **45.6%**;
- peak CUDA allocator memory rose from roughly **2,566.76 to 3,051.61 MB**, about **18.9%**; and
- inference cost stayed unchanged because both treatments deploy the same student architecture.

This is the classic attraction of offline distillation: pay for a teacher during training, then
retain only the student at inference. Whether +0.26 points is worth the added training cost depends
on how often training happens and how valuable final accuracy is.

The broader comparison between ordinary DeiT-S/16 (88.86%) and Distilled DeiT without an active
teacher (89.46%) shows a +0.60-point difference, but it is **not** a clean estimate of distillation.
Those rows use different upstream checkpoints and a different token/head structure. The paired
hard-KD table is the appropriate evidence for the teacher's fine-tuning effect.

## 7. Parameters and operation counts

| Architecture | Task parameters (M) | Measured GMACs at 224×224 |
| --- | ---: | ---: |
| EfficientFormer-L1 | **11.48** | **1.317** |
| ViT-S/16 | 21.70 | 4.250 |
| DeiT-S/16 | 21.70 | 4.250 |
| Distilled DeiT-S/16 | 21.74 | 4.272 |
| ConvNeXt-T | 27.90 | 4.470 |
| Swin-T | 27.60 | 4.509 |

These are live counts after replacing the original classifier with a 100-class head. The operation
counter uses `fvcore`'s MAC-style convention: one multiply-add is reported as one operation. It is
better to call these **MACs**, not universal FLOPs. Some operators are not counted by the tracer;
the exact unsupported-operator lists are preserved in
[`model_profiles.csv`](modal_full_results_20260816/report/model_profiles.csv).

EfficientFormer used about 47% fewer parameters and 69% fewer measured MACs than ViT. Those are
important advantages for storage, bandwidth, and some runtimes. They did not, however, produce a
69% throughput improvement on the L4. Operation count describes computational work at one level;
runtime also depends on kernel quality, memory traffic, parallelism, launch overhead, precision,
and batch shape.

## 8. Inference benchmark

### 8.1 Measurement protocol

All six architectures were benchmarked sequentially on one **NVIDIA L4** with:

- PyTorch `2.9.1+cu128`, torchvision `0.24.1+cu128`, and `timm 1.0.28`;
- FP32 and FP16;
- batch sizes 1 and 64;
- 50 warm-up iterations;
- 200 timed iterations per repeat;
- 5 repeats, or 1,000 timed samples per model/precision/batch case;
- CUDA events plus synchronization; and
- random input tensors already resident on the GPU.

The timings include the model forward pass and exclude image decoding, preprocessing, data loading,
host-to-device transfer, postprocessing, networking, and service overhead. Peak memory is PyTorch's
CUDA allocator peak, not whole-system memory or energy use. All **24 of 24** configured benchmark
cases completed.

### 8.2 Main FP16 results

| Model | Batch-1 latency (ms) | Batch-1 images/s | Batch-64 latency (ms) | Batch-64 images/s | Batch-64 peak MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-T | 5.377 | 185.98 | 57.764 | 1,107.96 | 728.76 |
| ViT-S/16 | **4.694** | **213.04** | **32.072** | **1,995.49** | **320.05** |
| Swin-T | 9.133 | 109.49 | 84.709 | 755.52 | 970.56 |
| EfficientFormer-L1 | 6.209 | 161.06 | 32.634 | 1,961.13 | 354.11 |
| DeiT-S/16 | 4.723 | 211.73 | 32.136 | 1,991.55 | **320.05** |
| Distilled DeiT-S/16 | 4.723 | 211.73 | 32.342 | 1,978.86 | 321.56 |

ViT and DeiT are effectively tied at batch 64: their reported means differ by less than 0.2%.
EfficientFormer is also in the same broad FP16 throughput band despite having far fewer MACs. Swin
is the slowest and has the largest inference allocator peak in this environment.

![Accuracy versus FP16 batch-64 throughput](modal_full_results_20260816/report/figures/accuracy_throughput.png)

### 8.3 Precision changes the ranking

| Model | FP32 b1 images/s | FP16 b1 images/s | FP32 b64 images/s | FP16 b64 images/s |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-T | 209.30 | 185.98 | 500.35 | 1,107.96 |
| ViT-S/16 | **233.61** | **213.04** | 598.39 | **1,995.49** |
| Swin-T | 123.19 | 109.49 | 359.25 | 755.52 |
| EfficientFormer-L1 | 173.24 | 161.06 | **1,128.16** | 1,961.13 |
| DeiT-S/16 | 228.91 | 211.73 | 599.09 | 1,991.55 |
| Distilled DeiT-S/16 | 228.13 | 211.73 | 607.31 | 1,978.86 |

FP16 was slower than FP32 at batch 1 for every model. At such a small batch, conversion and kernel
launch overhead can outweigh tensor-core savings. At batch 64, FP16 was substantially faster for
every architecture. EfficientFormer led the FP32 batch-64 table, while ViT narrowly led the FP16
table. A statement such as “model A is faster” is incomplete unless it names the hardware,
precision, batch size, software stack, and timing boundary.

The raw means, medians, p90 values, standard deviations, sample counts, and memory measurements are
available in
[`inference_benchmark.csv`](modal_full_results_20260816/report/inference_benchmark.csv).

## 9. Interpretation by model family

### Plain ViT: stronger than its reputation under transfer learning

ViT-S/16 reached 89.22% and delivered the best measured FP16 batch-64 throughput with low inference
memory. This does not contradict the claim that plain ViT can be data hungry from scratch. It shows
that a pretrained ViT can be a very practical small-data fine-tuning baseline. Global attention is
not automatically too expensive at a 14×14 token grid.

### DeiT: the recipe is a load-bearing idea

Standard DeiT averaged 88.86% across three seeds, while the distilled checkpoint family did better.
The central lesson is not that the DeiT block is universally superior to ViT—the architectures are
close and the upstream checkpoints differ. The lesson is that augmentation, regularization,
optimization, and teacher supervision deserve first-class experimental treatment.

### Swin: hierarchy is useful, but it is not a free speedup

Swin-T reached a competitive 89.54%, illustrating that windowed attention and hierarchical feature
maps work. In this PyTorch/L4 setup, however, it trained longest, used the most peak memory, and had
the lowest inference throughput. Window locality changes asymptotic attention behavior, but shifted
windows, tensor layouts, and kernel implementations still determine wall-clock performance. Swin's
hierarchy may be more valuable for detection and segmentation than this classification-only project
can reveal.

### EfficientFormer: compact does not mean universally fastest

EfficientFormer-L1 was clearly the smallest model and had the fewest measured MACs. Its 86.83%
accuracy was 2.39 points below ViT and 2.89 points below hard-KD Distilled DeiT. On the L4 it nearly
matched ViT's FP16 batch-64 throughput, but did not beat it. The compact model may still be preferable
when storage, download size, CPU/mobile execution, or energy dominates; this GPU experiment does not
measure those targets. A real mobile recommendation requires benchmarking the exported model on the
actual runtime and device.

### ConvNeXt: a strong CNN remains a relevant baseline

ConvNeXt-T reached 89.78%, the best single-seed architecture row, and served as an effective teacher.
It was slower and larger than ViT/DeiT in the FP16 L4 benchmark. This is exactly why a CNN-style
baseline belongs in the chapter: architectural novelty should be compared with a strong, modern
convolutional model, not with an obsolete strawman.

## 10. Decision guide

For this exact task and L4 software stack:

| Constraint | Sensible starting point | Reason |
| --- | --- | --- |
| Highest transformer accuracy | Hard-KD Distilled DeiT-S/16 | Best multi-seed transformer mean; no teacher at inference |
| FP16 GPU throughput | ViT-S/16 or DeiT-S/16 | Approximately 2,000 images/s at batch 64 |
| Lowest parameter/MAC budget | EfficientFormer-L1 | 11.48M parameters and 1.32 GMACs |
| CNN-style accuracy reference | ConvNeXt-T | 89.78% in its single completed run |
| Hierarchical features for downstream dense tasks | Swin-T, then benchmark again | Classification result alone does not value the full hierarchy |
| Minimal training complexity | Plain ViT/DeiT fine-tuning | Avoids the teacher's training-time cost |

This table is a starting point, not a deployment guarantee. If the target is a phone, browser, CPU,
TensorRT service, or different accelerator, export the shortlisted models and rerun the benchmark on
that exact stack.

## 11. Threats to validity and limitations

1. **Single-seed architecture rows.** ViT, Swin, EfficientFormer, and ConvNeXt were run once. Small
   differences between those rows should not be interpreted as statistically settled rankings.
2. **Checkpoint-level, not architecture-only, comparison.** The pretrained recipes and weights
   differ. Only the paired standard-versus-hard-KD runs isolate the active teacher treatment.
3. **Upsampled low-resolution data.** CIFAR-100 images begin at 32×32 and are resized to 224×224.
   This is a controlled transfer task, not a substitute for naturally high-resolution benchmarks.
4. **One GPU and software stack.** Kernel support can reorder models across hardware, framework
   versions, compilers, and deployment runtimes.
5. **No end-to-end serving measurement.** The inference benchmark excludes input pipelines, data
   transfer, postprocessing, network latency, power, and energy.
6. **No quantization or compiler optimization.** Results use eager PyTorch FP32/FP16, not INT8,
   `torch.compile`, TensorRT, Core ML, TFLite, or vendor-specific graph optimization.
7. **MAC tracing is incomplete by construction.** Unsupported operators are disclosed rather than
   silently treated as proof of exact FLOPs.
8. **Finite training budget.** Many runs selected epoch 19 or 20. Longer schedules could change the
   result, but would require a fresh controlled run.
9. **No dense prediction task.** Classification does not fully test the practical value of Swin's
   multi-scale hierarchy.

## 12. Reproducibility record and artifact map

The completed local result bundle is
[`modal_full_results_20260816`](modal_full_results_20260816). Important files are:

| Artifact | Purpose |
| --- | --- |
| [`benchmark.json`](modal_full_results_20260816/benchmark.json) | Complete hardware/software metadata and raw inference measurements |
| [`run_results.csv`](modal_full_results_20260816/report/run_results.csv) | One row for every training run |
| [`epoch_metrics.csv`](modal_full_results_20260816/report/epoch_metrics.csv) | Per-epoch learning curves and timing |
| [`aggregate_results.csv`](modal_full_results_20260816/report/aggregate_results.csv) | Grouped mean and standard deviation |
| [`distillation_gains.csv`](modal_full_results_20260816/report/distillation_gains.csv) | Seed-matched KD differences |
| [`model_profiles.csv`](modal_full_results_20260816/report/model_profiles.csv) | Parameters, MACs, and unsupported tracing operators |
| [`inference_benchmark.csv`](modal_full_results_20260816/report/inference_benchmark.csv) | All 24 timing and memory cases |
| [`report.md`](modal_full_results_20260816/report/report.md) | Automatically generated compact report |
| [Hugging Face checkpoints](https://huggingface.co/Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book/tree/main/chapter-04-efficient-and-scalable-vision-transformers/checkpoints) | Thirteen public model-only Safetensors files |
| [Hugging Face manifest](https://huggingface.co/Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book/blob/main/chapter-04-efficient-and-scalable-vision-transformers/manifest.json) | Model paths, metrics, source hashes, and exported hashes |

### Hugging Face distribution

Model-only inference exports are stored in the public repository
[`Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book`](https://huggingface.co/Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book),
under `chapter-04-efficient-and-scalable-vision-transformers/`. The Hub bundle contains 13
Safetensors files totaling 1.06 GiB, a machine-readable provenance manifest, exported SHA-256
checksums, the full experiment configuration, tables, figures, and a Hub-adapted copy of this
report. Every remote file size matched the local export, and a fresh download was checksum-verified
and loaded strictly into its `timm` architecture.

Before public release, the six official upstream model-card licenses were checked: five report
Apache-2.0 and the Swin-T source reports MIT. The published inference files contain no optimizer,
scheduler, gradient-scaler, or random-number-generator state.

After all 13 public Hugging Face files passed path, byte-size, and LFS SHA-256 verification, the
local `modal_full_results_20260816/checkpoints` directory was deleted at the user's request. The
book project workspace now contains no `.pt` or `.safetensors` weight files. The Modal volume remains
the durable source for complete resumable `best.pt` and `latest.pt` run artifacts.

To download and verify every published model file:

```bash
hf download \
  Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book \
  --include "chapter-04-efficient-and-scalable-vision-transformers/checkpoints/*" \
  --local-dir transformer-vision-book

cd transformer-vision-book/chapter-04-efficient-and-scalable-vision-transformers/checkpoints
shasum -a 256 -c SHA256SUMS
```

Each line should end in `OK`. A mismatch means that file should be downloaded again before use.

## 13. Reproducing the experiment

From the project root, install and validate the environment as described in the README, then choose
one of the following paths.

### Run the complete workflow on Modal

```bash
uv run modal run modal_app.py --preset full --stage all
```

The teacher runs first, the students run with bounded concurrency, the inference suite uses one L4,
and the report stage follows. Completed checkpoints are resumable, so rerunning after interruption
does not intentionally discard finished epochs.

Individual stages can also be run explicitly:

```bash
uv run modal run modal_app.py --preset full --stage train
uv run modal run modal_app.py --preset full --stage benchmark
uv run modal run modal_app.py --preset full --stage report
```

### Run on a local CUDA machine

```bash
uv run vision-bench all --preset full --device cuda
```

For a first instructional pass, use the shorter preset documented in the README before committing
to the full multi-seed run.

### Regenerate the report from existing artifacts

```bash
uv run vision-bench report --preset full
```

### Run the quality gates

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
```

At handoff, the project passed **41 tests with 1 hardware-dependent MPS test skipped**, and both
Ruff and mypy completed cleanly.

## 14. Loading a downloaded checkpoint

The public files are model-only Safetensors state dictionaries. The following example downloads and
restores the validation-selected ViT on CPU without relying on a local project checkpoint:

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

For Distilled DeiT, disable distilled-training output before ordinary evaluation if the installed
`timm` model exposes the switch:

```python
if hasattr(model, "set_distilled_training"):
    model.set_distilled_training(False)
```

Use the chapter model card's checkpoint-native preprocessing instructions so resize, interpolation,
mean, and standard deviation match training. The published files support inference and further
fine-tuning, but they do not contain optimizer, scheduler, scaler, or RNG state for exact resume.
Complete resumable training artifacts remain on the Modal volume.

## 15. Final takeaway

No single number answered this project. ViT and DeiT were excellent FP16 GPU throughput baselines;
hard distillation bought a repeatable but modest accuracy gain; EfficientFormer minimized size and
MACs without becoming universally faster; Swin's valuable hierarchy carried real implementation
cost on this classifier; and ConvNeXt remained highly competitive. The practical method is the
lasting result: select checkpoints without touching the test set, repeat the comparison that is
meant to be causal, disclose uncertainty, profile live models, and benchmark the exact deployment
shape instead of choosing from FLOPs alone.

That is the chapter's main engineering lesson: efficient vision is not a property of an architecture
name. It is an accuracy–training–memory–latency trade-off measured under a stated workload.

## 16. Primary references

- Dosovitskiy et al., [*An Image Is Worth 16×16 Words: Transformers for Image Recognition at
  Scale*](https://arxiv.org/abs/2010.11929) — ViT.
- Touvron et al., [*Training data-efficient image transformers & distillation through
  attention*](https://arxiv.org/abs/2012.12877) — DeiT and the distillation token.
- Liu et al., [*Swin Transformer: Hierarchical Vision Transformer using Shifted
  Windows*](https://arxiv.org/abs/2103.14030) — window attention, shifts, and hierarchy.
- Li et al., [*EfficientFormer: Vision Transformers at MobileNet
  Speed*](https://arxiv.org/abs/2206.01191) — compact, latency-oriented design.
- Liu et al., [*A ConvNet for the 2020s*](https://arxiv.org/abs/2201.03545) — ConvNeXt.
- Mehta and Rastegari, [*MobileViT*](https://arxiv.org/abs/2110.02178); Wu et al.,
  [*TinyViT*](https://arxiv.org/abs/2207.10666); and Tu et al.,
  [*MaxViT*](https://arxiv.org/abs/2204.01697) — practical variants for the chapter's paper map.

The exact executable checkpoint identifiers are pinned in
[`src/vision_bench/models.py`](src/vision_bench/models.py). Architecture names alone are not enough
to reproduce a pretrained-checkpoint comparison.
