# Interpreting results

The report generator separates raw runs, grouped accuracy, distillation pairs, hardware profiles,
and visualizations. This prevents one attractive chart from hiding which measurements are directly
comparable.

## Generate the result package

```bash
uv run vision-bench benchmark --preset full --device cuda
uv run vision-bench report --preset full
```

The report directory contains Markdown for the book, CSVs for independent analysis, PNG figures,
and a JSON summary for automation. No result is hard-coded in the repository.

## Accuracy

Top-1 accuracy is the percentage of examples whose highest-scoring class is correct. Top-5 is the
percentage whose true class appears among the five highest scores. CIFAR-100 makes top-5 informative,
but top-1 is the primary model-selection metric.

For three-seed rows, the report shows arithmetic mean ± sample standard deviation. This describes
observed seed variation; with only three values it is not a precise confidence interval. Single-seed
rows intentionally omit `± 0` because zero would imply knowledge the experiment does not have.

## Distillation gain

The defensible training-time distillation contrast is:

`Distilled DeiT hard KD − Distilled DeiT standard`, paired by seed.

This holds the student checkpoint family and architecture fixed. Ordinary DeiT versus Distilled DeiT
is still interesting, but it mixes a distillation-token architecture, pretrained checkpoint history,
and fine-tuning behavior. Describe it as a practical comparison rather than an isolated causal gain.

Check both the average gain and the three paired directions. A positive mean driven by one seed is a
weaker story than a consistent improvement across all seeds.

## Parameters and MACs

Parameters approximate model storage and some training memory, but activations and optimizer states
also matter. MAC estimates describe graph arithmetic for one input. They do not capture kernel
fusion, memory traffic, parallelism, launch overhead, or runtime support. That is why the project
measures actual latency and throughput too.

Inspect `unsupported_operators` in `model_profiles.csv`. If an architecture has unsupported traced
operators, treat its measured MAC count as a lower-bound-like estimate and use the retained
model-card number as a reasonableness comparison—not as a silent replacement.

## Latency and throughput

- Batch-1 median and p90 latency are useful for interactive, one-request-at-a-time systems.
- Batch-64 images/second is useful for a saturated batch service.
- Peak allocator memory answers whether the tested batch fits; it is not total machine memory.

Always state device, precision, batch size, input resolution, software versions, and whether data
loading is included. Here, data loading is excluded to isolate model execution.

## Training behavior

The validation-accuracy plot can reveal slow adaptation, instability, or an early plateau. Use
`epoch_metrics.csv` to compare loss and accuracy rather than inferring optimization quality from one
final number. Hard-KD rows expose classification and teacher loss separately; their scales need not
match, but divergence or a flat term is worth investigating.

Training images/second and duration are diagnostic, not the main hardware ranking: augmentation,
validation frequency, teacher forwards, and transient system load differ from inference timing.
Hard KD is expected to train more slowly because it executes a frozen teacher for every batch.

## Making a recommendation

A useful conclusion is conditional:

- Accuracy-first: choose the highest validation-selected test result, but acknowledge seed coverage.
- Throughput-first: filter to the deployment batch/precision and choose on measured images/second.
- Memory-constrained: first exclude models that exceed the memory budget, then compare accuracy.
- Balanced: inspect the Pareto frontier—models for which no other candidate is both more accurate
  and faster under the same timing setting.

Avoid averaging accuracy, parameters, and throughput into an arbitrary single score unless a real
product requirement supplies the weights.

## Required limitations paragraph

A complete book-project write-up should mention:

1. pretrained checkpoints have different upstream recipes, so the broad comparison is not causal;
2. CIFAR-100 is upsampled from 32 to 224 pixels;
3. practical baseline models use one seed while DeiT contrasts use three;
4. inference results apply to the recorded hardware/runtime only; and
5. classification does not establish behavior on detection, segmentation, or mobile deployment.
