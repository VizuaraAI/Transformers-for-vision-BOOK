# Experiment protocol

This page is the pre-registered contract for the project. Change it only by creating a new preset;
do not tune one model with information that the others did not receive.

## Dataset and split

The task is 100-class image classification on CIFAR-100. The official 50,000-image training split is
stratified with split seed `2027`:

- full training: 450 examples per class, 45,000 total;
- full validation: 50 examples per class, 5,000 total; and
- test: the untouched official 10,000-image test split.

The selected integer indices and a SHA-256 checksum are stored in `data/splits/`. Every model and
seed uses the same train/validation indices. Quick and smoke presets create separate balanced,
smaller manifests and are marked tutorial-only.

The validation split chooses the best epoch. Test labels do not affect training, early stopping,
hyperparameters, or model selection.

## Input pipeline

Training applies, in order:

1. random `32 × 32` crop with four-pixel reflected padding;
2. random horizontal flip;
3. RandAugment with two operations and magnitude 9;
4. bicubic resize to `224 × 224`;
5. conversion to a tensor and checkpoint-native normalization; and
6. random erasing with probability 0.25 in the full and quick presets.

Validation and test use deterministic bicubic resize, tensor conversion, and the same
checkpoint-native normalization. Geometry and augmentation policy are shared; normalization follows
the pretrained checkpoint because changing its expected input distribution would unfairly damage
transfer. The code refuses hard distillation if teacher and student normalization differ.

CIFAR images contain far less native detail than ImageNet-sized inputs. Upsampling to 224 pixels is
necessary for a simple checkpoint comparison, but it is also an explicit limitation.

## Shared fine-tuning recipe

| Setting | Full value |
| --- | --- |
| Epochs | 20 |
| Batch size | 64 |
| Optimizer | AdamW |
| Backbone learning rate | `5e-5` |
| Classifier learning rate | `5e-4` |
| Weight decay | 0.05 (biases, 1-D parameters, and model-declared tokens/position terms excluded) |
| Schedule | 2-epoch linear warmup, then cosine to `1e-6` |
| Label smoothing | 0.1 |
| Gradient clipping | global norm 1.0 |
| Precision | CUDA AMP; FP32 on CPU/MPS |
| Epoch selection | highest validation top-1; validation loss breaks ties |

The classifier receives a higher learning rate because it starts as a new 100-class layer. All
backbone parameters remain trainable. The suite uses fixed epoch budgets rather than model-specific
early stopping so training opportunity is comparable.

## Distillation protocol

ConvNeXt-T is fine-tuned first on exactly the same training split. Its best validation checkpoint is
frozen, placed in evaluation mode, and used only to produce an argmax class for each augmented input.

For hard token distillation, Distilled DeiT exposes two outputs:

- the class-token head receives smoothed cross-entropy against the true label; and
- the distillation-token head receives ordinary cross-entropy against the teacher's argmax class.

The combined loss is:

`L = (1 - α) L_class + α L_teacher`, with `α = 0.5`.

Teacher inference runs without gradients. At evaluation, the two student heads are averaged. In the
standard Distilled DeiT control, there is no live teacher; cross-entropy on the averaged logits sends
label supervision to both heads. This control keeps the student architecture fixed.

## Run matrix

| Model/training mode | Full seeds | Purpose |
| --- | --- | --- |
| ConvNeXt-T, standard | 42 | Teacher and CNN reference |
| ViT-S/16, standard | 42 | Plain transformer baseline |
| Swin-T, standard | 42 | Hierarchical/window baseline |
| EfficientFormer-L1, standard | 42 | Compact practical baseline |
| DeiT-S/16, standard | 42, 43, 44 | Recipe-oriented DeiT result |
| Distilled DeiT-S, standard | 42, 43, 44 | Same-architecture control |
| Distilled DeiT-S, hard KD | 42, 43, 44 | Active distillation treatment |

Only the three-seed DeiT comparisons receive a sample standard deviation. Single-seed values should
be described as practical measurements, not stable population estimates.

## Metrics

Training records each epoch's loss, top-1, top-5, images/second, duration, peak allocator memory,
learning rate, and cumulative elapsed time. Hard-KD runs additionally record the two loss terms.

Final evaluation records top-1 and top-5 test accuracy and saves predictions for later error
analysis. Parameter count comes from live tensors, not a copied paper table.

The inference benchmark uses random in-memory `224 × 224` tensors and therefore excludes decoding,
transforms, and data loading. For each precision/batch pair it performs configured warmup iterations,
then synchronized timed forwards across repeats. It reports mean, median, p90, standard deviation,
images/second, and peak allocator memory. FP16 reference rows are CUDA-only.

MACs are traced with fvcore, where a fused multiply-add is counted as one operation. Unsupported
operators are preserved alongside the count; the report also retains published/model-card expected
GMACs as a reasonableness check.

## Throughput validity rule

Compare throughput only within one `benchmark.json`: same device, software environment, input size,
precision, batch size, warmup, and timing method. Batch 1 approximates online latency; batch 64
measures a throughput-oriented service. Neither automatically predicts a phone, browser, or CPU
runtime.
