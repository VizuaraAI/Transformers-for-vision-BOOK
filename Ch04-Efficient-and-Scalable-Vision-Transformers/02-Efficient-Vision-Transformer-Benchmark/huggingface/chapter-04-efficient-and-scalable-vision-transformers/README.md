# Chapter 4 — Efficient and scalable vision transformers

## ViT, DeiT, Swin, EfficientFormer, and hard distillation on CIFAR-100

This folder contains the validation-selected inference weights and measured artifacts for Chapter
4's controlled image-classification project. Six architectures were fine-tuned from pinned `timm`
checkpoints on the same fixed CIFAR-100 split. The experiment measured accuracy, training behavior,
parameters, MAC-style operation counts, accelerator memory, latency, and throughput.

The published `.safetensors` files contain **model parameters only**. They are derived from the
verified `best.pt` files selected by validation top-1 accuracy; optimizer and resume state are not
included.

## Experiment summary

- Dataset: CIFAR-100
- Split: 45,000 train / 5,000 validation / 10,000 official test
- Input: 224×224
- Training: 20 epochs, AdamW, mixed precision, batch size 64
- Hardware: NVIDIA L4
- Configuration fingerprint:
  `7b87d57ae5546600c460245a81431288db34c3e40a44c2e83da01acc4160d731`
- Completed training runs: 13
- Completed inference cases: 24

## Accuracy results

| Model | Mode | Seeds | Test top-1 (%) | Published checkpoint folder |
| --- | --- | --- | ---: | --- |
| ConvNeXt-T | Standard | 42 | 89.78 | `checkpoints/convnext-t/standard/seed-42` |
| ViT-S/16 | Standard | 42 | 89.22 | `checkpoints/vit-s16/standard/seed-42` |
| Swin-T | Standard | 42 | 89.54 | `checkpoints/swin-t/standard/seed-42` |
| EfficientFormer-L1 | Standard | 42 | 86.83 | `checkpoints/efficientformer-l1/standard/seed-42` |
| DeiT-S/16 | Standard | 42, 43, 44 | 88.86 ± 0.03 | `checkpoints/deit-s16/standard/seed-*` |
| Distilled DeiT-S/16 | Standard | 42, 43, 44 | 89.46 ± 0.27 | `checkpoints/deit-distilled-s16/standard/seed-*` |
| Distilled DeiT-S/16 | Hard KD | 42, 43, 44 | **89.72 ± 0.11** | `checkpoints/deit-distilled-s16/hard-kd/seed-*` |

The `±` value is the sample standard deviation across seeds. Single-seed rows do not have an
uncertainty estimate.

## Exact upstream model identifiers

| Export folder | `timm` model identifier | Upstream model-card license |
| --- | --- | --- |
| `convnext-t` | `convnext_tiny.fb_in1k` | Apache-2.0 |
| `vit-s16` | `vit_small_patch16_224.augreg_in1k` | Apache-2.0 |
| `swin-t` | `swin_tiny_patch4_window7_224.ms_in1k` | MIT |
| `efficientformer-l1` | `efficientformer_l1.snap_dist_in1k` | Apache-2.0 |
| `deit-s16` | `deit_small_patch16_224.fb_in1k` | Apache-2.0 |
| `deit-distilled-s16` | `deit_small_distilled_patch16_224.fb_in1k` | Apache-2.0 |

All exported models have a 100-class task head. Instantiate with `pretrained=False`; loading the
published state replaces every parameter.

## Download and load one model

Install the runtime packages:

```bash
python -m pip install huggingface_hub safetensors timm torch
```

Then download and restore the ViT seed-42 model:

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

For Distilled DeiT evaluation, disable training-time tuple output when supported:

```python
if hasattr(model, "set_distilled_training"):
    model.set_distilled_training(False)
```

## Preprocessing

Use checkpoint-native `timm` preprocessing. For a PIL image:

```python
from timm.data import create_transform, resolve_model_data_config

data_config = resolve_model_data_config(model)
transform = create_transform(**data_config, is_training=False)
input_tensor = transform(pil_image).unsqueeze(0)
```

The classifier outputs CIFAR-100 class indices. Class labels and the fixed split are defined by the
chapter project; do not interpret the output as the original ImageNet-1K label space.

## Hard-distillation result

Hard KD held the Distilled DeiT architecture and seed fixed while adding a frozen, fine-tuned
ConvNeXt teacher during training. Test top-1 changed by:

| Seed | Standard | Hard KD | Gain |
| ---: | ---: | ---: | ---: |
| 42 | 89.74 | 89.83 | +0.09 points |
| 43 | 89.21 | 89.61 | +0.40 points |
| 44 | 89.42 | 89.72 | +0.30 points |

The mean paired gain was +0.26 percentage points. The teacher is not required when loading or
deploying a hard-KD student.

## Artifact map

```text
chapter-04-efficient-and-scalable-vision-transformers/
├── README.md
├── EXPERIMENT_REPORT.md
├── manifest.json
├── configs/
│   └── full.yaml
├── checkpoints/
│   ├── SHA256SUMS
│   ├── convnext-t/standard/seed-42/model.safetensors
│   ├── vit-s16/standard/seed-42/model.safetensors
│   ├── swin-t/standard/seed-42/model.safetensors
│   ├── efficientformer-l1/standard/seed-42/model.safetensors
│   ├── deit-s16/standard/seed-{42,43,44}/model.safetensors
│   └── deit-distilled-s16/{standard,hard-kd}/seed-{42,43,44}/model.safetensors
└── results/
    ├── benchmark.json
    ├── aggregate_results.csv
    ├── run_results.csv
    ├── inference_benchmark.csv
    └── figures/
```

`manifest.json` is the authoritative model-to-file mapping and includes per-run metrics and both
source and exported checksums.

## Verify downloaded weights

From the chapter folder:

```bash
cd checkpoints
shasum -a 256 -c SHA256SUMS
```

Every line should end in `OK`.

## Interpretation and limitations

The main finding is that theoretical efficiency and measured runtime are different quantities.
EfficientFormer used the fewest parameters and MACs, while ViT/DeiT achieved the highest FP16
batch-64 throughput on the L4. Swin was competitive in accuracy but slower in this implementation.
Hard distillation improved all three paired seeds at extra training cost and no additional inference
cost.

These results are not a universal leaderboard. Most broad architecture rows use one seed; upstream
pretraining recipes differ; CIFAR-100 images are enlarged from 32×32 to 224×224; inference excludes
data loading; and the timing ranking is specific to the recorded hardware and software stack. See
`EXPERIMENT_REPORT.md` for the complete analysis.

## Upstream references and terms

The artifacts are fine-tuned derivatives of the exact `timm` checkpoints listed above. The license
column records the metadata returned by their official Hugging Face model cards at publication.
Review those source cards and terms, along with CIFAR-100's terms, before redistribution or
commercial use. Publishing a derived weight file does not remove upstream attribution or license
obligations.
