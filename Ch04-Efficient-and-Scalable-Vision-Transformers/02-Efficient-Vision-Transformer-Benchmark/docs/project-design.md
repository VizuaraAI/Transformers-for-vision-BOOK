# Project design

## Research question

Under one modest transfer-learning budget, how do a plain vision transformer, a data-efficient ViT
checkpoint, a hierarchical windowed transformer, and a compact latency-oriented transformer trade
classification accuracy for model and systems cost?

The project is a practical checkpoint comparison, not a controlled architecture-only study.
Published pretrained weights encode different pretraining datasets, augmentations, regularization,
teachers, and optimization recipes. That is useful when choosing something to deploy, but it limits
causal claims about any single design feature.

## Why these models

### ViT-S/16: global-attention baseline

ViT tokenizes an image into non-overlapping patches, adds positional information, and applies global
self-attention. For `N` tokens, the attention map grows with `N²`; high-resolution images therefore
increase both computation and activation memory quickly. ViT also begins with weaker locality and
translation priors than a CNN, so the original training story relied heavily on large-scale
pretraining. The selected small checkpoint keeps the experiment feasible while preserving this
plain design.

### DeiT-S/16: the training-recipe story

DeiT showed that strong augmentation, regularization, optimization, and distillation could make a
ViT-style model competitive using ImageNet-scale data. Ordinary DeiT-S is architecturally very close
to ViT-S. Its role here is to demonstrate that “efficient” can refer to the data/training recipe,
not only fewer layers or cheaper attention.

The project also includes Distilled DeiT-S, which adds a distillation token and second classifier.
It is fine-tuned once without a live teacher and once with hard teacher labels. Those matched runs
are the cleanest estimate of training-time distillation gain in this project.

### Swin-T: hierarchy and shifted windows

Swin restricts attention to local windows, shifts the window partition between blocks so information
crosses prior boundaries, and merges patches as depth increases. The result resembles a CNN feature
pyramid: spatial resolution falls while channel capacity grows. Window attention changes scaling
from global quadratic attention over the full token grid to attention over bounded windows, making
the hierarchy more practical for dense and higher-resolution vision tasks.

### EfficientFormer-L1: compact practical candidate

EfficientFormer was designed around real latency as well as nominal operation count. It uses
efficient convolution-like token processing through much of the network and reserves more expensive
attention for later stages. It is the project's constrained/mobile candidate because its parameter
and MAC budgets are substantially below the small ViT/Swin models while it remains available as a
well-supported pretrained `timm` checkpoint.

### ConvNeXt-T: teacher and reference

ConvNeXt modernizes a convolutional backbone using design choices that make the comparison with
transformers informative. It is not included to turn the project into a CNN survey. Its two roles
are to produce hard targets for Distilled DeiT and to provide one CNN-style accuracy/efficiency
reference.

## Load-bearing ideas versus the paper map

The chapter should teach a few reusable mechanisms in depth:

- patch tokenization and global-attention scaling;
- a strong recipe and distillation as forms of data efficiency;
- bounded windows, shifted connections, patch merging, and hierarchy; and
- the difference between FLOPs/MACs and observed latency.

MobileViT, TinyViT, and MaxViT belong in the broader paper map unless the reader chooses an extension:

| Family | Useful idea | Why it is not another main run |
| --- | --- | --- |
| MobileViT | CNN locality plus transformer context in a mobile block | Adds another hybrid and mobile-runtime question |
| TinyViT | Distillation and efficient hierarchical design | Overlaps the compact and distillation axes |
| MaxViT | Block and grid attention combine local and global exchange | More compute than needed for the constrained baseline |

Keeping one compact model makes the experiment interpretable and affordable. Adding every named
paper would expand the leaderboard without isolating more mechanisms.

## Claims the project can support

- measured accuracy and learning behavior on the frozen CIFAR-100 protocol;
- paired hard-distillation gain for the matched Distilled DeiT seeds;
- exact parameter counts and trace-based MAC estimates for these implementations;
- latency, throughput, and allocator memory on the recorded device and settings; and
- conditional model recommendations under the measured constraints.

## Claims it cannot support

- a universal ranking across datasets, resolutions, runtimes, or devices;
- an architecture-only explanation for ViT versus DeiT checkpoint differences;
- statistical uncertainty for the single-seed practical baselines;
- mobile efficiency from an L4 measurement alone;
- energy efficiency without a power measurement; or
- conclusions about detection or segmentation from classification only.

## Primary papers

- [An Image Is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [EfficientFormer](https://arxiv.org/abs/2206.01191)
- [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545)
- [MobileViT](https://arxiv.org/abs/2110.02178),
  [TinyViT](https://arxiv.org/abs/2207.10666), and
  [MaxViT](https://arxiv.org/abs/2204.01697) for the paper map
