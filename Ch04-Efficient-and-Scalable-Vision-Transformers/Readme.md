# Chapter 4: Efficient and Scalable Vision Transformers

This chapter contains two complementary projects. Start with the compact notebook to understand
DeiT's architecture and distillation mechanism, then use the controlled benchmark to compare the
accuracy and systems trade-offs of practical transformer families.

## Projects

### 1. [Data-Efficient Image Transformers (DeiT) from Scratch](01-Data-Efficient-Image-Transformers)

A beginner-friendly notebook that builds a small DeiT student in PyTorch and trains it on MNIST
with knowledge distillation from a frozen ResNet-50 teacher. It makes the class token,
distillation token, dual heads, and distillation loss visible in one linear workflow.

Use this project when the learning goal is implementation-level understanding of DeiT.

### 2. [Efficient Vision Transformer Benchmark](02-Efficient-Vision-Transformer-Benchmark)

A reproducible, production-style CIFAR-100 experiment comparing pretrained ViT-S, DeiT-S,
Distilled DeiT-S, Swin-T, and EfficientFormer-L1 under one fixed protocol. ConvNeXt-T is included
as the teacher and CNN-style reference. The project measures validation and test accuracy,
learning behavior, parameter count, MACs, accelerator memory, latency, and throughput.

It includes:

- pinned Python dependencies and validated YAML presets;
- a tested `vision_bench` package and command-line interface;
- guided notebooks and step-by-step documentation;
- local and Modal GPU instructions with checkpoint resumption;
- the completed NVIDIA L4 result tables, figures, and technical report; and
- links and checksums for the public model-only Safetensors archive on Hugging Face.

Use this project when the learning goal is a fair architecture comparison, experimental design,
or reproducible benchmarking.

## Recommended reading order

1. Run the DeiT-from-scratch notebook to learn the mechanism.
2. Read the benchmark's [project overview](02-Efficient-Vision-Transformer-Benchmark/README.md).
3. Follow its quick preset before attempting the full multi-seed experiment.
4. Compare your output with the completed [experiment report](02-Efficient-Vision-Transformer-Benchmark/EXPERIMENT_REPORT.md).

The benchmark deliberately excludes private credentials, downloaded datasets, virtual
environments, generated caches, and local model weights. Its released inference weights are
available from the public Hugging Face repository linked in the project README.
