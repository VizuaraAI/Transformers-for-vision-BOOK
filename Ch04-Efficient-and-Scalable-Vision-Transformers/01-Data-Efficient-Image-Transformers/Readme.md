# Ch 04: Data-Efficient Image Transformers (DeiT) from Scratch

Build a DeiT model from scratch in PyTorch with knowledge distillation from a CNN teacher, trained on MNIST.

## What's inside

The notebook implements the full DeiT pipeline: a ResNet-50 teacher model distills its knowledge into a lightweight Vision Transformer student. It covers patch embeddings, the learnable `[CLS]` and `[DIST]` tokens, dual classification heads, and a combined KD + cross-entropy loss.



**Config** — `dim=16`, `heads=4`, `layers=4`, `patch_size=7`, `image_size=28`, teacher: ResNet-50 (ImageNet pretrained)

## Quick start

```bash
pip install torch torchvision tqdm matplotlib
```

Open `DEIT_from_scratch.ipynb` and run all cells.

## Notebook outline

| # | Section | Description |
|---|---------|-------------|
| 1–2 | Setup & Data | Imports, hyperparameters, MNIST loading with RGB conversion, train/test DataLoaders |
| 3 | Teacher model | Pretrained ResNet-50, frozen backbone, modified classification head for 10 classes |
| 4 | Student model | Patch embedding, `[CLS]` and `[DIST]` tokens, positional embeddings, transformer encoder, dual classification heads |
| 5 | Knowledge distillation | KD loss (temperature-scaled KL divergence + cross-entropy), alpha balancing |
| 6 | Training | AdamW optimizer, 10-epoch training loop with combined KD loss |
| 7 | Evaluation & Inference | Test accuracy, sample predictions with ground truth visualization |
