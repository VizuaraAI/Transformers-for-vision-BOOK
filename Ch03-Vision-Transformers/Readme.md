# Ch 03: Fine-Tuning a Vision Transformer for Image Classification

Fine-tune a pretrained ViT-Base model on the Oxford-IIIT Pet dataset to classify 37 cat and dog breeds using PyTorch.

## What's inside

The notebook starts with a [ViT-Base](https://huggingface.co/google/vit-base-patch16-224) model pretrained on ImageNet and adapts it for pet breed classification on the [Oxford-IIIT Pet dataset](https://huggingface.co/datasets/timm/oxford-iiit-pet) (7,349 images, 37 breed classes). It walks through data loading, ViT-specific preprocessing, partial layer freezing, training, and evaluation.

## Quick start

```bash
pip install torch torchvision torchmetrics transformers datasets tqdm scikit-learn matplotlib
```

Open `Fine-Tuning_a_Vision_Transformer_for_Image_Classification.ipynb` and run all cells.

## Notebook outline

| # | Section | Description |
|---|---------|-------------|
| 1.1–1.3 | Setup | Imports, device selection, reproducibility, hyperparameters |
| 1.4–1.5 | Dataset exploration | Loading Oxford-IIIT Pet dataset, visualizing sample images |
| 1.6–1.8 | Preprocessing | ViT image processor, train/val transforms, DataLoaders |
| 1.9–1.11 | Model | Loading pretrained ViT-Base, inspecting structure, freezing early layers |
| 1.12 | Pre-training inference | Model predictions before any fine-tuning |
| 1.13–1.14 | Training | Optimizer, scheduler, loss function, training and validation loop |
| 1.15 | Training curves | Visualizing loss and accuracy over epochs |
| 1.16–1.17 | Evaluation | Post-training inference, confusion matrix |
