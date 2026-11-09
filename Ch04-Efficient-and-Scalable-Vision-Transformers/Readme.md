# Chapter 4: Efficient and Scalable Vision Transformers

This chapter includes three illustrated notebooks that build DeiT, Swin, and a Looped Vision
Transformer from scratch. Each notebook contains its figures; no separate
image or illustration folder is needed.

## Illustrated notebooks

| Model | Notebook | What you will build |
| --- | --- | --- |
| DeiT | [DeiT_from_scratch.ipynb](01-DeiT-from-Scratch/DeiT_from_scratch.ipynb) | A CIFAR-10 ViT student with class and distillation tokens, a trained ResNet-18 teacher, and hard and soft distillation. |
| Swin | [Swin_from_scratch.ipynb](02-Swin-from-Scratch/Swin_from_scratch.ipynb) | A CIFAR-10 Swin model with window attention, shifted windows, relative-position bias, and patch merging, compared with a plain ViT and an unshifted Swin. |
| Looped ViT | [Looped_ViT_from_scratch.with_figures.ipynb](03-Looped-Vision-Transformer/Looped_ViT_from_scratch.with_figures.ipynb) | A CIFAR-100 comparison of a six-block ViT, a one-block ViT, and one shared block run six times, all trained from their final predictions. |

Open a notebook to read the explanations and view the embedded illustrations. DeiT and Swin
include saved outputs; Looped ViT includes chapter reference results and a saved prediction example.
To run it in Colab, upload the notebook and select a GPU runtime. For local use, install the
dependencies in your Jupyter environment:

```bash
pip install torch torchvision numpy matplotlib ipython
```

Run each notebook from its own model folder so its downloaded data and checkpoints remain
separate. Run the cells in order; datasets are downloaded by the notebook. For a short CPU run,
reduce `TRAIN_SUBSET` and `EPOCHS` in the configuration cell, and reduce `TEACHER_EPOCHS` for DeiT.

## Recommended reading order

1. Start with [DeiT](01-DeiT-from-Scratch/DeiT_from_scratch.ipynb) to learn teacher-guided distillation.
2. Continue with [Swin](02-Swin-from-Scratch/Swin_from_scratch.ipynb) to explore windows and hierarchical representations.
3. Work through [Looped ViT](03-Looped-Vision-Transformer/Looped_ViT_from_scratch.with_figures.ipynb) to compare parameter counts, block uses, and accuracy when one block is reused six times.
