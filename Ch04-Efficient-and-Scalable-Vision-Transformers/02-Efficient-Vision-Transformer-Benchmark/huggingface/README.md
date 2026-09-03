---
language:
  - en
tags:
  - book
  - computer-vision
  - transformers
  - pytorch
  - reproducibility
---

# Transformer Vision: From Image, Video to World Models & Robotics — Book

This repository is the model-artifact companion to the technical book **Transformer Vision: From
Image, Video to World Models & Robotics**. It keeps the book's trained weights, experiment
configuration, evaluation tables, and usage notes together without pretending that every chapter
uses one interchangeable model architecture.

## Why the repository is organized by chapter

Each chapter asks a different technical question and may use a different architecture, dataset,
checkpoint format, preprocessing pipeline, or output type. Chapter-level folders create a stable
boundary around those differences:

```text
chapter-NN-short-title/
├── README.md                 # chapter-specific model card and usage
├── configs/                  # resolved experiment configuration
├── checkpoints/              # model, treatment, and seed subfolders
├── results/                  # metrics, profiles, and figures
├── EXPERIMENT_REPORT.md      # interpreted experiment narrative
└── manifest.json             # machine-readable provenance
```

Inside a chapter, checkpoint folders are further separated by model, training treatment, and seed.
This prevents a ViT checkpoint from being confused with a Swin, language, video, world-model, or
robotics checkpoint from another part of the book.

## Available chapters

| Chapter | Artifact folder | Status |
| --- | --- | --- |
| 4 — Efficient and scalable vision transformers | [`chapter-04-efficient-and-scalable-vision-transformers`](chapter-04-efficient-and-scalable-vision-transformers) | Complete |

Folders for other chapters will be added only after their experiments and validation are complete.
Empty placeholder folders are intentionally avoided.

## Download one chapter

Using the Hugging Face CLI:

```bash
hf download \
  Mayank022/Transformer-Vision-From-Image-Video-to-World-Models-and-Robotics-Book \
  --include "chapter-04-efficient-and-scalable-vision-transformers/*" \
  --local-dir transformer-vision-book
```

Readers who need only one seed can download a single file with `hf_hub_download`; each chapter card
contains exact examples.

## File-format policy

Published inference weights use **Safetensors**. They contain the model state only and exclude
optimizer, scheduler, gradient-scaler, and random-number-generator state. This makes the public
artifact smaller and avoids pickle execution during weight loading. Full resumable training
checkpoints remain in the experiment's controlled storage and are not required for inference.

Every chapter includes SHA-256 checksums and a JSON manifest connecting each exported file to:

- the exact upstream checkpoint identifier;
- architecture and training mode;
- random seed and selected epoch;
- validation and test metrics;
- configuration fingerprint; and
- the checksum of the original training checkpoint.

## Important scope note

This is a **multi-model book archive**, not a single `from_pretrained()` repository. Always open the
README inside the relevant chapter before loading a file. Model construction and preprocessing are
chapter-specific.

## License and attribution

The repository contains fine-tuned derivatives of upstream research checkpoints. For Chapter 4,
the official Hugging Face model cards identify five source checkpoints as Apache-2.0 and the Swin-T
source checkpoint as MIT. Each chapter lists the exact identifiers and source-license metadata.
Retain applicable notices and review the upstream model and dataset terms before making another
redistribution or commercial derivative; the book's documentation does not replace those licenses.
