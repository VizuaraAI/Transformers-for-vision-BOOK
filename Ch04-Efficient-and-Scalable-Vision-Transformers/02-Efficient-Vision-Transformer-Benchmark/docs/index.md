# Efficient and scalable vision transformers

This is the executable project for a technical-book chapter on DeiT, Swin, and practical vision
transformers under data and compute constraints. You will fine-tune several pretrained families on
one classification task, add teacher-guided DeiT training, profile each architecture on the same
hardware, and build a report that makes accuracy and systems cost visible together.

The project answers four questions:

1. How do plain ViT, a DeiT recipe, a Swin-style hierarchy, and a compact hybrid behave under the
   same transfer-learning budget?
2. Does active hard distillation improve the same Distilled DeiT student in this setting?
3. Which model is preferable when parameters, memory, latency, or throughput matter alongside
   accuracy?
4. Which conclusions are supported by this experiment, and which require a broader paper map?

Start with the [chapter project guide](chapter-project.md). It separates inexpensive learning checks
from reportable experimentation and tells you what to inspect at each milestone.

!!! warning "Quick numbers are not final numbers"

    The `smoke` and `quick` presets are deliberately small. Every artifact records
    `tutorial_only: true`; never mix those measurements into the full chapter table.
