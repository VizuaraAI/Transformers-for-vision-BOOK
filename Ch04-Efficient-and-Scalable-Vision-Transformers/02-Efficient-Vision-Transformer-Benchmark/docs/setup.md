# Setup

## Requirements

- Python 3.12 (the project intentionally pins one minor version)
- `uv` 0.5 or newer, or a conventional virtual environment and `pip`
- approximately 10 GB of free disk space for dependencies, pretrained weights, data, and artifacts
- an NVIDIA CUDA GPU for practical full training and reference throughput measurements

CPU and Apple MPS are supported for learning and diagnostics, but they are not the reference
hardware. A full suite on CPU is not a reasonable use of time.

## Install with uv

From the project directory:

```bash
uv sync --extra dev --extra docs
```

This creates `.venv`, installs the exact locked Python environment, and exposes the `vision-bench`
command through `uv run`. You do not have to activate the environment.

```bash
uv run vision-bench doctor
```

The report should show Python 3.12, installed versions rather than `not installed`, at least several
gigabytes of free disk, and the accelerator you intend to use.

## Install with pip

If `uv` is unavailable, create and activate a Python 3.12 environment, then install the project:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,docs]'
vision-bench doctor
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The remaining commands are the
same.

## Verify model contracts without downloads

Start with two representative models:

```bash
uv run vision-bench smoke --models vit,deit_distilled --device auto
```

Each model is initialized randomly, receives one synthetic `224 × 224` image, and must emit a
`[1, 100]` classifier tensor. Distilled DeiT is also switched into its explicit two-head training
mode. Expand to every architecture only after this succeeds:

```bash
uv run vision-bench smoke --device auto
```

This can take several minutes on CPU. The reported forward time includes model setup effects and is
not a benchmark.

## CUDA notes

The full reference environment pins PyTorch and uses CUDA automatic mixed precision. Confirm that
`doctor` reports `cuda_available: true`, then request the device explicitly in real runs:

```bash
uv run vision-bench smoke --models vit --device cuda
```

If PyTorch cannot see the GPU, fix the driver/PyTorch installation before changing project code.
Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for
platform-specific wheel guidance.

## Apple MPS and CPU

`--device auto` selects CUDA, then MPS, then CPU. CUDA-only AMP is disabled automatically on the
other backends. The benchmark records FP16 rows as skipped outside CUDA so a local convenience run
cannot be mistaken for the chapter's FP16 reference measurement.

## Build the documentation

```bash
uv run mkdocs serve
```

Open the local URL printed by MkDocs. Use `uv run mkdocs build --strict` before publishing the
chapter companion materials.

## Open the notebooks

The development extras include JupyterLab and a Python kernel:

```bash
uv run jupyter lab
```

Open the notebooks in numeric order. Training cells are opt-in, so merely opening or running the
analysis notebook does not launch an expensive experiment.
