# Running on Modal

The cloud launcher is designed for a reportable full run without requiring readers to administer a
GPU server. It uses NVIDIA L4 GPUs, a persistent Modal Volume, bounded concurrency, and resumable
per-epoch checkpoints.

!!! danger "Cloud jobs cost money"

    Check the current [Modal pricing page](https://modal.com/pricing), account spending limits, and
    dashboard before starting. Rehearse `quick` first. Stop an accidental App from the Modal
    dashboard or CLI; closing a terminal is not a reliable cancellation strategy.

## Architecture of the launcher

1. Build a Python 3.12 image from the pinned `pyproject.toml` dependencies and local package source.
2. Mount `vision-transformer-benchmark-data` at `/vol` for data, model caches, checkpoints, and reports.
3. Train ConvNeXt-T first so every hard-KD run has a teacher.
4. Fan out the remaining independent runs with at most four L4 containers, each with eight CPU
   cores for the input pipeline.
5. Commit each epoch to the Volume; retries resume from `latest.pt` in a fresh container.
6. Benchmark all architectures sequentially in one L4 container.
7. Generate figures and tables in a CPU container.

Four containers keep checkpoint commits below the documented high-contention range and put a clear
upper bound on simultaneous GPU spend. Each function can retry three times and each training attempt
has a 24-hour timeout.

## Authenticate

Install the project locally, then connect the Modal CLI to your account. For an interactive local
setup, use:

```bash
uv sync --extra dev --extra docs
uv run modal setup
```

For CI, a temporary shell, or project-scoped authentication, use the checked-in placeholder file:

```bash
cp .env.example .env
# Edit .env and replace both placeholder values before continuing.
set -a
source .env
set +a
uv run modal token info
```

The required variables are `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The optional
`MODAL_ENVIRONMENT` variable selects a named Modal Environment; leave it commented to use the active
profile or workspace default. A populated `.env` is ignored by Git, while `.env.example` contains
placeholders only and is safe to commit.

The shell does not load `.env` automatically, so source it in each new shell before invoking Modal.
These variables authenticate the local Modal CLI; they are not copied into remote functions by this
application. The benchmark downloads public upstream checkpoints and does not require an additional
Hugging Face secret to run. See Modal's
[authentication documentation](https://modal.com/docs/guide/trigger-deployed-functions#authentication)
for the precedence between environment variables and the local Modal profile.

Modal's [Apps documentation](https://modal.com/docs/guide/apps) explains local entrypoints; the
[GPU guide](https://modal.com/docs/guide/gpu) lists current accelerator names and behavior.

## Rehearse with the quick preset

```bash
uv run modal run modal_app.py --preset quick --stage all
```

This still launches billable resources, but it verifies image building, weight downloads, Volume
access, distillation ordering, timing, and report generation with a smaller training workload.

## Run the full project

```bash
uv run modal run modal_app.py --preset full --stage all
```

The entrypoint waits for the teacher, uses `starmap` to distribute the other 12 runs, waits for all
of them, then performs timing and reporting. Modal's [scaling guide](https://modal.com/docs/guide/scale)
documents the mapped-call behavior.

## Run or repeat one stage

```bash
uv run modal run modal_app.py --preset full --stage train
uv run modal run modal_app.py --preset full --stage benchmark
uv run modal run modal_app.py --preset full --stage report
```

These stages are idempotent at their intended boundary. Completed training runs return immediately;
benchmark and report artifacts are replaced with measurements from the current invocation.

## Inspect and download artifacts

```bash
uv run modal volume ls vision-transformer-benchmark-data /artifacts/full
uv run modal volume get \
  vision-transformer-benchmark-data \
  /artifacts/full/report \
  ./downloaded-full-report
```

Use `uv run modal volume get --help` if your installed Modal CLI changes the destination syntax.
The [Volumes guide](https://modal.com/docs/guide/volumes) explains commits, reloads, and concurrent
write constraints.

## Failure and resumption behavior

At the end of every epoch, the application atomically writes checkpoints and explicitly commits the
Volume. If a container is preempted or a transient error triggers a retry, the new container reloads
the Volume and resumes the run. A crash inside an epoch repeats that epoch; it never fabricates a
partial metric row as complete.

If the local entrypoint stops while remote mapped calls are still active, inspect the Modal dashboard
before relaunching. Relaunching is safe after prior calls have stopped because run directories are
configuration-qualified and completed runs are detected.

## Benchmark fairness in the cloud

Training jobs may use different physical L4 instances, which is acceptable for model fitting. The
reference inference benchmark deliberately runs all architectures sequentially inside one L4
function. Its environment metadata is recorded in `benchmark.json`. Do not combine timing rows from
separate benchmark invocations into one ranking.
