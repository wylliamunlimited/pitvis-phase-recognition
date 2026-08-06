# PitVis Surgical Phase Recognition

Automatic recognition of surgical steps (phases) in endoscopic pituitary surgery,
using the [PitVis Challenge](https://arxiv.org/abs/2409.01184) dataset
(EndoVis / MICCAI 2023, Das et al. 2024).

**New to this repo?** [`notes/walkthrough.md`](notes/walkthrough.md) is the guided tour:
what the surgery is, what every annotation column means, how the data flows through the
pipeline, and which line of which file to read next. For the machine-learning side from
the ground up — what an embedding is and how ours are generated —
[`notes/embeddings.md`](notes/embeddings.md).

**What's next?** [`notes/roadmap.md`](notes/roadmap.md) tracks everything left to build,
phased: data engineering → end-to-end pipelining → models → app.

## The task

The dataset contains 25 full-length videos of endoscopic TransSphenoidal Approach
(eTSA) pituitary surgery — 2,600 to 8,600 seconds each, 40 GB total — with
per-second annotations of the surgical step being performed (14 steps such as
*sellotomy*, *durotomy*, *tumour excision*, plus a background class) and of the
instruments in view. Given a video, the goal is to predict the surgical step at
every second.

This is a temporal segmentation problem with real-world difficulties:

- **Heavy class imbalance** — *tumour excision* covers 23.9% of annotated time,
  *nasal packing* 0.06% (a single video).
- **Long sequences** — hours of video per case, so per-frame recognition benefits
  strongly from temporal modeling.
- **Data quirks** — one video (19) is missing annotations, one video (24) runs at
  25 fps instead of 24, and annotations are one row longer than the extractable
  frames. All of these are verified and handled explicitly (see `CLAUDE.md` for
  the full data notes).

## Approach

Two-stage pipeline, standard for surgical phase recognition:

1. **Frame features** — decode each video at 1 fps and embed every frame with a
   frozen ImageNet-pretrained ResNet-50 (2048-d). Extraction is resumable; done
   once, cached under `data/features/`.
2. **Step classification** — models over the cached features, starting with a
   frame-wise linear probe (no temporal context, the floor) and moving to
   temporal models (e.g. TCN / GRU / transformer over the feature sequence).

Train/val split follows the paper: videos 01, 12, 21, 24, 25 for validation, the
rest for training (19 train videos in practice, since video 19 has no labels).
Evaluation is the official challenge metric, not an approximation of it:
`(macro F1 + normalised edit score) / 2`, with the rare classes (background,
*gasket seal construct*, *nasal packing*) excluded from scoring, computed **per
video and mean-averaged** as in the paper. The organisers' scoring code is
vendored verbatim as `src/pitvis/evaluation/official.py` and called directly, so reported
numbers are directly comparable to the paper. Per-class recall/F1 and a
confusion matrix are printed alongside as diagnostics.

## Layout

```
src/pitvis/
  paths.py                  every filesystem location, defined once
  data/
    inventory.py            probe videos + verify annotation invariants
    extract_features.py     1 fps decode -> frozen ResNet-50 features (resumable)
    verify_cache.py         integrity check of the feature cache
    dataset.py              per-video (T, 2048) features + labels, split constants
  models/
    arst.py                 CITI's task-1 architecture (spatial + TeCNO + ARST)
  training/
    registry.py             the only list of trainable models — add one here
    arst.py                 three-stage training + auto-regressive inference
    baseline.py             frame-wise linear probe baseline
  inference/
    predict.py              mp4 -> per-second steps + segments; no labels needed
  evaluation/
    official.py             organisers' scoring code, vendored verbatim — do not edit
    metric.py               official metric per video + mean±std, plus diagnostics

tests/test_eval.py          pins the metric to the official code
notes/walkthrough.md        the domain, the data, and the pipeline — start here
notes/embeddings.md         what the feature cache is and how embeddings are made
notes/citi-baseline.md      the CITI reproduction: architecture, faithfulness, results
notes/citi-dataflow.md      the same cascade traced with real tensor dimensions
notes/data-dictionary.md    every annotation column and what each integer means
notes/roadmap.md            phased plan of remaining work
notes/inventory.md          generated dataset inventory
data/features/              cached per-video features.npy + labels.npy (gitignored)
data/arst/                  CITI checkpoints + result.json (gitignored)
26531686/                   raw PitVis download (gitignored, read-only)
```

## Setup

Python dependencies are managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml`
and `uv.lock` are tracked, so the environment is reproducible. Python 3.13 is pinned
via `.python-version` — uv will fetch it if you don't have it.

```sh
uv sync                           # create .venv and install the locked dependencies
```

`ffmpeg` / `ffprobe` are also required and are **not** Python packages — install them
separately (`brew install ffmpeg`).

Place the raw PitVis download at `26531686/` in the project root (gitignored,
treated as read-only), with the videos, `annotations_*.csv`, and `map_*.csv` directly
inside it.

## Usage

Each package under `src/pitvis/` has a `run.py` that runs that directory end to end.
Start here:

```sh
uv run pitvis-data      # inventory -> extract -> verify  (the whole data pipeline)
uv run pitvis-train     # train every registered model    (baseline, then arst)
uv run pitvis-predict --video case.mp4   # point a trained model at any video
uv run pitvis-eval      # score an existing checkpoint, no retraining
uv run pitvis-models    # shape + parameter trace through the cascade (~1 s)
```

Every runner takes `--dry-run` to print the plan without executing, and
`--only` / `--skip` to run part of a workflow:

```sh
uv run pitvis-data --dry-run              # what would run, in what order
uv run pitvis-data --only verify --probe  # just the slow integrity check
uv run pitvis-data --videos 1 2 3         # limit extraction to 3 videos
uv run pitvis-train arst --ablations       # one model plus its variants
uv run pitvis-train --list                 # what models are registered
```

Models are named positionally and come from `training/registry.py`, so adding a
model needs no new console script:

```sh
uv run pitvis-train arst              # just ARST
uv run pitvis-train baseline arst     # both, in the order given
uv run pitvis-train arst --no-cci     # unknown flags pass through to the model
```

The individual stages are still addressable when you want one thing:

```sh
uv run pitvis-inventory        # sanity-check the raw data, write notes/inventory.md
uv run pitvis-extract          # one-time feature extraction (all 25 videos)
uv run pitvis-verify           # integrity-check the feature cache
uv run pytest                  # verify the metric against the official code
```

These are console scripts declared in `pyproject.toml`, so each maps to exactly one
module's `main()` — and the runners call those same `main()`s rather than a copy, so
`pitvis-data` and `pitvis-extract` cannot drift apart. Every module is also runnable
directly if you prefer (`uv run python -m pitvis.training.arst --help`).

`uv run` syncs the environment first, so there is no venv to activate. To add or change
a dependency, use `uv add <pkg>` / `uv remove <pkg>` rather than editing `pyproject.toml`
by hand, and commit the updated `uv.lock`.

Torch resolves to the default PyPI wheels, which give CPU + MPS on macOS. For a CUDA
box, add a `[[tool.uv.index]]` entry pointing at the appropriate PyTorch index.
