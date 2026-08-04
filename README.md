# PitVis Surgical Phase Recognition

Automatic recognition of surgical steps (phases) in endoscopic pituitary surgery,
using the [PitVis Challenge](https://arxiv.org/abs/2409.01184) dataset
(EndoVis / MICCAI 2023, Das et al. 2024).

**New to this repo?** [`notes/walkthrough.md`](notes/walkthrough.md) is the guided tour:
what the surgery is, what every annotation column means, how the data flows through the
pipeline, and which line of which file to read next.

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
vendored verbatim as `src/official_metric.py` and called directly, so reported
numbers are directly comparable to the paper. Per-class recall/F1 and a
confusion matrix are printed alongside as diagnostics.

## Layout

```
src/inventory.py          probe videos + verify annotation invariants -> notes/inventory.md
src/extract_features.py   1 fps decode -> frozen ResNet-50 features (resumable)
src/dataset.py            per-video (T, 2048) features + labels, train/val split constants
src/official_metric.py    organisers' scoring code, vendored verbatim — do not edit
src/eval.py               official metric per video + mean±std, plus pooled diagnostics
src/train_baseline.py     frame-wise linear probe baseline
tests/test_eval.py        pins eval.py to the official metric
notes/inventory.md        generated dataset inventory
notes/walkthrough.md      guide to the domain, the data, and the pipeline — start here
data/features/            cached per-video features.npy + labels.npy (gitignored)
26531686/                 raw PitVis download (gitignored, read-only)
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

```sh
uv run python src/inventory.py         # sanity-check the raw data, write notes/inventory.md
uv run python src/extract_features.py  # one-time feature extraction (all 25 videos)
uv run python src/train_baseline.py    # train + evaluate the linear probe
uv run pytest                          # verify eval.py against the official metric
```

`uv run` syncs the environment first, so there is no venv to activate. To add or change
a dependency, use `uv add <pkg>` / `uv remove <pkg>` rather than editing `pyproject.toml`
by hand, and commit the updated `uv.lock`.

Torch resolves to the default PyPI wheels, which give CPU + MPS on macOS. For a CUDA
box, add a `[[tool.uv.index]]` entry pointing at the appropriate PyTorch index.
