# PitVis Surgical Phase Recognition

Automatic recognition of surgical steps (phases) in endoscopic pituitary surgery,
using the [PitVis Challenge](https://arxiv.org/abs/2409.01184) dataset
(EndoVis / MICCAI 2023, Das et al. 2024).

## The task

The dataset contains 25 full-length videos of endoscopic TransSphenoidal Approach
(eTSA) pituitary surgery — 2,600 to 8,600 seconds each, ~84 GB total — with
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
Evaluation matches the challenge metric conventions: macro F1 with the rare
classes (background, *gasket seal construct*, *nasal packing*) excluded from
scoring, so numbers are comparable to the paper.

## Layout

```
src/inventory.py          probe videos + verify annotation invariants -> notes/inventory.md
src/extract_features.py   1 fps decode -> frozen ResNet-50 features (resumable)
src/dataset.py            per-video (T, 2048) features + labels, train/val split constants
src/eval.py               macro F1 (challenge exclusions), per-class accuracy, confusion matrix
src/train_baseline.py     frame-wise linear probe baseline
notes/inventory.md        generated dataset inventory
data/features/            cached per-video features.npy + labels.npy (gitignored)
26531686/                 raw PitVis download (gitignored, read-only)
```

## Usage

Requires Python 3.13, PyTorch (MPS/CUDA), timm, pandas, scikit-learn, and ffmpeg.

```sh
python src/inventory.py           # sanity-check the raw data, write notes/inventory.md
python src/extract_features.py    # one-time feature extraction (all 25 videos)
python src/train_baseline.py      # train + evaluate the linear probe
```
