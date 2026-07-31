# PitVis Surgical Phase Recognition

Surgical step (phase) recognition on the PitVis dataset — endoscopic pituitary surgery
(endoscopic TransSphenoidal Approach, eTSA), per-second step and instrument annotations.

Personal git/commit/branch conventions live in `~/.claude/CLAUDE.md` and apply here.
This file records project-specific facts and decisions.

## Data

Raw data lives at `26531686/` in the project root. It is **not** tracked by git — treat it
as read-only input. Source: PitVis Challenge (EndoVis / MICCAI-2023), Das et al. 2024,
https://arxiv.org/abs/2409.01184.

```
26531686/
  video_{01..25}.mp4          25 videos, 40 GB total
  annotations_{n}.csv         24 files — annotations_19.csv does not exist (see below)
  map_steps.csv
  map_instruments.csv         note: plural, despite what README.txt calls it
  README.txt
  video_encoder_details.txt
```

The upstream `dreets/pitvis` GitHub repo contains **no** annotation-parsing code and no
CSVs — only evaluation metrics, a frame extractor, and a Docker submission example. Do not
go looking there for a reference parser; it isn't one.

## Annotation schema

`annotations_{n}.csv` — five integer columns, no nulls, verified across all 24 files:

```
int_video,int_time,int_step,int_instrument1,int_instrument2
```

- `int_video` — constant per file, matches the filename.
- `int_time` — **one row per elapsed second**, contiguous `0..N-1`. No gaps, no duplicates.
- `int_step` — a single integer in `{-1} ∪ {1..14}`. **There is no step 0.**
- `int_instrument1` / `int_instrument2` — the instrument label is a *pair of columns*, not
  a list. `int_instrument2 == -2` means no secondary instrument (85.3% of rows). The pair
  is sorted ascending except in 4 rows dataset-wide.

`int_step == -1` and `int_instrument1 == -1` coincide exactly (10,476 rows, zero
disagreement in either direction). Background is one consistent state.

### Map files are not uniquely keyed

Do **not** load either map into an `int -> str` dict without handling collisions:

- Steps: `-1` maps to three names — `operation_ended`, `operation_not_started`,
  `out_of_patient`. The CSV collapses all three, so **the distinction is unrecoverable**
  from the annotations. Treat `-1` as a single background class.
- Instruments: `0` maps to both `no_visible_instrument` and `occluded_image_inside_patient`.
- Step 1's name has a **trailing space** (`"nasal corridor creation "`). Always `.strip()`.

## Class encoding

Steps are encoded **15-way**, matching the challenge baseline (`nn.Linear(..., 15)` in the
upstream Docker example):

```
-1  -> 0     (background)
 k  -> k     for k in 1..14
```

Train on all 15. Rarity-based exclusions happen at **evaluation** time only, never by
dropping rows from training.

## Evaluation

The challenge metric is `(macro F1 + normalised edit score) / 2`, excluding classes
`[-1, 11, 13]`. That exclusion is a **rarity** exclusion, not an index offset:

- step 11 (`gasket seal construct`) appears in only 2 videos
- step 13 (`nasal packing`) appears in only 1 video

**`src/official_metric.py` is the organisers' `helper_scripts/evaluation_steps.py`, vendored
verbatim** (commit `b1cb307`, sha256 in the file header). Do not edit it and do not
reimplement the metric — `src/eval.py` calls it, so the headline number is the challenge's
number by construction. If upstream ever changes, re-vendor and diff.

`src/eval.py` recovers the F1/edit split by replicating the vendored function's two internal
calls, then **asserts** the two halves recombine to what the vendored one-shot function
returns. `tests/test_eval.py` pins all of this; run `uv run pytest` after touching eval.

### Aggregation: per video, then mean

Das et al. 2024 §evaluation: scores are *"mean-averaged across the 8-testing-videos"*,
reported as mean±std, **"not pooled frame-wise"**. So `eval.evaluate` takes
`[(vid, y_true, y_pred), ...]` and never concatenates.

Pooling is not a harmless approximation — it **inflates the score**. Concatenation merges the
last segment of one video with the first of the next and lets opposite per-video errors cancel
in the frame-wise F1. `test_pooling_videos_flatters_the_score` demonstrates 0.583 pooled vs
0.417 honest on a two-video toy case.

### Three official behaviours that look like bugs and must be preserved

A "cleaner" reimplementation would silently diverge from the challenge on all three:

1. **Exclusion filters by ground truth only.** A model that *predicts* -1/11/13 on a retained
   row is not filtered. Since `f1_score` is called with **no `labels=`**, sklearn infers the
   label set from the union of cleaned trues *and* preds, so that class joins the macro
   average at F1 = 0 and drags the score down. `eval.report` counts these as `leaked`.
   Corollary worth exploiting: **masking classes 0/11/13 out of the argmax at inference can
   only raise the official metric.** Not yet implemented in any model.
2. **`zero_division=1`** in the `f1_score` call.
3. **The edit score runs after exclusion**, so removed rows splice the sequence and the
   segments either side of a gap merge. `[1,1,bg,bg,1,1]` is ONE segment, not three.

Our own additions — per-class recall/F1 and the 15-way confusion matrix — are **pooled** and
use a fixed 12-class label set for stability. They are diagnostics, labelled as such in the
output, and are not part of the reported metric.

Edge case: if every ground-truth row of a video is an excluded class, the official code
divides by zero. `eval.evaluate_video` raises a clear `ValueError` instead of patching the
vendored file. All 12 scored classes are present in all 5 val videos, so this does not arise
on our split.

## Step distribution (115,586 labeled seconds, 24 videos)

| Step | Name | % | Videos |
|---|---|---|---|
| -1 | background | 9.06 | 24 |
| 1 | nasal corridor creation | 2.45 | 24 |
| 2 | anterior sphenoidotomy | 9.32 | 24 |
| 3 | septum displacement | 1.16 | 24 |
| 4 | sphenoid sinus clearance | 15.30 | 24 |
| 5 | sellotomy | 14.19 | 24 |
| 6 | durotomy | 5.34 | 24 |
| 7 | tumour excision | 23.87 | 24 |
| 8 | haemostasis | 11.87 | 24 |
| 9 | synthetic_graft_placement | 3.06 | 18 |
| 10 | fat graft placement | 1.94 | 22 |
| 11 | gasket seal construct | 0.73 | 2 |
| 12 | dural sealant | 0.77 | 23 |
| 13 | nasal packing | 0.06 | 1 |
| 14 | debris clearance | 0.87 | 18 |

Heavily imbalanced — step 7 is 23.9%, step 13 is 0.06%. Prefer macro-averaged metrics.

## Video properties

- **Resolution is uniform**: 1280x720, H.264, all 25 videos.
- **fps is not uniform**: 24 fps everywhere except **`video_24.mp4`, which is 25 fps**.
  Any 1-fps decode must read fps per video, not assume 24.
- Durations range 2,645 s to 8,645 s.

## Frame/label alignment (the off-by-one)

For **every** video, annotation rows are exactly one more than the 1-fps-extractable frames:

```
ann_rows == ceil(nb_frames / round(fps)) + 1
```

Every video also ends in a run of `-1` background (6 to 147 seconds), so the extra trailing
row is always background.

**Decision: truncate the labels to the frame count.** Features and labels are both length
`ceil(nb_frames / round(fps))`. The dropped row is verified background in all 24 videos.

## Train/validation split

From Das et al. 2024 (arXiv 2409.01184), verbatim: *"25-annotated-videos were provided. A
20-training to 5-validation (01, 12, 21, 24, 25) split was suggested but not enforced."*
The split was chosen so each class holds an approximate 4:1 train:val annotation ratio.

```
VAL   = [1, 12, 21, 24, 25]                                              # 5 videos
TRAIN = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,20,22,23]               # 19 videos
```

`TRAIN` is the paper's 20 minus video 19, which has no labels (see below) — so our split is
**19/5**, not 20/5. Keep both lists as explicit constants in `src/dataset.py`; do not derive
the split by arithmetic.

Note the paper's separate 8-video *testing* set is private and was never released. All 25
videos we have are "training" videos in challenge terms; the 20/5 is a split within them.

Also note `video_24.mp4` — the lone 25 fps outlier — is in the validation set.

## Known gap: video 19

`annotations_19.csv` does not exist — not in `26531686/`, and not in the source
`26531686.zip` (which contains 53 files). No other CSV contains `int_video == 19`.
`video_19.mp4` is present and intact (4,455 s). The paper does not mention video 19 as
excluded, so this is a gap in the download, not an upstream design decision.

**Decision: proceed with the 24 labeled videos.** Video 19 belongs to the paper's training
set, so the loss costs one training video and leaves validation untouched — val results stay
directly comparable to the paper; training is on 19/20 of the intended data.

## Environment

Dependencies are managed with **uv**. `pyproject.toml` + `uv.lock` are tracked; `.venv/` is
not. Python is pinned to **3.13** via `.python-version`.

- Run scripts as `uv run python src/<script>.py`. Do not use bare `python` — the machine's
  default interpreter is 3.14 with a different, unmanaged set of packages.
- Change dependencies with `uv add` / `uv remove`, not by hand-editing `pyproject.toml`, and
  commit the resulting `uv.lock`.
- `package = false` in `[tool.uv]` — `src/` is a set of entry-point scripts using flat imports
  (`from dataset import ...`), not an installable package. There are no `__init__.py` files and
  nothing gets built.
- **`ffmpeg` / `ffprobe` are hard requirements and are not Python packages** — both
  `inventory.py` and `extract_features.py` shell out to them, so uv cannot supply them.
- Resolved versions worth knowing: torch 2.13 (MPS available on this Mac), **pandas 3.0**.
  The pandas 3.0 major bump was verified against every pandas path in `src/` — `read_csv`,
  `dropna`, `iterrows`, `pd.isna`, `.to_numpy()` — all pass.
- Torch comes from default PyPI (CPU + MPS). A CUDA host needs a `[[tool.uv.index]]` entry.

## Layout

```
pyproject.toml            uv-managed dependencies (package = false)
uv.lock                   pinned resolution — tracked, commit changes to it
.python-version           3.13
src/inventory.py          per-video duration, resolution, fps, annotated seconds, step distribution
src/extract_features.py   1 fps decode, frozen timm resnet50 (num_classes=0) -> data/features/
src/dataset.py            per-video (T, D) features + aligned labels, train/val split
src/official_metric.py    VENDORED official challenge metric — do not edit
src/eval.py               official metric per video + mean±std, plus pooled diagnostics
src/train_baseline.py     frame-wise linear probe baseline
tests/test_eval.py        pins eval.py to the official metric — `uv run pytest`
notes/inventory.md        generated by src/inventory.py
data/features/            per-video features.npy + labels.npy (gitignored)
26531686/                 raw PitVis download (gitignored, read-only)
```

`extract_features.py` must be **resumable** — decoding 40 GB is expensive; skip videos whose
outputs already exist and are the expected length.
