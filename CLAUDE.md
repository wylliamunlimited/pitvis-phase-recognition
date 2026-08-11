# PitVis Surgical Phase Recognition

Surgical step (phase) recognition on the PitVis dataset — endoscopic pituitary surgery
(endoscopic TransSphenoidal Approach, eTSA), per-second step and instrument annotations.

Personal git/commit/branch conventions live in `~/.claude/CLAUDE.md` and apply here.
This file records project-specific facts and decisions.

## How to explain things here — "grounded explanation"

The default mode for any ML explanation in this repo. Not a tutorial and not a code
walkthrough: **an explanation anchored to a real artifact, pitched to spark a
follow-up question rather than close the topic.** `notes/embeddings.md` is the
reference example; that file exists because this style worked.

The recipe:

1. **Start from what the thing *is*, not what the code does.** "A feature vector is
   2,048 numbers summarising one frame" comes before any mention of
   `extract_features.py`.
2. **Every number is real and freshly read.** Load the `.npy`, run the snippet, print
   the shape — then quote *that* output. Never estimate, never round from memory,
   never describe an artifact without opening it. Real numbers are the whole
   difference between this and prose.
3. **Demonstrate rather than assert where it's cheap.** Regenerating one embedding
   from scratch and diffing it against the cache taught more in six lines than a
   paragraph would have. Reach for this whenever a claim is checkable in seconds.
4. **Explain the *why* behind one design choice, not all of them.** Pick the single
   most load-bearing idea (e.g. "2,048 isn't a chosen number — it's ResNet-50's last
   block channel count") and let the rest be reference.
5. **Stop at reasoning depth, not completeness.** Enough to think with. Detail that
   doesn't change a decision goes in `walkthrough.md` or gets left out.
6. **End with the open thread.** Name what this makes possible or blocks — usually a
   roadmap item — and ask what to dig into. The explanation should hand back a
   choice.

Applies to conversation *and* to notes. If an explanation lands, offer to save it —
see the doc layers under **Layout**.

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
  a list, so **two is a structural maximum**. `int_instrument2 == -2` means no secondary
  instrument (85.3% of rows). Among rows where slot 2 holds a real instrument the pair is
  sorted ascending with **zero** violations; the 4 rows with `int_instrument2 == 0` are an
  anomaly (0 = "nothing visible", but the unused-slot sentinel is -2), not an ordering
  violation. Slot 2 only ever takes 6 of the 18 ids, and 98.6% of two-instrument rows have
  suction as the secondary — the pair is "working instrument + suction", not two co-equal
  tools. Full breakdown in `notes/data-dictionary.md`.

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

This section records the **rules that must not change**. For what each metric
actually measures, why the challenge picked it and what it catches that the others
miss, see `notes/metrics.md`.

The challenge metric is `(macro F1 + normalised edit score) / 2`, excluding classes
`[-1, 11, 13]`. That exclusion is a **rarity** exclusion, not an index offset:

- step 11 (`gasket seal construct`) appears in only 2 videos
- step 13 (`nasal packing`) appears in only 1 video

**`evaluation/official.py` is the organisers' `helper_scripts/evaluation_steps.py`,
vendored verbatim** (commit `b1cb307`, sha256 in the file header). Do not edit it and do
not reimplement the metric — `evaluation/metric.py` calls it, so the headline number is
the challenge's number by construction. If upstream ever changes, re-vendor and diff.

`evaluation/metric.py` recovers the F1/edit split by replicating the vendored function's
two internal calls, then **asserts** the two halves recombine to what the vendored
one-shot function returns. `tests/test_eval.py` pins all of this; run `uv run pytest` after touching eval.

### The paper's Equation 3 is wrong — do not "fix" the code to match it

Das et al. 2024 §3.4.2 defines `Edit-score = 1 / Lev`. The organisers' code computes
`1 - Lev / max(len(s), len(t))` — the normalised segmental edit score of Lea et al.,
which the paper itself cites two sentences earlier as `[31]`.

**The code is authoritative.** `1/Lev` divides by zero on a perfect prediction, and the
reported values rule it out: CITI's 64.7 edit would imply `Lev ≈ 1.55`, i.e. about one
and a half edits across an entire multi-hour operation (ground-truth segment counts in
our val videos are 78-182). Eq 3 is a typo in the write-up.

### Only online models are permitted

Das et al. 2024 §3.2, verbatim: *"Only online models were permitted: only information
from frames up to and including the current frame can be used to classify the current
frame."*

This constrains feature engineering permanently. In particular **normalised time
(`t / (T-1)`, "% of duration passed") is not legal** — it requires the total duration,
which is information from the end of the video. It is also strongly informative (mean
normalised time per step is a near-monotone ladder, sd as low as 0.017), so it will
silently inflate any score that uses it. Causal substitutes: absolute elapsed time, or
running summaries of the model's own past predictions.

Fixed-*lag* post-processing appears to have been tolerated — both CITI's CCI (n=10) and
TSO-NCT's threshold smoothing (7 frames) decide frame `t` after seeing later frames.
Report lagged and strictly-causal variants side by side rather than conflating them.

### Aggregation: per video, then mean

Das et al. 2024 §evaluation: scores are *"mean-averaged across the 8-testing-videos"*,
reported as mean±std, **"not pooled frame-wise"**. So `metric.evaluate` takes
`[(vid, y_true, y_pred), ...]` and never concatenates.

Pooling is not a harmless approximation — it **inflates the score**. Concatenation merges the
last segment of one video with the first of the next and lets opposite per-video errors cancel
in the frame-wise F1. `test_pooling_videos_flatters_the_score` demonstrates 0.583 pooled vs
0.417 honest on a two-video toy case.

### Task 2 (instruments) is scored differently — do not reuse task-1 conventions

`evaluation/official_instruments.py` is `helper_scripts/evaluation_instruments.py`
vendored verbatim (commit `ebc82dd`, sha256 in the header; note the upstream
default branch is `trunk`, not `main`). It differs from the steps metric in four
ways, all deliberate upstream:

- **`average="weighted"`, not macro.** Support-dominated: ids 16/0/8/13 carry
  ~91% of positives.
- **Background rows are KEPT.** `remove_background_insts` is defined but its call
  is commented out, so out-of-patient frames become all-zero rows and are scored.
- **No edit score** — F1 only. Instruments are multi-label, so a sequence cannot
  be collapsed by `groupby`.
- **No rarity exclusions.** All 19 classes (ids 0..18) are scored, and **class 0
  ("no visible instrument") is a real class**, not a sentinel — 31.5% of frames.

**The vendored function has a real defect, preserved deliberately.**
`hot_encode_insts` fits a separate `MultiLabelBinarizer` on trues and on preds,
so when they observe different class sets the column ORDERS diverge and
`f1_score` compares them positionally. It fires on 5/5 of our val videos, and
through the official path three different constant strategies all score an
identical 0.1383. We keep it as the headline (the number is the challenge's by
construction) but `evaluate_video` sets `column_order_diverged` and `report()`
warns. The name-aligned score is printed alongside.

**The paper contradicts its own code**: §3.4.3 and Table 6's header say *macro*,
the script computes *weighted*. Unlike the Eq-3 case, nothing rules either out —
so all three numbers are printed and **no leaderboard comparability is claimed**.

### Three official behaviours that look like bugs and must be preserved

A "cleaner" reimplementation would silently diverge from the challenge on all three:

1. **Exclusion filters by ground truth only.** A model that *predicts* -1/11/13 on a retained
   row is not filtered. Since `f1_score` is called with **no `labels=`**, sklearn infers the
   label set from the union of cleaned trues *and* preds, so that class joins the macro
   average at F1 = 0 and drags the score down. `metric.report` counts these as `leaked`.
   Corollary worth exploiting: **masking classes 0/11/13 out of the argmax at inference can
   only raise the official metric.** Not yet implemented in any model.
2. **`zero_division=1`** in the `f1_score` call.
3. **The edit score runs after exclusion**, so removed rows splice the sequence and the
   segments either side of a gap merge. `[1,1,bg,bg,1,1]` is ONE segment, not three.

Our own additions — per-class recall/F1 and the 15-way confusion matrix — are **pooled** and
use a fixed 12-class label set for stability. They are diagnostics, labelled as such in the
output, and are not part of the reported metric.

Edge case: if every ground-truth row of a video is an excluded class, the official code
divides by zero. `metric.evaluate_video` raises a clear `ValueError` instead of patching the
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
**19/5**, not 20/5. Keep both lists as explicit constants in `src/pitvis/data/dataset.py`;
do not derive the split by arithmetic.

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

- Run the console scripts (`uv run pitvis-<name>`), or `uv run python -m pitvis.<module>`.
  Do not use bare `python` — the machine's default interpreter is 3.14 with a different,
  unmanaged set of packages.
- Change dependencies with `uv add` / `uv remove`, not by hand-editing `pyproject.toml`, and
  commit the resulting `uv.lock`.
- **`pitvis` is a real package** (`src/pitvis/`, hatchling, installed editable into `.venv`
  by `uv sync`). Imports are absolute and fully qualified — `from pitvis.data.dataset import
  load_split`. Entry points are the console scripts in `[project.scripts]`; run those rather
  than invoking files by path. Tests import the package the same way, so there is no
  `pythonpath` hack in the pytest config.
  *(This reverses the earlier `package = false` / flat-imports decision. That was right when
  `src/` held four scripts; it stopped being right at nine modules across four concerns.)*
- **`ffmpeg` / `ffprobe` are hard requirements and are not Python packages** — both
  `inventory.py` and `extract_features.py` shell out to them, so uv cannot supply them.
- Resolved versions worth knowing: torch 2.13 (MPS available on this Mac), **pandas 3.0**.
  The pandas 3.0 major bump was verified against every pandas path in `src/` — `read_csv`,
  `dropna`, `iterrows`, `pd.isna`, `.to_numpy()` — all pass.
- Torch comes from default PyPI (CPU + MPS). A CUDA host needs a `[[tool.uv.index]]` entry.

## Feature cache is multi-space

`data/features/<space>/video_NN/` — one directory per feature space, named in
`src/pitvis/data/spaces.py`. `resnet50` (2048-d, the original) and
`dinov2_vitb14` (768-d, ViT-B/14 @224) coexist; `--space` selects.

- **The hashed payload is frozen.** `space_id` hashes
  `{backbone, feature_dim, target_fps, transform}` and nothing else. Adding a
  key moves `resnet50` off id `67912d3efc6852e7` and invalidates 940 MB of
  correct features. `Space.name` and `model_kwargs` stay outside it —
  `model_kwargs` only ever changes things that already surface in `transform`.
- A pre-space `data/features/video_NN/` layout is detected and refused with a
  pointer to `pitvis-extract --migrate`, which renames rather than re-decoding.
- `resolve_data_config` reports the CHECKPOINT's native input size, not the
  model's. DINOv2 ships at 518, so a model built at 224 needs an explicit
  `input_size` override or the transform feeds it 518 and it raises.

## Model selection: cross-validate, score val once

Variants are ranked by **5-fold cross-validation over the 19 training videos**
(`src/pitvis/data/folds.py`, frozen literals), and **VAL is scored exactly once
for the winner**. Five videos with a per-video std near 0.05 cannot rank four
variants, and Das et al. measure a −47-point val→test collapse for instruments
against −7 for steps — ranking on VAL ranks noise and turns VAL into a
selection set.

- **Primary `macro_f1`**, because it is what the paper names for task 2 and the
  only metric that moves when a dead class comes alive.
- **Guard on the official `metric`**: no regression beyond one std of the
  19-video spread. It is weighted and support-dominated, so a variant can raise
  macro while lowering the headline.
- Folds are frozen so every variant sees the identical partition. `zero_division=1`
  inflates folds missing a class — a constant offset across variants, recorded
  rather than fixed.
- **`data/instruments/sano.pt` is never overwritten.** Variants write to
  `data/instruments/v2/<variant>/`, so `pitvis-predict` and the app keep
  working against the reproduction.

## Two registries, deliberately separate

- **`training/registry.py`** — the ONLY list of trainable models. Adding one is
  one `Model(...)` entry; `pitvis-train <name>` then works with no CLI or
  packaging change.
- **`inference/checkpoints.py`** — where each model's weights land.
  `pitvis-predict --steps-model arst-v2:best`, `--list-models`.

They stay apart because `main()` is a training entry point while a checkpoint
is an artifact that may not exist yet: `pitvis-train --list` answers *what can
I train*, `--list-models` answers *what have I trained*, and on any given
machine those differ.

Checkpoints carry their own tags — `space`, `variant`, and `mask_excluded`
(steps) or `arch`/`thresholds` (instruments) — each defaulting to what the
original reproductions, which predate all of them, were trained with.
**Inference must honour those tags**: the step winner masks 0/11/13 out of the
argmax, and ignoring the tag would discard most of its advantage while still
reporting its name. Standardisation stats are never resolved separately from
the weights.

## Layout

```
pyproject.toml                 uv-managed deps, console scripts, hatchling build
uv.lock                        pinned resolution — tracked, commit changes to it
.python-version                3.13

src/pitvis/
  paths.py                     ROOT/RAW/DATA/FEATURES/MANIFEST/CKPT/PREDICTIONS/
                               NOTES + PACKAGE/APP_ASSETS — the ONLY place a
                               filesystem location is computed. TWO anchors:
                               ROOT is the repo, PACKAGE is the installed
                               package. They diverge inside a wheel.
  pipeline.py                  Stage record + --only/--skip/--dry-run plumbing
                               shared by the four run.py workflow runners
  data/
    run.py                     WORKFLOW: inventory -> extract -> verify
    inventory.py               per-video duration, resolution, fps, annotated
                               seconds, step distribution
    extract_features.py        1 fps decode, frozen timm resnet50 (num_classes=0)
                               -> data/features/; writes manifest.json (feature
                               space + per-video provenance); refuses to extract
                               into a cache from a different feature space
    verify_cache.py            integrity check of the whole cache — run after any
                               extraction; --probe adds the slow, annotation-
                               independent ffprobe length check
    dataset.py                 per-video (T, D) features + labels, train/val split;
                               also STEP_NAMES/step_name — the ONE definition
  app/
    run.py                     `uv run pitvis-app` — CLI + serve. NOT a pipeline
                               runner (a server has one stage and blocks)
    server.py                  HTTP mechanics ONLY: Range, gzip, SSE, static,
                               Host validation. Knows nothing about surgery
    api.py                     route table; handlers are (Request) -> Response
    catalogue.py               what cases exist + cache/prediction/truth state
    case.py                    build_case() -> the case document (the data model)
    jobs.py                    on-demand inference, one worker, stdout -> SSE
    media.py                   ffprobe metadata, single-frame JPEG (5.4 seam)
    names.py                   step colour ramp; imports names, defines none
    assets/                    index.html, app.css, js/ — no build step
  inference/
    run.py                     WORKFLOW: mp4 -> steps + instruments, one pass
    predict.py                 decode -> embed -> both task heads; no labels needed
  models/
    run.py                     WORKFLOW: shape/param trace through all 3 stages
                               — the executable form of notes/step-variants.md — the task-1 iteration: masking, class weights,
                               DINOv2, and the winning configuration
  notes/instrument-variants.md — the task-2 iteration: variants tested,
                               the CV protocol, and the winning configuration
  notes/citi-dataflow.md
    arst.py                    CITI's task-1 architecture: spatial embedding +
                               TeCNO + ARST (banded causal mask)
    lstm.py                    SANO's task-2 architecture: causal windowed LSTM,
                               19 sigmoid outputs (multi-label)
  training/
    run.py                     WORKFLOW: `pitvis-train <model ...>`, registry-driven
    registry.py                the ONLY list of trainable models — add one here
    arst.py                    three-stage training + CCI auto-regressive inference
    baseline.py                frame-wise linear probe baseline
    instruments.py             SANO task-2 training (BCE, windowed minibatches)
  evaluation/
    run.py                     WORKFLOW: score an existing checkpoint, no retrain
    official.py                VENDORED official STEPS metric — do not edit
    official_instruments.py    VENDORED official INSTRUMENT metric — do not edit
    metric.py                  task-1 metric per video + mean±std, plus pooled
                               diagnostics
    instruments.py             task-2 metric; reports the official number, the
                               name-aligned one, and macro (see below)

tests/test_eval.py             pins evaluation/metric.py to the official metric
tests/test_app_range.py        pins HTTP Range parsing (see below)
tests/test_app_case.py         pins the case document + the probability additions
notes/                         see the doc-layer section below
data/features/                 per-video features.npy + labels.npy (gitignored)
data/arst/                     CITI checkpoints, standardize.npz, result.json
predictions/<stem>/            pitvis-predict output (gitignored)
26531686/                      raw PitVis download (gitignored, read-only)
```

Commands are console scripts declared in `pyproject.toml`, so the CLI surface and the
import graph cannot drift apart. Each package has a `run.py` that runs that directory
as one workflow, plus per-stage scripts for when you want a single step:

```
uv run pitvis-data              inventory -> extract -> verify
uv run pitvis-train [model ...] registry-driven; bare command trains ALL
uv run pitvis-predict --video   mp4 -> steps AND instruments; labels optional
uv run pitvis-eval              score an existing checkpoint, no retraining
uv run pitvis-models            shape/param trace (~1 s smoke test)
uv run pitvis-app               play a case beside the model's output

uv run pitvis-train --list      what models exist
uv run pitvis-inventory   uv run pitvis-extract   uv run pitvis-verify   uv run pytest
```

Only `pitvis-data` and `pitvis-train` use `pipeline.py`, and only `pitvis-data`
takes the full `--only`/`--skip`/`--dry-run`/`--continue-on-error` set;
`pitvis-train` declares `--dry-run` and `--continue-on-error` itself because its
stages are positional. `pitvis-predict`, `pitvis-eval` and `pitvis-app` are
single-workflow scripts and have none of them — `pipeline.py` sequences a finite
stage list and prints a timing summary, which a one-stage command does not need
and a blocking server would never reach.

## App (`pitvis-app`) — decisions

Reasoning and measured numbers live in `notes/app.md`. These are the rules.

- **No web framework, and no build step.** `http.server` + hand-written Range,
  native ES modules, no npm. The only thing starlette would have added is a
  Range-capable file response, and Range is the most load-bearing behaviour
  here, so it gets written and tested regardless. `pitvis-app` adds **zero**
  dependencies.
- **Range is mandatory, not an optimisation.** Every PitVis video has box order
  `ftyp, free, mdat, moov` — the index is the last ~1.3 MB of a multi-gigabyte
  file, so a browser cannot start playback without fetching a tail range.
  `parse_range` is a pure function pinned by `tests/test_app_range.py` against
  video_25's real offsets. Do not "simplify" it into a whole-file send.
- **Validate the `Host` header.** Binding to `127.0.0.1` is not enough: without
  it, any page the user visits reaches the server by DNS rebinding and can
  stream patient video off loopback.
- **Assets are anchored on `paths.APP_ASSETS` (= `PACKAGE`), never `ROOT`.**
  `ROOT` walks out of the package and does not exist in an installed wheel.
  `src/pitvis/app/assets/` is the first non-`.py` content in the package;
  `tests/test_app_case.py` resolves it through `importlib.resources` so the
  failure shows up from inside the install, not only on a clone.
- **Probabilities are PRE-CCI.** `cci_decode(..., return_probs=True)` records
  the decoder's distribution at the moment of decision, which the consistency
  constraint may then override — 3.8% of seconds on video_25. Confidence is
  therefore defined as `probs[t][emitted]`, **not** `max()`: it reads low
  exactly where CCI is holding a phase the frame does not support, and that is
  the signal. Never label it "model confidence" unqualified.
- **`(-1, -2)` means different things by source.** In annotations it is
  out-of-patient; in a prediction it can only mean nothing cleared the
  threshold, because SANO has no out-of-patient class. 26% of video_19. The
  collision is resolved once in `case._instrument_state` and the wire format
  carries a `state` string, never the raw pair.
- **Categories float; they never stack in a scrolling column.** A fixed rail
  puts every category in one column, and the moment it overflows it scrolls —
  the one interaction that guarantees you cannot see two things at once. Steps
  and instruments are two halves of one judgement. `panels.js` gives each
  category an independently collapsible, draggable panel; `_bounds()` reserves
  the transport strip, because a panel takes pointer events and one covering
  PLAY eats the click rather than merely hiding it. STATUS is collapsed by
  default — its live values are already burned into the frame — which is what
  lets all fourteen worklist rows fit with nothing to scroll.
- **Panels may sit over the image, and that is measured, not assumed.** Across
  the 5 validation videos at 3 timestamps each, the left gutter is optical
  black for at least 217 of 1280 px (17.0%) in every frame. The right is **not**
  reliably dead — `video_01` runs out to x=1176, leaving 8%. Re-measure before
  trusting either number for a new default.
- **No `backdrop-filter` on a panel.** A blur behind an element overlapping the
  video makes the compositor re-read the video layer every frame, and this
  surface has already lost the video once to a compositing bug. Flat alpha.
- **The default view hides the analyst layer.** Confidence, ground truth,
  agreement, scores and per-class probabilities are behind `[ + DETAIL ]`. The
  visible layer answers "what is happening now"; the hidden one answers "how
  well is the model doing". Six stacked timeline lanes reads as a video editor.
- **Honesty elements are load-bearing.** The research banner, the amber
  train-split chip, the stated absence of ground truth, and always-numeric
  confidence exist because the model scores 0.331 and a composed surface makes
  any number on it read as authority. Do not trim them for cleanliness.
- **One inference worker, permanently.** `redirect_stdout` swaps a
  process-global `sys.stdout`.
- `renderTimeline(ctx, doc, geom, opts)` **is pure**. That is what makes
  corrections and multi-case comparison cheap rather than a rewrite.

**`.gitignore` patterns must be anchored.** An unanchored `data/` matches
`src/pitvis/data/` as well as the repo-root cache, which silently kept that
package's `run.py` and `__init__.py` out of git — `uv run pitvis-data` was broken
on a fresh clone while working perfectly locally. Anything ignored at the repo
root gets a leading slash; leave only genuinely recursive patterns
(`__pycache__/`, `.venv/`) unanchored. Note `git mv` moves *tracked* files, which
stay tracked regardless of ignore rules — so the breakage only shows up in files
created after a rename, and only on a clone.

Three structural rules:

- **`run.py` orchestrates, never reimplements.** A runner selects and sequences
  stages; the behaviour lives in the stage module's `main(argv)`, which is also what
  the per-stage console script calls. There is one definition of "extract features".
  Every `main()` takes `argv: list[str] | None = None` so runners can compose them
  without touching `sys.argv`.
- **Models live in `training/registry.py`, not in `pyproject.toml`.** Adding a model is
  one `Model(...)` entry; `pitvis-train <name>` then works with no CLI or packaging
  change, and the bare `pitvis-train` picks it up automatically — the default set is
  *derived* from `REGISTRY`, never hand-listed. `ORDER_HINT` fixes the order of the
  models it names and nothing else; a model missing from it still runs.
  `tests/test_registry.py` pins this, because it broke once already. Per-model console scripts (`pitvis-train-arst`) are gone — three places to
  edit meant three chances to forget one.
- **One path from pixels to features.** `extract_features.embed_video` is used by both
  cache extraction and `pitvis-predict`, so a prediction is always computed in the
  feature space the checkpoint was trained on.
- **Never recompute a path.** Import from `pitvis.paths`. Five modules used to derive
  `ROOT` independently, each encoding "I live one level below the repo root" — a
  constraint that broke the moment anything moved.
- **`extract_features.py` must stay resumable** — decoding 40 GB is expensive; skip
  videos whose outputs already exist and are the expected length.

### The docs are layered by depth, on purpose

Do not merge them. Each has a different reader in a different moment:

- **`CLAUDE.md`** — decisions only, terse. What we chose and what must not change.
- **`notes/where-we-are.md`** — the orientation layer, and a dated snapshot rather
  than a permanent reference: vocabulary, the iterations so far with their numbers,
  what to run next, and what to carry to another machine. Read it first after time
  away; re-date it when it goes stale rather than leaving stale numbers standing.
- **`notes/embeddings.md`** — conceptual, assumes nothing, every number read off the
  real cache. The grounded-explanation layer; entry point for the ML side.
- **`notes/walkthrough.md`** — reasoning and domain, with `file.py:NN` pointers.
  A code tour; assumes ML fluency.
- **`notes/roadmap.md`** — what is left to build, phased.
- **`notes/citi-baseline.md`** — the CITI reproduction: *why* the architecture is what
  it is, faithfulness, results.
- **`notes/citi-dataflow.md`** — the same model as a shape trace: *what shape the data
  is* at every hop. Reference layer; read alongside `citi-baseline.md`, not instead.
- **`notes/data-dictionary.md`** — every annotation column and what each integer means,
  with real distributions. The reference layer for the *data*, as `citi-dataflow.md` is
  for the model. `CLAUDE.md` keeps only the decisions; look things up there.
- **`notes/metrics.md`** — what macro F1, the edit score and weighted F1 each measure,
  why the challenge picked them, and what each catches that the others miss. The
  reference layer for *scoring*. `CLAUDE.md` keeps the rules that must not change;
  the reasoning lives there.
- **`notes/instruments.md`** — the SANO task-2 reproduction: why not the rank-1 model,
  the metric's column-ordering defect, results.
- **`notes/app.md`** — the review surface: why Range is load-bearing, the
  `(-1, -2)` collision, what pre-CCI confidence means, and why the default view
  hides most of what the repo can measure. Same layer as the two above.

`walkthrough.md` §8 and `embeddings.md` deliberately cover the same extraction stage
at two depths. They are cross-linked, not deduplicated. When adding docs, pick the
layer first.
