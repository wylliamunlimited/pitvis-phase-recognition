# Roadmap

Everything left to build, in the order we intend to build it. This is a working
document: phases get checked off and amended as decisions are made, and each
completed item should end up reflected in `CLAUDE.md` (decisions) or the code.

The order is deliberate — **data engineering, then end-to-end pipelining, then
models, then app.** Modeling ideas are cheap and the temptation is to start
there, but every model we try will be bottlenecked by the same missing plumbing
(no checkpoints, no reusable normalisation, no sequence loader). Building that
once is faster than working around it three times.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started · **(D)** needs a
decision from us before it can be built.

---

## Phase 0 — Foundation (done)

What already exists and has been verified by running it, not just by reading it.

- [x] Raw data inventoried and invariants verified — `src/inventory.py`,
      `notes/inventory.md`. Resolution uniform, fps *not* uniform (video 24),
      annotation off-by-one confirmed on all 24 labeled videos.
- [x] Annotation semantics pinned down — no step 0, `-1` is a collapsed
      three-way background, map files are not uniquely keyed. See `CLAUDE.md`.
- [x] 1 fps decode + frozen ResNet-50 feature extraction — `src/extract_features.py`.
      Resumable. Smoke-tested on video 07: 2,646 frames in 47 s (~56 fps) on MPS.
- [x] Per-video loaders and the 19/5 train/val split — `src/dataset.py`.
- [x] Official challenge metric vendored verbatim — `src/official_metric.py`.
- [x] Evaluation aligned to the challenge: per video, mean-averaged, with the
      three official quirks preserved — `src/eval.py`, pinned by `tests/test_eval.py`
      (23 tests, all passing).
- [x] Frame-wise linear probe baseline written — `src/train_baseline.py`.
- [~] **First full feature extraction run.** Until now `data/` did not exist —
      the pipeline had never been executed end to end and no baseline number
      existed. Running now; ~35 min for all 25 videos, ~1 GB of cache
      (120,018 frames × 2048 × 4 B).

Getting evaluation provably correct *before* any model exists is the right
order, and it is the strongest part of the foundation. The weakest part is that
nothing downstream of feature extraction has ever produced an artifact.

---

## Phase 1 — Data engineering

Goal: the feature cache becomes a trustworthy, self-describing asset, and every
model we later write can get the data shape it needs without re-inventing a
loader.

- [ ] **1.1 Cache verifier.** A script that checks the whole cache at once:
      every expected video present, `features.npy` length matches
      `ceil(nb_frames / round(fps))` from the inventory, `labels.npy` matches
      features length, dtypes are `float32`/`int64`, no NaN/Inf, label values in
      `0..14`. `extract_features.py` asserts these at *write* time; nothing
      re-checks them afterwards, so a truncated or half-written file is
      currently invisible until training silently misbehaves.

- [ ] **1.2 Cache manifest.** Write a `data/features/manifest.json` recording
      backbone name, timm model tag, the resolved transform config, target fps,
      per-video frame counts, and extraction timestamp. Right now `features.npy`
      carries no provenance whatsoever — swapping the backbone and re-extracting
      only some videos would silently mix two feature spaces in one training
      run, and nothing would catch it.

- [ ] **1.3 Normalisation statistics as a saved artifact.** `train_baseline.py`
      computes train-split mean/std inline (`src/train_baseline.py:45`) and
      discards them when the process exits. Any inference path must apply the
      *same* transform, so these have to become a saved artifact keyed to the
      cache + split. This is a correctness blocker for the app, not a
      convenience.

- [ ] **1.4 Sequence dataset.** `src/dataset.py` is 37 lines of whole-array
      loading. Temporal models need: full-video sequences (an MS-TCN trains on
      one whole video per batch — 8,645 × 2048 fits in memory comfortably),
      fixed-length windows with stride and padding, and a collate that handles
      variable lengths with a mask. Build all three behind one interface so
      model code never touches `.npy` paths.

- [ ] **1.5 Imbalance utilities.** Class frequencies from the train split →
      inverse-frequency and effective-number class weights, plus a balanced
      sampler. The metric is macro-averaged over a 23.9% / 0.06% distribution
      while the baseline uses unweighted cross-entropy, so this is a direct
      mismatch between what we optimise and what we are scored on.

- [ ] **1.6 Generalised extraction path.** `extract_features.py` is hardcoded to
      `26531686/video_{n:02d}.mp4` and the `annotations_{n}.csv` convention
      (`src/extract_features.py:65`, `:116`). Accept an arbitrary video path with
      optional labels. Required by `predict.py` in Phase 2 — an app cannot only
      work on the 25 videos we happen to have.

- [ ] **1.7 (D) Frame access for end-to-end fine-tuning.** Extraction saves
      features *only*, never frames, so fine-tuning the backbone — the single
      largest accuracy lever, given ImageNet ResNet-50 against endoscopic video
      is a severe domain gap — has no data path at all today. Two options:
      decode 1 fps frames to disk as JPEGs (≈4 GB at 256 px shorter side, ≈25 GB
      at native 720p), or an on-the-fly decoding `Dataset` (no storage cost,
      much slower per epoch). **Decision needed**, but it can wait until Phase 3
      proves frozen features are the ceiling.

---

## Phase 2 — End-to-end pipelining

Goal: one command trains any model, saves everything needed to reproduce and to
re-run it on a new video, and writes a comparable score. Nothing here is
model-specific.

- [ ] **2.1 Config.** A single typed config (dataclass, CLI-overridable) covering
      data, model, optimiser, and run identity. Today each script grows its own
      ad-hoc `argparse` block and defaults live in the flag definitions.

- [ ] **2.2 Shared training loop.** Seeding, device selection, epoch loop,
      validation, early stopping / best-checkpoint selection on the official
      metric, and logging — written once, reused by every model. Currently
      `train_baseline.py` inlines all of it in 60 lines and any second model
      would copy-paste it.

- [ ] **2.3 Checkpointing.** Save model weights **plus** normalisation stats,
      config, label encoding, and the feature-cache manifest hash in one
      artifact. `train_baseline.py` currently trains, prints, and discards the
      model — no run so far has produced anything reusable. This is the single
      blocker shared by both the modeling and app tracks.

- [ ] **2.4 Run artifacts.** Per-run directory: `config.json`, `metrics.json`
      (the full dict from `eval.evaluate`, per video and aggregate),
      per-video predictions as `.npy`, and the console report. Makes runs
      diffable instead of scrollback-dependent.

- [ ] **2.5 `src/predict.py`.** Video path → features (1.6) → checkpoint (2.3) →
      per-second step predictions, emitted both as a raw array and as merged
      `(start_s, end_s, step)` segments. Optional ground-truth scoring when
      labels are supplied. This is the piece that turns the repo from an
      experiment into something that can be pointed at a new case.

- [ ] **2.6 Reproducibility check.** Same seed + same config → same metric.
      Worth one test, because MPS non-determinism is easy to mistake for a real
      model improvement.

---

## Phase 3 — Models

Ordered cheapest-first. Each step gets a row in the results table (Phase 4)
before the next one starts, so we always know what an idea actually bought.

- [ ] **3.1 Baseline, completed.** Record the linear probe's official metric as
      the floor, then add the three known gaps: class weighting (1.5),
      checkpointing (2.3), and masking classes 0/11/13 out of the argmax at
      inference. `CLAUDE.md` already establishes that the last one *can only
      raise* the official metric, since exclusion filters by ground truth only —
      it is a free win that no model currently takes.

- [ ] **3.2 Temporal model over frozen features — MS-TCN.** The standard strong
      baseline for surgical phase recognition, and the natural next step: the
      linear probe has no temporal context at all, so its edit score is near the
      floor by construction. Expect the largest single jump here.

- [ ] **3.3 Alternatives.** Bi-GRU/LSTM and a windowed transformer over the same
      features, for comparison against 3.2 under an identical harness.

- [ ] **3.4 Post-processing.** Segment smoothing and order priors. Surgical steps
      follow a largely monotonic sequence (corridor → sphenoidotomy → sellotomy →
      durotomy → excision → closure), which the frame-wise metric ignores but the
      *edit score* rewards directly.

- [ ] **3.5 (D) Instruments as an auxiliary task.** Every annotation row carries
      `int_instrument1` / `int_instrument2` and we currently discard both. The
      instrument in view is strongly predictive of the step, and multi-task
      supervision is a well-established win on this kind of data. Cheap to try
      once 1.4 exists.

- [ ] **3.6 (D) End-to-end fine-tuned backbone.** Gated on 1.7. Highest expected
      gain, highest cost.

---

## Phase 4 — Evaluation and analysis

The metric itself is done and tested; what is missing is everything *around* it.

- [ ] **4.1 Results table.** One row per run in `notes/results.md`: config, macro
      F1, edit score, official metric ± std, date, commit. Append-only.
- [ ] **4.2 Error analysis.** Which steps get confused (the confusion matrix
      exists but has never been looked at on real predictions), where segment
      boundaries drift, and which of the 5 val videos drive the variance.
- [ ] **4.3 Ablations.** Temporal context length, class weighting on/off,
      post-processing on/off.
- [ ] **4.4 Split-variance caveat.** Five validation videos is a small sample and
      the reported std is across only those five. Worth stating explicitly
      wherever we quote a number, and worth a cross-validation run before
      believing any small improvement.

---

## Phase 5 — App

Deliberately last: nothing here is buildable until 2.3 and 2.5 exist.

- [ ] **5.1 (D) Surface.** CLI report, local web UI, or a served API — decide
      before building, since it determines everything below.
- [ ] **5.2 Inference service.** Wraps 2.5. Model loaded once, videos processed
      on request, progress reported (a full case is 45–140 min of video, so this
      is not a sub-second request).
- [ ] **5.3 Timeline visualisation.** Predicted step timeline against ground
      truth where available, with per-segment confidence.
- [ ] **5.4 Packaging.** Reproducible run instructions, model artifact
      distribution.

---

## Cross-cutting risks

- **Frozen ImageNet features are probably the ceiling.** Everything through 3.4
  builds on a backbone that has never seen an endoscope. If temporal modeling
  plateaus well below the paper, 3.6 is the reason.
- **The val split is 5 videos.** Small improvements will not be distinguishable
  from split noise (4.4).
- **Video 19 has no labels** — we train on 19 videos, not the paper's 20, so our
  numbers are not exactly comparable on the training side even though validation
  is untouched.
- **Classes 11 and 13 are essentially unlearnable** (2 videos and 1 video
  respectively) and are excluded from scoring anyway. Do not spend effort there.
