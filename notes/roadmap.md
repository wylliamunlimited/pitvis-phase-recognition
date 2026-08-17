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

- [x] Raw data inventoried and invariants verified — `src/pitvis/data/inventory.py`,
      `notes/reference/inventory.md`. Resolution uniform, fps *not* uniform (video 24),
      annotation off-by-one confirmed on all 24 labeled videos.
- [x] Annotation semantics pinned down — no step 0, `-1` is a collapsed
      three-way background, map files are not uniquely keyed. See `CLAUDE.md`.
- [x] 1 fps decode + frozen ResNet-50 feature extraction — `src/pitvis/data/extract_features.py`.
      Resumable. Smoke-tested on video 07: 2,646 frames in 47 s (~56 fps) on MPS.
- [x] Per-video loaders and the 19/5 train/val split — `src/pitvis/data/dataset.py`.
- [x] Official challenge metric vendored verbatim — `src/pitvis/evaluation/official.py`.
- [x] Evaluation aligned to the challenge: per video, mean-averaged, with the
      three official quirks preserved — `evaluation/metric.py`, pinned by
      `tests/test_eval.py`
      (23 tests, all passing).
- [x] Frame-wise linear probe baseline written — `src/pitvis/training/baseline.py`.
- [x] **First full feature extraction run.** Completed 2026-08-04: all 25
      videos, 120,018 frames, 939 MB. Verified end to end by
      `src/pitvis/data/verify_cache.py --probe` (every check passing).

Getting evaluation provably correct *before* any model exists is the right
order, and it is the strongest part of the foundation. The weakest part is that
nothing downstream of feature extraction has ever produced an artifact.

---

## Phase 1 — Data engineering

Goal: the feature cache becomes a trustworthy, self-describing asset, and every
model we later write can get the data shape it needs without re-inventing a
loader.

- [x] **1.1 Cache verifier.** `src/pitvis/data/verify_cache.py`. Checks features
      (dtype, shape, finiteness), labels (re-derived byte-for-byte from the raw
      annotation CSVs, which also re-verifies the off-by-one alignment), the
      video-19 special case, and manifest consistency. `--probe` adds the
      ffprobe length check that is independent of the annotations. Exit code 0
      iff clean; verified to catch injected NaN/truncation/drift.

- [x] **1.2 Cache manifest.** `data/features/manifest.json`, written by
      `extract_features.py`: feature space (backbone, transform config, target
      fps, content-hash id) plus per-video frames/fps/labels/timestamp.
      Extraction refuses to write into a cache whose manifest describes a
      different feature space, so mixed-backbone caches fail loudly instead of
      silently. Checkpoints (2.3) should record `space.id`.

- [ ] **1.3 Normalisation statistics as a saved artifact.** `train_baseline.py`
      computes train-split mean/std inline (`src/pitvis/training/baseline.py:45`) and
      discards them when the process exits. Any inference path must apply the
      *same* transform, so these have to become a saved artifact keyed to the
      cache + split. This is a correctness blocker for the app, not a
      convenience.

- [ ] **1.4 Sequence dataset.** `src/pitvis/data/dataset.py` is 37 lines of whole-array
      loading. Temporal models need: full-video sequences (an MS-TCN trains on
      one whole video per batch — 8,645 × 2048 fits in memory comfortably),
      fixed-length windows with stride and padding, and a collate that handles
      variable lengths with a mask. Build all three behind one interface so
      model code never touches `.npy` paths.

- [~] **1.5 Imbalance utilities.** Done for task 2, still open for task 1.
      `training/instruments_v2.py` computes capped inverse-frequency
      `pos_weight` from the fold's own training videos, and it is the single
      largest win measured so far: macro F1 0.296 → 0.401 out of fold, with
      classes never predicted going 7/19 → 0/19. The steps model still trains
      on unweighted cross-entropy over a 23.9% / 0.06% distribution while being
      scored macro, so the same mismatch remains there.
      No balanced *sampler* was needed — `pos_weight` reweights the loss without
      changing the effective epoch length, which keeps a variant comparable to
      its control on the same compute budget.

- [ ] **1.6 Generalised extraction path.** `extract_features.py` is hardcoded to
      `26531686/video_{n:02d}.mp4` and the `annotations_{n}.csv` convention
      (`src/pitvis/data/extract_features.py:63`, `:116`). Accept an arbitrary video path with
      optional labels. Required by `predict.py` in Phase 2 — an app cannot only
      work on the 25 videos we happen to have.

- [x] **1.7 (D) Frame access for end-to-end fine-tuning.** Resolved in favour
      of a JPEG cache on disk: `pitvis-frames` writes 1 fps centre-square 384 px
      JPEGs to `data/frames/384/`. Measured **120,018 frames, 3.6 GB,
      33.7 KB/frame, ~19 min** — the on-the-fly decoding `Dataset` was rejected
      because it pays the decode cost once per epoch rather than once. A space
      with `source="frames"` reads it, which is mandatory for a fine-tuned
      encoder: it was tuned on those exact crops, and feeding it differently
      framed pixels would measure the preprocessing mismatch as much as the
      model.

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
      (the full dict from `metric.evaluate`, per video and aggregate),
      per-video predictions as `.npy`, and the console report. Makes runs
      diffable instead of scrollback-dependent.

- [x] **2.5 `pitvis/inference/predict.py`.** ✅ Now runs BOTH tasks off one
      feature pass — steps (ARST) and instruments (SANO). Video path → features (1.6) → checkpoint (2.3) →
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

- [x] **3.4 Task 2: instrument recognition.** ✅ SANO's joint-winning model
      (frozen ResNet-50 -> causal 5-window LSTM -> 19 sigmoid outputs, BCE),
      `uv run pitvis-train instruments`. The official metric is vendored
      alongside, including a column-ordering defect that is preserved but
      surfaced. Results and the val->test caveat in `notes/models/instruments.md`.
      Note this is the *standalone* task-2 model; 3.5 below is still open.

- [x] **3.5 (D) Instruments as an auxiliary task.** Done in the other
      direction: `models/lstm.py` carries a 15-way step head alongside the
      19-way instrument head, trained and discarded, which is SANO's own
      "step (just for training)" design. `--no-aux-step` ablates it. The
      reverse — instruments supervising the *step* model — is still untested.

- [x] **3.6 (D) End-to-end fine-tuned backbone — pulled, on ResNet-50.**
      `pitvis-finetune` trains one backbone with two heads (15-way CE steps,
      19-way BCE instruments) on the frame cache, and augments — random crop,
      flip, colour jitter, three rows the faithfulness tables had marked "we
      cannot" purely for want of pixels. A 5-epoch pilot moved **mean AP
      0.271 → 0.445 with 19/19 classes improving**.

      **End to end it does not win the headline, and how it loses is the
      finding.** VAL scored once per arm, `best` recipe, frozen DINOv2 vs
      fine-tuned ResNet-50: steps metric 0.4610 → 0.4425 (macro 0.4420 →
      0.4658, **edit 0.4801 → 0.4193**); instruments official 0.5572 → 0.3805
      (**macro 0.3792 → 0.4783**). It is better on the rare classes macro
      weights equally and worse on the four carrying ~91% of positives, and on
      steps it trades segment stability for per-frame accuracy — which is what
      a frame-wise objective with no temporal term should be expected to do.
      Numbers in [`step-variants.md` §6](models/step-variants.md) and
      [`instrument-variants.md` §6](models/instrument-variants.md).

- [x] **3.6b Fine-tune DINOv2 — done, and it is the largest gain in the
      project.** The pilot went to ResNet-50 because ViT-B trains at 29 img/s
      against 96 — the cheap backbone answers "does fine-tuning help at all"
      for a quarter of the cost. It does, but it was applied to the backbone
      that *loses* frozen, so this applied it to the one that wins frozen.

      **It took two runs, and the difference between them is the recipe, not
      the idea.** Run 1 (uniform 1e-4, 50 epochs, no validation anywhere)
      destroyed the representation: mean AP 0.350 → 0.270, steps 0.4610 →
      0.3500. Run 2 (backbone lr 1e-5, layer-wise decay 0.75, 200-step warmup,
      early stopping on three TRAIN videos held out by video — stopped at
      **epoch 2**) reversed it: **mean AP 0.523 with 19/19 classes improved,
      steps 0.4610 → 0.5608, instruments macro 0.3792 → 0.5333.** Eleven of
      twelve scored step classes improve and durotomy goes 0.000 → 0.573 F1.
      Numbers in [`step-variants.md` §7](models/step-variants.md) and
      [`instrument-variants.md` §6](models/instrument-variants.md).

- [ ] **3.6c Cross-validate `dinov2_ft` honestly.** Everything in 3.6b is a
      single VAL measurement. A ranking needs one encoder per fold — six
      fine-tunes, `STAGE=all`, ~23 h on an L4 — because a single encoder
      trained on all of TRAIN leaks into every fold (measured: steps macro read
      0.917). It also needs a harness change first: each fold must read
      features from its own encoder, which one `--space` cannot express. See
      [`infra/`](../infra/README.md).

---

- [ ] **3.7 Post-processing.** Segment smoothing and order priors. Surgical steps
      follow a largely monotonic sequence (corridor → sphenoidotomy → sellotomy →
      durotomy → excision → closure), which the frame-wise metric ignores but the
      *edit score* rewards directly.

## Phase 4 — Evaluation and analysis

The metric itself is done and tested; what is missing is everything *around* it.

- [ ] **4.1 Results table.** One row per run in `notes/models/results.md`: config, macro
      F1, edit score, official metric ± std, date, commit. Append-only.
- [ ] **4.2 Error analysis.** Which steps get confused (the confusion matrix
      exists but has never been looked at on real predictions), where segment
      boundaries drift, and which of the 5 val videos drive the variance.
- [ ] **4.3 Ablations.** Temporal context length, class weighting on/off,
      post-processing on/off.
- [x] **4.4 Split-variance caveat.** `data/folds.py` + `training/crossval.py`.
      Variants are ranked by 5-fold cross-validation over the 19 training
      videos — each held out exactly once, scored per-video-then-mean — and VAL
      is touched once, for the winner. Folds are frozen literals so every
      variant sees the identical partition.
      It earned its keep immediately: DINOv2 alone gains +0.021 macro against a
      ±0.048 fold spread, which on the five-video split could easily have read
      as a real improvement and been shipped.

---

## Phase 5 — App

Deliberately last: nothing here is buildable until 2.3 and 2.5 exist.

The target is a **minimal app with an agentic explanation layer**: an agent
that walks the viewer through a case by interacting with the video itself —
circling a region, then captioning it, and so on — and that *adapts to the
output of the ML model* (predicted step, segment boundaries, confidence).
The phase model supplies the *when*; the agent supplies the narration and
the *where*.

- [x] **5.1 (D) Surface — DECIDED: a local web UI.** `uv run pitvis-app`,
      stdlib HTTP server, no build step, zero new dependencies. A `<canvas>`
      sits over the video with a `Layer` registry, sized in video pixels, so
      5.4 is a layer rather than a refactor. See `notes/surfaces/app.md`.
- [~] **5.2 Inference service.** The cheap half is done: a case with cached
      features can be predicted from the page (~45 s), one worker, stdout
      streamed to the browser as SSE. Still missing: videos OUTSIDE the feature
      cache, which need a full 1 fps decode (10–25 min) and a warm long-lived
      process rather than an in-process call. The app refuses those and prints
      the command.
- [x] **5.3 Timeline visualisation.** Predicted steps against ground truth,
      per-segment confidence, and a lane marking where the two disagree. Note
      confidence required 5.6 first — nothing persisted a probability before.
- [ ] **5.4 (D) Agentic explanation layer.** The agent consumes 2.5's output
      (per-second steps, merged segments, confidences) plus sampled frames, and
      decides what to point at and what to say, per segment — e.g. circle the
      relevant region, then caption it in surgical-context language, adapting
      when the model is uncertain or the step sequence is atypical. **Open
      decision: spatial grounding.** The step classifier is temporal-only —
      nothing we build in Phases 1–3 localises anything in the frame. Options:
      a VLM that can point/box on frames at explanation time, or the instrument
      annotations (3.5) as a supervised hook. Related: the 3.5 auxiliary task
      becomes more valuable if its predictions feed the agent.
- [~] **5.5 Packaging — and the "public repo" gap.** Run instructions and
      licensing are done: AGPL-3.0 with `NOTICE` scoping the vendored
      organiser scripts out of the grant, and a README that says up front what
      a fresh clone can run without the 40 GB download (`pytest` and
      `pitvis-models`, and nothing else).

      **Still open, in the order that matters for anyone arriving cold:**

      - [ ] **Screenshots.** `docs/app-default.png` and `docs/app-detail.png`
            are referenced by the README and do not exist yet, so the images
            render broken. Capture from a running `pitvis-app`, then blur the
            surgical frame — the dataset is CC BY-NC-ND 4.0 and a screenshot
            with the interface composited around dataset video is arguably a
            derivative work, which ND forbids:

                uv run python scripts/blur_frame.py raw.png --preview
                uv run python scripts/blur_frame.py raw.png --out docs/app-default.png

            Almost nothing is lost by blurring: the point of those images is
            the step card, the confidence readout and the timeline, not the
            anatomy. Instructions also live in `docs/README.md`.
      - [ ] **Checkpoint distribution.** `data/arst/citi.pt` and
            `data/instruments/sano.pt` are small and gitignored. Publishing
            them as a release asset would let someone run `pitvis-predict` on
            their own video without a training run — though they would still
            need a video, so this is a weaker win than it first looks.
      - [ ] **The app is unreachable on a clone.** Everything it needs —
            dataset, feature cache, checkpoints, predictions — is gitignored,
            so `pitvis-app` prints "no cases found". The screenshots are the
            cheap mitigation; a genuinely runnable demo would need a
            redistributable sample case, which CC BY-NC-ND rules out.
- [x] **5.6 Confidence as an artifact.** `cci_decode` and `predict_video` take
      a keyword-only `return_probs`; `pitvis-predict --probs` writes
      `step_probs.npy` (T, 15) and `instrument_probs.npy` (T, 19). Additive by
      construction — `predictions.csv` is byte-identical before and after.
      **The step distribution is PRE-CCI**: it is the decoder's belief at the
      moment of decision, which the consistency constraint may then override
      (3.8% of seconds on video_25). Confidence is therefore `p(emitted step)`,
      not `max`, so it reads low exactly where CCI is holding a phase.
- [ ] **5.7 Human-in-the-loop correction.** Confirm or override a predicted
      step, persist to `predictions/<id>/corrections.json`, export as an
      `annotations_NN.csv`. Seam exists: `doc.corrections` ships empty,
      `segments[].source` is `"model"`, `/corrections` is routed and 501s.
- [ ] **5.8 Live / streaming input.** Seam exists: the clock is a `TimeSource`
      interface and `VideoTimeSource` is one implementation, so no renderer
      knows a file is involved.
- [ ] **5.9 Multi-case comparison.** Seam exists: case documents are
      self-contained (comparison is N fetches) and `renderTimeline` is a pure
      function of its arguments.

---

## Phase 6 — Task-2 iteration (done)

Recorded in full in [`notes/models/instrument-variants.md`](models/instrument-variants.md).

- [x] **6.1 Multi-space feature cache.** `data/features/<space>/`, named in
      `data/spaces.py`. The hashed payload is frozen, so the existing cache
      migrated by rename rather than re-extraction and still verifies at
      `67912d3efc6852e7`.
- [x] **6.2 Cross-validation harness.** See 4.4.
- [x] **6.3 Variants tested.** control / weighted / thresholds / dinov2 /
      composed. Winner is pos_weight + per-class thresholds on DINOv2, and it
      took classes never predicted from **9/19 to 0/19**. Numbers in
      [`instrument-variants.md` §4](models/instrument-variants.md).
- [x] **6.4 Wired into the product.** `pitvis-predict` and the app dispatch on
      the checkpoint's arch/space tags and embed per space. `sano.pt` is
      untouched and still reproduces byte for byte.
- [ ] **6.5 Four classes are emitted but not usable.** Classes 1, 4, 12 and 17
      are predicted 1, 25, 19 and 7 times against supports of 184–492, so
      "0 never predicted" flatters. They cleared the bar of being emitted
      without becoming useful, and that is the next honest target.
- [x] **6.6 The same treatment for task 1.** `training/arst_v2.py`, same
      protocol and the same folds. Winner is argmax masking + class weights on
      DINOv2. Masking alone was the largest single lever (+0.062 macro out of
      fold) and had been sitting unclaimed as an ablation flag. Numbers in
      [`step-variants.md` §4](models/step-variants.md).
- [x] **6.7 The CV harness is task-agnostic.** `crossval.Task` holds the
      loader, scorer and ranking metric, so both tasks share the fold logic and
      the leaderboard instead of a second copy.
- [x] **6.8 Per-class thresholds for steps — tried, and it loses.** Task 2's
      second-largest win has no direct analogue: steps are multi-class with one
      argmax rather than 15 independent sigmoids. The equivalent is a
      prior-corrected argmax (logit adjustment, Menon et al. 2021), and it
      degrades **monotonically in tau**. The cause is double correction — the
      winning recipe already carries inverse-frequency class weights, and the
      sign of the adjustment's effect flips when those are removed. Training-
      time correction beats inference-time correction on this task, the
      opposite of task 2. Numbers in
      [`step-variants.md` §8](models/step-variants.md).

### The finding worth carrying forward

**A frozen-backbone swap must not be tested first.** On BOTH tasks, DINOv2
alone landed inside the fold spread — +0.021 macro on instruments, +0.029 on
steps — and on both it became the best variant once the loss and decision rule
were fixed (+0.055 and +0.014 respectively, winning on every metric). The
representation gain is real and it is masked by the imbalance defect. Testing
the backbone first and stopping at the null would have retired a true
hypothesis, twice.

**And the same mistake was nearly made a third time, from the other end.**
Fine-tuning DINOv2 produced the worst result in the repo (3.6b run 1) and the
best (run 2), from the same code, the same data and the same augmentation. Only
the optimisation recipe differed. A fine-tune that fails tells you nothing
about whether fine-tuning helps until you can see the failure happening — run 1
had no validation split at all, so fifty epochs of damage were indistinguishable
from fifty epochs of learning from inside the loop.

The generalisable form: **every expensive experiment needs a cheap falsifier
fixed before it runs.** Here that was the AP probe, which fits on TRAIN and
scores VAL, so it reads the representation without the temporal model in the
way. It called run 1 correctly (0.270, abandon) and run 2 correctly (0.523,
proceed) — and it also called the ResNet-50 case correctly as *equivocal*
(0.445, a moderate gain that did not carry end to end). A large AP gain
carried; a moderate one did not.

---

## Phase 7 — Serving without Python

Not planned; it happened because the question "could this run where a device
vendor's runtime is the only runtime" turned out to be answerable in an evening.
Reasoning and the fidelity result: [`deployment.md`](surfaces/deployment.md).

- [x] **7.1 Export the step cascade to ONNX.** Cut at the seam the decoder
      already has — `front.onnx` for the memory, `decode.onnx` for one position
      — with the auto-regressive loop, the CCI probes and the argmax masking
      left in the host language, because they are control flow over a graph
      rather than a graph. Needs the dynamo exporter: the legacy path bakes the
      trace-time sequence length into `MultiheadAttention` and runs only at T=64.
- [x] **7.2 Verify per second, not approximately.** `--verify` rolls the
      exported graphs through the whole CCI decode against torch. Memory agrees
      to 4.6e-06 and predictions agree **exactly, 4337/4337 seconds on
      video_25**. Float noise is expected; a differing second is not, because
      the output feeds a surface that reads as clinical.
- [x] **7.3 A host-language decoder.** `rust/pitvis-serve`, ort 2.0 over the
      two graphs. The rollout exists twice now — `cci_decode` stays the
      reference — and 7.2 is what keeps the copies honest.
- [ ] **7.4 Serve task 2.** `pitvis-export` already writes and verifies the
      instrument graph; `main.rs` loads the step model and nothing else. The
      exported half is the finished half.
- [ ] **7.5 Take pixels, not features.** The binary reads a float32 feature
      blob, so the backbone is still Python. Exporting the encoder makes it
      genuinely standalone — and for DINOv2 that is the larger half of the
      compute, so it is also where the interesting engineering is.
- [ ] **7.6 (D) Is this worth continuing before the model is better?** 0.561
      served at 200 Hz is still 0.561. 3.6b has now landed and raised that
      number, which makes 7.4/7.5 more attractive — and 7.5 harder, since the
      best encoder is now our own fine-tuned ViT-B rather than a stock timm
      checkpoint. Deciding this belongs with 3.6c.

---

## Cross-cutting risks

- **Frozen ImageNet features are probably the ceiling.** Everything except 3.6
  builds on a backbone that has never seen an endoscope. If temporal modeling
  plateaus well below the paper, 3.6 is the reason.
- **The val split is 5 videos.** Small improvements will not be distinguishable
  from split noise (4.4).
- **Video 19 has no labels** — we train on 19 videos, not the paper's 20, so our
  numbers are not exactly comparable on the training side even though validation
  is untouched.
- **Classes 11 and 13 are essentially unlearnable** (2 videos and 1 video
  respectively) and are excluded from scoring anyway. Do not spend effort there.
