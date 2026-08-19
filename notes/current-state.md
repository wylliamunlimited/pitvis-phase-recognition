# Current state — the system as it stands

*Snapshot: 2026-08-17. Written for the question "can I demo this, and what may
I claim while doing it?"*

Companion to [`where-we-are.md`](where-we-are.md), which is the *research*
orientation — vocabulary, iterations, what to run next. This one describes the
**system**: what is trained, what is wired to what, how a case flows through
the app, and which claims the evidence actually supports.

Results detail lives in the model notes and is linked, not restated.

---

## 1. Can it be demoed? Yes — one case, today

`predictions/video_25/` holds output from the current best models on both
tasks, generated 2026-08-17. `uv run pitvis-app` plays it.

video_25 is a **validation** video: never trained on, by any stage of the
pipeline. So what the demo shows is generalisation, not fit — which is the
harder and more honest thing to show.

```sh
uv run pitvis-app          # then open video_25
```

Three caveats that belong in the narration, not in the small print:

- **Every other case is unpredicted**, and if you generate one in the app it
  will use the *default* models, not the best ones (§5). Demo video_25 only.
- The scores shown are **this video alone**, not the 5-video mean±std that the
  notes and README quote.
- Confidence is **pre-CCI** — it reads low exactly where the consistency
  constraint overrode the frame. That is signal, not error. See
  [`app.md`](surfaces/app.md).

---

## 2. The ML design, end to end

Four stages. Stages 1–2 run once and are cached; 3–4 are minutes.

```
mp4 (1280x720, 24 fps)
  │  decode at 1 fps                                    data/frames/384/
  ▼
frame JPEGs ──► ENCODER ──────────────────────────────► data/features/<space>/
  │             DINOv2 ViT-B/14 @224, fine-tuned            (T, 768) per video
  │             on PitVis frames, 2 epochs
  ▼
features (T, 768)
  │
  ├──► TASK 1  spatial embed ─► TeCNO ─► ARST decoder ──► step per second (15-way)
  │            + inverse-freq class weights
  │            + argmax masking of classes 0/11/13
  │            + CCI auto-regressive rollout (lag 10)
  │
  └──► TASK 2  causal windowed LSTM (W=5, 2x512) ───────► 19 sigmoids, top-2
               + pos_weight BCE
               + per-class thresholds
```

**Why a two-stage pipeline at all.** Embedding 40 GB of video is hours; a
training run over cached features is ~100 s. Every experiment in this repo was
affordable because the expensive half happens once. The cost is that the
encoder cannot learn from the temporal loss — which is exactly the limitation
§3's fine-tuning worked around, and exactly why the fine-tuned encoder trades
per-frame accuracy against segment stability.

**The encoder is the load-bearing choice.** Four feature spaces coexist
(`resnet50`, `dinov2_vitb14`, `resnet50_ft`, `dinov2_ft`), selected by
`--space`. Swapping it moved the step metric further than every architecture
and loss change combined.

### Where the numbers are

| | steps (challenge metric) | instruments (macro / aligned-w) |
|---|---|---|
| published reproductions | 0.3425 | 0.2556 / 0.6234 |
| **current best** | **0.5608** ±0.052 | **0.5333** / **0.8416** |

Both on the 5 validation videos, scored once. Per-class movement, what each
iteration tested, and the full leaderboards:
[`step-variants.md`](models/step-variants.md),
[`instrument-variants.md`](models/instrument-variants.md).

**On instruments, read macro or aligned-weighted, not the official number.**
The vendored metric compares label columns positionally and collapses whenever
a model's predicted class set differs from the truth's — it reports 0.3220 for
a model that beats the previous best on all five videos. The per-video proof is
in `instrument-variants.md`. The official number is still the headline
everywhere it appears, because that is what makes it the challenge's number by
construction.

---

## 3. Is the training sound? — the integrity audit

Every claim below was re-checked against the artifacts on 2026-08-17, not
inferred from the code's intent.

| risk | status | evidence |
|---|---|---|
| encoder saw validation videos | **clean** | `backbone.pt` records `trained_on = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18]`; `VAL ∩ trained_on = ∅` |
| early stopping selected on VAL | **clean** | stopped on 3 videos carved from TRAIN (20, 22, 23), held out **by video**, not by frame |
| standardisation fitted on VAL | **clean** | `arst_v2.py:304` / `instruments_v2.py:317` load TRAIN only; VAL is first touched at the scoring line |
| decision thresholds fitted on VAL | **clean** | `crossfit_thresholds(train, ...)`, 2-fold within TRAIN |
| illegal normalised-time feature | **clean** | no `t/(T-1)` anywhere in `src/` — it needs the total duration, i.e. the end of the video |
| model is strictly online | **fixed-lag, disclosed** | CCI emits frame *t* after observing *t+10*. The challenge permitted this (TSO-NCT smooths over 7). `--no-cci` gives the strictly causal variant, which scores lower |
| checkpoint tags survive a save/load | **clean** | `checkpoints.read_tags` is the one decoder, and falls back to `args["mask_excluded"]` for reproductions written before the tag existed; pinned by `tests/test_checkpoints.py` |
| argmax masking | **legal, and an exploit** | the official metric filters by ground truth only, so predicting an excluded class can only hurt. Masking 0/11/13 was the largest single lever in iteration 2. It is a scoring-rule exploit, not a modelling gain — say so |

**The one real limitation, and it is not leakage.** The fine-tuned encoder was
trained on 16 TRAIN videos, so it cannot be cross-validated over folds drawn
from TRAIN — the encoder has seen most of every fold's held-out set. That makes
**iteration 4b a single VAL measurement rather than a ranking.** VAL itself is
untouched, so the number is honest; what is missing is the error bar that
repeated measurement would give. Closing it costs six fine-tunes
([`infra/README.md`](../infra/README.md)).

This has bitten once already and was caught: an earlier encoder trained on all
19 TRAIN videos produced a CV macro of 0.917, which was the size of the leak
rather than of an improvement. Both entries were deleted and
`crossval.check_no_leak` now refuses the configuration.

### Against the paper

Das et al. Table 8 benchmarks the same five validation videos: **CITI 70** on
steps, and CITI 88 / SDS-HD 89 / SANO 81 on instruments.

**Those numbers are not a target line, and the reason cuts in our favour.**
Those five videos were part of every competing team's *training* data, so their
Table 8 figures measure fit; ours measure generalisation on videos no stage of
our pipeline ever saw. The paper supplies its own evidence for how large that
gap is: it reports a **−47-point val→test collapse for instruments** against
−7 for steps, and SDS-HD scored **89 on validation and 41.7 on test**.

The defensible statement for a demo is therefore *not* "we are 14 points behind
CITI". It is: **the reproduction scored 0.3425 and the current model scores
0.5608 on held-out video, a 64% relative improvement, under stricter conditions
than the published validation figures were measured under.** The private
8-video test set was never released, so no comparison to the actual leaderboard
is possible in either direction.

---

## 4. The app, as it stands

`uv run pitvis-app` — no web framework, no build step, no npm. `http.server`
plus hand-written Range, native ES modules. It adds **zero** dependencies.

### The flow

```
pitvis-app
  └─ catalogue.cases()          25 videos found in 26531686/
       │                        + per-case: cache state, truth present?,
       │                          prediction present? stale?
       ▼
  GET /  ──► index.html + ES modules
  GET /api/cases                the case list
  GET /api/cases/<id>           the CASE DOCUMENT — case.build_case()
       │                        steps, instruments, truth, probabilities,
       │                        segments, scores, colour ramp
       ▼
  GET /api/cases/<id>/video     Range-served mp4  ◄── load-bearing, see below
  GET /api/cases/<id>/frame     single JPEG via ffmpeg
  POST /api/cases/<id>/predict  on-demand inference ─► SSE at /api/jobs/<id>/events
```

**Range is mandatory, not an optimisation.** Every PitVis mp4 has box order
`ftyp, free, mdat, moov` — the seek index is the last ~1.3 MB of a multi-GB
file, so a browser cannot begin playback without a tail range request.
`parse_range` is pinned by `tests/test_app_range.py` against video_25's real
byte offsets.

### What it looks like

Both shots are **video_25 at 46:40**, the model mid *tumour excision*, from the
current best checkpoints. The endoscopic image is blurred in every committed
screenshot — the dataset is CC BY-NC-ND 4.0 and a UI composited around a
surgical frame is arguably a derivative work. The interface text is untouched,
which is what these are here to show.

![the default view](../docs/app-default.png)

**The default view answers "what is happening now".** The step is burned into
the frame corners PACS-fashion — `[07] TUMOUR EXCISION` bottom-left, the
instruments bottom-right — so the answer survives being photographed off a
screen. The worklist on the left is the whole procedure at once: elapsed time
per step, revisit counts (`×5`), and greyed rows for steps that have not
happened. Nothing scrolls.

![the detail view](../docs/app-detail.png)

**`[ + DETAIL ]` reveals the analyst layer** — instrument usage totals, ground
truth with a MATCH/MISS verdict, the four scores, and six timeline lanes
(step, confidence, truth, errors, tools).

Two things in that second image are worth pointing at in a demo, because they
are the app arguing against its own authority:

- **`this video alone — NOT the 5-video mean±std`**, sitting directly under the
  scores rather than in a footer. A caveat belongs with the number it
  qualifies.
- **The POST-PROCESSING panel explaining itself**: *"CCI HOLD — the decoder
  preferred SPHENOID SINUS CLEARANCE at 0.99, but the consistency constraint is
  holding the previous step pending 10 s of agreement. The confidence shown is
  the probability of the step actually displayed, which is why it reads low
  here."* That is the pre-CCI confidence rule made visible at the moment it
  bites, rather than documented somewhere the viewer will not look.

### What the surface actually does

| | |
|---|---|
| **step burn-in** | the current step drawn into the frame corners, PACS-fashion |
| **worklist** | all 14 steps as a procedure checklist — done / current / pending |
| **instrument panel** | what is in view now, and a cumulative usage record |
| **timeline** | one progress strip by default; six lanes behind `[ + DETAIL ]` |
| **floating panels** | each category independently collapsible and draggable, over the video. They never stack in a scrolling column — steps and instruments are two halves of one judgement and must be visible together |
| **`[ + DETAIL ]`** | reveals the analyst layer: confidence, ground truth, agreement, per-class probabilities, official scores |
| **on-demand inference** | POST a case, watch stdout stream back over SSE |

The default view answers *what is happening now*; the detail layer answers *how
well is the model doing*. That split is deliberate — six stacked lanes reads as
a video editor, not clinical software. Reasoning, measured dead-gutter numbers,
and the caveat placement rules: [`app.md`](surfaces/app.md).

### Also built, not part of the app

The step cascade exports to ONNX and runs from a **Rust binary** with no Python
runtime, verified exactly per second — 4337 of 4337 on video_25.
[`deployment.md`](surfaces/deployment.md).

---

## 5. Known gaps — what a demo must not claim

1. ~~**The app's default models are not the best models.**~~ **Fixed.**
   `jobs.py` still builds its argv with no `--steps-model`, but the default no
   longer resolves by name. `checkpoints.default()` now ranks by each
   checkpoint's own recorded `macro_f1` (`result.json` beside the weights), so
   `best@dinov2_ft` (0.6147) wins over `best` (0.4420) instead of losing to it
   on alphabetical order. Macro rather than the official `metric` deliberately:
   task 2's official number carries the column-ordering defect and would rank
   the fine-tuned encoder last. `--list-models` prints the score that decided
   it. Falls back to the old `:best` convention when nothing has been scored.
2. ~~**Staleness tracking ignores task-1 variants.**~~ **Fixed.**
   `_checkpoint_mtime()` hand-listed the two reproductions plus
   `data/instruments/v2/*/model.pt`, and `data/arst/v2/` was simply absent, so
   training a new step model left every prediction reading as current. It now
   derives the list from `checkpoints.available()` — the registry that already
   knows where every family's weights live — so a family added there is covered
   without touching the app. The hand-maintained list was a second copy of
   `FAMILIES`, and the two drifted the moment a family was added.
3. **No cross-validated number for `dinov2_ft`** — §3.
4. **Four instrument classes are emitted but unusable** (roadmap 6.5): bipolar
   forceps is predicted 320 times against 49 true instances. "0 classes never
   predicted" flatters.
5. ~~**`pitvis-eval` has no `--space`.**~~ **Fixed, and it was worse than
   "limited".** It loaded ResNet-50 features unconditionally *and* took its
   standardisation stats from `data/arst/` regardless of which checkpoint was
   named. Cross-width pairs failed loudly on `load_state_dict`, but
   `resnet50`/`resnet50_ft` are both 2048-d and `dinov2_vitb14`/`dinov2_ft`
   both 768-d — so scoring a fine-tuned checkpoint against frozen features
   loaded cleanly and reported a wrong number silently. It now takes the space
   from the checkpoint's tag, the stats from beside the weights, and honours
   the `mask_excluded` and `logit_adjust` tags; `--space` still overrides and
   warns when it disagrees with the tag. Nothing above is affected — every
   number came through `pitvis-train`, which does take `--space`.
6. **Panels overlap the burn-in at the default layout.** Visible in both
   screenshots: the PROCEDURE STEPS panel covers the video's top-left `CASE 25`
   label, and in the detail view INSTRUMENT USE clips the timecode. Panels are
   draggable, so it is a default-position issue rather than a layout failure.

Every item above is now fixed except 3, 4 and 6 — the missing cross-validated
number for `dinov2_ft`, the four unusable instrument classes, and the default
panel positions. None of those is a correctness bug in the pipeline.

Also fixed alongside these: `pitvis-predict --list-models` required `--video`,
so the one command that answers "which model is the default here" could not be
run without naming a file it would never open.

---

## 6. Provenance of the shipped encoder

Worth knowing because the directory name is misleading:

`data/backbone/dinov2-50ep/backbone.pt` is **not** a 50-epoch encoder. The tag
was carried over from the failed first run; the weights inside stopped at
**epoch 2** under early stopping, and the checkpoint records `best_epoch: 2`
and the correct 16 training videos. The `result.json` beside it still carries
the first run's summary and should not be trusted — `backbone.pt` and
`data/features/dinov2_ft/manifest.json` agree with each other and are the
authority.
