# Reproducing SANO — PitVis task 2 (instrument recognition)

Task 1 is step recognition (`citi-baseline.md`). This is task 2: which
instruments are visible, per second. Different problem, different winner,
different metric — and a metric with a defect worth knowing about.

Code: `src/pitvis/models/lstm.py` (architecture),
`src/pitvis/training/instruments.py` (training),
`src/pitvis/evaluation/instruments.py` (metric).

---

## 1. Who won task 2, and why we built SANO rather than the rank-1 model

Das et al. 2024, Table 6 — "19-instruments multi-label online recognition
(task-2) rankings … across the 8-testing-videos (mean±std)":

| Rank | Team | Macro-F1 |
|---|---|---|
| 1 | SDS-HD | 41.7±15.4 |
| **2** | **SANO** | **41.6±06.3** |
| 3 | CITI | 35.1±18.5 |
| 4 | SK | 34.0±17.0 |
| 5 | GMAI | 27.8±08.7 |
| 6 | UNI-ANDES-23 | 27.5±13.5 |

**CITI — who won task 1 — placed third here.** The paper notes the reversal
directly (§6.4): "CITI's model is identical to its previous task models … the
strong step recognition (1st) compensates for the poorer instrument recognition
(3rd)." And §6.3: the simple CNN+LSTM models "are able to outperform … more
sophisticated models that utilise temporal decoders; positional encoding; and
multi-task training (CITI and UNI-ANDES-23)."

So ARST is not the answer for instruments. CITI's task-2 output is a sigmoid
head on their Swin ST-encoder; ARST is tagged ⟨step⟩ only.

**We built SANO, not SDS-HD.** £500 went to "joint 1st (1st & 2nd)" (§6.3), so
both are challenge-designated winners, and the 0.1-point gap is far inside
either error bar. The deciding factors:

| | SDS-HD | SANO |
|---|---|---|
| backbone | ResNet152 + EfficientNetB7 + SwinL | **ResNet50** |
| temporal | 3 × LSTM (15/15/12 window) | 5-window LSTM |
| fusion | "balanced ensemble" — **rule unspecified** | none |
| std | ±15.4 | **±06.3** |

SANO's backbone *is* our feature cache. SDS-HD would need three full
re-extractions of 40 GB, and its ensemble weighting is never stated in the
paper, so any reproduction would be a guess wearing its name. SANO is also less
than half as variable across test videos.

SANO, §5.5 verbatim: *"For task-2 their model consisted of 2-stages: the trained
CNN was frozen; followed by a 5-window LSTM for both instrument (task-2) and
step (just for training) classification."*

---

## 2. Task 2 is not task 1

| | task 1 (steps) | task 2 (instruments) |
|---|---|---|
| problem | multi-class, exactly 1 | multi-label, **0, 1 or 2** |
| head | softmax + CE | **sigmoid + BCE** |
| averaging | `average="macro"` | **`average="weighted"`** |
| background rows | excluded | **kept**, as all-zero rows |
| edit score | yes | **none** — F1 only |
| rarity exclusions | 11 and 13 dropped | **none** |

Two of these are easy to get wrong by assuming task 1's conventions carry over:

**Class 0 is a scored class, not a sentinel.** After the official encoder pops
the `-1` and `-2` columns, ids 0..18 remain — 19 classes. Id 0 is "no visible
instrument", 31.5% of frames. Under support-weighted averaging, ids 16, 0, 8 and
13 carry ~91% of positives.

**Out-of-patient frames survive.** `remove_background_insts` exists upstream but
its call is commented out, so those rows become all-zero targets and are scored.
The steps metric does the opposite.

The paper is internally inconsistent on the class count — Tables 6 and 7 say
19, Figure 4 says 18. The code settles it at 19.

---

## 3. The metric has a defect, and it is load-bearing

`hot_encode_insts` fits a **separate** `MultiLabelBinarizer` on truths and on
predictions, then appends whichever columns are missing:

```python
df_trues_encoded = pd.DataFrame(mlb.fit_transform(df_trues), columns=mlb.classes_, ...)
df_preds_encoded = pd.DataFrame(mlb.fit_transform(df_preds), columns=mlb.classes_, ...)
for int_inst in ls_range:
    if int_inst not in df_trues_encoded.columns.to_list():
        df_trues_encoded[int_inst] = [0] * len(df_trues_encoded)
```

When truths and predictions observe different class sets, the two column orders
diverge — and `f1_score` on DataFrames compares **positionally**. Measured on a
three-frame example:

```
trues column order: [0, 8, 13, 16, 1, 2, 3, 4] ...
preds column order: [0, 16, 1, 2, 3, 4, 5, 6] ...
aligned by name : 0.600000
positional      : 0.333333   <- what the vendored function returns
```

**It fires on 5/5 of our validation videos.** It will fire on essentially any
real prediction, because a model rarely emits exactly the set of classes present
in the truth.

How badly it distorts things: through the official path, three genuinely
different constant strategies score *identically*.

| constant strategy | official | aligned-weighted |
|---|---|---|
| always suction (16) | 0.1383 | 0.1994 |
| always nothing-visible (0) | 0.1383 | 0.1383 |
| always {0, 16} | 0.1383 | **0.3376** |

A metric that cannot distinguish "always suction" from "always suction and
nothing-visible" is not measuring what it claims to.

**What we do about it.** The same thing `metric.py` does with the task-1 quirks:
preserve the vendored behaviour as the headline, because the point of vendoring
is that our number is the challenge's number by construction — but never let it
be silent. `evaluate_video` reports `column_order_diverged` per video and
`report()` prints a warning. Alongside the official figure we print the same
metric with columns aligned by name.

### And the paper disagrees with its own code

§3.4.3 and Table 6's column header both say **macro**-F1. The shipped script
computes **weighted**. Nothing in either source reveals which produced the
published 41.7 / 41.6.

This is the same shape of conflict as the Eq-3 edit-score bug in `CLAUDE.md`,
and it gets the same resolution — **the code is authoritative** — but with one
difference: for task 1 the reported values ruled out the paper's formula, and
here nothing rules out either. So all three numbers are printed, labelled, and
**no claim of leaderboard comparability is made**.

---

## 4. What we implemented

```
(T, 2048) frozen ResNet-50 cache      standardised on the train split
  -> causal window of 5 frames        [t-4 .. t], left-padded
  -> LSTM, 2 layers, unidirectional
  -> Linear(512, 19) -> sigmoid       instruments, BCE
  -> Linear(512, 15)                  steps, CE, auxiliary (training only)
```

**7,365,666 parameters.** Adam at 2e-4, 10 epochs, batch 1024 — the values in
Table 3.

**Unidirectional is not a tuning choice.** The challenge permits only online
models (§3.2), so a bidirectional LSTM would invalidate every number. Verified
empirically rather than by inspection: perturbing frames `t ≥ 25` leaves every
prediction for `t < 25` bit-identical.

Windows are sampled as a flat shuffled index over `(video, t)` rather than
looped per video. Each position is independent given its own window, so this is
honest minibatch SGD — and it avoids materialising 84,666 × 5 × 2048 floats
(~3.5 GB) at once.

The auxiliary step head is faithful to "step (just for training)". It is trained
and discarded; `--no-aux-step` ablates it.

### Faithfulness

| Component | Published | Ours | Faithful? |
|---|---|---|---|
| backbone | ResNet50, fine-tuned then frozen | ResNet50, **never fine-tuned** | ✗ same limitation as task 1 |
| temporal | 5-window LSTM | same | ✓ |
| head | sigmoid | same | ✓ |
| loss | BCE (+ CE for the step head) | same | ✓ |
| multitask training | yes | yes (`--no-aux-step` to ablate) | ✓ |
| optimiser | Adam 2e-4 | same | ✓ |
| epochs | 10 | 10 | ✓ |
| batch size | 64 | 1024 | ~ ours are windows, not images |
| resizing / augmentation | 384², rotation, reflection, colour | **none** | ✗ we cannot — no pixels |
| data balancing | instrument upsampling | **none** | ✗ see below |

The two ✗ rows are the same root cause as task 1: `extract_features.py` saves
embeddings and discards pixels, so there is no augmentation surface and no way
to fine-tune. SANO's instrument upsampling is worth revisiting — it is
implementable on cached features (resample the window index) and the paper calls
class imbalance the dominant task-2 difficulty.

**Decision threshold.** The paper does not state SANO's. Only UNI-ANDES-23
documents one (0.4) and they placed last. Ours is `--threshold`, default 0.5,
plus a hard cap at two instruments — the label is a pair of columns, so three is
structurally impossible.

---

## 5. Results

Val split, seed 0, defaults unless stated.

| config | official | aligned-w | macro |
|---|---|---|---|
| always suction (constant) | 0.1383 | 0.1994 | 0.1132 |
| always {0, 16} (constant) | 0.1383 | 0.3376 | 0.1378 |
| SANO, 1 epoch | 0.0728 | 0.2613 | 0.1254 |
| **SANO, 10 epochs (faithful)** | **0.2336 ± 0.0381** | **0.6309 ± 0.0401** | 0.2513 ± 0.0569 |

Training takes **81 s** on MPS — the frozen cache again buying a cheap
experiment. The model clears every constant baseline on both the official
(0.2336 vs 0.1383) and the aligned (0.6309 vs 0.3376) reading, so it is
genuinely learning rather than exploiting the metric.

### It learned four classes and gave up on the rest

Per-instrument F1, pooled across the val videos:

| id | name | support | predicted | F1 |
|---|---|---|---|---|
| 16 | suction | 11,971 | 13,971 | 0.779 |
| 0 | no visible instrument | 9,275 | 9,196 | 0.775 |
| 8 | kerrisons | 3,567 | 2,002 | 0.551 |
| 13 | ring curette | 4,314 | 1,741 | 0.504 |
| 11 | pituitary rongeurs | 909 | 117 | 0.144 |
| 5 | freer elevator | 226 | 54 | 0.229 |
| 1, 4, 6, 7, 12, 14, 17 | seven others | 49–412 | **0** | **0.000** |

**Nine of the nineteen classes are never predicted at all.** The four the model
does learn are exactly the four carrying ~91% of positives.

This is not a surprise — it is the failure the paper names, reproduced. §6.3:
"Instruments are frequently misclassified as instrument-0 (no instrument) and
instrument-16 (suction). This is to be expected as they are the dominant
classes, **suggesting one way to overcome these incorrect predictions is through
data balancing.**"

Data balancing is precisely what we left out. SANO upsampled five instrument
classes; SDS-HD upsampled `{07, 10, 11, 12, 15}` and downsampled the rest. Our
faithfulness table marks it ✗, and the per-class column is the cost of that.

Note also what weighted averaging does here: a model that predicts nine classes
at F1 0.000 still scores 0.63 aligned-weighted, because those nine hold under 9%
of positives between them. The macro reading (0.2513) is the honest one for
per-class competence — which is an argument that the paper's *stated* metric was
the better choice, even though its code computes something else.

### These numbers are not comparable to Table 6

Two independent reasons, and both are large:

1. **Different videos.** The leaderboard scores 8 private test videos that were
   never released. We score the 5 suggested validation videos, which were part
   of every team's training data.
2. **The paper measures a −47% val→test collapse for instruments** (§6.5),
   against −7% for steps: "This is likely due to overfitting to the small number
   of images of each minor instrument class." SDS-HD scored **89 on validation
   and 41.7 on test.** Table 8 gives the validation-split benchmark as CITI 88,
   SDS-HD 89, SANO 81, UNI-ANDES-23 79.

So a strong validation number here would mean very little. Treat Table 6 as a
direction, not a target.

---

## 6. The open thread

The frozen backbone is the same ceiling as task 1, and instruments should be
*more* sensitive to it than steps: distinguishing eighteen surgical tools is a
fine-grained visual problem, and the paper says two of them (micro doppler and
tissue glue applicator) "look identical from a static image, and can only be
distinguished by the action performed and the wider surgical context."

Three things worth trying, cheapest first:

1. **Instrument upsampling**, as SANO did — implementable on cached features by
   reweighting the window index, no re-extraction.
2. **A longer window.** SANO used 5 frames; SDS-HD used 12-15. At 1 fps that is
   still only seconds of context, and steps needed ~17 minutes.
3. **A stronger backbone**, the shared answer to both tasks' ceilings.

The measurement that would settle the first question: does the aligned-weighted
score move when rare instruments are upsampled, or is it pinned by the features?
