# The annotation data dictionary

Every column of `annotations_{n}.csv`, what each integer means, and what the
real distribution looks like. Reference layer — read it when you need to know
what a value *is*; `CLAUDE.md` records what we decided to *do* about it, and
`walkthrough.md` explains the surgery.

All numbers here are read off the 24 annotation files, not copied from the
challenge paper.

```
24 files, 115,586 rows, videos 01-25 except 19 (annotations_19.csv does not exist)
```

---

## 1. The five columns

```
int_video,int_time,int_step,int_instrument1,int_instrument2
```

All five are **int64, no nulls, in every file** — verified across all 115,586
rows. There is no header variation and no missing-value sentinel like `NaN`;
every "missing" state is encoded as a negative integer instead (§4).

| column | range | distinct | meaning |
|---|---|---|---|
| `int_video` | 1–25 | 24 | video number; constant within a file, matches the filename |
| `int_time` | 0–8645 | — | elapsed **seconds**, contiguous `0..N-1`, no gaps or duplicates |
| `int_step` | `-1`, `1..14` | 15 | the surgical step — exactly one per second |
| `int_instrument1` | `-1`, `0..18` | 20 | primary instrument slot |
| `int_instrument2` | `-2`, `0`, `5..17` | 8 | secondary instrument slot |

Two things that look like typos but aren't:

- **There is no step 0.** The step ids run `1..14` with `-1` for background.
  Our 15-way training encoding maps `-1 -> 0`, which is why class 0 means
  background in the *code* and `-1` means background in the *data*. Predictions
  written for the challenge must be decoded back to `-1` (`metric.decode`).
- **`int_instrument2` has a narrower domain than `int_instrument1`** — 8
  distinct values against 20. That is not a data error; see §3.

---

## 2. `int_step` — the surgical step

One integer per second. The label is the step *being performed*, not the step
the surgeon is transitioning toward.

| id | name | rows | % | videos | segments | median | longest |
|---|---|---|---|---|---|---|---|
| **-1** | background | 10,476 | 9.06 | 24 | 634 | 7 s | 321 s |
| 1 | nasal corridor creation | 2,837 | 2.45 | 24 | 45 | 47 s | 220 s |
| 2 | anterior sphenoidotomy | 10,767 | 9.32 | 24 | 95 | 56 s | 701 s |
| 3 | septum displacement | 1,337 | 1.16 | 24 | 51 | 22 s | 66 s |
| 4 | sphenoid sinus clearance | 17,690 | 15.30 | 24 | 197 | 53 s | 447 s |
| 5 | sellotomy | 16,398 | 14.19 | 24 | 117 | 57 s | 820 s |
| 6 | durotomy | 6,178 | 5.34 | 24 | 51 | 106 s | 356 s |
| 7 | tumour excision | 27,595 | 23.87 | 24 | 130 | 85 s | **1,570 s** |
| 8 | haemostasis | 13,717 | 11.87 | 24 | 192 | 39 s | 716 s |
| 9 | synthetic graft placement | 3,536 | 3.06 | **18** | 42 | 53 s | 273 s |
| 10 | fat graft placement | 2,248 | 1.94 | **22** | 58 | 24 s | 295 s |
| 11 | gasket seal construct | 841 | 0.73 | **2** | 9 | 41 s | 271 s |
| 12 | dural sealant | 886 | 0.77 | **23** | 44 | 12 s | 205 s |
| 13 | nasal packing | 72 | 0.06 | **1** | **1** | 72 s | 72 s |
| 14 | debris clearance | 1,008 | 0.87 | **18** | 44 | 17 s | 80 s |

### Reading the table

**The imbalance is severe and it is a video-count problem, not just a row
count.** Step 13 is not merely rare at 0.06% — it is a *single 72-second
segment in a single video*. Step 11 appears in 2 videos. Those two are exactly
the classes the challenge metric excludes (along with background), and that
exclusion is a **rarity** decision, not an index offset:

```python
EXCLUDED_RAW = [-1, 11, 13]   # scored classes are the other 12
```

**Background is interstitial, not just top-and-tail.** 634 segments with a
median of 7 seconds — the scope leaving the patient between steps, over and
over. Only a handful are the long runs at the start and end of a case.

**Segment length spans two orders of magnitude.** Median 12 s (dural sealant)
to 106 s (durotomy); the longest single segment is 26 minutes of tumour
excision. That matters for temporal models: TeCNO's cascaded receptive field is
1,021 frames ≈ 17 minutes, so the longest step segments *exceed* what the model
can see at once (`citi-dataflow.md` §5).

### `map_steps.csv` is not uniquely keyed

`-1` maps to **three** names:

```
-1,operation_ended
-1,operation_not_started
-1,out_of_patient
```

The annotations collapse all three into `-1`, so **the distinction is
unrecoverable** — you cannot tell "before the operation started" from "scope
withdrawn mid-case". Treat `-1` as one background class. Loading the map into a
plain `dict` silently keeps only the last name.

Also: **step 1's name has a trailing space** (`"nasal corridor creation "`).
Always `.strip()`.

---

## 3. `int_instrument1` / `int_instrument2` — the instruments

The instrument label is a **pair of columns, not a list**. Two is therefore the
hard maximum, structurally — there is no third column. The organisers' scoring
code enforces it: `evaluation_instruments.py` asserts every ground-truth frame
is exactly length 2 and rejects a prediction longer than 2.

| id | name | frames | % |
|---|---|---|---|
| 16 | suction | 41,603 | 35.99 |
| 8 | kerrisons | 15,381 | 13.31 |
| 13 | ring curette | 12,067 | 10.44 |
| 11 | pituitary rongeurs | 2,745 | 2.37 |
| 10 | nasal cutting forceps | 2,067 | 1.79 |
| 15 | stealth pointer | 1,911 | 1.65 |
| 3 | cup forceps | 1,876 | 1.62 |
| 14 | spatula dissector | 1,705 | 1.48 |
| 5 | freer elevator | 1,148 | 0.99 |
| 9 | micro doppler | 930 | 0.80 |
| 7 | irrigation syringe | 879 | 0.76 |
| 2 | cottle | 792 | 0.69 |
| 12 | retractable knife | 628 | 0.54 |
| 6 | haemostatic foam | 522 | 0.45 |
| 17 | surgical drill | 484 | 0.42 |
| 4 | dural scissors | 471 | 0.41 |
| 18 | tissue glue | 345 | 0.30 |
| 1 | bipolar forceps | 233 | 0.20 |

### How many instruments are actually visible

| state | rows | % |
|---|---|---|
| out of patient (`i1 = -1`) | 10,476 | 9.06 |
| in patient, nothing visible (`i1 = 0`) | 36,363 | 31.46 |
| exactly one instrument | 51,719 | 44.75 |
| **exactly two instruments** | **17,026** | **14.73** |

Two instruments is the minority case — under 15% of the operation. Nearly a
third of all annotated time has the scope inside the patient with **no visible
instrument at all**.

### The slots are not symmetric

This is the non-obvious part, and it changes how you'd model the pair.
`int_instrument2` only ever takes **six** real values:

```
5 freer_elevator · 11 pituitary_rongeurs · 13 ring_curette
14 spatula_dissector · 16 suction · 17 surgical_drill
```

Twelve of the eighteen instruments **never** appear in slot 2. And of the
17,026 two-instrument rows, **16,796 (98.6%) have suction as the secondary**:

| pair | rows |
|---|---|
| ring curette + suction | 10,392 |
| kerrisons + suction | 2,360 |
| spatula dissector + suction | 874 |
| cup forceps + suction | 811 |
| irrigation syringe + suction | 682 |

So in practice the pair is not "two co-equal instruments" — it is **"a working
instrument, plus suction in the other hand"**. Suction also appears alone in
slot 1 for 24,795 frames. Any model treating the two columns as an unordered
set is technically correct (the official metric multi-hot encodes them) but is
throwing away a strong regularity.

### The pair is sorted ascending — with two genuine anomalies

Among rows where slot 2 holds a real instrument (`> 0`), **zero** violate
ascending order. Four rows have `int_instrument2 == 0`, which is itself the
anomaly — `0` means "nothing visible", and the unused-slot sentinel is `-2`:

| video | time | step | i1 | i2 | why it's odd |
|---|---|---|---|---|---|
| 1 | 1412 | 4 | 9 | 0 | micro doppler visible, but slot 2 says "nothing visible" |
| 12 | 2759 | 6 | 0 | 0 | double zero — slot 2 should be `-2` |
| 12 | 2760 | 6 | 0 | 0 | same |
| 25 | 1057 | 8 | 6 | 0 | haemostatic foam visible, slot 2 says "nothing visible" |

Four rows out of 115,586. Harmless in practice, but it means `i1 > i2` is not a
safe ordering assertion unless you first exclude `i2 <= 0`.

### `map_instruments.csv` is not uniquely keyed either

`0` maps to **two** names:

```
0,no_visible_instrument
0,occluded_image_inside_patient
```

"I can see there is no instrument" and "the view is obscured so I cannot tell"
are collapsed into the same integer. Like the step `-1` collapse, this is
unrecoverable.

---

## 4. The three kinds of nothing

The most common source of parsing bugs. Negative values are sentinels, and the
two slots use **different** ones:

| value | in `int_step` | in `int_instrument1` | in `int_instrument2` |
|---|---|---|---|
| `-2` | never | never | **no secondary instrument** (85.3% of rows) |
| `-1` | **background** | **out of patient** | never occurs (0 rows) |
| `0` | never (no step 0) | **in patient, none visible / occluded** | 4 anomalous rows only |

Read as prose: `-1` means "the scope is outside the patient", `0` means "inside
the patient but nothing visible", and `-2` means "this column is unused". Three
different kinds of nothing, and conflating them will quietly corrupt any
instrument model.

### Background is one consistent state

`int_step == -1` and `int_instrument1 == -1` coincide **exactly** — all 10,476
rows, zero disagreement in either direction. When the scope is out of the
patient, there is no step and no instrument; the two labels never disagree
about it. That is a useful invariant: `inventory.py` asserts it per video, so a
future re-download that broke it would fail loudly.

---

## 5. Alignment with the video

Annotation rows are always exactly **one more** than the extractable 1 fps
frames:

```
ann_rows == ceil(nb_frames / round(fps)) + 1
```

The extra row is the final second, for which no frame exists. Every video ends
in a background run of 6–147 seconds, so the dropped row is verified `-1` in
all 24 videos. Extraction truncates labels to the frame count and asserts both
facts (`data/extract_features.py:179`).

That truncation is why the labeled corpus is **115,562** frames rather than the
115,586 annotation rows here — exactly 24 rows, one per video.

Also note `int_time` is in **seconds, not frames**. The videos run at 24 fps
(except video 24 at 25 fps), so row `t` corresponds to raw frame `t * fps`.

---

## 6. What we currently use

Only `int_step`. `extract_features.py` reads that one column and writes
`labels.npy`; the instrument columns are never loaded.

That is a deliberate scoping decision — instruments are task 2 of the challenge
— but it leaves real signal on the table. Instruments are strongly tied to
steps (ring curette + suction is overwhelmingly tumour excision), they are
already annotated per second, and they cost nothing extra to load. If the
per-frame ceiling is set by the features rather than the classifier — which the
macro-F1 evidence in `citi-baseline.md` §6 suggests — an auxiliary instrument
head is one of the few ways to add supervision without new data.

The open question that would decide it: does predicting instruments as an
auxiliary task improve step recognition, or does it just spend capacity? Cheap
to test — the labels are sitting in the same CSV.
