# The app — `uv run pitvis-app`

The counterpart to [`citi-baseline.md`](citi-baseline.md) and
[`instruments.md`](instruments.md): what was built, why it is shaped this way,
and the three places where showing the truth took real work.

Everything the repo produced before this was a CSV. `pitvis-predict` writes
4,337 rows of `int_time,int_step` and prints a scoring table. That is enough to
*measure* a model and not nearly enough to *look* at one. This closes roadmap
**5.1** (the surface decision, which was an explicit open question) and **5.3**
(timeline visualisation), plus the cheap half of 5.2.

```sh
uv run pitvis-app                      # opens on the first predicted case
uv run pitvis-app --case video_25
uv run pitvis-app --video path/to/case.mp4
```

---

## 1. What it shows, and what it hides

The default view carries four things: the video, the current step, the
instruments in view, and one strip saying where you are in the operation.

Nothing else. Not confidence, not ground truth, not the score, not the
per-class probabilities. All of that is behind `+ DETAIL`.

That split is the main design decision and it is worth stating plainly. The two
layers answer different questions:

| question | who asks it | where it lives |
|---|---|---|
| what is happening right now? | anyone watching | always visible |
| how well is the model doing? | whoever is evaluating it | `+ DETAIL` |

An interface that answers both at once answers neither well, and the first gate
is not comprehension — it is whether someone opens it twice. Six stacked
timeline channels reads as a video editor, and a video editor is a workspace
you commit to rather than a display you glance at.

The preference persists in `localStorage`: someone who wants the analyst layer
wants it every time.

---

## 2. Why HTTP Range is the load-bearing part

The least glamorous code in `app/server.py` is the reason the thing works at
all. Read the box layout of any PitVis video:

```
$ python3 - <<'EOF'   # 26531686/video_25.mp4
ftyp  size=32              at=0
free  size=8               at=32
mdat  size=1,066,486,506   at=40
moov  size=1,338,182       at=1,066,486,546
EOF
```

`moov` — the index a demuxer needs before it can decode anything — is the
**last 1.3 MB of a 1.07 GB file**. These are not `+faststart`. A browser
therefore cannot begin playback until it has fetched a range from the end, and
a server that ignores `Range` produces a black rectangle that never resolves.

So `parse_range` is a pure function with its own test file, pinned against the
real byte offsets above. The forms that actually arrive, in order:

```
bytes=0-              Chrome's opener  (Safari sends bytes=0-1 first)
bytes=-1338182        the moov tail, suffix form
bytes=533000000-      every seek
```

Open-ended requests are capped at 8 MiB, which is conformant — a server may
return fewer bytes than asked — and is what keeps scrubbing responsive: 8 MiB
is about 34 s of PitVis video, so a playthrough is a sequence of short
responses instead of one gigabyte-long transfer aborted on every seek.

**No web framework.** The one thing starlette would have provided is a
Range-capable file response, and since that behaviour is the single most
load-bearing thing here it gets hand-written and tested regardless. Once the
test exists the framework buys nothing: eight routes, no auth, no forms, no
validation, and the expensive work is a torch rollout on a worker thread, which
async does not help. `pitvis-app` adds **zero** dependencies.

Two consequences of using `http.server` that are worth knowing because their
symptoms are so misleading:

- `BaseHTTPRequestHandler` speaks HTTP/1.0 by default, closing the socket after
  every response. With 100+ range requests per case that is 100+ TCP
  handshakes, and server-sent events never stream at all.
- A `<video>` aborts its outstanding range request on every seek, which arrives
  as `BrokenPipeError` mid-write. Unhandled, the console fills with tracebacks
  and a perfectly healthy app looks like it is crashing.
- **The same teardown arrives on the read side, and a `try/except` around the
  writes cannot catch it.** Keep-alive means the thread finishes a response and
  parks in `handle_one_request` → `rfile.readline()` waiting for the next
  request on that socket. When the video resets the connection instead, the
  exception surfaces *above* every `except` in the module, so socketserver
  catches it and its default `handle_error` prints a full traceback — one per
  reset, and a single seek can produce several. Measured: 8 forced resets give
  8 `ConnectionResetError` tracebacks before the fix and 0 after, with the
  server still answering 200.

  `Server.handle_error` filters `TEARDOWN` and delegates everything else to the
  default. The list is deliberately narrow: all four are `OSError` subclasses,
  so the tempting `except OSError` would also swallow a missing video file and
  every permission fault in the component that has no framework beneath it.
  `tests/test_app_server.py` pins both directions.

And one that is a genuine vulnerability rather than an annoyance: binding to
`127.0.0.1` is **not** sufficient. Without validating the `Host` header, any web
page the user visits can reach the server by DNS rebinding and stream patient
video off loopback. `_host_ok` is the fix.

---

## 3. `(-1, -2)` means two different things

The sharpest data problem in the app, and the one most likely to have shipped
silently wrong.

`instruments.csv` writes an unused secondary slot as `-2` and an absent primary
as `-1`. In the **annotations** that pair means *the scope is out of the
patient*. In a **prediction** it cannot mean that, because SANO's head is 19
sigmoids and has no out-of-patient class at all — `multihot_to_pairs` reuses
the same pair as padding for an all-zero row (`evaluation/instruments.py`, "the
vendored padding rule").

Measured on the two cases that existed when this was written:

| case | rows written `(-1, -2)` | of |
|---|---|---|
| video_19 | 1,167 | 4,456 (**26.2%**) |
| video_25 | 559 | 4,337 (12.9%) |

Rendering those as "out of patient" would tell a viewer the scope had left the
patient for a quarter of an operation. So the collision is resolved once, at
the case-builder boundary, and the wire format never carries the raw pair:

```python
def _instrument_state(slot1, slot2, *, truth):
    if slot1 == -1:
        return "out_of_patient" if truth else "none"
    return "two" if slot2 != -2 else "one"
```

The UI then says `nothing above 0.50 · closest: nasal cutting forceps 0.498`,
which is both true and more useful than a blank. Class 0 — "no visible
instrument / occluded" — is a *third* distinct thing: a real, scored, predicted
class covering 31.5% of frames, and it renders as a named instrument.

*(The three decimals on the runner-up are deliberate. It sits just under the
threshold by definition, and at two decimals a 0.498 prints as "0.50" directly
beside "nothing above 0.50", which reads as a contradiction.)*

---

## 4. Confidence, and the fact that it is pre-CCI

Neither model persisted a probability before this. Both computed the logits and
discarded them one character later:

```python
p = int(step(t, prev).argmax())        # the distribution existed, briefly
```

`cci_decode` and `predict_video` now take a keyword-only `return_probs=False`,
so none of their five existing callers changed, and `pitvis-predict --probs`
writes `step_probs.npy` (T, 15) and `instrument_probs.npy` (T, 19).

The change is purely additive, and that is tested rather than asserted:
`predictions.csv` for video_25 is **byte-identical** before and after, and the
scores are unmoved at 0.3311 / 0.2699.

**The caveat that matters.** The recorded distribution is the decoder's belief
*before* the consistency constraint. CCI can revert a predicted transition, so
on those seconds `probs[t].argmax()` disagrees with the label actually emitted.
Measured: **164 of 4,337 seconds (3.8%)** on video_25, 160 of 4,456 (3.6%) on
video_19.

Two ways to define confidence there, and the choice is the whole point:

- `probs[t].max()` — always looks confident, hides the disagreement.
- `probs[t][emitted]` — reads **low** exactly where CCI is holding a phase the
  current frame does not support.

The second is what ships. A low number there is not noise; it is the constraint
working, and the detail panel names it:

> **CCI HOLD** — the decoder preferred TUMOUR EXCISION at 0.63, but the
> consistency constraint is holding the previous step pending 10 s of
> agreement. Confidence above is the probability of the step actually shown.

---

## 5. Honesty is load-bearing, not decoration

ARST scores **0.331** on the challenge metric and gets **40.5%** of seconds
right on video_25. A composed, clinical-looking surface makes any number on it
read as authority. Four things exist specifically to resist that, and none of
them should be trimmed for cleanliness:

1. **`RESEARCH — NOT FOR CLINICAL USE`** in the header, non-dismissible.
2. **The split chip turns amber on a training video.** video_02 reads 0.891
   frame accuracy against video_25's 0.405 — roughly twice as good, and
   meaningless as a measure of generalisation. Without saying which split a
   case belongs to, the app would flatter the model by a wide margin depending
   on which case you happened to open.
3. **Missing ground truth is stated, never blank.** video_19 has no
   `annotations_19.csv` — a gap in the download, not an exclusion. Its truth
   lane says so in words. An empty lane would read as "all background".
4. **Confidence is always a number**, never a bar alone, and scores are
   labelled *this video alone — NOT the 5-video mean±std* that the paper and
   the README quote.

Task 2 shows both its numbers, labelled: the vendored `metric` (0.270, column
ordering defect included, because that is the challenge's number by
construction) and the name-aligned `weighted` (0.646). They differ by 2.4×;
showing one unlabelled would mislead.

---

## 6. Design

Light ground, nine colours, one accent (`#0F7B6C`), a light sans with tabular
numerals, corner brackets instead of filled borders, one easing and two
durations.

**Type is light sans, not mono.** The first pass set everything in a monospace
family. Mono plus uppercase plus wide tracking is the house style of a tactical
briefing, and it made a clinical research tool read as one — the surface
signalled *operator* when it needed to signal *instrument*. Body is now 300
weight, and the big step numeral is 150.

Mono survives in exactly two places: the inference console and the copyable
command in the veil, where column alignment is the whole point. Everything else
carries `font-variant-numeric: tabular-nums`, so timecodes and probabilities
still hold their columns without the surface looking like a terminal. Tracking
was retuned throughout — values chosen against mono metrics are too loose for
a sans at the same size.

**Phases are one hue, not fifteen.** Every step gets the same slate at a rising
darkness (`hsl(200 20% L)`, L from 79% down to 42%), so a case reads
left-to-right as a deepening and the *shape* of an operation — where the
boundaries fall, how long each stage ran — is legible before a single label is
read. Identity is carried by the number, which every segment renders as text.
Colour carries structure. That is what keeps the palette at two colours instead
of sixteen, and it means nothing depends on hue discrimination.

The **bracket** is the motif: `.brk` draws four corners from eight background
gradients, so anything can be framed with one class and no extra DOM, and the
colour animates — which is how a step change announces itself.

**The bracket marks information, never a control.** It frames what you *read* —
the video, a card, the case picker, `[ VAL SPLIT ]`. Controls originally echoed
it typographically (`[ PLAY ]`, `[ + DETAIL ]`, `[ RE-RUN ]`), which was the
mistake: if the same mark means both "look at this" and "click this", it means
neither. Buttons now carry no literal brackets and are **filled boxes** instead. A
readout is a region you look at; a control is a surface you press. The first
attempt at separating them used a hairline underline, which read as a
hyperlink — technically distinct from the bracket, but still not something that
looks pressable. Area is what makes a control legible as one.

State is carried entirely by colour, on one property, with no movement and no
shadow: `--raised` at rest, a 12% accent tint under the pointer, 22% while
held, and **solid accent with white text while a toggle is on** — the only
place in the app a control reads as filled. 2px of radius and no more: enough
to separate a control from the square-cornered readouts around it, not enough
to break the orthogonal language.

The case picker follows the same rule. It *is* a control, so it lost its
bracket and became a box too — leaving it framed while the buttons beside it
were filled would have been the same confusion in reverse.

**Reveal is animated, because `display` is not.** `.more` elements go
`display: none` → `revert`, and no transition can span that. So the analyst
layer runs a `reveal` keyframe instead — a 5px rise and a fade over `--d2` —
staggered 30/80/130 ms down the rail so it assembles top-down rather than
snapping in as one block. The footer grows at the same time: `--tl` is set from
JS, and `#app` transitions `grid-template-rows`, so lanes and cards arrive
together instead of the timeline jumping ahead of them.

All of it collapses under `prefers-reduced-motion: reduce`. Motion here is
affordance rather than decoration, so it reduces to nothing rather than merely
being shortened.

That motif is also the seam. Roadmap 5.4 is an agent that circles a region of
the frame and captions it, and "circle a region" is the same bracket, moved and
resized. `overlay.js` already owns a canvas over the video sized to its
displayed rect, with a `Layer` registry and video-pixel coordinates. Nothing
registers a layer yet.

---

## 7. Where the rest plugs in

| roadmap | seam that exists today |
|---|---|
| 5.4 agentic overlay | `CanvasHost` + `Layer` on the video; `/api/cases/{id}/frame?t=` ships; `POST /explain` is routed and 501s |
| 5.7 correction | `doc.corrections` is present and empty; `segments[].source` is `"model"`; `/corrections` routed |
| 5.8 live input | the clock is a `TimeSource` interface — `VideoTimeSource` wraps `<video>`, a stream implementation swaps in and no renderer changes |
| 5.9 comparison | case documents are self-contained, so comparison is N fetches; `renderTimeline` is a pure function of its arguments, which is what makes that cheap |

The purity rule on `renderTimeline(ctx, doc, geom, opts)` is not style. Passing
different segments is how a corrected timeline renders, and calling it N times
is how comparison works. A renderer that reached for module state would need
rewriting for either.

---

## 8. Things that will bite

- **Static assets under `src/pitvis/` are a first for this repo.** An editable
  install hides a packaging failure completely — the same shape as the
  `.gitignore` bug in `5473f86`, which worked locally and broke on every clone.
  `paths.APP_ASSETS` is anchored on `PACKAGE`, not `ROOT` (which walks out of
  the package and does not exist in a wheel), and `tests/test_app_case.py`
  resolves `index.html` through `importlib.resources` so a regression fails
  from inside the installed package.
- **`inference/run.py` used to compute `Path("predictions")`** — CWD-relative,
  so running it from a subdirectory wrote somewhere nothing else looked. Now
  `paths.PREDICTIONS`.
- **`report()` prints and `evaluate()` returns numpy.** The app calls
  `evaluate_video` per case: `mean` at n=1 is not a mean, and `pooled` holds
  ndarrays that `json.dumps` raises on.
- **MPS is not bit-deterministic**, so a re-run may not reproduce a stored
  prediction. Every case carries `computed_at` and its checkpoint.
- **One inference worker, permanently.** `redirect_stdout` swaps a
  process-global `sys.stdout`, so two jobs would interleave their logs, and two
  torch rollouts would contend for one MPS device.
