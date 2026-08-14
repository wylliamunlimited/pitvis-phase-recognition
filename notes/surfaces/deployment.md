# Serving the model without Python

The component-layer note for `pitvis-export` and `rust/pitvis-serve`, sibling to
[`citi-baseline.md`](../models/citi-baseline.md), [`instruments.md`](../models/instruments.md) and
[`app.md`](app.md): what was built, why it is shaped this way, and what it does
**not** yet cover.

Everything else in this repo assumes a Python process with torch in it. That is
right for research and wrong for anything embedded near an operating theatre,
where the runtime is whatever the device vendor ships. This is the path out.

```sh
uv run pitvis-export --steps-model arst-v2:best --verify 25
cargo run --release --manifest-path rust/pitvis-serve/Cargo.toml -- data/onnx feats.bin
```

---

## 1. Why the graph is cut where it is

ARST is not a forward pass. It decodes **auto-regressively over its own past
labels**, and the consistency constraint re-runs the decoder on *speculative*
inputs before it will accept a transition. Neither of those is a graph — they
are control flow over a graph.

So the export does not try to trace the rollout. It cuts at the seam the
decoder already has, and leaves the loop in the host language:

```
front.onnx    (T, D) standardised features -> (1, T, 512) memory
              spatial -> TeCNO -> ARST.encode.  Once per video.
decode.onnx   (1, L, 512) memory window + (1, L, 512) decoder input
              -> (15,) logits at the last position.  L <= width + 1.
```

The loop, the CCI probes and the argmax masking are written twice — once in
`training/arst.py:cci_decode`, which stays the reference, and once in
`rust/pitvis-serve/src/steps.rs`. That duplication is deliberate and it is what
§3 exists to police.

**The decoder input is built outside the graph.** `ARST.decode` adds a
sinusoidal positional slice to a phase embedding, and both tables are *fixed
buffers*, not learned. Passing the window in directly keeps a dynamic slice out
of the graph, and a table lookup plus an add is trivial in any host language.
Both tables ship in `tables.npz`.

Two export mechanics are load-bearing and owned by
`src/pitvis/inference/export.py`'s module docstring rather than restated here:
why it needs the **dynamo** exporter (the legacy path bakes the trace-time
sequence length into `MultiheadAttention` and runs only at T=64), and the exact
tensor contract of each graph. Read that before changing either.

---

## 2. The bar is exact agreement, not close agreement

`--verify` rolls the exported graphs through the **whole** CCI decode and
compares against the torch path second by second. On video_25:

| | |
|---|---|
| memory tensors | agree to **4.6e-06** |
| predictions | agree **exactly — 4337 of 4337 seconds** |

Float noise in the activations is expected and fine. A single differing second
is not, and the reason is specific to this project rather than general
fastidiousness: a step prediction feeds a surface that reads as clinical. An
interface that says `[08] HAEMOSTASIS` in 15px type over a surgical frame has
already spent whatever credibility a "roughly equivalent port" would cost.

That bar is also what makes the duplicated decode loop safe to have. Two
implementations of a stateful rollout will drift; `--verify` is the thing that
notices.

---

## 3. What it does not do yet

**The Rust binary serves steps only.** `pitvis-export` exports both heads —
`--instruments-model` is a real flag and the instrument graph is written and
verified — but `rust/pitvis-serve/src/main.rs` loads `StepModel` and nothing
else. Task 2 is exported and unserved.

**It starts from features, not from pixels.** `main.rs` reads a `.bin` of raw
little-endian float32 features. The backbone is not in the bundle, so nothing
here decodes a video or embeds a frame — that is still `pitvis-extract`'s job in
Python. A genuinely standalone binary needs the encoder exported too, and for
DINOv2 that is the larger half of the compute.

**There is no server.** The name is aspirational: it is a CLI that prints one
step per line to stdout. No HTTP, no streaming, no batching.

None of these are defects — they are the boundary of what was verified. The
piece that was hard was proving the rollout survives the port, and that is done.

---

## 4. Where the artifacts live

```
data/onnx/
  front.onnx      the encoder half — spatial, TeCNO, ARST.encode
  decode.onnx     one decoder position
  tables.npz      the fixed positional and phase-embedding tables
  meta.json       feature_dim, width, class count, source checkpoint
  bin/*.bin       float32 C-order sidecars; shapes recorded alongside
```

`data/` is gitignored, so the bundle is a build artifact and not a committed
one. Regenerating it is seconds given a checkpoint and a cached video.

---

## 5. Roadmap

Tracked as **Phase 7** in [`roadmap.md`](../roadmap.md). The open ends are the two
in §3 — serving task 2, and exporting the encoder so the input is pixels rather
than a feature blob — plus the question of whether any of it is worth doing
before the model itself is better than 0.461.
