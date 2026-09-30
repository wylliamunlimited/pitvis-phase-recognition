# Why the backbone was tried last

A grounded explanation of one methodological decision: the encoder was the
obvious first thing to fix, every diagnosis pointed at it, and on both tasks it
was changed last — the frozen swap only after the loss and the decision rule,
the fine-tune after that.

This note owns the *reasoning*, not the results. Every number in it belongs to
another file and is linked rather than restated — see the ownership table in
`CLAUDE.md`. What is original here is the argument for the ordering, and one
calculation nobody else in the repo does.

Companion to [`embeddings.md`](embeddings.md): that one explains what the
features *are*, this one explains why we stopped trying to improve them until
three other things were fixed.

---

## 1. The question is not "what is cheapest"

It is **what a null result licenses you to conclude** — and that depends
entirely on what else is broken when you run the experiment.

The frozen-backbone swap was the obvious opening move, and
[`step-variants.md`](models/step-variants.md) §1 says so in its own diagnosis:
two reproductions, both roughly half the published score, sharing *exactly one*
deviation from their published counterparts — a frozen ImageNet backbone. Every
finger points at the encoder.

Run that experiment first and it comes back null. DINOv2 alone lands **inside
the fold spread on both tasks** — the two deltas and the two spreads are in
[`step-variants.md`](models/step-variants.md) §4 and
[`instrument-variants.md`](models/instrument-variants.md) §4. A careful reader
calls it noise, closes the hypothesis and moves on.

**It was not noise.** Composed with the loss and decision-rule fixes, the same
backbone is the best variant on every metric on both tasks.
[`roadmap.md`](roadmap.md) owns the finding and states the consequence plainly:
testing the backbone first and stopping at the null *"would have retired a true
hypothesis, twice."*

---

## 2. Why the null was a lie — mechanism, not statistics

The task-2 control never emits **nine of its nineteen classes at all** — on
VAL, per [`instrument-variants.md`](models/instrument-variants.md) §1 and the
§4 VAL table. (Read the split before quoting any of these: the out-of-fold
leaderboard in §4 reports a different count for the same variant because it
averages over folds, and §6 counts only the classes the probe examines. Three
numbers, three measurements, one easy conflation.)

```
   frame ──► encoder ──► features ──► loss + threshold ──► prediction
                             │                   │
               the signal is HERE                └── and is DISCARDED here
```

A representation can only pay off through a decision rule that lets it reach
the output. While a class is never predicted, its features are irrelevant: you
can improve them arbitrarily and the metric will not move by one part in a
thousand. **The imbalance defect masks the representation gain**, so the two
are not independent variables you may vary in either order.

That is the whole argument. The ordering is not a preference about effort; it
is that one of these levers is *gated* by the other, and a gated lever's null
result says nothing about the lever.

---

## 3. The instrument that can see a gated quantity

The above would be a nice story and nothing more, except that average precision
can measure the features directly, past the decision rule. AP is computed from
the *ranking*, so it is independent of both the threshold and the class's base
rate — it reports what is in the representation regardless of what the loss
does with it. `uv run pitvis-probe` fits a balanced one-vs-rest logistic
regression per class on frozen features, on TRAIN, and scores VAL.

The per-class table is owned by
[`instrument-variants.md`](models/instrument-variants.md) §6, and its finding is
the one worth carrying: **rarity does not predict difficulty.** One of the
rarest instruments in the set is nearly separable while a far commoner one is
invisible. What predicts difficulty is whether the encoder can see the
instrument — and for six of nineteen it cannot.

That is a claim about the *encoder*, obtained without touching the encoder.
It is what moved the next lever from the decision rule to the backbone on
evidence rather than on intuition, and it is the reason the backbone came
fourth rather than never.

---

## 4. The calculation that says the experiment was never runnable

Suppose you insisted on settling the frozen swap on its own terms. How much
data would that take? Using the deltas and spreads from the two leaderboards,
at 80% power and alpha 0.05:

```python
# unpaired two-sample sizing: n per arm ~= 7.849 * (sd/d)**2
for task, sd, d in [("steps", 0.093, 0.029), ("instruments", 0.048, 0.021)]:
    print(task, round(7.849 * (sd / d) ** 2))
```

```
steps         81 videos per arm
instruments   41 videos per arm
```

**The dataset has 19 training videos.** Read these as an upper bound — a paired
test over the same videos does better than the unpaired approximation, and the
spreads are per-video score spreads rather than the spread of the paired
difference. The conclusion survives the caveat by a wide margin: the experiment
that would have answered "does a better encoder help" *in isolation* was not
runnable at any budget, because the data does not exist.

Fixing the loss first was therefore not a shortcut around the statistics. It
was the only way to raise the effect size above what 19 videos can resolve.

---

## 5. Cost is real, and it is the secondary argument

Fine-tuning needs **six** encoders rather than one. A backbone fine-tuned on
all of TRAIN and then cross-validated over folds drawn from that same set
produces features that already encode the held-out videos' labels;
[`infra/README.md`](../infra/README.md) owns the measurement of that leak and
the cost table for the honest alternative — one encoder per fold plus one on all
of TRAIN.

Against that, the masking change is a one-line edit to an argmax
([`CLAUDE.md`](../CLAUDE.md) records why it can only raise the official metric,
and [`step-variants.md`](models/step-variants.md) §4 what it was worth).

So cost and information point the same way here, which is a coincidence worth
not relying on. **Sequence by information; let cost break ties.** Had the
expensive lever been the ungated one, it would have had to go first anyway.

---

## 6. The same mistake, nearly made a third time

Fine-tuning DINOv2 produced both the worst result in the project and the best,
from identical code, data and augmentation — only the optimisation recipe
differed. Both runs are in [`step-variants.md`](models/step-variants.md) §7 and
[`instrument-variants.md`](models/instrument-variants.md) §6.

The generalisation is sharper than "tune your hyperparameters":

> **A fine-tune that fails tells you nothing about whether fine-tuning helps
> until you can watch the failure happen.**

Run 1 had no validation split anywhere, so fifty epochs of destroying the
representation were indistinguishable, from inside the loop, from fifty epochs
of learning. The null looked like evidence for the third time. And when run 2
added early stopping on three TRAIN videos held out *by video*, it fired at
epoch 2 — so the overwhelming majority of run 1's compute had been actively
making the encoder worse.

Three nulls, three different reasons, one shape: **the measurement was not
capable of seeing the thing it was being read as evidence about.**

---

## 7. The rule, and what it is not

Not "backbone last" as a ritual. The rule is:

> A lever whose effect is gated by another defect must be tested **after** that
> defect, or its null is uninterpretable. And you need an instrument that can
> see the gated quantity to know which situation you are in.

Two corollaries the repo now runs on:

- **A one-factor ablation is only valid for an ungated factor.** Backbone swaps
  are the canonical trap because they feel like the most fundamental change
  available, which makes their null feel the most conclusive.
- **Before running an experiment, ask what its null would license.** If the
  answer is "nothing, until X is fixed", fix X first — however much cheaper the
  experiment looks.

---

## 8. What this reasoning has not earned

Everything above rests on effect sizes measured against a fold spread that was
never independently bounded. The two v2 winners drift between training runs of
identical configuration, by more than DINOv2-alone's isolated delta:
[`current-state.md`](current-state.md) §5, gap 8 owns that table, and
[`citi-baseline.md`](models/citi-baseline.md) §6 owns why a seed does not pin a
run on this machine.

So the composed result sits comfortably above the drift and the isolated one may
be pure seed. **The §4 calculation is also the argument for bounding the
spread**, which is why [`where-we-are.md`](where-we-are.md) §5 puts it first:
until the run-to-run spread is known, every delta in this note smaller than it
is a conjecture with a number attached.

The open thread: what would that measurement cost — how many repeat runs at
what compute to resolve a delta of 0.02 — and does the answer make the
non-determinism worth fixing at its source instead?
