# Ensembling — a plan, not a result

Nothing here has been run. This is the design, the reasoning, and the
falsifiers, written before any number exists so the protocol cannot be chosen
after seeing which one flatters us.

Companion to [`step-variants.md`](step-variants.md) and
[`instrument-variants.md`](instrument-variants.md), which own results.

---

## 1. Why this is worth trying, and why now

**SDS-HD won task 2 by ensembling**, and Das et al. never state the fusion
rule. That is the single largest untested idea from the challenge, and it is
the one the paper leaves most room to rediscover.

It also fits what this repo has learned the hard way. Two directions have now
been pushed to their ends:

| direction | outcome |
|---|---|
| the decision rule | paid on task 2 (+0.099 macro), **null on task 1** — logit adjustment loses, monotonically |
| the encoder | fine-tuned ResNet-50 a wash; DINOv2 **once destroyed and once won** — see the update below |

Ensembling is the remaining lever that needs no new encoder, no GPU, and no
new data. Everything it needs is already cached.

> **Update, 2026-08-16 — the encoder row changed after this was written.**
> Re-running the DINOv2 fine-tune with a fixed recipe produced the largest gain
> in the project: steps 0.4610 → **0.5608**, instruments macro 0.3792 →
> **0.5333**. That does not weaken the case for ensembling, but it does change
> the members: `dinov2_ft` is now the strongest single space and belongs in
> every ensemble here, where this plan originally listed it as a member that
> was "worse individually". §2B below is updated; the rest stands as written,
> deliberately, since the protocol must not be chosen after seeing results.

**And the variance argues for it directly.** Our per-video std on the step
metric is **±0.092**, on the instrument headline **±0.225**. Those are enormous
relative to the differences we have been chasing. Averaging is the standard
answer to variance that large, and unlike every other idea left, its benefit
*grows* with the noise rather than being hidden by it.

---

## 2. What to ensemble — three axes, cheapest first

The members must be **decorrelated**, or averaging buys nothing. Ranked by
decorrelation per unit of compute:

### A. Seeds (cheapest, most certain, least interesting)

Same recipe, `--seed 0..4`. Training is ~95 s for the step cascade and ~41 s
for instruments, so five members is under ten minutes total.

Decorrelation comes only from initialisation and batch order. That is real —
MPS is not bit-deterministic and the same seed already moves the third decimal
— but it is the weakest axis. **Its value is as the floor**: whatever seed
averaging buys is the part of any other ensemble's gain that is *not* about
diversity.

### B. Feature spaces (most promising)

`resnet50` (2048-d, ImageNet CNN) and `dinov2_vitb14` (768-d, self-supervised
ViT) are genuinely different encoders — different architecture, different
pretraining data, different failure modes. `resnet50_ft` and `dinov2_ft` add
two more, and one being *worse individually* does not preclude it helping: an
ensemble wants members that are wrong in different places.

*Updated:* `dinov2_ft` is now the best single space, not a weak member, so the
first ensemble to try is **`dinov2_ft` + `dinov2_vitb14`** — the same encoder
before and after adaptation, which is the cheapest genuinely-decorrelated pair
available and directly tests whether fine-tuning traded away something the
frozen version still has. `resnet50` joins as the architecture-diverse third.

This is the axis most likely to pay, and the AP probe already hints at it —
tissue glue scores 0.767 on frozen DINOv2 while cup forceps scores 0.055, and
the ResNet numbers are not the same shape.

### C. Architectures (most work)

ARST against a plain TeCNO or MS-TCN for steps; the SANO LSTM against a
windowed transformer for instruments. Roadmap 3.2/3.3 already scope these, and
they must exist before they can be ensembled — so this axis is gated on work
that is not ensembling.

**Start with B, use A as the floor, treat C as future.**

---

## 3. How to combine — and why this is not obvious for either task

### Task 2 (instruments) — easy, do it first

Nineteen independent sigmoids. Average the **probabilities** across members,
then apply the existing per-class thresholds to the mean.

One subtlety: the per-class taus were fitted against a *single* model's
probability distribution. Averaging shrinks variance and pulls probabilities
toward the middle, so the old thresholds will be systematically mis-set. **The
taus must be re-fitted on the ensemble's out-of-bag probabilities**, using the
same 2-fold cross-fitting `instruments_v2.py` already implements. Skipping that
would report the ensemble's gain net of a threshold mismatch it did not need
to have.

### Task 1 (steps) — genuinely awkward, and the reason to think first

Averaging is not well-defined here, because ARST decodes **auto-regressively
over its own past predictions**. Three options, in increasing fidelity:

1. **Vote on final labels.** Per second, take the majority over members.
   Trivial, but throws away all confidence, and a 2-2-1 split needs a
   tie-break that is effectively arbitrary.
2. **Average the pre-CCI probabilities, then decode once.** `cci_decode`
   already exposes these via `return_probs`. But the probabilities are recorded
   *at the moment of decision*, each conditioned on that member's own past
   labels — so they are not comparable in the way an ensemble assumes. This is
   the cheap option and its weakness must be stated wherever the number is.
3. **Ensemble inside the loop.** Run all members' decoders in lockstep,
   average their logits at each step, and feed the *shared* argmax back to
   every member as its previous label. This is the only version where the
   members genuinely agree on a trajectory, and it makes CCI well-defined
   again — one sequence, one consistency check.

**Option 3 is the right one and it is a real change** to `cci_decode` (it must
accept a list of models). Option 2 is an afternoon and would say whether the
idea is worth that work.

---

## 4. Protocol — fixed before running anything

Unchanged from the existing rules, which is the point:

- **Rank on 5-fold CV over the 19 TRAIN videos**, frozen folds, never on VAL.
- **Primary `macro_f1`**, guarded on the official metric within one std of the
  control spread.
- **VAL scored exactly once**, for the winner only, and only if CV says there
  is a winner.
- Every member trains on the **fold's own training videos**. An ensemble whose
  members saw different data than the fold allows is the leak we already made
  once with the encoder.

**Falsifier, stated in advance:** if a 5-member ensemble does not beat the best
single model by more than the fold spread on `macro_f1`, ensembling is dead for
this dataset and the note records that instead. Given ±0.092 on steps, that is
a demanding bar — deliberately.

---

## 5. Expected value, honestly

Ensembles reliably buy *something*; the question is whether it survives a
per-video-then-mean metric on five videos. My estimate, and it is an estimate:

- **Task 2, feature-space ensemble: most likely to pay.** Two genuinely
  different encoders, a metric that rewards rare-class recall, and a decision
  rule that can be re-fitted to exploit the averaged distribution.
- **Task 1, option 2: probably a small gain, possibly none.** The edit score
  rewards stability and averaging is stabilising, but the pre-CCI probability
  objection is real.
- **Seeds alone: small.** Worth running only to calibrate the others.

**What it will not do** is close the gap to CITI's 70. Ensembling narrows
variance around a ceiling set by the representation; it does not raise the
ceiling. If the goal is the leaderboard rather than the best model this repo
can honestly build, the encoder is the answer.

*That last sentence has since been acted on and it was right* — the fixed
recipe moved steps to 0.5608, closing about half the gap that stood when this
was written, and doing it by raising the ceiling rather than narrowing variance
around it. The prediction stands as a check on the estimates above: ensembling
is now the *cheap* remaining lever, not the *best* one.

---

## 6. Order of work

1. Task 2, feature-space ensemble of `dinov2_ft` + `dinov2_vitb14`,
   probabilities averaged, taus re-fitted, CV. **~20 minutes, no new
   training.** (Was `resnet50` + `dinov2_vitb14`; reordered because the best
   single member changed. The falsifier's bar moves with it — beat
   `dinov2_ft` alone, not the frozen space.)
2. If that clears the bar: add `resnet50` and `resnet50_ft` as members and
   re-run. Tests whether individually-worse members still help.
3. Task 1, option 2 (average pre-CCI probabilities). **~1 hour.**
4. Only if 3 pays: option 3, the lockstep decoder. **A day, and a real change
   to `cci_decode`.**
5. Seeds, at any point, as the floor to compare against.

Stop at the first step that fails the falsifier.
