# notes/

Four layers, and which one a document belongs to is decided before it is
written. `CLAUDE.md` holds the *decisions*; these hold the reasoning behind
them. **Every fact has exactly one owner and everywhere else links** — a
scoreboard maintained in four files is a scoreboard that goes stale in three.

## Start here

| | |
|---|---|
| [**where-we-are.md**](where-we-are.md) | orientation snapshot — the vocabulary, where the numbers got to, what to run next. A **dated** snapshot: re-date it when it goes stale rather than leaving old numbers standing. |
| [**roadmap.md**](roadmap.md) | everything left to build, in phases, with the decisions each item is waiting on |

## The tour — read in order, once

| | |
|---|---|
| [**walkthrough.md**](walkthrough.md) | the surgery, the data and the pipeline, with `file.py:NN` pointers. Assumes ML fluency. |
| [**embeddings.md**](embeddings.md) | what the feature cache *is*, from the ground up, every number read off the real cache. Assumes nothing — the entry point for the ML side. |

## [reference/](reference/) — look things up

Consulted, not read. Each owns one lookup surface.

| | |
|---|---|
| [data-dictionary.md](reference/data-dictionary.md) | every annotation column and what each integer means |
| [metrics.md](reference/metrics.md) | what each metric measures and why the challenge picked it |
| [citi-dataflow.md](reference/citi-dataflow.md) | the task-1 cascade as a shape trace, every hop |
| [inventory.md](reference/inventory.md) | **generated** by `pitvis-inventory` — do not hand-edit |

## [models/](models/) — what was built, and what beat it

Reproductions first, then the iteration on top of each. The reproduction notes
stay reproductions: improvements go in the variant notes, so "what did the paper
do" never gets tangled with "what did we do next".

| | |
|---|---|
| [citi-baseline.md](models/citi-baseline.md) | the CITI task-1 reproduction: architecture, faithfulness, results |
| [instruments.md](models/instruments.md) | the SANO task-2 reproduction, and a defect in the official metric |
| [step-variants.md](models/step-variants.md) | the task-1 iteration — masking, class weights, DINOv2 |
| [instrument-variants.md](models/instrument-variants.md) | the task-2 iteration, and §2 owns the CV protocol |

## [surfaces/](surfaces/) — how the model reaches a person

| | |
|---|---|
| [app.md](surfaces/app.md) | the review surface: why Range is load-bearing, the `(-1,-2)` collision, pre-CCI confidence |
| [deployment.md](surfaces/deployment.md) | serving without Python: where the ONNX cut falls, the per-second fidelity bar |

## Not here

[`infra/README.md`](../infra/README.md) is the cloud fine-tuning runbook. It
lives beside the scripts rather than in this directory because you read it with
a terminal open, not to understand the project.

---

`walkthrough.md` §8 and `embeddings.md` deliberately cover the same extraction
stage at two depths. They are cross-linked, not deduplicated.
