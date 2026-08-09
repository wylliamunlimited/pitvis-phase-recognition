"""Out-of-fold cross-validation for instrument variants.

WHAT THIS IS FOR. Ranking variants on the 5 validation videos would rank noise
— per-video std is 0.04-0.05 and the paper measures a -47 point val->test
collapse for instruments — and it would turn VAL into a selection set, which
is the one thing that makes a final number untrustworthy. So variants are
ranked here, inside TRAIN, and VAL is scored once for the winner.

HOW THE SCORE IS FORMED. Each of the 19 training videos is held out by exactly
one fold (`data/folds.py`), so the union of the five folds' held-out
predictions is 19 `(vid, y_true, y_pred)` triples, **every one produced by a
model that never saw that video**. Those 19 go through a single
`evaluation.instruments.evaluate` call, which gives per-video-then-mean±std in
the challenge's own aggregation convention — a far more stable statistic than
the mean of five fold means, and it reuses the existing scorer unchanged.
Per-fold means are kept as a secondary spread diagnostic only.

THE PRE-REGISTERED RANKING, fixed before any variant ran:

  primary    macro_f1  — the metric Das et al. names for task 2, and the only
                         one of the three that moves when a dead class comes
                         alive. 9 of 19 classes currently sit at F1 0.000.
  guard      metric    — the official (defect-included) number must not
                         regress by more than one std of the 19-video spread.
                         It is weighted and support-dominated, so a variant
                         trading id-16 precision for id-17 recall can raise
                         macro while lowering the headline.
  reported   weighted  — the name-aligned control on `metric`.

LEAKAGE. Standardisation statistics are computed inside `fit`, on the fold's
training videos only. Computing them over all of TRAIN — which the single-split
trainer legitimately does — would leak held-out feature statistics into every
fold.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Callable

import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path

from pitvis.data.dataset import TRAIN, load_split, load_split_instruments
from pitvis.data.folds import folds as fold_ids
from pitvis.evaluation import instruments as inst_eval
from pitvis.evaluation import metric as step_eval
from pitvis.paths import CKPT, CKPT_INSTRUMENTS

# fit(train_videos, args, device) -> predict(features (T, D)) -> targets (T, ...)
PredictFn = Callable[[np.ndarray], np.ndarray]
FitFn = Callable[[list[tuple[int, np.ndarray, np.ndarray]], object, torch.device], PredictFn]


@dataclass(frozen=True)
class Task:
    """What differs between the two challenge tasks, gathered in one place.

    The harness itself is task-agnostic: hold out a fold, fit, predict, and
    aggregate per-video-then-mean. Only the loader, the scorer and which metric
    ranks change, so they are data rather than a second copy of the harness.
    """

    name: str
    loader: Callable          # (videos, space) -> [(vid, features, target)]
    evaluate: Callable        # per_video -> {videos, mean, std, pooled}
    metrics: tuple[str, ...]
    primary: str              # what ranks
    guard: str                # what must not regress
    cv_dir: Path


INSTRUMENTS = Task(
    name="instruments", loader=load_split_instruments,
    evaluate=inst_eval.evaluate, metrics=inst_eval.METRICS,
    primary="macro_f1", guard="metric",
    cv_dir=CKPT_INSTRUMENTS / "cv",
)

STEPS = Task(
    name="steps", loader=load_split,
    evaluate=step_eval.evaluate, metrics=step_eval.METRICS,
    primary="macro_f1", guard="metric",
    cv_dir=CKPT / "cv",
)


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def cross_validate(fit: FitFn, args, dev: torch.device, *, variant: str,
                   space: str, task: Task = INSTRUMENTS, k: int = 5,
                   quiet: bool = False) -> dict:
    """Fit `k` models, score every training video out of fold, aggregate once."""
    t0 = time.time()
    # Loaded once and sliced. 19 videos at 2048-d is ~693 MB; reloading it per
    # fold would turn a 5-minute sweep into a disk-bound one.
    split = task.loader(TRAIN, space)
    by_vid = {vid: (f, inst) for vid, f, inst in split}

    per_fold, out_of_fold = [], []
    for i, held in enumerate(fold_ids(k)):
        train = [(v, *by_vid[v]) for v in TRAIN if v not in set(held)]
        predict = fit(train, args, dev)

        fold_preds = []
        for v in held:
            feats, truth = by_vid[v]
            fold_preds.append((v, truth, predict(feats)))
        out_of_fold.extend(fold_preds)

        m = task.evaluate(fold_preds)["mean"]
        per_fold.append({"fold": i, "held_out": list(held), **m})
        if not quiet:
            print(f"  fold {i} ({len(held)} held out): "
                  + "  ".join(f"{k_}={m[k_]:.4f}" for k_ in task.metrics))

    # The headline: one scorer call over all 19, never pooled frame-wise.
    result = task.evaluate(out_of_fold)
    pooled = result["pooled"]
    # Two different failures, both worth counting. `dead` is F1 exactly 0 —
    # the class is never got right. `never_predicted` is the stricter one: the
    # model never emits it at all, which is the shape of the current defect
    # (9 of 19). Every class has non-zero pooled support across 19 videos, so
    # neither count is an artifact of an absent class.
    dead = int((np.asarray(pooled["per_class_f1"]) == 0).sum())
    # The steps scorer reports a confusion matrix rather than a predicted
    # count; its column sums are the same thing.
    if "predicted" in pooled:
        predicted = np.asarray(pooled["predicted"])
    else:
        predicted = np.asarray(pooled["confusion_matrix"]).sum(axis=0)
    never = int((predicted == 0).sum())

    entry = {
        "variant": variant,
        "space": space,
        "task": task.name,
        "k": k,
        "videos": {str(v): result["videos"][v] for v in result["videos"]},
        "mean": result["mean"],
        "std": result["std"],
        "per_fold": per_fold,
        "pooled": {
            "per_class_f1": [float(x) for x in pooled["per_class_f1"]],
            "support": [int(x) for x in pooled["support"]],
            "predicted": [int(x) for x in predicted],
        },
        "dead_classes": dead,
        "never_predicted": never,
        "seconds": round(time.time() - t0, 1),
        "git": git_rev(),
        "args": {k_: v for k_, v in vars(args).items()} if hasattr(args, "__dict__") else {},
    }

    task.cv_dir.mkdir(parents=True, exist_ok=True)
    (task.cv_dir / f"{variant}.json").write_text(json.dumps(entry, indent=2) + "\n")

    if not quiet:
        print(f"\n  {variant}: " + "  ".join(
            f"{k_} {entry['mean'][k_]:.4f}±{entry['std'][k_]:.4f}" for k_ in task.metrics))
        # Counts are over each task's own class set: 19 instruments, but 12
        # SCORED steps for F1 and 15 for the confusion matrix. Hardcoding 19
        # here reported "7/19 never predicted" for a 15-class task.
        n_f1 = len(pooled["per_class_f1"])
        n_pred = len(predicted)
        print(f"  dead classes (F1 exactly 0): {dead}/{n_f1}   "
              f"never predicted at all: {never}/{n_pred}   "
              f"[{entry['seconds']:.0f}s]")
    return entry


def load_entries(task: Task = INSTRUMENTS) -> list[dict]:
    """Every CV result written so far for one task, by variant name."""
    if not task.cv_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(task.cv_dir.glob("*.json"))]


def summarise(entries: list[dict], task: Task = INSTRUMENTS) -> str:
    """The leaderboard. Ranked by the task's primary metric, guard reported."""
    if not entries:
        return "no cross-validation results yet — run with --cv"

    ranked = sorted(entries, key=lambda e: -e["mean"][task.primary])
    control = next((e for e in entries if e["variant"] == "control"), None)

    cols = list(task.metrics)
    head = f"{'variant':<20}{'space':<16}" + "".join(f"{c:>18}" for c in cols)
    rows = [head + f"{'dead':>6}{'never':>7}{'sec':>7}", "-" * len(head + " " * 20)]
    for e in ranked:
        m, sd = e["mean"], e["std"]
        rows.append(
            f"{e['variant']:<20}{e['space']:<16}"
            + "".join(f"{m[c]:>11.4f}\u00b1{sd[c]:.4f}" for c in cols)
            + f"{e['dead_classes']:>6}{e.get('never_predicted', -1):>7}"
              f"{e['seconds']:>7.0f}"
        )

    if control:
        rows.append("")
        rows.append(f"deltas vs control — ranked on {task.primary}, "
                    f"guarded on {task.guard}:")
        tol = control["std"][task.guard]
        for e in ranked:
            if e["variant"] == "control":
                continue
            dp = e["mean"][task.primary] - control["mean"][task.primary]
            dg = e["mean"][task.guard] - control["mean"][task.guard]
            ok = "PASS" if dg >= -tol else f"FAIL (>{tol:.4f} regression)"
            rows.append(f"  {e['variant']:<20} {task.primary} {dp:+.4f}   "
                        f"{task.guard} {dg:+.4f}   guard {ok}")
        rows.append(f"\n  guard tolerance = control's {task.guard} std = {tol:.4f}")
    return "\n".join(rows)
