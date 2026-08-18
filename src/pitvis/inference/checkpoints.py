"""Where each trainable model writes its checkpoint — the inference-side registry.

`training/registry.py` is the only list of trainable models. This is its
counterpart for the other direction: given a model NAME, where are its weights,
its standardisation statistics, and which feature space were they trained on.

Without it, `pitvis-predict` could only be pointed at raw paths, so using a
model meant knowing the layout by heart — and the layout is exactly the sort of
thing that drifts. Naming a model instead means `pitvis-predict --steps arst-v2`
works the same way `pitvis-train arst-v2` does.

The spec grammar is `<model>` or `<model>:<variant>`:

    arst                 data/arst/citi.pt              the CITI reproduction
    arst-v2:best         data/arst/v2/best/model.pt     a step variant
    instruments          data/instruments/sano.pt       the SANO reproduction
    instruments-v2:best  data/instruments/v2/best/...   an instrument variant

`<model>` alone on a v2 family resolves to `:best`, since that is the variant
the leaderboard selected.

WHY THIS IS NOT IN training/registry.py. A model's `main` is a training entry
point; its checkpoint is an artifact that may or may not exist yet. Keeping
them apart means `pitvis-train --list` answers "what can I train" and
`pitvis-predict --list-models` answers "what have I trained", which are
different questions with different answers on any given machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pitvis.data import spaces
from pitvis.paths import CKPT, CKPT_INSTRUMENTS

STEPS = "steps"
INSTRUMENTS = "instruments"


@dataclass(frozen=True)
class Checkpoint:
    name: str          # the spec that resolved to this
    task: str          # STEPS | INSTRUMENTS
    path: Path
    stats: Path

    @property
    def exists(self) -> bool:
        return self.path.exists() and self.stats.exists()

    @property
    def result(self) -> Path:
        """Where this checkpoint's own scoring lands. Beside the weights."""
        return self.path.parent / "result.json"

    def score(self) -> float | None:
        """This checkpoint's recorded PRIMARY_METRIC, or None if never scored.

        Read rather than recomputed: every trainer writes `result.json` beside
        the weights it just saved, so the number and the weights cannot drift.
        """
        import json
        if not self.result.exists():
            return None
        try:
            mean = json.loads(self.result.read_text()).get("mean") or {}
            return float(mean[PRIMARY_METRIC])
        except (ValueError, KeyError, TypeError):
            return None

    def meta(self) -> dict:
        """Tags recorded at training time. See `read_tags` for the rules."""
        import torch
        if not self.path.exists():
            return {}
        return read_tags(
            torch.load(self.path, map_location="cpu", weights_only=False), self.task)


# name -> (task, root, reproduction filename). v2 families hang variants off
# <root>/v2/<variant>/model.pt; the reproduction is the bare file at the root.
FAMILIES: dict[str, tuple[str, Path, str | None]] = {
    "arst": (STEPS, CKPT, "citi.pt"),
    "arst-v2": (STEPS, CKPT, None),
    "instruments": (INSTRUMENTS, CKPT_INSTRUMENTS, "sano.pt"),
    "instruments-v2": (INSTRUMENTS, CKPT_INSTRUMENTS, None),
}

DEFAULT_VARIANT = "best"

# What ranks two trained checkpoints against each other. Macro F1, not the
# official `metric`, and deliberately: task 2's official number carries the
# vendored column-ordering defect, which reads the fine-tuned encoder as the
# WORST model tried (0.3220) where macro reads it as the best (0.5333). Macro
# is also the pre-registered primary in `crossval.Task` for both tasks, so this
# agrees with how variants were selected in the first place.
PRIMARY_METRIC = "macro_f1"


def read_tags(ckpt: dict, task: str = STEPS) -> dict:
    """Decode a loaded checkpoint dict into the tags inference must honour.

    THE ONE PLACE THIS IS DECIDED. It used to be decided in three — here,
    in `predict.load_checkpoint`, and implicitly in `evaluation/run.py` — and
    they disagreed, which is how `pitvis-train arst --mask-excluded` came to
    write a checkpoint that read back as unmasked.

    Every key defaults to what the original reproductions, which predate all
    of them, were trained with. Two rules are worth stating:

    - `mask_excluded` falls back to `args["mask_excluded"]` before it falls
      back to False. The reproduction trainer records the flag only inside
      `args`, so reading the top level alone silently un-masks a model that
      was trained masked — and masking is worth ~0.076 on the official metric.
    - `logit_adjust` is the ARRAY, not the tau it came from. `tau * log(prior)`
      is computed from the training labels of the split the model was fitted
      on, so it cannot be reconstructed at inference from tau alone. `prior_tau`
      rides along for provenance only.
    """
    args = ckpt.get("args") or {}
    return {
        "space": ckpt.get("space", spaces.DEFAULT),
        "variant": ckpt.get("variant", "reproduction"),
        "arch": ckpt.get("arch", "sano-lstm" if task == INSTRUMENTS else "arst"),
        "mask_excluded": bool(ckpt.get("mask_excluded",
                                       args.get("mask_excluded", False))),
        "thresholds": ckpt.get("thresholds"),
        "prior_tau": float(ckpt.get("prior_tau") or 0.0),
        "logit_adjust": ckpt.get("logit_adjust"),
    }


def resolve(spec: str) -> Checkpoint:
    """`<model>` or `<model>:<variant>` -> where its weights live."""
    model, _, variant = spec.partition(":")
    if model not in FAMILIES:
        raise SystemExit(
            f"unknown model {model!r}. Registered: {', '.join(sorted(FAMILIES))}"
        )
    task, root, reproduction = FAMILIES[model]

    if reproduction and not variant:
        return Checkpoint(spec, task, root / reproduction, root / "standardize.npz")
    if reproduction and variant:
        raise SystemExit(
            f"{model!r} has no variants — it is the published reproduction. "
            f"Did you mean {model}-v2:{variant}?"
        )
    d = root / "v2" / (variant or DEFAULT_VARIANT)
    return Checkpoint(spec, task, d / "model.pt", d / "standardize.npz")


def available(task: str | None = None) -> list[Checkpoint]:
    """Every checkpoint actually on disk, optionally filtered by task."""
    out: list[Checkpoint] = []
    for model, (t, root, reproduction) in sorted(FAMILIES.items()):
        if task and t != task:
            continue
        if reproduction:
            c = Checkpoint(model, t, root / reproduction, root / "standardize.npz")
            if c.exists:
                out.append(c)
        else:
            for d in sorted((root / "v2").glob("*")) if (root / "v2").exists() else []:
                c = Checkpoint(f"{model}:{d.name}", t, d / "model.pt",
                               d / "standardize.npz")
                if c.exists:
                    out.append(c)
    return out


def default(task: str) -> Checkpoint | None:
    """The best trained checkpoint for a task, by its own recorded score.

    WHY THIS IS NOT A NAME RULE. It was one — "the first checkpoint whose name
    ends `:best`" — and `available()` returns names sorted, so `best` sorted
    ahead of `best@dinov2_ft` and the default resolved to the model the
    fine-tuned encoder beat by 0.0998 on steps. Both are legitimately called
    "best"; one is best on frozen DINOv2 and the other best overall. A name
    cannot express that, and the scores can.

    So: rank by PRIMARY_METRIC read from each checkpoint's own `result.json`,
    and fall back to the old name convention only when nothing on this machine
    has been scored. Ties and unscored checkpoints keep `available()`'s sorted
    order, so the answer is deterministic either way.

    This picks among artifacts that already exist; it is NOT the variant
    selection protocol. That is 5-fold cross-validation inside TRAIN
    (`training/crossval.py`), and it stays that way — VAL is scored once, for
    the winner. Choosing which already-trained file to load by default is a
    convenience, and it is reported by `--list-models` rather than hidden.
    """
    have = available(task)
    if not have:
        return None
    scored = [(c.score(), c) for c in have]
    scored = [(s, c) for s, c in scored if s is not None]
    if scored:
        return max(scored, key=lambda sc: sc[0])[1]
    for c in have:
        if c.name.endswith(f":{DEFAULT_VARIANT}"):
            return c
    return have[0]


def describe() -> str:
    """What is trained on this machine, for `--list-models`."""
    have = available()
    if not have:
        return ("no checkpoints found — train one first:\n"
                "    uv run pitvis-train arst\n"
                "    uv run pitvis-train instruments")
    w = max(len(c.name) for c in have)
    lines = []
    for task in (STEPS, INSTRUMENTS):
        picked = default(task)
        for c in available(task):
            m = c.meta()
            mark = " (default)" if picked and c.name == picked.name else ""
            sc = c.score()
            score = f"  {PRIMARY_METRIC}={sc:.4f}" if sc is not None else "  (unscored)"
            lines.append(f"  {c.name:<{w}}  {c.task:<11} space={m.get('space','?')}"
                         f"  variant={m.get('variant','?')}{score}{mark}")
    return "trained checkpoints:\n" + "\n".join(lines)
