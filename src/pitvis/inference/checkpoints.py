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

    def meta(self) -> dict:
        """Tags recorded at training time.

        Every key has a default that reproduces the original models, which
        predate all of them: `space` defaults to resnet50, `mask_excluded` to
        False, `thresholds` to None (meaning the caller's global bar).
        """
        import torch
        if not self.path.exists():
            return {}
        ck = torch.load(self.path, map_location="cpu", weights_only=False)
        return {
            "space": ck.get("space", spaces.DEFAULT),
            "variant": ck.get("variant", "reproduction"),
            "arch": ck.get("arch", "sano-lstm" if self.task == INSTRUMENTS else "arst"),
            "mask_excluded": ck.get("mask_excluded", False),
            "thresholds": ck.get("thresholds"),
        }


# name -> (task, root, reproduction filename). v2 families hang variants off
# <root>/v2/<variant>/model.pt; the reproduction is the bare file at the root.
FAMILIES: dict[str, tuple[str, Path, str | None]] = {
    "arst": (STEPS, CKPT, "citi.pt"),
    "arst-v2": (STEPS, CKPT, None),
    "instruments": (INSTRUMENTS, CKPT_INSTRUMENTS, "sano.pt"),
    "instruments-v2": (INSTRUMENTS, CKPT_INSTRUMENTS, None),
}

DEFAULT_VARIANT = "best"


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
    """The best trained checkpoint for a task, preferring a v2 winner.

    A machine that only ran the reproductions still gets those; one that ran
    the iteration gets the model the leaderboard selected, without having to
    know either path.
    """
    have = available(task)
    if not have:
        return None
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
            lines.append(f"  {c.name:<{w}}  {c.task:<11} space={m.get('space','?')}"
                         f"  variant={m.get('variant','?')}{mark}")
    return "trained checkpoints:\n" + "\n".join(lines)
