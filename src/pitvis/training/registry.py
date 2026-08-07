"""The model registry — one place that knows what models exist.

Adding a model used to mean touching three files: the module, a new
`pitvis-train-<name>` entry in `[project.scripts]`, and the training runner's
hardcoded stage list. That is three chances to forget one, and the CLI drifting
from the code is exactly the failure the console scripts were meant to prevent.

Now a model is registered here and `uv run pitvis-train <name>` works
immediately — no pyproject edit, no CLI edit, no reinstall.

To add a model:

    1. Write `pitvis/training/<name>.py` with `main(argv: list[str] | None)`.
    2. Add one `Model(...)` entry to REGISTRY below.

`uv run pitvis-train --list` prints the registry, so it is also the answer to
"what can I train?".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pitvis.training import arst, baseline, instruments


@dataclass(frozen=True)
class Model:
    """One trainable model.

    `main` is the module's own argparse entry point, so the model owns its
    flags and `pitvis-train <name> --help` shows them. `ablations` names
    variants worth running together; each value is extra flags for `main`.
    """

    name: str
    summary: str
    main: Callable[[list[str] | None], None]
    ablations: dict[str, list[str]] = field(default_factory=dict)


REGISTRY: dict[str, Model] = {
    m.name: m
    for m in [
        Model(
            name="baseline",
            summary="frame-wise linear probe on frozen features — the floor",
            main=baseline.main,
        ),
        Model(
            name="instruments",
            summary="SANO's PitVis task-2 joint winner: frozen features + causal LSTM",
            main=instruments.main,
            ablations={
                "no-aux-step": ["--no-aux-step"],
            },
        ),
        Model(
            name="arst",
            summary="CITI's PitVis-2023 task-1 winner: spatial + TeCNO + ARST",
            main=arst.main,
            ablations={
                "no-cci": ["--no-cci"],
                "width-0": ["--width", "0"],
                "masked": ["--mask-excluded"],
            },
        ),
    ]
}

# A bare `pitvis-train` means TRAIN ALL. This list only fixes the ORDER of the
# models it names — cheapest first, so a broken feature cache fails in seconds
# rather than after a three-stage run. Anything registered and not named here
# still runs, appended afterwards in name order.
#
# It is deliberately not the selection list. A hand-maintained "what runs by
# default" would reintroduce exactly the drift the registry exists to remove:
# adding a model would mean editing its Model(...) entry AND remembering to add
# it here. One place to edit, or it will be forgotten.
ORDER_HINT = ["baseline", "arst"]


def get(name: str) -> Model:
    """Look up a model, with a useful error rather than a KeyError."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown model {name!r}. Registered: {', '.join(sorted(REGISTRY))}\n"
            f"Run `uv run pitvis-train --list` for details."
        ) from None


def default_order() -> list[str]:
    """Every registered model, ORDER_HINT first, then the rest by name.

    Derived from REGISTRY rather than hand-listed, so a newly registered model
    joins the default run automatically.
    """
    named = [n for n in ORDER_HINT if n in REGISTRY]
    return named + sorted(set(REGISTRY) - set(named))


def resolve(names: list[str] | None) -> list[Model]:
    """Model objects for `names`, or every registered model when none given."""
    return [get(n) for n in (names or default_order())]


def describe() -> str:
    width = max(len(n) for n in REGISTRY)
    lines = []
    for name in default_order():
        m = REGISTRY[name]
        lines.append(f"  {m.name:<{width}}  {m.summary}")
        if m.ablations:
            lines.append(f"  {'':<{width}}  ablations: {', '.join(m.ablations)}")
    return "\n".join(lines)
