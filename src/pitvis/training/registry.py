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

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Model:
    """One trainable model.

    `module` names the module whose `main(argv)` trains it, and `main`
    imports it on first use. The module is NOT imported when this registry is
    built: `pitvis-train --list` and `--dry-run` need only `name`, `summary`
    and `ablations`, and eagerly importing all five trainers cost them 2.25 s
    of torch and scikit-learn to print a table. Adding a model is still one
    `Model(...)` entry — that rule is unchanged, only the import is deferred.

    The module owns its own argparse, so `pitvis-train <name> --help` shows
    that model's flags. `ablations` names variants worth running together;
    each value is extra flags for `main`.
    """

    name: str
    summary: str
    module: str
    ablations: dict[str, list[str]] = field(default_factory=dict)

    @property
    def main(self) -> Callable[[list[str] | None], None]:
        """The module's entry point, imported on first access."""
        return importlib.import_module(self.module).main


REGISTRY: dict[str, Model] = {
    m.name: m
    for m in [
        Model(
            name="baseline",
            summary="frame-wise linear probe on frozen features — the floor",
            module="pitvis.training.baseline",
        ),
        Model(
            name="instruments",
            summary="SANO's PitVis task-2 joint winner: frozen features + causal LSTM",
            module="pitvis.training.instruments",
            ablations={
                "no-aux-step": ["--no-aux-step"],
            },
        ),
        Model(
            name="arst-v2",
            summary="step variants — argmax masking, class weights, DINOv2",
            module="pitvis.training.arst_v2",
            ablations={
                "masked": ["--variant", "masked"],
                "weighted": ["--variant", "weighted"],
                "dinov2": ["--variant", "dinov2"],
            },
        ),
        Model(
            name="instruments-v2",
            summary="instrument variants — weighted loss, per-class thresholds, DINOv2",
            module="pitvis.training.instruments_v2",
            ablations={
                "weighted": ["--variant", "weighted"],
                "thresholds": ["--variant", "thresholds"],
                "dinov2": ["--variant", "dinov2"],
            },
        ),
        Model(
            name="arst",
            summary="CITI's PitVis-2023 task-1 winner: spatial + TeCNO + ARST",
            module="pitvis.training.arst",
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
