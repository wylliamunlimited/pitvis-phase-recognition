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

from pitvis.training import arst, baseline


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

# Default training order when no model is named. Cheapest first, so a broken
# feature cache fails in seconds rather than after the three-stage run.
DEFAULT_ORDER = ["baseline", "arst"]


def get(name: str) -> Model:
    """Look up a model, with a useful error rather than a KeyError."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown model {name!r}. Registered: {', '.join(sorted(REGISTRY))}\n"
            f"Run `uv run pitvis-train --list` for details."
        ) from None


def resolve(names: list[str] | None) -> list[Model]:
    """Model objects for `names`, or DEFAULT_ORDER when none are given."""
    return [get(n) for n in (names or DEFAULT_ORDER)]


def describe() -> str:
    width = max(len(n) for n in REGISTRY)
    lines = []
    for name in DEFAULT_ORDER + sorted(set(REGISTRY) - set(DEFAULT_ORDER)):
        m = REGISTRY[name]
        lines.append(f"  {m.name:<{width}}  {m.summary}")
        if m.ablations:
            lines.append(f"  {'':<{width}}  ablations: {', '.join(m.ablations)}")
    return "\n".join(lines)
