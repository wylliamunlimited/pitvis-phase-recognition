"""Train one or more registered models: `uv run pitvis-train [model ...]`.

Models are named positionally and come from the registry in `registry.py`, so
adding a model needs no change here and no new console script:

    uv run pitvis-train                 # every registered model, cheapest first
    uv run pitvis-train arst            # just ARST
    uv run pitvis-train baseline arst   # both, in the order given
    uv run pitvis-train --list          # what can I train?

Unrecognised flags are forwarded verbatim to the model, so each model keeps
ownership of its own arguments and `pitvis-train arst --help-model` shows them:

    uv run pitvis-train arst --no-cci --width 0
    uv run pitvis-train baseline --epochs 30

`--ablations` additionally runs each selected model's registered variants (for
ARST: CCI off, banded mask off, excluded classes masked out of the argmax).
Each is a full retrain, so expect roughly 4x the runtime.

Running baseline and ARST together is the point of the default: the probe's
edit score (~0.01) against ARST's (~0.35) is the whole argument for the
architecture, and it is only credible when both numbers come from one command
on one machine.

Note ARST writes data/arst/{citi.pt,result.json,standardize.npz} and the
ablations overwrite them in turn; the final state on disk is the last variant
that ran. Read the printed table, not the file, when comparing.
"""

import argparse

from pitvis.pipeline import Stage, execute
from pitvis.training import registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Registered models:\n" + registry.describe()
               + "\n\nUnrecognised flags are forwarded to the model.",
    )
    ap.add_argument("models", nargs="*", metavar="MODEL",
                    help="models to train (default: all, cheapest first)")
    ap.add_argument("--list", action="store_true",
                    help="print the model registry and exit")
    ap.add_argument("--ablations", action="store_true",
                    help="also run each model's registered ablations")
    ap.add_argument("--help-model", action="store_true",
                    help="show the selected model's own --help and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run, then exit")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a failing model instead of stopping")
    args, passthrough = ap.parse_known_args(argv)

    if args.list:
        print("Registered models:\n" + registry.describe())
        return 0

    models = registry.resolve(args.models)

    if args.help_model:
        models[0].main(["--help"])          # argparse exits inside
        return 0

    stages = []
    for m in models:
        stages.append(Stage(m.name, m.summary,
                            lambda m=m: m.main(list(passthrough))))
        if args.ablations:
            for label, extra in m.ablations.items():
                stages.append(Stage(
                    f"{m.name}:{label}",
                    f"ablation: {' '.join(extra)}",
                    lambda m=m, extra=extra: m.main(passthrough + extra),
                ))

    return execute(stages, args, "pitvis training")


if __name__ == "__main__":
    raise SystemExit(main())
