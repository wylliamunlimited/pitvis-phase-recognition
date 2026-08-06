"""Run the training workflow: linear probe, then the CITI/ARST model.

    baseline    frame-wise linear probe on the frozen features — no temporal
                context, so it establishes the floor a temporal model must beat
    arst        CITI's three-stage model (spatial -> TeCNO -> ARST) plus
                auto-regressive inference with the consistency constraint

Both stages score on the same 5 validation videos with the same official
metric, so running them together is what makes the comparison meaningful:
the probe's edit score (~0.01) versus ARST's (~0.35) is the entire argument
for the architecture, and it is only credible when both numbers come from
one command on one machine.

`--ablations` additionally runs the three ARST variants that isolate each
published claim — CCI off, banded mask off, excluded classes masked out of the
argmax. Each is a full retrain, so expect roughly 4x the runtime.

Note that ARST writes data/arst/{citi.pt,result.json,standardize.npz} and the
ablations overwrite them in turn; the final state on disk is the last variant
that ran. Read the printed table, not the file, when comparing.

Usage:
    uv run pitvis-train                       # baseline -> arst
    uv run pitvis-train --only arst           # skip the probe
    uv run pitvis-train --only arst --ablations
    uv run pitvis-train --dry-run
"""

import argparse

from pitvis.pipeline import Stage, add_selection_args, execute, select
from pitvis.training import arst, baseline

STAGES = ["baseline", "arst"]

# name -> extra flags. Each is a separate full training run.
ABLATIONS = {
    "arst:no-cci": ["--no-cci"],
    "arst:width-0": ["--width", "0"],
    "arst:masked": ["--mask-excluded"],
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selection_args(ap, STAGES)
    ap.add_argument("--ablations", action="store_true",
                    help="also run the three ARST ablations (each a full retrain)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confusion", action="store_true",
                    help="print the 15-way confusion matrix for each run")
    ap.add_argument("--epochs-arst", type=int,
                    help="override ARST epochs (useful for a fast smoke run)")
    args = ap.parse_args(argv)

    common = ["--seed", str(args.seed)]
    if args.confusion:
        common.append("--confusion")
    arst_argv = list(common)
    if args.epochs_arst is not None:
        arst_argv += ["--epochs-arst", str(args.epochs_arst)]

    stages = [
        Stage("baseline", "frame-wise linear probe — the floor",
              lambda: baseline.main(common)),
        Stage("arst", "CITI three-stage model + CCI inference",
              lambda: arst.main(arst_argv)),
    ]
    chosen = select(stages, args)

    if args.ablations and any(s.name == "arst" for s in chosen):
        for name, extra in ABLATIONS.items():
            chosen.append(Stage(name, f"ARST ablation: {' '.join(extra)}",
                                lambda e=extra: arst.main(arst_argv + e)))

    return execute(chosen, args, "pitvis training workflow")


if __name__ == "__main__":
    raise SystemExit(main())
