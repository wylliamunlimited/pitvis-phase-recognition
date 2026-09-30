"""Run the whole data pipeline: raw download -> verified feature cache.

    inventory   probe all 25 videos, check the annotation invariants,
                write notes/inventory.md
    extract     decode at 1 fps, embed with the selected space's frozen
                backbone, write data/features/<space>/video_NN/{features,
                labels}.npy + manifest.json
    verify      re-check that cache against the raw annotations and the manifest

Order matters and is not arbitrary: `inventory` asserts the annotation
invariants that `extract` then relies on, and `verify` re-derives labels
straight from the CSVs, so it is only meaningful once `extract` has written
something. Running them in this order means a bad download is caught before
40 GB of decoding rather than after.

`extract` is resumable — videos whose outputs already exist at the expected
length are skipped — so re-running this whole workflow after a partial
extraction is cheap and is the intended way to finish an interrupted run.

Usage:
    uv run pitvis-data                        # inventory -> extract -> verify
    uv run pitvis-data --dry-run              # show the plan, run nothing
    uv run pitvis-data --only verify --probe  # just the slow integrity check
    uv run pitvis-data --skip inventory       # cache work only
    uv run pitvis-data --videos 1 2 3         # limit extraction to 3 videos
    uv run pitvis-data --space dinov2_vitb14  # extract AND verify that space
"""

import argparse

from pitvis.data import extract_features, inventory, spaces, verify_cache
from pitvis.pipeline import Stage, add_selection_args, execute, select

STAGES = ["inventory", "extract", "verify"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages run in the order listed above; --only preserves that order.",
    )
    add_selection_args(ap, STAGES)
    ap.add_argument("--videos", nargs="+", type=int, metavar="N",
                    help="restrict extraction to these video numbers (default: all 25)")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"),
                    help="override device autodetection for extraction")
    ap.add_argument("--space", default=None, choices=spaces.names(),
                    help="feature space to extract into and verify "
                         f"(default: {spaces.DEFAULT})")
    ap.add_argument("--probe", action="store_true",
                    help="verify: also re-run ffprobe per video (slow, but the only "
                         "length check independent of the annotations)")
    args = ap.parse_args(argv)

    # --space is built ONCE and handed to every stage that takes it. It used to
    # be appended to extract's argv only, so `pitvis-data --space dinov2_ft`
    # extracted 25 videos into dinov2_ft and then verified resnet50 — printing
    # OK for a cache nothing had checked, or dying on "manifest.json missing"
    # after a 25-minute extraction that had in fact succeeded.
    space_argv = ["--space", args.space] if args.space else []

    extract_argv = [str(v) for v in (args.videos or [])] + space_argv
    if args.device:
        extract_argv += ["--device", args.device]
    verify_argv = space_argv + (["--probe"] if args.probe else [])

    stages = [
        Stage("inventory", "probe videos, check annotation invariants",
              lambda: inventory.main([])),
        Stage("extract", "1 fps decode -> frozen backbone features (resumable)",
              lambda: extract_features.main(extract_argv)),
        Stage("verify", "re-check that cache against annotations + manifest",
              lambda: verify_cache.main(verify_argv)),
    ]
    return execute(select(stages, args), args, "pitvis data pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
