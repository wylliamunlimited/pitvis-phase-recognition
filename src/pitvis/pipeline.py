"""Shared plumbing for the per-package `run.py` workflow scripts.

Each subpackage exposes a `run.py` that chains its stages into one command.
Those scripts must stay *orchestrators*: they select and sequence stages and
never reimplement one. A stage's behaviour lives in its own module's `main()`,
which is also what the per-stage console script calls — so there is exactly one
definition of what "extract features" means, and `pitvis-data` and
`pitvis-extract` cannot drift apart.

This module holds the parts every runner needs: the stage record, the argparse
flags for selecting stages, and the execution loop that prints banners, times
each stage and decides what a failure does.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

BANNER = "=" * 74


@dataclass(frozen=True)
class Stage:
    """One step of a workflow.

    `run` takes no arguments — runners close over whatever the stage needs, so
    that argument translation happens once, where the flags are defined.
    """

    name: str
    summary: str
    run: Callable[[], None]


def add_selection_args(ap: argparse.ArgumentParser, stages: list[str]) -> None:
    """Add the stage-selection flags shared by every runner."""
    choices = ", ".join(stages)
    ap.add_argument("--only", nargs="+", metavar="STAGE", choices=stages,
                    help=f"run only these stages, in workflow order ({choices})")
    ap.add_argument("--skip", nargs="+", metavar="STAGE", choices=stages,
                    help="run everything except these stages")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the stages that would run, then exit")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going after a failing stage instead of stopping")


def select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    """Apply --only / --skip, preserving the declared workflow order."""
    if args.only and args.skip:
        raise SystemExit("--only and --skip are mutually exclusive")
    chosen = stages
    if args.only:
        chosen = [s for s in stages if s.name in set(args.only)]
    elif args.skip:
        chosen = [s for s in stages if s.name not in set(args.skip)]
    if not chosen:
        raise SystemExit("stage selection left nothing to run")
    return chosen


def execute(chosen: list[Stage], args: argparse.Namespace, title: str) -> int:
    """Run the selected stages. Returns a process exit code.

    A stage that raises SystemExit is treated as that stage failing rather than
    as the workflow exiting, so one failing stage cannot silently truncate the
    run without a report.
    """
    plan = " -> ".join(s.name for s in chosen)
    print(f"{BANNER}\n{title}\n  plan: {plan}\n{BANNER}")

    w = max(len(s.name) for s in chosen)
    if args.dry_run:
        for s in chosen:
            print(f"  {s.name:<{w}}  {s.summary}")
        print("\ndry run — nothing executed")
        return 0

    results, t_all = [], time.time()
    for i, stage in enumerate(chosen, 1):
        print(f"\n--- [{i}/{len(chosen)}] {stage.name} — {stage.summary} ---")
        t0 = time.time()
        try:
            stage.run()
        except SystemExit as e:              # a stage's own sys.exit(...)
            code = e.code if isinstance(e.code, int) else 1
            if code == 0:
                results.append((stage.name, "ok", time.time() - t0))
                continue
            results.append((stage.name, f"FAILED ({e.code})", time.time() - t0))
            if not args.continue_on_error:
                _summary(results, time.time() - t_all, w, aborted=True)
                return code
        except Exception as e:
            results.append((stage.name, f"FAILED ({type(e).__name__}: {e})",
                            time.time() - t0))
            if not args.continue_on_error:
                _summary(results, time.time() - t_all, w, aborted=True)
                raise
        else:
            results.append((stage.name, "ok", time.time() - t0))

    _summary(results, time.time() - t_all, w)
    return 0 if all(r[1] == "ok" for r in results) else 1


def _summary(results, elapsed: float, w: int = 10, aborted: bool = False) -> None:
    print(f"\n{BANNER}")
    for name, status, secs in results:
        print(f"  {name:<{w}}  {status:<28} {secs:6.1f}s")
    tail = "  (aborted — rerun with --continue-on-error to push past it)" if aborted else ""
    print(f"  {'total':<{w}}  {'':<28} {elapsed:6.1f}s{tail}\n{BANNER}")
