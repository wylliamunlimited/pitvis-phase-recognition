"""Which accelerator to run on, and how to seed a run. One definition of each.

A leaf module, like `paths.py`: it imports nothing from this package, so
anything may import it without creating a cycle. That matters here, because
`device_of` previously lived in `training/arst.py` and `training/instruments.py`
and `evaluation/run.py` and `inference/run.py` each reached into a *training*
module to get it — an inverted dependency purely for `torch.cuda.is_available()`,
which `evaluation/run.py` carried an apologetic comment about.

**Two orderings existed and both are kept, deliberately.** Four sites preferred
MPS over CUDA; `training/backbone.py` preferred CUDA over MPS. On a machine with
one accelerator the two agree, so unifying them looks free — but this project
fine-tunes on a rented CUDA host (`infra/`) and does everything else on a Mac,
and a host with both would silently change which device the fine-tune used.
`prefer` names the choice instead of hiding it.

`seed_everything` is the same story: four mains seeded torch and numpy, one
seeded torch only and left numpy's global RNG untouched, and one hardcoded a
seed with no flag. Reported numbers depend on this, so it gets one definition
that seeds both.
"""

import numpy as np
import torch


def device_of(name: str | None = None, *, prefer: str = "mps") -> torch.device:
    """The device to run on: `name` if given, else the best one available.

    `prefer` breaks the tie on a host that has both CUDA and MPS. It is not a
    performance claim — on any machine with one accelerator every value gives
    the same answer.
    """
    if name:
        return torch.device(name)
    order = ("cuda", "mps") if prefer == "cuda" else ("mps", "cuda")
    for candidate in order:
        if candidate == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if candidate == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed torch and numpy together.

    Neither pins the arithmetic — MPS reduction kernels are not bit-deterministic
    between runs, which is why `citi-baseline.md` records three different scores
    for one configuration. This fixes the initialisation and the shuffling, and
    that is all it claims to do.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
