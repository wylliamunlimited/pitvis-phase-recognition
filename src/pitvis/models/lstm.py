"""SANO's PitVis-2023 task-2 model: frozen CNN features into a windowed LSTM.

SANO placed joint 1st in instrument recognition (Das et al. 2024 Table 6,
41.6±06.3; £500 was awarded to joint 1st, i.e. ranks 1 and 2). SDS-HD edged the
table by 0.1 points but did so with a three-encoder ensemble — ResNet152,
EfficientNetB7 and SwinL, each with its own LSTM, fused by an unspecified
"balanced ensemble". SANO is both reproducible here and far more stable
(±6.3 against SDS-HD's ±15.4).

The published description, §5.5 verbatim:

    "For task-2 their model consisted of 2-stages: the trained CNN was frozen;
    followed by a 5-window LSTM for both instrument (task-2) and step (just for
    training) classification."

Which maps onto this repo exactly: the backbone is ResNet-50 — the same network
that produced our feature cache — and "frozen, then a temporal model on top" is
the shape the cache was built for.

    (T, 2048) cached features
      -> causal window of W frames        [t-W+1 .. t], left-padded
      -> LSTM, 2 layers, unidirectional
      -> Linear(hidden, 19) -> sigmoid    instruments (task 2)
      -> Linear(hidden, 15)               steps (auxiliary, training only)

Three things worth knowing before changing any of it:

**Unidirectional is not a tuning choice.** The challenge permits only online
models — "only information from frames up to and including the current frame"
(Das et al. §3.2). A bidirectional LSTM reads the future and would invalidate
every number. The windowing is left-padded for the same reason.

**19 outputs, not 18.** After the official encoder pops the -1/-2 columns, ids
0..18 remain, and id 0 ("no visible instrument") is a scored class holding 31.5%
of frames — not a background sentinel. The paper is internally inconsistent here
(Tables 6-7 say 19, Figure 4 says 18); the code settles it at 19.

**Sigmoid + BCE, not softmax.** Task 2 is multi-label: zero, one or two
instruments may be present (§3.1). Table 3 gives SANO's final activation as
Sigmoid and its instrument loss as BCE.

The auxiliary step head is faithful to "step (just for training)" — SANO trains
it and discards it. It is ablatable via `--no-aux-step` so its contribution is
measurable rather than assumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_INSTRUMENTS = 19        # ids 0..18 survive the official column popping
NUM_STEPS = 15              # auxiliary head, 15-way encoded
WINDOW = 5                  # "5-window LSTM", SANO §5.5
HIDDEN = 512
LAYERS = 2


def causal_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, T, D) -> (B, T, window, D), each row the W frames ending at t.

    Left-padded by repeating frame 0, so position t sees [t-W+1 .. t] and never
    a future frame. Uses `unfold`, so this is a view-based reshape rather than a
    T-long Python loop — it stays cheap at T ≈ 8,600.
    """
    B, T, D = x.shape
    pad = x[:, :1].expand(B, window - 1, D)
    padded = torch.cat([pad, x], dim=1)                  # (B, T+W-1, D)
    return padded.unfold(1, window, 1).permute(0, 1, 3, 2)


class SanoLSTM(nn.Module):
    """Frozen-feature windowed LSTM with a multi-label instrument head."""

    def __init__(self, in_dim: int = 2048, hidden: int = HIDDEN,
                 layers: int = LAYERS, window: int = WINDOW,
                 num_instruments: int = NUM_INSTRUMENTS,
                 num_steps: int = NUM_STEPS, dropout: float = 0.2,
                 aux_step: bool = True):
        super().__init__()
        self.window = window
        self.aux_step = aux_step
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,                          # online-only rule
        )
        self.drop = nn.Dropout(dropout)
        self.instruments = nn.Linear(hidden, num_instruments)
        self.steps = nn.Linear(hidden, num_steps) if aux_step else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(B, T, D) -> instrument logits (B, T, 19), step logits (B, T, 15)|None.

        Every position is decoded from the final hidden state of its own window,
        so the T positions are independent given their windows — that is what
        makes the model strictly causal without any masking machinery.
        """
        B, T, D = x.shape
        w = causal_windows(x, self.window)                # (B, T, W, D)
        out, _ = self.lstm(w.reshape(B * T, self.window, D))
        h = self.drop(out[:, -1])                         # last step of each window
        inst = self.instruments(h).view(B, T, -1)
        step = self.steps(h).view(B, T, -1) if self.steps is not None else None
        return inst, step


def decide(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sigmoid logits -> (…, 19) binary, capped at the structural maximum of 2.

    The label is a pair of columns, so no frame can carry three instruments. The
    paper does not state SANO's decision rule — only UNI-ANDES-23 documents a
    threshold (0.4), and they placed last — so this is ours: take everything
    above `threshold`, and if more than two clear it, keep the two highest.
    """
    prob = torch.sigmoid(logits)
    keep = prob >= threshold
    over = keep.sum(-1) > 2
    if over.any():
        top2 = prob.topk(2, dim=-1).indices
        capped = torch.zeros_like(keep)
        capped.scatter_(-1, top2, True)
        keep = torch.where(over.unsqueeze(-1), capped & keep, keep)
    return keep.to(torch.int8)


def decide_per_class(logits: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """As `decide`, but with one threshold per class — and capped by MARGIN.

    A sibling rather than a change to `decide`: that one is pinned by the SANO
    reproduction and must keep returning what it returns today.

    Why the cap changes. `decide` breaks a 3-way tie by raw probability, which
    is the right ordering only while every class shares a threshold. Once
    class 17 clears at 0.15 and class 16 at 0.60, raw probability is no longer
    comparable across classes — the frequent class wins every tie by
    construction and the rare one can never survive the cap, which is exactly
    the failure per-class thresholds exist to fix. Ranking by `prob - tau`
    (how far past its own bar each class cleared) restores the comparison.

    `thresholds` is (19,) and broadcasts over any leading dimensions.
    """
    prob = torch.sigmoid(logits)
    keep = prob >= thresholds
    over = keep.sum(-1) > 2
    if over.any():
        margin = prob - thresholds
        top2 = margin.topk(2, dim=-1).indices
        capped = torch.zeros_like(keep)
        capped.scatter_(-1, top2, True)
        keep = torch.where(over.unsqueeze(-1), capped & keep, keep)
    return keep.to(torch.int8)
