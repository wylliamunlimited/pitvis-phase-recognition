"""CITI's PitVis-2023 task-1 winning model: ARST + TeCNO feature extractor.

CITI (Xiaoyang Zou, Guoyan Zheng — Shanghai Jiao Tong University) placed 1st in
PitVis-2023 task-1 with 62.9±9.7 on the challenge metric (61.1 macro-F1 / 64.7
edit). Their submission applies their own prior method:

    ARST: auto-regressive surgical transformer for phase recognition from
    laparoscopic videos. Zou, Liu, Wang, Tao, Zheng.
    Comput Methods Biomech Biomed Eng Imaging Vis 11(4), 2023.
    arXiv:2209.01148

The pipeline has three trained stages, each frozen before the next is trained
(ARST §2.1-2.2, §3.3):

    frames -> ResNet-50 -> 2048-d -> Linear -> 512-d  Z_t   (spatial embedding)
    Z_1:T  -> TeCNO (2-stage causal TCN)     -> 512-d  F_t   (temporal feature)
    F_1:T + shifted phase labels -> ARST     -> 15-way logits

ARST itself is a one-layer encoder-decoder transformer (Vaswani et al. 2017)
with two departures from the standard recipe, both load-bearing:

1. A BANDED causal mask, not an upper-triangular one. Position t attends only
   to [t-W, t]. The paper's ablation (Table 2) puts the optimum at W=5 on
   Cholec80 — accuracy falls off on both sides (W=0: 84.8, W=5: 87.6, W=40:
   83.6). The stated rationale is that long-range past is noisy for the current
   decision once recent predictions are available.

2. The decoder is AUTO-REGRESSIVE OVER PHASE LABELS. It consumes its own
   shifted past predictions, so the model learns p(y_t | y_0:t-1, F_1:t) rather
   than a per-frame posterior. This is what buys temporal consistency: the
   transition structure is modelled explicitly instead of being smoothed on
   afterwards. Teacher forcing parallelises training; inference is a true
   sequential rollout (see cci_decode in train_arst.py).

Both encoder and decoder self-attention use the banded causal mask, so the
whole network is causal — required by the challenge rule that "only information
from frames up to and including the current frame can be used to classify the
current frame" (Das et al. 2024 §3.2).

DEVIATIONS from the published method are documented in notes/citi-baseline.md.
The one that matters: ARST fine-tunes ResNet-50 on the target data for 50
epochs; we cannot, because extraction discards the pixels (roadmap 1.7), so our
spatial stage trains only the 2048->512 projection on top of frozen
ImageNet features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512
N_HEADS = 8
BAND_WIDTH = 5      # ARST Table 2 optimum
TCN_LAYERS = 8      # dilated causal layers per TeCNO stage
CCI_N = 10          # consistency-constraint lookahead, ARST §2.3


def banded_causal_mask(length: int, width: int, device) -> torch.Tensor:
    """Additive attention mask, ARST eq. 1.

    Returns (L, L) with 0 where attention is permitted and -inf elsewhere.
    Query t may attend to keys [t-width, t] — never the future, and never
    further back than `width`. width=0 degenerates to self-only attention,
    the W=0 row of the paper's ablation.
    """
    i = torch.arange(length, device=device)
    delta = i[:, None] - i[None, :]                 # query index - key index
    ok = (delta >= 0) & (delta <= width)
    return torch.zeros(length, length, device=device).masked_fill(~ok, float("-inf"))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, ARST eq. 4-5 (Vaswani et al. 2017)."""

    def __init__(self, d_model: int = D_MODEL, max_len: int = 16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x (B, L, D). `offset` is the absolute position of x[:, 0] — needed
        because inference rolls out over a trailing window, not from frame 0."""
        return x + self.pe[offset:offset + x.size(1)].unsqueeze(0)


class SegmentedPhaseEmbedding(nn.Module):
    """Phase-label embedding, ARST eq. 3.

    A d_model vector is split into `num_classes` equal segments; phase i sets
    its whole segment to 1 and leaves the rest 0. With 15 classes and d=512
    each segment is 34 wide and the last 2 dimensions are always 0.

    The point of this over plain one-hot is distance: any two phases differ in
    2*seg coordinates rather than 2, so the decoder input carries a much
    stronger signal. It is a FIXED encoding, not learned — hence a buffer.

    Index `num_classes` is the start-of-sequence symbol (all zeros), used for
    the shifted decoder input at t=0 where no previous prediction exists.
    """

    def __init__(self, num_classes: int, d_model: int = D_MODEL):
        super().__init__()
        seg = d_model // num_classes
        table = torch.zeros(num_classes + 1, d_model)
        for i in range(num_classes):
            table[i, i * seg:(i + 1) * seg] = 1.0
        self.register_buffer("table", table, persistent=False)
        self.num_classes, self.seg = num_classes, seg

    @property
    def sos(self) -> int:
        return self.num_classes

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx]


# --------------------------------------------------------------------------
# Stage 1: spatial embedding (frozen ResNet-50 features -> 512-d)
# --------------------------------------------------------------------------

class SpatialEmbedding(nn.Module):
    """ARST §2.1: 2048-d pooled ResNet-50 feature -> 512-d Z_t, plus the
    frame-wise classifier head used to train it.

    In the paper the ResNet-50 is fine-tuned end to end and this projection is
    learned as part of it. We train the projection alone on cached frozen
    features — see notes/citi-baseline.md.
    """

    def __init__(self, in_dim: int = 2048, d_model: int = D_MODEL, num_classes: int = 15):
        super().__init__()
        self.project = nn.Linear(in_dim, d_model)
        self.classify = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.project(x)
        return z, self.classify(F.relu(z))


# --------------------------------------------------------------------------
# Stage 2: TeCNO (Czempiel et al. 2020) — two cascaded causal TCNs
# --------------------------------------------------------------------------

class CausalDilatedResidual(nn.Module):
    """One dilated causal residual layer. Left-padding only, so no frame ever
    sees its own future."""

    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation                     # kernel 3, causal
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.point = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv(F.pad(x, (self.pad, 0))))
        return x + self.drop(self.point(y))


class TCNStage(nn.Module):
    def __init__(self, in_ch: int, ch: int, num_classes: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, ch, 1)
        self.layers = nn.ModuleList(
            CausalDilatedResidual(ch, 2 ** i, dropout) for i in range(layers)
        )
        self.out = nn.Conv1d(ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return self.out(h), h                       # logits, hidden


class TeCNO(nn.Module):
    """Two-stage causal TCN. Stage 2 refines stage 1's class posteriors.

    8 layers with dilations 1,2,...,128 give each stage a causal receptive
    field of 1 + 2*(1+2+...+128) = 511 frames; cascading two stages roughly
    doubles it. At 1 fps that is ~17 minutes of history, which is why ARST can
    get away with a banded transformer mask of only W=5 on top — the long
    context is already in F_t.

    `hidden` from stage 2 is the 512-d F_t that feeds ARST (ARST §2.1).
    """

    def __init__(self, in_dim: int = D_MODEL, ch: int = D_MODEL,
                 num_classes: int = 15, layers: int = TCN_LAYERS, dropout: float = 0.5):
        super().__init__()
        self.stage1 = TCNStage(in_dim, ch, num_classes, layers, dropout)
        self.stage2 = TCNStage(num_classes, ch, num_classes, layers, dropout)

    def forward(self, z: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """z (B, T, in_dim) -> ([stage1_logits, stage2_logits], F (B, T, ch))."""
        x = z.transpose(1, 2)
        l1, _ = self.stage1(x)
        l2, h2 = self.stage2(F.softmax(l1, dim=1))
        return [l1.transpose(1, 2), l2.transpose(1, 2)], h2.transpose(1, 2)


# --------------------------------------------------------------------------
# Stage 3: ARST
# --------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d_model)
        )
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.drop(a))
        return self.norm2(x + self.drop(self.ff(x)))


class DecoderLayer(nn.Module):
    """Masked self-attention over past phase embeddings, then cross-attention
    with Q from the decoder and K,V from the encoder (ARST §2.2)."""

    def __init__(self, d_model: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, mem: torch.Tensor,
                self_mask: torch.Tensor, cross_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.self_attn(y, y, y, attn_mask=self_mask, need_weights=False)
        y = self.norm1(y + self.drop(a))
        c, _ = self.cross_attn(y, mem, mem, attn_mask=cross_mask, need_weights=False)
        y = self.norm2(y + self.drop(c))
        return self.norm3(y + self.drop(self.ff(y)))


class ARST(nn.Module):
    """One-layer encoder-decoder transformer with banded causal masking.

    encode() is independent of the decoder, so inference computes the memory
    once for the whole video and then rolls the decoder forward frame by frame.
    """

    def __init__(self, num_classes: int = 15, d_model: int = D_MODEL,
                 heads: int = N_HEADS, width: int = BAND_WIDTH,
                 ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.width = width
        self.pos = PositionalEncoding(d_model)
        self.phase = SegmentedPhaseEmbedding(num_classes, d_model)
        self.encoder = EncoderLayer(d_model, heads, ff, dropout)
        self.decoder = DecoderLayer(d_model, heads, ff, dropout)
        self.head = nn.Linear(d_model, num_classes)

    def encode(self, feats: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """feats (B, T, D) -> memory (B, T, D)."""
        mask = banded_causal_mask(feats.size(1), self.width, feats.device)
        return self.encoder(self.pos(feats, offset), mask)

    def decode(self, mem: torch.Tensor, prev: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """mem (B, T, D); prev (B, T) shifted phase indices -> logits (B, T, C)."""
        mask = banded_causal_mask(mem.size(1), self.width, mem.device)
        y = self.pos(self.phase(prev), offset)
        return self.head(self.decoder(y, mem, mask, mask))

    def forward(self, feats: torch.Tensor, prev: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.decode(self.encode(feats, offset), prev, offset)

    def shift(self, labels: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing input: [SOS, y_0, ..., y_{T-2}] (ARST §2.2)."""
        sos = torch.full((labels.size(0), 1), self.phase.sos,
                         dtype=torch.long, device=labels.device)
        return torch.cat([sos, labels[:, :-1]], dim=1)
