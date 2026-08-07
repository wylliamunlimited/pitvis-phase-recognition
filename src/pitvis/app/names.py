"""Display names and the step colour ramp — one source, shared by CSS and canvas.

Names are imported, never redefined: `STEP_NAMES`/`step_name` from
`data.dataset`, `INSTRUMENT_NAMES` from `evaluation.instruments`. This module
adds only what is purely presentational.

**The ramp is one hue, not fifteen.** A distinct colour per step would be a
rainbow: sixteen hues carrying no relationship, where adjacent steps look
maximally unlike and the eye has nothing to hold onto. Instead every step gets
the same hue at a rising lightness, so a timeline reads left-to-right as a
brightening and the *shape* of an operation — where the boundaries fall, how
long each stage ran — is legible before a single label is read.

Identity is carried by the number, which every segment renders as text. Colour
carries structure. That split is what lets the whole palette stay two colours
instead of sixteen, and it means the display degrades gracefully for the ~8% of
men with colour-vision deficiency, since nothing depends on hue discrimination.
"""

from __future__ import annotations

import colorsys

from pitvis.data.dataset import STEP_NAMES, step_name  # noqa: F401  (re-exported)
from pitvis.evaluation.instruments import INSTRUMENT_NAMES

# Raw challenge encoding: background is -1, then steps 1..14 in surgical order.
STEP_ORDER = [-1, *range(1, 15)]

_HUE = 200 / 360          # a cool slate; the endoscopic image is warm
_SAT = 0.20               # barely chromatic — a greyscale with a cast
# On a light ground the ramp DARKENS as the operation proceeds, so a case reads
# left-to-right as a deepening. Inverted from what a dark ground would want.
_L_LO, _L_HI = 0.79, 0.42  # step 1 -> step 14
_L_BACKGROUND = 0.91      # paler than any real step: absence reads as absence


def _hex(h: float, s: float, ll: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, ll, s)
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def step_ramp() -> dict[int, str]:
    """Raw step label -> hex. Keys are ints; JSON will stringify them."""
    ramp = {-1: _hex(_HUE, _SAT * 0.4, _L_BACKGROUND)}
    for k in range(1, 15):
        t = (k - 1) / 13
        ramp[k] = _hex(_HUE, _SAT, _L_LO + t * (_L_HI - _L_LO))
    return ramp


def payload() -> dict:
    """Everything the frontend needs to name and colour a label."""
    return {
        "steps": {str(k): step_name(k, raw=True) for k in STEP_ORDER},
        "instruments": {str(k): v for k, v in INSTRUMENT_NAMES.items()},
        "step_order": STEP_ORDER,
        "ramp": {str(k): v for k, v in step_ramp().items()},
    }
