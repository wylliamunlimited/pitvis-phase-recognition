#!/usr/bin/env python3
"""Blur the video region of a `pitvis-app` screenshot.

    uv run python scripts/blur_frame.py shot.png
    uv run python scripts/blur_frame.py shot.png --out docs/app-default.png
    uv run python scripts/blur_frame.py shot.png --preview     # outline only
    uv run python scripts/blur_frame.py shot.png --box 40,90,1180,660

**Why this exists.** The PitVis dataset is CC BY-NC-ND 4.0 — attribution,
non-commercial, and *no derivatives*. A screenshot with the app's interface
composited around a surgical frame is arguably a derivative work, which is a
grey area not worth arguing in public. Blurring the frame removes the question.

Very little is lost. What a reader is meant to look at is the step card, the
confidence readout and the timeline; the anatomy is backdrop. A softened frame
still reads unmistakably as "endoscopic video is playing here".

The default region is found by **detection, not by hardcoded pixels**: the
video element is the one large near-black rectangle on a light ground, so the
script looks for it and falls back to the layout proportions in `app.css` if
it cannot find one. Always eyeball `--preview` before committing an image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# Fractions of the window, from app.css: header 54px, timeline >=86px, rail
# 340px, stage padding 20px. Only used when detection fails.
FALLBACK = (0.012, 0.075, 0.760, 0.700)      # x, y, w, h
DARK = 70                                    # luma below this counts as "video"


def detect(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the largest dark region — the letterboxed video."""
    g = img.convert("L")
    w, h = g.size
    # Downsample first: we want the shape of a 1000px rectangle, not per-pixel
    # accuracy, and this makes the scan trivial rather than clever.
    step = max(1, min(w, h) // 240)
    small = g.resize((w // step, h // step))
    px = small.load()
    sw, sh = small.size

    xs, ys = [], []
    for y in range(sh):
        for x in range(sw):
            if px[x, y] < DARK:
                xs.append(x)
                ys.append(y)
    if len(xs) < (sw * sh) * 0.04:           # too little dark area to be video
        return None

    box = (min(xs) * step, min(ys) * step,
           (max(xs) + 1) * step, (max(ys) + 1) * step)
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw < w * 0.25 or bh < h * 0.25:       # not a big rectangle; distrust it
        return None
    return box


def fallback_box(img: Image.Image) -> tuple[int, int, int, int]:
    w, h = img.size
    x, y, fw, fh = FALLBACK
    return (int(x * w), int(y * h), int((x + fw) * w), int((y + fh) * h))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, help="default: <name>-blurred.png")
    ap.add_argument("--radius", type=int, default=28, help="blur radius (28)")
    ap.add_argument("--box", help="x,y,w,h in pixels — overrides detection")
    ap.add_argument("--preview", action="store_true",
                    help="outline the region instead of blurring it")
    args = ap.parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"not found: {args.image}")

    img = Image.open(args.image).convert("RGB")

    if args.box:
        try:
            x, y, w, h = (int(v) for v in args.box.split(","))
        except ValueError:
            raise SystemExit("--box wants four integers: x,y,w,h") from None
        box, how = (x, y, x + w, y + h), "given"
    else:
        found = detect(img)
        box, how = (found, "detected") if found else (fallback_box(img), "fallback")

    print(f"image  {img.size[0]}x{img.size[1]}")
    print(f"region {box}  ({how})")
    if how == "fallback":
        print("       no dark rectangle found — check with --preview before using")

    out = args.out or args.image.with_name(f"{args.image.stem}-blurred.png")
    if args.preview:
        shown = img.copy()
        ImageDraw.Draw(shown).rectangle(box, outline=(255, 0, 0), width=4)
        out = args.out or args.image.with_name(f"{args.image.stem}-preview.png")
        shown.save(out)
    else:
        region = img.crop(box).filter(ImageFilter.GaussianBlur(args.radius))
        img.paste(region, box)
        img.save(out)

    print(f"wrote  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
