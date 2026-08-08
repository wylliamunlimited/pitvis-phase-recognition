Screenshots for the top-level README.

Capture from a running `uv run pitvis-app`, then blur the surgical frame before
committing — the dataset is CC BY-NC-ND 4.0 and a composited screenshot is
arguably a derivative work:

    uv run python scripts/blur_frame.py raw.png --preview        # check the region
    uv run python scripts/blur_frame.py raw.png --out docs/app-default.png

Expected files:

    app-default.png   the default view — step, instruments, one timeline strip
    app-detail.png    with [ + DETAIL ] on — confidence, truth, errors, tools
