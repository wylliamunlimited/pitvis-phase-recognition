"""Serve the review surface: `uv run pitvis-app`.

    uv run pitvis-app                          # opens on the first ready case
    uv run pitvis-app --case video_25          # a specific one
    uv run pitvis-app --video path/to/case.mp4 # a video outside 26531686/
    uv run pitvis-app --port 9000 --no-open

Plays a case and shows, second by second, the step the model believes the
operation is in and the instruments it believes are in view — against ground
truth where the annotations exist.

This runner deliberately does NOT use `pipeline.Stage`/`execute` like the other
four. That machinery sequences a finite list of stages and prints a timing
summary at the end; a server has one stage and blocks until Ctrl-C, so the
banner would be noise and the summary would never print. Left as a plain
argparse main on purpose — please don't "fix" it.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from pitvis.app import api, catalogue, server
from pitvis.paths import APP_ASSETS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--case", help="case id to open, e.g. video_25")
    ap.add_argument("--video", type=Path,
                    help="register a video outside 26531686/ and open it")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: loopback; changing this "
                         "exposes patient video to the network)")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--no-open", dest="open", action="store_false",
                    help="do not launch a browser")
    ap.add_argument("--verbose", action="store_true",
                    help="log every request, including each video range")
    args = ap.parse_args(argv)

    if not (APP_ASSETS / "index.html").exists():
        raise SystemExit(
            f"app assets missing at {APP_ASSETS}. This means the package was "
            f"built without its static files — reinstall with `uv sync`."
        )

    opening = args.case
    if args.video:
        if not args.video.exists():
            raise SystemExit(f"video not found: {args.video}")
        opening = catalogue.register(args.video)

    known = catalogue.cases()
    if not known:
        raise SystemExit(
            "no cases found. Expected videos at 26531686/video_NN.mp4, or pass "
            "--video path/to/case.mp4."
        )
    if opening and opening not in known:
        raise SystemExit(
            f"unknown case {opening!r}. Available: {', '.join(known)}"
        )
    if not opening:
        ready = [c for c in known.values() if c.prediction.get("available")]
        opening = (ready or list(known.values()))[0].case_id

    httpd = server.serve(api.handle, args.host, args.port, quiet=not args.verbose)
    url = f"http://{args.host}:{httpd.server_port}/?case={opening}"

    ready = sum(1 for c in known.values() if c.prediction.get("available"))
    cached = sum(1 for c in known.values() if c.features_cached)
    print(f"pitvis-app  {len(known)} case(s) — {ready} predicted, "
          f"{cached} with cached features")
    print(f"            opening {opening}")
    print(f"\n    {url}\n")
    print("Ctrl-C to stop.")

    if args.open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
