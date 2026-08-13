"""The route table. Handlers are pure `(Request) -> Response`.

No sockets here and no HTTP mechanics — those live in `server.py`. Keeping the
two apart is what makes the stdlib server a reversible decision rather than a
commitment: swapping in an ASGI framework replaces `server.py` alone.

Errors carry a `hint` holding the exact command that would fix them. That is
the convention the CLI already uses — `predict.py` raises `SystemExit("...
train a model first:\\n    uv run pitvis-train arst")` — carried over a
different transport rather than reinvented.

Four routes are declared but return 501. They are the seams for the interactive
work this iteration deliberately stops short of (roadmap 5.6-5.9). Declaring
the URL shape now costs nothing and means the frontend, the docs and any future
client agree on it before anything is built against a guess.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pitvis.app import catalogue, media, names
from pitvis.app.server import (
    FileResponse,
    Request,
    Response,
    StreamResponse,
    serve_static,
)
from pitvis.paths import APP_ASSETS


def _json(payload, status: int = 200) -> Response:
    return Response(status, json.dumps(payload).encode(), "application/json")


def _err(status: int, code: str, message: str, **extra) -> Response:
    return _json({"error": {"code": code, "message": message, **extra}}, status)


NOT_IMPLEMENTED = {
    "/corrections": "human-in-the-loop step correction (roadmap 5.7)",
    "/explain": "agentic explanation layer (roadmap 5.4)",
    "/compare": "multi-case comparison (roadmap 5.9)",
    "/live": "streaming input (roadmap 5.8)",
}


def handle(req: Request):
    path = req.path.rstrip("/") or "/"

    if path == "/" and req.method in ("GET", "HEAD"):
        return serve_static(APP_ASSETS, "index.html")

    if path.startswith("/static/") and req.method in ("GET", "HEAD"):
        return serve_static(APP_ASSETS, path[len("/static/"):])

    if path == "/api/names":
        return _json(names.payload())

    if path == "/api/cases":
        return _json({"cases": [c.to_json() for c in catalogue.cases().values()],
                      "cache_state": catalogue.cache_state()})

    m = re.fullmatch(r"/api/cases/([^/]+)(/[^/]*)?", path)
    if m:
        return _case_route(req, m.group(1), m.group(2) or "")

    m = re.fullmatch(r"/api/jobs/([^/]+)(/events)?", path)
    if m:
        return _job_route(req, m.group(1), bool(m.group(2)))

    for prefix, what in NOT_IMPLEMENTED.items():
        if path.startswith("/api" + prefix) or prefix in path:
            return _err(501, "not_implemented", f"{what} is not built yet")

    return _err(404, "not_found", f"no route for {req.path}")


def _case_route(req: Request, case_id: str, sub: str):
    ref = catalogue.get(case_id)
    if ref is None:
        return _err(404, "unknown_case", f"no case {case_id!r}")

    if sub in ("", "/"):
        from pitvis.app.case import build_case
        try:
            return _json(build_case(ref))
        except FileNotFoundError as exc:
            return _err(404, "no_prediction", str(exc), hint=_predict_hint(ref))

    if sub == "/video":
        return FileResponse(ref.video, "video/mp4")

    if sub == "/frame":
        try:
            t = int(req.query.get("t", "0"))
        except ValueError:
            return _err(400, "bad_request", "t must be an integer second")
        return Response(200, media.frame_jpeg(ref.video, max(0, t)), "image/jpeg",
                        gzip_ok=False)

    if sub == "/instrument_probs":
        from pitvis.app.case import instrument_probs
        probs = instrument_probs(ref)
        if probs is None:
            return _err(404, "no_probs", "no instrument_probs.npy for this case",
                        hint=_predict_hint(ref))
        return _json({"threshold": None, "probs": probs})

    if sub == "/predict" and req.method == "POST":
        from pitvis.app import jobs
        if not ref.features_cached:
            if catalogue.cache_state() == "legacy":
                # The features exist; the cache just predates the per-space
                # layout. Telling someone to sit through a 20-minute decode
                # when a rename would do is the wrong instruction, not a
                # rounding error.
                return _err(409, "cache_not_migrated",
                            "features for every case are on disk but still in "
                            "the pre-space cache layout, so nothing can find "
                            "them. Migrating is a rename, not a re-extraction.",
                            hint="uv run pitvis-extract --migrate")
            return _err(409, "not_cached",
                        "this video has no cached features, so predicting it "
                        "means a full 1 fps decode (10-25 min). Run it yourself.",
                        hint=_predict_hint(ref))
        job = jobs.submit(ref)
        return _json({"job_id": job.id, "state": job.state}, 202)

    return _err(404, "not_found", f"no route for {req.path}")


def _job_route(req: Request, job_id: str, events: bool):
    from pitvis.app import jobs
    job = jobs.get(job_id)
    if job is None:
        return _err(404, "unknown_job", f"no job {job_id!r}")
    if events:
        return StreamResponse(jobs.stream(job))
    try:
        since = int(req.query.get("since", "0"))
    except ValueError:
        since = 0
    return _json(job.snapshot(since))


def _predict_hint(ref) -> str:
    """The exact command that produces what this case is missing."""
    parts = ["uv run pitvis-predict", "--video", _rel(ref.video)]
    if ref.truth:
        parts += ["--labels", _rel(ref.truth)]
    parts.append("--probs")
    return " ".join(parts)


def _rel(p: Path) -> str:
    from pitvis.paths import ROOT
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)
