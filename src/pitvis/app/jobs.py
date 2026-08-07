"""Run inference on request, and stream its console output to the browser.

The work itself is `pitvis.inference.run.main(argv)` called in-process. That is
the repo's orchestration rule applied to a server: `data/run.py` composes stages
as `lambda: extract_features.main(argv)` rather than reimplementing them, and
the same applies here. There is one definition of "predict a video", so the app
and `pitvis-predict` cannot drift.

**`report()` printing to stdout is a feature here, not an obstacle.** The
official scoring table, the leaked-class note and the instrument
column-divergence warning all stream into the browser verbatim, in the exact
wording the CLI uses. Nothing parses that text — the structured numbers come
from `evaluate()` in `case.py`. The log is the log.

**Exactly one worker, permanently.** `redirect_stdout` swaps a process-global
`sys.stdout`, so two concurrent jobs would interleave their output into each
other's logs; and two torch rollouts would contend for one MPS device, which is
slower than running them in sequence. This is not a throughput knob — please
don't turn it into a pool.

Progress is milestones, not a percentage. `pitvis-predict` prints a cache-hit
line, then decodes silently for ~40 s, then prints a timing line. There is no
honest fraction to report, so the UI shows the log, a coarse phase inferred
from line prefixes, and elapsed time. Inventing a progress bar here would mean
adding instrumentation to `inference/run.py` for the app's benefit, which is
the app leaking into the pipeline.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

from pitvis.app.catalogue import CaseRef

MAX_LINES = 4000
MAX_SUBSCRIBERS = 8          # ThreadingHTTPServer spawns threads without bound
KEEPALIVE_S = 15.0

_ids = itertools.count(1)
_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}
_queue: "queue.Queue[Job]" = queue.Queue()
_worker: threading.Thread | None = None


@dataclass
class Job:
    id: str
    case_id: str
    argv: list[str]
    state: str = "queued"            # queued | running | done | failed
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    _subs: list["queue.Queue[tuple[str, str]]"] = field(default_factory=list)

    def emit(self, kind: str, payload: str) -> None:
        with _lock:
            if kind == "line":
                self.lines.append(payload)
                del self.lines[:-MAX_LINES]
            subs = list(self._subs)
        for q in subs:
            with contextlib.suppress(queue.Full):
                q.put_nowait((kind, payload))

    def set_state(self, state: str) -> None:
        self.state = state
        self.emit("state", state)

    def snapshot(self, since: int = 0) -> dict:
        with _lock:
            lines = self.lines[since:]
            cursor = len(self.lines)
        return {
            "id": self.id, "case_id": self.case_id, "state": self.state,
            "lines": lines, "cursor": cursor, "error": self.error,
            "returncode": self.returncode,
            "elapsed_s": round((self.finished_at or time.time())
                               - self.started_at, 1) if self.started_at else None,
        }


class _Tee(io.TextIOBase):
    """Split stdout: through to the real console, and out as SSE lines."""

    def __init__(self, job: Job, real):
        self.job, self.real, self.buf = job, real, ""

    def write(self, s: str) -> int:
        with contextlib.suppress(Exception):
            self.real.write(s)
            self.real.flush()
        self.buf += s
        while "\n" in self.buf:
            line, _, self.buf = self.buf.partition("\n")
            self.job.emit("line", line)
        return len(s)

    def flush(self) -> None:
        pass


def submit(ref: CaseRef) -> Job:
    """Queue a prediction for `ref`, or return the one already in flight."""
    with _lock:
        for job in _jobs.values():
            if job.case_id == ref.case_id and job.state in ("queued", "running"):
                return job

    argv = ["--video", str(ref.video), "--probs"]
    if ref.truth:
        argv += ["--labels", str(ref.truth)]

    job = Job(id=f"j{next(_ids)}", case_id=ref.case_id, argv=argv)
    with _lock:
        _jobs[job.id] = job
    _queue.put(job)
    _ensure_worker()
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="pitvis-jobs",
                                       daemon=True)
            _worker.start()


def _loop() -> None:
    while True:
        job = _queue.get()
        _run(job)


def _run(job: Job) -> None:
    from pitvis.inference import run as inference_run

    job.started_at = time.time()
    job.set_state("running")
    job.emit("line", f"$ uv run pitvis-predict {' '.join(job.argv)}")
    tee = _Tee(job, sys.stdout)
    try:
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            job.returncode = inference_run.main(job.argv)
    except SystemExit as exc:
        # main() exits this way for a missing video or an untrained checkpoint,
        # and the message it carries already IS the "run this instead" hint.
        job.returncode = exc.code if isinstance(exc.code, int) else 1
        job.error = str(exc)
    except Exception as exc:                        # noqa: BLE001
        import traceback
        job.error = f"{type(exc).__name__}: {exc}"
        for line in traceback.format_exc().splitlines():
            job.emit("line", line)
        job.returncode = 1
    finally:
        if tee.buf:
            job.emit("line", tee.buf)
        job.finished_at = time.time()
        job.set_state("done" if job.returncode == 0 and not job.error else "failed")


def stream(job: Job):
    """Server-sent events for one job. Ends when the job does."""
    q: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=2048)
    with _lock:
        if len(job._subs) >= MAX_SUBSCRIBERS:
            yield b"event: error\ndata: too many listeners\n\n"
            return
        backlog = list(job.lines)
        state = job.state
        job._subs.append(q)

    try:
        for line in backlog:                       # so a late listener catches up
            yield _sse("line", line)
        yield _sse("state", state)
        if state in ("done", "failed"):
            yield _sse("done", job.state)
            return

        while True:
            try:
                kind, payload = q.get(timeout=KEEPALIVE_S)
            except queue.Empty:
                yield b": keepalive\n\n"
                continue
            yield _sse(kind, payload)
            if kind == "state" and payload in ("done", "failed"):
                yield _sse("done", payload)
                return
    finally:
        with _lock:
            if q in job._subs:
                job._subs.remove(q)


def _sse(event: str, data: str) -> bytes:
    # Every line of a multi-line payload needs its own `data:` prefix.
    body = "".join(f"data: {ln}\n" for ln in data.split("\n"))
    return f"event: {event}\n{body}\n".encode()
