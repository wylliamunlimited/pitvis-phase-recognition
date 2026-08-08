"""HTTP mechanics. This module knows nothing about surgery.

Everything domain-shaped lives in `api.py` as pure `handle(Request) -> Response`
functions; this file does sockets, Range, gzip, SSE framing and static files.
The split is the escape hatch: if this ever needs to become a real ASGI app,
`api.py` is untouched and only this file is replaced.

Why the stdlib and not starlette+uvicorn. The one thing a framework would have
given us is a Range-capable file response — and Range on a multi-gigabyte
non-faststart file is the single most load-bearing behaviour in this app, so it
gets hand-written and unit-tested either way. Once that test exists the
framework buys nothing: there is no auth, no forms, no validation, no
templating, and the expensive work is a torch rollout on a worker thread, which
async does not help. Zero new dependencies for eight routes.

The stdlib does come with sharp edges, all handled below and all of which
produce mystifying symptoms if missed:

- `BaseHTTPRequestHandler` speaks **HTTP/1.0** by default, closing the socket
  after every response. A video issues 100+ range requests, so that is 100+ TCP
  handshakes, and an SSE stream never streams at all. Overriding
  `protocol_version` fixes it but makes an accurate `Content-Length` mandatory
  on every response, or the client hangs waiting for bytes that never come.
- A `<video>` element aborts its outstanding range request on every seek, which
  surfaces as `BrokenPipeError` mid-write. Unhandled, the console fills with
  tracebacks and a perfectly healthy app looks like it is crashing.
- The same teardown also arrives on the **read** side, and that one cannot be
  caught where the writes are. Keep-alive leaves the thread parked in
  `handle_one_request` -> `rfile.readline()`, above every try/except here, so
  socketserver's default `handle_error` prints the traceback. `Server.
  handle_error` filters exactly the teardown errors and nothing else.
- Without `Host` validation, any web page the user visits can reach this server
  by DNS rebinding and stream patient video off loopback. Binding to 127.0.0.1
  is not sufficient protection on its own.
"""

from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import parse_qs, unquote, urlsplit

# A browser tearing down a connection is routine, not an error. One definition,
# used by every guard in this file: the per-write `except`s below and
# Server.handle_error, which catches the same teardowns arriving on the read
# side. Stays narrow and explicit -- all four subclass OSError, so the tempting
# `except OSError` would also swallow a missing video and every permission
# fault in the one component with no framework beneath it.
TEARDOWN = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
            TimeoutError)

CHUNK = 256 * 1024
MAX_RANGE = 8 * 1024 * 1024      # cap on an open-ended `bytes=N-`
GZIP_MIN = 1024

FULL = "full"                    # serve the whole entity, 200
UNSATISFIABLE = "unsatisfiable"  # 416


# --------------------------------------------------------------------------
# Range


def parse_range(header: str | None, size: int, max_span: int = MAX_RANGE):
    """`Range` header -> (start, end) inclusive, or FULL, or UNSATISFIABLE.

    A pure function so it can be tested without a socket, which matters: this
    is the code path that decides whether a 1 GB video plays at all.

    What browsers actually send for a moov-at-end MP4, in order: `bytes=0-`
    (Chrome) or a two-byte probe then `bytes=0-` (Safari); then a tail range
    for the index, either `bytes=<near-end>-` or the suffix form `bytes=-N`;
    then a fresh `bytes=N-` on every seek. So the open-ended form dominates and
    the suffix form must work.

    Capping an open-ended range at `max_span` is conformant — a server may
    return fewer bytes than asked for — and it is what keeps seeking responsive:
    8 MiB is ~34 s of PitVis video, so a playthrough is a sequence of short
    responses rather than one gigabyte-long transfer aborted on every seek.
    """
    if not header:
        return FULL
    header = header.strip()
    if not header.startswith("bytes="):
        return FULL                    # RFC 7233: an unknown unit MAY be ignored
    spec = header[6:].strip()
    if "," in spec or "-" not in spec:
        return FULL                    # multi-range is legal and never sent here
    first, _, last = spec.partition("-")
    first, last = first.strip(), last.strip()
    try:
        if not first:                  # suffix form: the last N bytes
            n = int(last)
            if n <= 0:
                return UNSATISFIABLE
            start, end = max(0, size - n), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
    except ValueError:
        return FULL
    if start < 0 or start >= size or start > end:
        return UNSATISFIABLE
    return start, min(end, size - 1, start + max_span - 1)


# --------------------------------------------------------------------------
# Request / Response


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: bytes = b""


@dataclass
class Response:
    """A complete in-memory response."""

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    gzip_ok: bool = True


@dataclass
class FileResponse:
    """A file served with Range support. Never gzipped."""

    path: Path
    content_type: str = "application/octet-stream"
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamResponse:
    """An open-ended byte stream — server-sent events.

    Length is unknown, so this is the one response that cannot use keep-alive.
    """

    chunks: Iterator[bytes]
    content_type: str = "text/event-stream"
    headers: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Handler


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pitvis"
    sys_version = ""

    # set by serve()
    dispatch: Callable[[Request], object]
    quiet: bool = True

    def do_GET(self) -> None:      # noqa: N802
        self._run()

    def do_HEAD(self) -> None:     # noqa: N802
        self._run()

    def do_POST(self) -> None:     # noqa: N802
        self._run()

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)

    def log_error(self, fmt: str, *args) -> None:
        # Client-side aborts (a seek cancelling a range request) are normal.
        if not self.quiet:
            super().log_error(fmt, *args)

    def _host_ok(self) -> bool:
        """Reject anything not addressed to loopback by name.

        Without this the same-origin policy is defeasible by DNS rebinding: a
        page on any origin resolves its own hostname to 127.0.0.1, and its
        fetches arrive here as same-origin requests carrying that hostname.
        Comparing the Host header against the names we actually listen on is
        what closes it.
        """
        host = self.headers.get("Host", "")
        if not host:
            return False
        if host.startswith("["):                        # [::1]:8420
            name, _, rest = host.partition("]")
            name, port = name[1:], rest.lstrip(":")
        else:
            name, _, port = host.partition(":")
        if name not in ("127.0.0.1", "localhost", "::1"):
            return False
        return not port or port == str(self.server.server_port)

    def _run(self) -> None:
        if not self._host_ok():
            self._send(Response(403, b'{"error":{"code":"bad_host",'
                                     b'"message":"loopback only"}}'))
            return
        split = urlsplit(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        req = Request(
            method=self.command,
            path=unquote(split.path),
            query={k: v[0] for k, v in parse_qs(split.query).items()},
            headers={k.lower(): v for k, v in self.headers.items()},
            body=self.rfile.read(length) if length else b"",
        )
        try:
            resp = self.dispatch(req)
        except Exception as exc:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            resp = Response(500, _error("internal", str(exc)))

        if isinstance(resp, FileResponse):
            self._send_file(resp)
        elif isinstance(resp, StreamResponse):
            self._send_stream(resp)
        else:
            self._send(resp)

    def _write(self, data: bytes) -> bool:
        """True if the bytes went out. False means the client hung up."""
        try:
            self.wfile.write(data)
            return True
        except TEARDOWN:
            self.close_connection = True
            return False

    def _send(self, resp: Response) -> None:
        body = resp.body
        headers = dict(resp.headers)
        if (resp.gzip_ok and len(body) >= GZIP_MIN
                and "gzip" in self.headers.get("Accept-Encoding", "")):
            body = gzip.compress(body, 6)
            headers["Content-Encoding"] = "gzip"
            headers["Vary"] = "Accept-Encoding"
        headers.setdefault("Content-Type", resp.content_type)
        headers["Content-Length"] = str(len(body))
        self.send_response(resp.status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self._write(body)

    def _send_file(self, resp: FileResponse) -> None:
        try:
            size = resp.path.stat().st_size
        except OSError:
            self._send(Response(404, _error("not_found", str(resp.path))))
            return

        rng = parse_range(self.headers.get("Range"), size)
        headers = dict(resp.headers)
        headers["Content-Type"] = resp.content_type
        headers["Accept-Ranges"] = "bytes"
        # Without this Chrome pins gigabytes of patient video in its disk cache.
        # Re-fetching over loopback costs nothing.
        headers["Cache-Control"] = "no-store"

        if rng is UNSATISFIABLE:
            headers["Content-Range"] = f"bytes */{size}"
            headers["Content-Length"] = "0"
            self.send_response(416)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            return

        if rng is FULL:
            start, end, status = 0, size - 1, 200
        else:
            start, end = rng
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"

        span = end - start + 1
        headers["Content-Length"] = str(span)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return

        try:
            with resp.path.open("rb") as f:
                f.seek(start)
                _copy_span(f, self.wfile, span)
        except TEARDOWN:
            self.close_connection = True

    def _send_stream(self, resp: StreamResponse) -> None:
        self.close_connection = True          # length unknown; no keep-alive
        self.send_response(200)
        self.send_header("Content-Type", resp.content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        for k, v in resp.headers.items():
            self.send_header(k, v)
        self.end_headers()
        for chunk in resp.chunks:
            if not self._write(chunk):
                return
            try:
                self.wfile.flush()
            except TEARDOWN:
                return


def _copy_span(src, dst, span: int) -> None:
    left = span
    while left > 0:
        block = src.read(min(CHUNK, left))
        if not block:
            return
        dst.write(block)
        left -= len(block)


def _error(code: str, message: str, hint: str | None = None) -> bytes:
    import json
    payload = {"error": {"code": code, "message": message}}
    if hint:
        payload["error"]["hint"] = hint
    return json.dumps(payload).encode()


# --------------------------------------------------------------------------


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        """Swallow the normal ways a browser drops a connection.

        The `except TEARDOWN` guards further up cover writes — the response
        being aborted mid-body. They cannot cover the *read* side: with
        keep-alive on (`protocol_version = "HTTP/1.1"`) a <video> holds several
        connections open and resets them on seek, on buffer-full, and on
        navigate-away. The serving thread is then parked in
        `handle_one_request` -> `rfile.readline()`, which is above every
        try/except in this module, so socketserver catches it and its default
        `handle_error` prints a full traceback per reset.

        The symptom is alarming and the cause is nothing: scrolling
        `ConnectionResetError` tracebacks from an app that is working
        perfectly. One seek can produce several.

        Everything that is not a teardown still gets the default traceback —
        this must not become a blanket suppressor.
        """
        if isinstance(sys.exception(), TEARDOWN):
            return
        super().handle_error(request, client_address)


def serve(dispatch, host: str, port: int, quiet: bool = True) -> Server:
    """Bind and return a server. Caller runs `serve_forever`."""
    handler = type("PitVisHandler", (Handler,),
                   {"dispatch": staticmethod(dispatch), "quiet": quiet})
    return Server((host, port), handler)


def serve_static(root: Path, rel: str) -> Response | FileResponse:
    """Serve `rel` from under `root`, refusing anything that escapes it."""
    import mimetypes
    target = (root / rel.lstrip("/")).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return Response(403, _error("forbidden", "path escapes the asset root"))
    if not target.is_file():
        return Response(404, _error("not_found", rel))
    ctype, _ = mimetypes.guess_type(target.name)
    body = target.read_bytes()
    return Response(200, body, ctype or "application/octet-stream",
                    {"Cache-Control": "no-cache"})
