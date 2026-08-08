"""Pins the server's connection-teardown handling.

A <video> element holds several keep-alive connections open and resets them on
every seek, on buffer-full, and on navigate-away. The serving thread takes that
reset inside `handle_one_request` -> `rfile.readline()`, which sits above every
try/except in `server.py`, so socketserver's default `handle_error` prints a
full traceback for each one. The symptom is pages of `ConnectionResetError`
from an app that is working perfectly.

The risk in fixing it is over-correcting into a blanket `except Exception`,
which would hide real faults in exactly the component that has no framework
underneath it. Hence both directions are pinned here: teardowns silent,
everything else still loud.

No sockets — `handle_error` only reads the in-flight exception, so the server
is instantiated without binding a port.
"""

import contextlib
import io

import pytest

from pitvis.app.server import TEARDOWN, Server


def handle(exc: BaseException) -> str:
    """Run `Server.handle_error` with `exc` in flight; return captured stderr."""
    server = object.__new__(Server)          # no socket, no bind, no cleanup
    buf = io.StringIO()
    try:
        raise exc
    except BaseException:
        with contextlib.redirect_stderr(buf):
            server.handle_error(None, ("127.0.0.1", 56689))
    return buf.getvalue()


@pytest.mark.parametrize("exc", [
    ConnectionResetError(54, "Connection reset by peer"),   # the observed one
    BrokenPipeError(32, "Broken pipe"),
    ConnectionAbortedError(53, "Software caused connection abort"),
    TimeoutError("timed out"),
])
def test_teardown_is_silent(exc):
    assert handle(exc) == ""


@pytest.mark.parametrize("exc", [
    ValueError("a real bug"),
    KeyError("missing"),
    RuntimeError("boom"),
])
def test_everything_else_still_reports(exc):
    out = handle(exc)
    assert "Exception occurred during processing of request" in out
    assert type(exc).__name__ in out


def test_teardown_list_is_narrow():
    """Guard against someone widening this to OSError or Exception.

    ConnectionResetError et al. are OSError subclasses, so `except OSError`
    would look like it works while also swallowing a missing video file, a
    permission error on the feature cache, and every other genuine I/O fault.
    """
    assert OSError not in TEARDOWN
    assert Exception not in TEARDOWN
    assert all(issubclass(e, OSError) for e in TEARDOWN)
