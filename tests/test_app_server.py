"""Pins the server's connection-teardown handling. See `server.py` for why it
exists; this file is about the two directions that must both hold.

The risk in silencing browser teardown is over-correcting into a blanket
`except Exception`, which would hide real faults in the one component with no
framework underneath it. So: teardowns silent, everything else still loud.

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


def test_everything_else_still_reports():
    """One case is enough: every non-teardown takes the same `super()` branch,
    so a second exception type would exercise no new code."""
    out = handle(ValueError("a real bug"))
    assert "Exception occurred during processing of request" in out
    assert "ValueError" in out


def test_teardown_list_is_narrow():
    """Guard against someone widening this to OSError.

    ConnectionResetError et al. are OSError subclasses, so `except OSError`
    would look like it works while also swallowing a missing video file, a
    permission error on the feature cache, and every other genuine I/O fault.
    The subclass assertion also rules out Exception, which is not an OSError.
    """
    assert OSError not in TEARDOWN
    assert all(issubclass(e, OSError) for e in TEARDOWN)
