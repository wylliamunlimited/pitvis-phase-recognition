"""Pins HTTP Range parsing — the code path that decides whether a case plays.

Every PitVis video has its `moov` index at the END of the file (box order
ftyp, free, mdat, moov), so a browser cannot begin playback until it has
fetched a tail range. Get this wrong and the symptom is not a subtle bug: the
video element shows a black rectangle and never recovers.

The numbers below are `26531686/video_25.mp4` as measured, not invented:

    total   1,067,824,728 bytes
    moov    1,338,182 bytes at offset 1,066,486,546

No sockets here — `parse_range` is a pure function, which is exactly why it was
written as one.
"""

import pytest

from pitvis.app.server import FULL, MAX_RANGE, UNSATISFIABLE, parse_range

SIZE = 1_067_824_728          # video_25.mp4, real
MOOV_AT = 1_066_486_546
MOOV_LEN = 1_338_182


# -- the forms browsers actually send --------------------------------------

def test_no_header_is_a_full_response():
    assert parse_range(None, SIZE) is FULL
    assert parse_range("", SIZE) is FULL


def test_chrome_opener_is_open_ended_from_zero():
    """`bytes=0-` — the dominant form, and the one a naive server gets wrong."""
    assert parse_range("bytes=0-", SIZE) == (0, MAX_RANGE - 1)


def test_safari_probes_two_bytes_first():
    assert parse_range("bytes=0-1", SIZE) == (0, 1)


def test_suffix_range_fetches_the_moov_tail():
    """`bytes=-1338182` must land exactly on the index at the end of the file."""
    assert parse_range(f"bytes=-{MOOV_LEN}", SIZE) == (MOOV_AT, SIZE - 1)


def test_seek_is_open_ended_from_the_seek_point():
    start = 533_000_000
    assert parse_range(f"bytes={start}-", SIZE) == (start, start + MAX_RANGE - 1)


def test_explicit_span():
    assert parse_range("bytes=100-199", SIZE) == (100, 199)


# -- clamping ---------------------------------------------------------------

def test_end_is_clamped_to_the_last_byte():
    """An end past EOF clamps to the final byte — checked inside the cap, so
    it is the file length doing the clamping and not `max_span`."""
    near = SIZE - 1000
    assert parse_range(f"bytes={near}-99999999999", SIZE) == (near, SIZE - 1)


def test_suffix_longer_than_the_file_starts_at_zero():
    """A suffix bigger than the file must clamp the START to 0, never negative.

    The end then falls under the usual cap, like any other open-ended request.
    """
    assert parse_range(f"bytes=-{SIZE * 2}", SIZE) == (0, MAX_RANGE - 1)
    assert parse_range(f"bytes=-{SIZE * 2}", 4096) == (0, 4095)


def test_open_ended_ranges_are_capped():
    """Returning fewer bytes than asked for is conformant, and deliberate.

    Uncapped, `bytes=0-` streams a gigabyte that the client aborts the instant
    anyone touches the scrubber.
    """
    assert parse_range("bytes=0-", SIZE, max_span=1024) == (0, 1023)


def test_the_cap_never_extends_a_short_explicit_range():
    assert parse_range("bytes=0-9", SIZE, max_span=1024) == (0, 9)


# -- refusals ---------------------------------------------------------------

def test_start_past_the_end_is_unsatisfiable():
    assert parse_range(f"bytes={SIZE}-", SIZE) is UNSATISFIABLE
    assert parse_range("bytes=99999999999-", SIZE) is UNSATISFIABLE


def test_zero_length_suffix_is_unsatisfiable():
    assert parse_range("bytes=-0", SIZE) is UNSATISFIABLE


def test_reversed_range_is_unsatisfiable():
    assert parse_range("bytes=200-100", SIZE) is UNSATISFIABLE


# -- degrade to a full response, never to an error --------------------------

@pytest.mark.parametrize("header", [
    "bytes=0-99,200-299",     # multi-range: legal, but no browser sends it here
    "items=0-99",             # an unknown unit — RFC 7233 says it MAY be ignored
    "bytes=abc-def",
    "bytes=",
    "nonsense",
])
def test_forms_we_do_not_serve_fall_back_to_the_whole_entity(header):
    assert parse_range(header, SIZE) is FULL


# -- static assets ----------------------------------------------------------

def test_static_serving_refuses_to_escape_the_asset_root(tmp_path):
    from pitvis.app.server import serve_static

    (tmp_path / "app.css").write_text("body{}")
    (tmp_path.parent / "secret.txt").write_text("private")

    assert serve_static(tmp_path, "app.css").status == 200
    assert serve_static(tmp_path, "../secret.txt").status == 403
    assert serve_static(tmp_path, "/../secret.txt").status == 403
    assert serve_static(tmp_path, "nope.css").status == 404
