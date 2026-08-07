"""Video metadata and single-frame extraction, both via ffmpeg.

`ffprobe`/`ffmpeg` are hard requirements of this repo already (`inventory.py`
and `extract_features.py` shell out to them), so nothing new is being demanded
here — but the app must fail with the same fixable message rather than a
FileNotFoundError traceback, so it reuses `predict.require_ffmpeg`.

Probing is memoised by (path, mtime, size): a probe costs ~0.4 s and the case
document is rebuilt on every request, so 25 videos would otherwise make the
catalogue route unusable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_probe_cache: dict[tuple[str, int, int], dict] = {}


def probe(video: Path) -> dict:
    """Container facts for one video: duration, geometry, fps, faststart."""
    st = video.stat()
    key = (str(video.resolve()), int(st.st_mtime), st.st_size)
    if key in _probe_cache:
        return _probe_cache[key]

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    meta = json.loads(out)
    stream = (meta.get("streams") or [{}])[0]
    num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
    fps = int(num) / int(den) if int(den or 0) else 0.0

    info = {
        "bytes": st.st_size,
        "duration_s": float(meta.get("format", {}).get("duration") or 0.0),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "faststart": _faststart(video),
    }
    _probe_cache[key] = info
    return info


def _faststart(video: Path) -> bool:
    """True when `moov` precedes `mdat` — i.e. the file can stream naively.

    Every PitVis video answers False: the box order is ftyp, free, mdat, moov,
    so the 1.3 MB index sits at the very end of a multi-gigabyte file and the
    browser cannot begin playback until it has fetched a tail range. That is
    the whole reason `server.py` implements HTTP Range rather than serving the
    file whole, and it is reported in the case document so the fact is visible
    rather than folklore.
    """
    import struct
    size = video.stat().st_size
    off = 0
    with video.open("rb") as f:
        while off < size:
            f.seek(off)
            head = f.read(16)
            if len(head) < 8:
                return False
            box = struct.unpack(">I", head[:4])[0]
            kind = head[4:8]
            if box == 1:
                if len(head) < 16:
                    return False
                box = struct.unpack(">Q", head[8:16])[0]
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if box <= 0:
                return False
            off += box
    return False


def frame_jpeg(video: Path, t: int, width: int = 640) -> bytes:
    """One JPEG at second `t`.

    Not used by the current UI. It exists because the agentic explanation layer
    (roadmap 5.4) needs to hand individual frames to a vision model, and the
    seam is worth having settled while the surrounding code is being written.

    `-ss` before `-i` seeks by keyframe, which is approximate but takes ~0.2 s
    on a 3 GB file rather than minutes of decoding.
    """
    from pitvis.inference.predict import require_ffmpeg
    require_ffmpeg()
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(int(t)), "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={int(width)}:-1",
         "-f", "image2", "-c:v", "mjpeg", "-"],
        capture_output=True, check=True,
    )
    return out.stdout
