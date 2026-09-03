"""ffprobe queries.

Keyframe positions are the important one: a stream copy can only start on a
keyframe, so knowing where they are is what lets us decide between a fast
copy and a frame-accurate re-encode.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from ..logging_setup import get_logger
from ..util.proc import require_binary, run

log = get_logger(__name__)


def media_duration(path: str | Path) -> float:
    """Container duration in seconds, 0.0 when ffprobe cannot tell."""
    try:
        result = run(
            [
                require_binary("ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            timeout=60,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        log.debug("ffprobe.duration_failed", extra={"path": str(path)})
        return 0.0


def video_dimensions(path: str | Path) -> tuple[int, int]:
    """(width, height), or (0, 0) if unknown."""
    try:
        result = run(
            [
                require_binary("ffprobe"),
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(path),
            ],
            timeout=60,
        )
        streams = json.loads(result.stdout).get("streams") or []
        if streams:
            return int(streams[0].get("width", 0)), int(streams[0].get("height", 0))
    except (subprocess.SubprocessError, ValueError, OSError, KeyError):
        log.debug("ffprobe.dimensions_failed", extra={"path": str(path)})
    return (0, 0)


@lru_cache(maxsize=64)
def keyframe_times(path: str, window_start: float = 0.0, window_end: float = 0.0) -> tuple[float, ...]:
    """Keyframe timestamps in ``path``, sorted.

    Reading keyframes for a whole 5-minute segment is cheap with
    ``-skip_frame nokey``; the optional window narrows it further via
    ``-read_intervals`` so cutting near the end of a long file does not scan
    from the start. Results are cached because several clips often land in
    the same segment.
    """
    args = [
        require_binary("ffprobe"),
        "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
    ]
    if window_end > window_start:
        # A little slack each side so the keyframe *before* the window is seen.
        args += ["-read_intervals", f"{max(0.0, window_start - 10):.3f}%{window_end + 10:.3f}"]
    args += [
        "-show_entries", "frame=pts_time",
        "-of", "csv=print_section=0",
        str(path),
    ]

    try:
        result = run(args, timeout=180)
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("ffprobe.keyframes_failed", extra={"path": path, "reason": str(exc)[:200]})
        return ()

    times: list[float] = []
    for line in result.stdout.splitlines():
        value = line.strip().rstrip(",")
        if not value:
            continue
        try:
            times.append(float(value))
        except ValueError:
            continue
    times.sort()
    log.debug("ffprobe.keyframes", extra={"path": Path(path).name, "count": len(times)})
    return tuple(times)
