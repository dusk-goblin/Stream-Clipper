"""Stream-time helpers.

All internal timestamps are *stream seconds*: floats measured from the moment
a recording session started. Wall-clock only enters at the edges (chat
messages arrive with a wall timestamp, manifests carry ISO dates).
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_ts() -> float:
    """Wall-clock UNIX seconds."""
    return datetime.now(timezone.utc).timestamp()


def iso(ts: float) -> str:
    """UNIX seconds -> ISO 8601 UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def hhmmss(seconds: float, millis: bool = False) -> str:
    """Stream seconds -> HH:MM:SS(.mmm), clamped at zero."""
    seconds = max(0.0, seconds)
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if millis:
        ms = int(round((seconds - whole) * 1000))
        if ms == 1000:  # rounding pushed us to the next second
            ms, secs = 0, secs + 1
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def srt_time(seconds: float) -> str:
    """Stream seconds -> SRT timestamp (comma decimal separator)."""
    return hhmmss(seconds, millis=True).replace(".", ",")


def parse_hhmmss(value: str) -> float:
    """'1:02:03.5', '02:03', '123' -> seconds. Raises ValueError otherwise."""
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    sign = 1.0
    if text.startswith("-"):
        sign, text = -1.0, text[1:]
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"not a timestamp: {value!r}")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return sign * total


def vod_offset(seconds: float) -> str:
    """Stream seconds -> the ``?t=`` fragment Twitch VOD URLs use."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    out = ""
    if hours:
        out += f"{hours}h"
    if hours or minutes:
        out += f"{minutes}m"
    return out + f"{secs}s"
