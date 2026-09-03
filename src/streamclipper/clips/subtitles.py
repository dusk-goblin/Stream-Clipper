"""Subtitle generation for burn-in.

Word timings are grouped into short caption cues rather than one cue per
utterance: a 20-second sentence as a single caption is unreadable on a phone.
Cues are clipped to the clip's own time range and rebased to zero, because
the burned-in file is applied to the cut clip, not the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..state.models import Word
from ..util.timefmt import srt_time


@dataclass
class Cue:
    start: float
    end: float
    text: str


def build_cues(
    words: Sequence[Word],
    clip_start: float,
    clip_end: float,
    max_chars: int = 42,
    max_seconds: float = 3.0,
    gap_break: float = 0.7,
) -> list[Cue]:
    """Group words into readable cues, rebased to the clip's own timeline."""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    cues: list[Cue] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(w.text for w in current).strip()
        if text:
            cues.append(
                Cue(
                    start=max(0.0, current[0].start - clip_start),
                    end=max(0.05, min(current[-1].end, clip_end) - clip_start),
                    text=text,
                )
            )
        current.clear()

    for word in inside:
        if current:
            pending = " ".join(w.text for w in current)
            too_long = len(pending) + 1 + len(word.text) > max_chars
            too_slow = word.end - current[0].start > max_seconds
            paused = word.start - current[-1].end > gap_break
            if too_long or too_slow or paused:
                flush()
        current.append(word)
    flush()

    # Never let a cue outlive the next one's start.
    for index in range(len(cues) - 1):
        cues[index].end = min(cues[index].end, cues[index + 1].start)
    return [c for c in cues if c.end > c.start]


def render_srt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{index}\n{srt_time(cue.start)} --> {srt_time(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def write_srt(
    path: Path, words: Sequence[Word], clip_start: float, clip_end: float
) -> Path | None:
    """Write an SRT for a clip. Returns None when there is nothing to caption."""
    cues = build_cues(words, clip_start, clip_end)
    if not cues:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(cues), encoding="utf-8")
    return path
