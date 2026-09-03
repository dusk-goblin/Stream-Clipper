"""Transcript shaping.

The transcript is stored as whisper utterances plus words at absolute stream
time. Downstream stages want *sentences*: units long enough to embed
meaningfully, each carrying a start and end. Livestream speech is rarely
punctuated cleanly, so this splits on punctuation where it exists and falls
back to whisper's own utterance breaks where it does not, then merges
fragments up to a minimum length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..state.models import Utterance, Word
from ..util.timefmt import hhmmss

# Sentence terminator followed by whitespace, keeping the terminator.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Sentence:
    """A unit of transcript with its own time span."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def _split_utterance(utterance: Utterance) -> list[Sentence]:
    """One utterance -> one or more sentences, with times interpolated."""
    text = utterance.text.strip()
    if not text:
        return []

    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    if len(parts) <= 1:
        return [
            Sentence(
                start=utterance.start,
                end=utterance.end,
                text=text,
                words=list(utterance.words),
            )
        ]

    words = list(utterance.words)
    sentences: list[Sentence] = []
    cursor = 0  # index into `words` as we hand them out in order
    span = max(1e-6, utterance.end - utterance.start)
    total_chars = sum(len(p) for p in parts)
    char_seen = 0

    for part in parts:
        if words:
            # Give this sentence as many words as its text accounts for.
            take = max(1, round(len(part) / max(1, total_chars) * len(words)))
            chunk = words[cursor : cursor + take]
            if not chunk:  # ran out; fall through to interpolation
                chunk = words[-1:]
            cursor = min(len(words), cursor + take)
            sentences.append(
                Sentence(
                    start=chunk[0].start, end=chunk[-1].end, text=part, words=chunk
                )
            )
        else:
            start = utterance.start + span * (char_seen / max(1, total_chars))
            char_seen += len(part)
            end = utterance.start + span * (char_seen / max(1, total_chars))
            sentences.append(Sentence(start=start, end=max(end, start + 0.01), text=part))

    # Any words the rounding left over belong to the last sentence.
    if words and cursor < len(words):
        sentences[-1].words.extend(words[cursor:])
        sentences[-1].end = max(sentences[-1].end, words[-1].end)
    return sentences


def _is_fragment(text: str, min_chars: int, fragment_chars: int) -> bool:
    """Whether a unit is too incomplete to stand as its own sentence.

    A short unit that ends in a terminator ("It was wild.") is a real short
    sentence and keeps its own boundary. A short unit that does not
    ("yeah", "so anyway") is a fragment and gets merged forward. Very short
    units are fragments either way -- "Right." carries no topical signal.
    """
    stripped = text.strip()
    if len(stripped) >= min_chars:
        return False
    if len(stripped) < fragment_chars:
        return True
    return not stripped.endswith((".", "!", "?", "\u2026"))


def sentences_from_utterances(
    utterances: Sequence[Utterance],
    min_chars: int = 40,
    max_chars: int = 400,
    fragment_chars: int = 15,
) -> list[Sentence]:
    """Flatten utterances into sentences of a usable size.

    Fragments are merged forward, because a two-word backchannel ("yeah,
    right") embeds to noise and produces spurious topic boundaries. Merging
    stops at ``max_chars`` so one run-on utterance cannot swallow a real
    transition, and complete short sentences are left alone.
    """
    raw: list[Sentence] = []
    for utterance in utterances:
        raw.extend(_split_utterance(utterance))

    merged: list[Sentence] = []
    for sentence in raw:
        if (
            merged
            and _is_fragment(merged[-1].text, min_chars, fragment_chars)
            and len(merged[-1].text) + len(sentence.text) <= max_chars
        ):
            previous = merged[-1]
            previous.text = f"{previous.text} {sentence.text}".strip()
            previous.end = sentence.end
            previous.words.extend(sentence.words)
        else:
            merged.append(sentence)

    # A trailing fragment has nothing to merge forward into; fold it back.
    if len(merged) >= 2 and _is_fragment(merged[-1].text, min_chars, fragment_chars):
        tail = merged.pop()
        if len(merged[-1].text) + len(tail.text) <= max_chars:
            merged[-1].text = f"{merged[-1].text} {tail.text}".strip()
            merged[-1].end = tail.end
            merged[-1].words.extend(tail.words)
        else:
            merged.append(tail)

    return merged


def slice_sentences(
    sentences: Sequence[Sentence], start: float, end: float
) -> list[Sentence]:
    """Sentences whose midpoint falls in ``[start, end)``.

    Midpoint rather than overlap, so a sentence straddling a boundary lands on
    exactly one side and clip text never duplicates.
    """
    return [s for s in sentences if start <= s.midpoint < end]


def excerpt_for(
    utterances: Sequence[Utterance], start: float, end: float, max_chars: int = 1200
) -> str:
    """Readable transcript text for a time range, truncated on a word boundary."""
    pieces = [
        u.text.strip()
        for u in utterances
        if u.start < end and u.end > start and u.text.strip()
    ]
    text = " ".join(pieces).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars * 0.6 else cut).rstrip() + "…"


def format_timestamped(
    sentences: Sequence[Sentence], max_chars: int | None = None
) -> str:
    """``[HH:MM:SS] text`` lines -- the shape the LLM prompts consume.

    Absolute timestamps let the model answer with boundary times directly,
    rather than with offsets we would have to translate.
    """
    lines: list[str] = []
    used = 0
    for sentence in sentences:
        line = f"[{hhmmss(sentence.start)}] {sentence.text}"
        if max_chars is not None and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def total_span(sentences: Sequence[Sentence]) -> tuple[float, float]:
    if not sentences:
        return (0.0, 0.0)
    return (sentences[0].start, max(s.end for s in sentences))
