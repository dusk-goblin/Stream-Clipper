"""Highlight ranking inside a topic.

Sweeps clip-length windows across a topic, scores each on chat rate, emote
spike and (optionally) LLM-rated clippability, then picks the best
non-overlapping windows.

The sweep is deliberately dumb and the selection is greedy: quality comes
from the signals, and keeping the search simple means a fixture transcript
plus a fixture chat log fully determine the output, which is what the tests
check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..config import HighlightConfig
from ..logging_setup import get_logger
from ..transcribe.transcript import Sentence
from .chat_signals import ChatProfile, emote_rate, message_rate, squash, zscores

log = get_logger(__name__)


@dataclass
class Candidate:
    """A window inside a topic, with its component scores."""

    start: float
    end: float
    chat_score: float = 0.0
    emote_score: float = 0.0
    llm_score: float = 0.0
    score: float = 0.0
    text: str = ""
    title: str = ""
    reason: str = ""
    raw: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "Candidate") -> bool:
        return self.start < other.end and self.end > other.start

    def breakdown(self) -> dict[str, float]:
        return {
            "chat": round(self.chat_score, 4),
            "emote": round(self.emote_score, 4),
            "llm": round(self.llm_score, 4),
            "total": round(self.score, 4),
            **{k: round(v, 4) for k, v in self.raw.items()},
        }


def sweep_windows(
    topic_start: float, topic_end: float, config: HighlightConfig
) -> list[tuple[float, float]]:
    """Candidate windows across a topic.

    Windows are generated at both the minimum and maximum clip length so a
    tight punchline and a longer bit both get a chance; the ranking decides
    which shape actually fits the moment.
    """
    duration = topic_end - topic_start
    if duration < config.clip_min_seconds:
        return []

    lengths = {config.clip_min_seconds, config.clip_max_seconds}
    midpoint = (config.clip_min_seconds + config.clip_max_seconds) / 2.0
    lengths.add(midpoint)

    windows: list[tuple[float, float]] = []
    for length in sorted(lengths):
        if length > duration:
            continue
        cursor = topic_start
        while cursor + length <= topic_end + 1e-6:
            windows.append((cursor, cursor + length))
            cursor += config.stride_seconds
    return windows


def snap_to_speech(
    start: float, end: float, sentences: Sequence[Sentence], config: HighlightConfig
) -> tuple[float, float]:
    """Nudge a window to sentence edges so a clip does not open mid-word.

    Only moves the edges if doing so keeps the clip inside the configured
    length bounds -- a clean cut is not worth a clip that is too short to
    stand alone or too long to hold attention.
    """
    if not sentences:
        return start, end

    starts = [s.start for s in sentences if abs(s.start - start) <= 4.0]
    new_start = min(starts, key=lambda t: abs(t - start)) if starts else start

    ends = [s.end for s in sentences if abs(s.end - end) <= 4.0]
    new_end = min(ends, key=lambda t: abs(t - end)) if ends else end

    length = new_end - new_start
    if config.clip_min_seconds <= length <= config.clip_max_seconds:
        return new_start, new_end
    return start, end


def text_for(sentences: Sequence[Sentence], start: float, end: float) -> str:
    return " ".join(
        s.text for s in sentences if s.start < end and s.end > start
    ).strip()


def score_candidates(
    windows: Sequence[tuple[float, float]],
    profile: ChatProfile,
    config: HighlightConfig,
) -> list[Candidate]:
    """Score windows on the chat signals alone (the LLM pass comes later)."""
    if not windows:
        return []

    rates = [message_rate(profile, s, e) for s, e in windows]
    emotes = [emote_rate(profile, s, e) for s, e in windows]

    chat_z = zscores(rates)
    emote_z = zscores(emotes)

    candidates: list[Candidate] = []
    for index, (start, end) in enumerate(windows):
        candidates.append(
            Candidate(
                start=start,
                end=end,
                chat_score=squash(chat_z[index]),
                emote_score=squash(emote_z[index]),
                raw={"msgs_per_sec": rates[index], "emotes_per_sec": emotes[index]},
            )
        )
    return candidates


def combine(candidate: Candidate, config: HighlightConfig, has_llm: bool) -> float:
    """Weighted total, renormalised when the LLM signal is unavailable.

    Without renormalisation a missing LLM score would silently cap every
    candidate at the sum of the remaining weights, pushing everything under
    ``min_score`` and emitting no clips at all.
    """
    weights = config.weights.normalised()
    if has_llm:
        return (
            weights.chat_rate * candidate.chat_score
            + weights.emote_spike * candidate.emote_score
            + weights.llm * candidate.llm_score
        )
    denominator = weights.chat_rate + weights.emote_spike
    if denominator <= 0:
        return candidate.chat_score
    return (
        weights.chat_rate * candidate.chat_score
        + weights.emote_spike * candidate.emote_score
    ) / denominator


def select(
    candidates: Sequence[Candidate], config: HighlightConfig
) -> list[Candidate]:
    """Greedy non-overlapping pick, best first, honouring ``per_topic``."""
    chosen: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if candidate.score < config.min_score:
            break
        if any(candidate.overlaps(other) for other in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= config.per_topic:
            break
    chosen.sort(key=lambda c: c.start)
    return chosen


def rank_topic(
    topic_start: float,
    topic_end: float,
    sentences: Sequence[Sentence],
    profile: ChatProfile,
    config: HighlightConfig,
    llm_scores: dict[int, float] | None = None,
    llm_titles: dict[int, str] | None = None,
    llm_reasons: dict[int, str] | None = None,
) -> list[Candidate]:
    """Full ranking for one topic.

    ``llm_scores`` is keyed by candidate index into the *scored* list, which
    is what ``attach_llm_scores`` produces. Pass None to rank on chat alone.
    """
    windows = sweep_windows(topic_start, topic_end, config)
    candidates = score_candidates(windows, profile, config)
    if not candidates:
        return []

    has_llm = bool(llm_scores)
    for index, candidate in enumerate(candidates):
        if llm_scores and index in llm_scores:
            candidate.llm_score = llm_scores[index]
        if llm_titles and index in llm_titles:
            candidate.title = llm_titles[index]
        if llm_reasons and index in llm_reasons:
            candidate.reason = llm_reasons[index]
        candidate.score = combine(candidate, config, has_llm)

    chosen = select(candidates, config)
    for candidate in chosen:
        candidate.start, candidate.end = snap_to_speech(
            candidate.start, candidate.end, sentences, config
        )
        candidate.text = text_for(sentences, candidate.start, candidate.end)

    log.debug(
        "rank.topic",
        extra={
            "windows": len(windows),
            "selected": len(chosen),
            "llm": has_llm,
            "top": round(chosen[0].score, 3) if chosen else 0.0,
        },
    )
    return chosen
