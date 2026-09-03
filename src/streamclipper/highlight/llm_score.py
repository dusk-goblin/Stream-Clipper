"""Apply the LLM clippability pass to ranked candidates.

Sending every swept window to the model would be wasteful and mostly
redundant -- neighbouring windows overlap heavily. So the chat signals
pre-filter: only the strongest, mutually distinct candidates are sent, and
the rest keep a neutral score.
"""

from __future__ import annotations

from typing import Sequence

from ..config import HighlightConfig
from ..logging_setup import get_logger
from ..segment.llm import LLMClient
from ..transcribe.transcript import Sentence
from .rank import Candidate, text_for

log = get_logger(__name__)

# Score given to candidates the LLM never saw. Mid-scale, so an unrated
# candidate is neither promoted nor buried relative to a rated one.
NEUTRAL = 0.4


def shortlist(
    candidates: Sequence[Candidate], limit: int, min_separation: float
) -> list[int]:
    """Indices of the best chat-scoring candidates, spread out in time."""
    order = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i].chat_score + candidates[i].emote_score,
        reverse=True,
    )
    picked: list[int] = []
    for index in order:
        candidate = candidates[index]
        if all(
            abs(candidate.start - candidates[other].start) >= min_separation
            for other in picked
        ):
            picked.append(index)
        if len(picked) >= limit:
            break
    return sorted(picked)


def attach_llm_scores(
    candidates: Sequence[Candidate],
    sentences: Sequence[Sentence],
    topic_label: str,
    llm: LLMClient,
    config: HighlightConfig,
) -> tuple[dict[int, float], dict[int, str], dict[int, str]]:
    """Score a shortlist. Returns ``(scores, titles, reasons)`` by index.

    An empty scores dict means the pass did not run, which the ranker reads as
    "fall back to chat-only weighting".
    """
    if not config.llm.enabled or not candidates or not llm.available:
        return {}, {}, {}

    indices = shortlist(
        candidates,
        limit=config.llm.max_candidates,
        min_separation=max(config.stride_seconds * 2, config.clip_min_seconds / 2),
    )
    payload = [
        (
            index,
            candidates[index].start,
            candidates[index].end,
            text_for(sentences, candidates[index].start, candidates[index].end),
        )
        for index in indices
    ]
    payload = [entry for entry in payload if entry[3].strip()]
    if not payload:
        return {}, {}, {}

    rated = llm.score_clippability(payload, topic_label)
    if not rated:
        return {}, {}, {}

    scores: dict[int, float] = {}
    titles: dict[int, str] = {}
    reasons: dict[int, str] = {}
    for index in range(len(candidates)):
        entry = rated.get(index)
        if entry is None:
            scores[index] = NEUTRAL
            continue
        scores[index] = entry.score
        if entry.title:
            titles[index] = entry.title
        if entry.reason:
            reasons[index] = entry.reason
    return scores, titles, reasons
