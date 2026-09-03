"""Semantic-drift boundary detection.

A TextTiling-style pass over sentence embeddings. For every gap between
consecutive sentences, compare the centroid of the ``w`` sentences before it
against the ``w`` after. Where the topic holds, those two block vectors point
the same way; where it turns, similarity falls into a valley.

Two details matter more than the core idea:

**Full windows only.** A gap near the start or end of the transcript has a
lopsided comparison -- two sentences against eight -- which reads as low
similarity for reasons that have nothing to do with topic. Those gaps are
scored but never offered as boundaries.

**Scale-free thresholds.** Absolute cosine values are not comparable across
backends: within-topic similarity runs around 0.6 for sentence-transformer
vectors and around 0.2 for TF-IDF. So the profile is normalised to its own
observed range before thresholding, and both configured thresholds are read
against that normalised series. ``similarity_threshold: 0.55`` therefore
means "in the lower 55% of the similarity range this stream actually
exhibited", which behaves the same whichever backend produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import SemanticConfig
from ..logging_setup import get_logger
from ..transcribe.transcript import Sentence
from .embeddings import Embedder, Vector, cosine, mean

log = get_logger(__name__)

# Below this spread the transcript has no drift to speak of and every
# "boundary" would be normalisation noise.
_MIN_RANGE = 1e-6


@dataclass
class GapScore:
    """One inter-sentence gap and how strongly it looks like a boundary."""

    index: int          # gap sits between sentence[index-1] and sentence[index]
    time: float         # stream seconds at the gap
    similarity: float   # raw cosine between the two block centroids
    normalised: float = 0.0  # similarity rescaled to the profile's own range
    depth: float = 0.0  # valley depth, measured on the normalised series
    full_window: bool = True


def similarity_profile(
    sentences: Sequence[Sentence],
    embeddings: Sequence[Vector],
    window: int,
) -> list[GapScore]:
    """Block similarity, normalised similarity and valley depth at every gap.

    Exposed separately from boundary selection so thresholds can be tuned
    against a real stream without re-embedding it.
    """
    n = len(sentences)
    if n < 2 or len(embeddings) != n:
        return []

    gaps: list[GapScore] = []
    for i in range(1, n):
        lo, hi = max(0, i - window), min(n, i + window)
        left = mean(embeddings[lo:i])
        right = mean(embeddings[i:hi])
        # The gap sits in the silence between two sentences; use its middle so
        # a cut lands in the pause rather than clipping a word.
        gap_time = (sentences[i - 1].end + sentences[i].start) / 2.0
        gaps.append(
            GapScore(
                index=i,
                time=gap_time,
                similarity=cosine(left, right),
                full_window=(i - lo >= window and hi - i >= window),
            )
        )

    _normalise(gaps)
    _score_depths(gaps)
    return gaps


def _normalise(gaps: list[GapScore]) -> None:
    """Rescale similarity to 0..1 across the profile's own observed range.

    Only full-window gaps set the range; the lopsided ones at either end
    would otherwise drag the minimum down and flatten everything else.
    """
    scored = [g.similarity for g in gaps if g.full_window] or [g.similarity for g in gaps]
    low, high = min(scored), max(scored)
    spread = high - low
    for gap in gaps:
        if spread < _MIN_RANGE:
            gap.normalised = 1.0     # no drift anywhere: nothing is a boundary
        else:
            gap.normalised = min(1.0, max(0.0, (gap.similarity - low) / spread))


def _score_depths(gaps: list[GapScore]) -> None:
    """Depth = how far this gap sits below the nearest peak on each side.

    Walk outward from each gap while similarity keeps rising; the value where
    it stops rising is that side's local peak. A gap that is not a local
    minimum scores zero, which is what keeps a long monotonic slide from
    registering as a string of boundaries.
    """
    n = len(gaps)
    for i, gap in enumerate(gaps):
        left_peak = gap.normalised
        j = i - 1
        while j >= 0 and gaps[j].normalised >= left_peak:
            left_peak = gaps[j].normalised
            j -= 1

        right_peak = gap.normalised
        k = i + 1
        while k < n and gaps[k].normalised >= right_peak:
            right_peak = gaps[k].normalised
            k += 1

        gap.depth = (left_peak - gap.normalised) + (right_peak - gap.normalised)


def semantic_boundaries(
    sentences: Sequence[Sentence],
    embedder: Embedder,
    config: SemanticConfig,
    min_gap_seconds: float = 0.0,
) -> list[GapScore]:
    """Gaps that look like topic changes, in time order.

    ``min_gap_seconds`` suppresses boundaries that sit too close to a stronger
    one -- a single transition often dips similarity across two or three
    consecutive gaps, and only the deepest is the real edge.
    """
    if len(sentences) < 2:
        return []

    embeddings = embedder.encode([s.text for s in sentences])
    gaps = similarity_profile(sentences, embeddings, config.window_sentences)

    candidates = [
        gap
        for gap in gaps
        if gap.full_window
        and gap.normalised <= config.similarity_threshold
        and gap.depth >= config.depth_threshold
    ]

    # Non-maximum suppression by depth.
    kept: list[GapScore] = []
    for gap in sorted(candidates, key=lambda g: g.depth, reverse=True):
        if all(abs(gap.time - other.time) >= min_gap_seconds for other in kept):
            kept.append(gap)

    kept.sort(key=lambda g: g.time)
    log.debug(
        "semantic.boundaries",
        extra={
            "sentences": len(sentences),
            "gaps": len(gaps),
            "candidates": len(candidates),
            "kept": len(kept),
        },
    )
    return kept
