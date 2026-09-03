"""Reconcile boundaries from the two signals into a topic timeline.

The semantic pass and the LLM pass disagree in predictable ways: the semantic
pass fires on vocabulary shifts inside one topic, the LLM pass rounds
timestamps and misses transitions near a window edge. Merging them:

1. cluster boundaries that land within ``tolerance`` seconds of each other --
   the two signals agreeing on one edge is the strongest evidence available,
   so an agreed boundary keeps the LLM's time (it carries the label) and gets
   a confidence bonus;
2. drop boundaries that would leave a topic under ``min_seconds``, keeping
   whichever of the pair is more confident;
3. split any topic over ``max_seconds``, preferring an interior semantic
   candidate that was suppressed in step 2 over an arbitrary midpoint.

Everything here is pure: given boundaries in, topics out. That is what makes
the segmentation testable against fixture transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from ..logging_setup import get_logger

log = get_logger(__name__)

SOURCE_SEMANTIC = "semantic"
SOURCE_LLM = "llm"
SOURCE_BOTH = "both"
SOURCE_START = "session-start"
SOURCE_SPLIT = "max-length"


@dataclass
class Boundary:
    """A candidate topic edge on the stream timeline."""

    time: float
    source: str
    confidence: float = 0.5
    # Populated by the LLM pass for the topic that *starts* here.
    label: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)

    def merged_with(self, other: "Boundary") -> "Boundary":
        """Combine two boundaries the signals agree on.

        The LLM side supplies the time and the labelling; agreement raises
        confidence but is capped, since two imperfect signals agreeing is not
        certainty.
        """
        llm, sem = (self, other) if self.source == SOURCE_LLM else (other, self)
        primary = llm if llm.source == SOURCE_LLM else self
        return Boundary(
            time=primary.time,
            source=SOURCE_BOTH,
            confidence=min(0.95, max(self.confidence, other.confidence) + 0.25),
            label=llm.label or sem.label,
            summary=llm.summary or sem.summary,
            tags=list(llm.tags or sem.tags),
        )


@dataclass
class TopicSpan:
    """A finished topic: a time range with its labelling."""

    start: float
    end: float
    label: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    method: str = ""
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def cluster_boundaries(
    boundaries: Sequence[Boundary], tolerance: float
) -> list[Boundary]:
    """Collapse boundaries within ``tolerance`` seconds into one each."""
    if not boundaries:
        return []
    ordered = sorted(boundaries, key=lambda b: b.time)
    clusters: list[Boundary] = [ordered[0]]
    for boundary in ordered[1:]:
        if boundary.time - clusters[-1].time <= tolerance:
            clusters[-1] = clusters[-1].merged_with(boundary)
        else:
            clusters.append(boundary)
    return clusters


def enforce_min_length(
    boundaries: Sequence[Boundary], start: float, end: float, min_seconds: float
) -> tuple[list[Boundary], list[Boundary]]:
    """Drop boundaries that would create a topic shorter than ``min_seconds``.

    Returns ``(kept, dropped)``. The dropped list is reused as split
    candidates when a topic later turns out to be too long -- a boundary that
    was merely inconveniently close is still better evidence than a midpoint.
    """
    kept: list[Boundary] = []
    dropped: list[Boundary] = []
    previous = start

    for boundary in sorted(boundaries, key=lambda b: b.time):
        if boundary.time <= start or boundary.time >= end:
            dropped.append(boundary)
            continue
        if boundary.time - previous < min_seconds:
            # Too close behind. Keep the stronger of the two edges.
            if kept and boundary.confidence > kept[-1].confidence:
                dropped.append(kept[-1])
                # Replacing the last edge must not orphan the one before it.
                floor = kept[-2].time if len(kept) >= 2 else start
                if boundary.time - floor >= min_seconds:
                    kept[-1] = boundary
                    previous = boundary.time
                    continue
                kept.pop()
                previous = floor
            dropped.append(boundary)
            continue
        kept.append(boundary)
        previous = boundary.time

    # A final topic below the minimum: fold it into its predecessor.
    while kept and end - kept[-1].time < min_seconds:
        dropped.append(kept.pop())

    return kept, dropped


def enforce_max_length(
    boundaries: Sequence[Boundary],
    start: float,
    end: float,
    max_seconds: float,
    min_seconds: float,
    fallbacks: Sequence[Boundary] = (),
) -> list[Boundary]:
    """Split any span longer than ``max_seconds``.

    Prefers a previously dropped boundary sitting legally inside the span;
    only when none exists does it cut at an even division.
    """
    edges = sorted(boundaries, key=lambda b: b.time)
    spare = sorted(fallbacks, key=lambda b: b.time)
    result: list[Boundary] = []

    marks = [start, *(b.time for b in edges), end]
    for index in range(len(marks) - 1):
        span_start, span_end = marks[index], marks[index + 1]
        if index > 0:
            result.append(edges[index - 1])
        if span_end - span_start <= max_seconds:
            continue

        cursor = span_start
        while span_end - cursor > max_seconds:
            target = cursor + max_seconds
            candidate = _best_split(
                spare, lower=cursor + min_seconds, upper=min(target, span_end - min_seconds)
            )
            if candidate is not None:
                result.append(replace(candidate, source=SOURCE_SPLIT))
                cursor = candidate.time
                continue

            # No usable evidence: divide the remainder evenly so we do not
            # leave a runt at the end.
            remaining = span_end - cursor
            pieces = max(2, int(remaining // max_seconds) + 1)
            step = remaining / pieces
            for piece in range(1, pieces):
                result.append(
                    Boundary(
                        time=cursor + step * piece,
                        source=SOURCE_SPLIT,
                        confidence=0.1,
                        label="",
                    )
                )
            break

    result.sort(key=lambda b: b.time)
    return result


def _best_split(
    candidates: Sequence[Boundary], lower: float, upper: float
) -> Boundary | None:
    """Highest-confidence candidate in ``[lower, upper]``, latest on a tie."""
    usable = [b for b in candidates if lower <= b.time <= upper]
    if not usable:
        return None
    return max(usable, key=lambda b: (b.confidence, b.time))


def merge_boundaries(
    semantic: Iterable[Boundary],
    llm: Iterable[Boundary],
    start: float,
    end: float,
    tolerance: float,
    min_seconds: float,
    max_seconds: float,
) -> list[Boundary]:
    """The full reconciliation: cluster, enforce minimum, enforce maximum."""
    if end <= start:
        return []

    clustered = cluster_boundaries([*semantic, *llm], tolerance)
    kept, dropped = enforce_min_length(clustered, start, end, min_seconds)
    final = enforce_max_length(
        kept, start, end, max_seconds, min_seconds, fallbacks=dropped
    )

    log.debug(
        "merge.boundaries",
        extra={
            "clustered": len(clustered),
            "after_min": len(kept),
            "final": len(final),
            "span": round(end - start, 1),
        },
    )
    return final


def boundaries_to_topics(
    boundaries: Sequence[Boundary],
    start: float,
    end: float,
    default_label: str = "Untitled topic",
) -> list[TopicSpan]:
    """Turn edges into spans. Each boundary labels the topic it opens."""
    if end <= start:
        return []

    opening = Boundary(time=start, source=SOURCE_START, confidence=1.0)
    edges = [opening, *sorted(boundaries, key=lambda b: b.time)]

    spans: list[TopicSpan] = []
    for index, edge in enumerate(edges):
        span_end = edges[index + 1].time if index + 1 < len(edges) else end
        if span_end <= edge.time:
            continue
        spans.append(
            TopicSpan(
                start=edge.time,
                end=span_end,
                label=edge.label or default_label,
                summary=edge.summary,
                tags=list(edge.tags),
                method=edge.source,
                confidence=edge.confidence,
            )
        )
    return spans
