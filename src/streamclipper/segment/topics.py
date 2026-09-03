"""Topic segmentation orchestration.

Runs both signals over the settled part of the transcript and commits the
resulting topics to the database. Committing is watermark-driven so it is
safe to call repeatedly while a stream is still running: only the region
between ``session.topics_watermark`` and the settle point is considered, and
the topic still in progress at the settle point is left uncommitted unless it
has already outgrown ``max_topic_seconds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import Config
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import Topic
from ..transcribe.transcript import Sentence, sentences_from_utterances, slice_sentences
from .embeddings import Embedder, get_embedder
from .llm import LLMClient
from .merge import (
    SOURCE_LLM,
    SOURCE_SEMANTIC,
    Boundary,
    boundaries_to_topics,
    merge_boundaries,
)
from .semantic import semantic_boundaries

log = get_logger(__name__)


@dataclass
class SegmentationResult:
    topics: list[Topic]
    watermark: float
    semantic_count: int = 0
    llm_count: int = 0


class TopicSegmenter:
    """Owns the embedder and the LLM client for a run."""

    def __init__(
        self,
        config: Config,
        db: Database,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self._embedder = embedder
        self._llm = llm

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.config.segment.embeddings)
        return self._embedder

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.config.llm)
        return self._llm

    # -- signals ----------------------------------------------------------

    def _semantic(self, sentences: Sequence[Sentence]) -> list[Boundary]:
        config = self.config.segment
        gaps = semantic_boundaries(
            sentences,
            self.embedder,
            config.semantic,
            min_gap_seconds=config.min_topic_seconds,
        )
        return [
            Boundary(
                time=gap.time,
                source=SOURCE_SEMANTIC,
                # Depth is unbounded above; a valley twice the threshold is
                # already as much evidence as this signal can give.
                confidence=min(
                    0.9, 0.35 + gap.depth / max(1e-6, config.semantic.depth_threshold * 4)
                ),
            )
            for gap in gaps
        ]

    def _llm_boundaries(
        self, sentences: Sequence[Sentence], start: float, end: float
    ) -> list[Boundary]:
        config = self.config.segment.llm
        if not config.enabled or not self.llm.available:
            return []

        boundaries: list[Boundary] = []
        window_start = start
        step = max(60.0, config.window_seconds - config.overlap_seconds)

        while window_start < end:
            window_end = min(end, window_start + config.window_seconds)
            window = slice_sentences(sentences, window_start, window_end)
            if window:
                for topic in self.llm.topic_boundaries(window, window_start, window_end):
                    # The first topic of a window is usually the one already in
                    # progress, not a new edge.
                    if topic.start <= window_start + 1.0:
                        continue
                    boundaries.append(
                        Boundary(
                            time=topic.start,
                            source=SOURCE_LLM,
                            confidence=topic.confidence,
                            label=topic.label,
                            summary=topic.summary,
                            tags=topic.tags,
                        )
                    )
            if window_end >= end:
                break
            window_start += step

        return boundaries

    # -- entry point ------------------------------------------------------

    def segment_session(self, session_id: int, final: bool = False) -> SegmentationResult:
        """Segment the settled transcript and commit whole topics.

        ``final=True`` (capture finished) commits everything to the end of the
        transcript, including the last topic.
        """
        session = self.db.get_session(session_id)
        if session is None:
            return SegmentationResult([], 0.0)

        config = self.config.segment
        start = session.topics_watermark
        transcribed = self.db.transcribed_seconds(session_id)
        end = transcribed if final else transcribed - config.settle_seconds

        if end - start < config.min_topic_seconds:
            log.debug(
                "segment.too_early",
                extra={
                    "session_id": session_id,
                    "available": round(max(0.0, end - start), 1),
                    "need": config.min_topic_seconds,
                },
            )
            return SegmentationResult([], start)

        utterances = self.db.utterances(session_id, start, end)
        sentences = sentences_from_utterances(utterances)
        if len(sentences) < 2:
            return SegmentationResult([], start)

        semantic = self._semantic(sentences)
        llm = self._llm_boundaries(sentences, start, end)

        boundaries = merge_boundaries(
            semantic,
            llm,
            start=start,
            end=end,
            tolerance=config.merge_tolerance,
            min_seconds=config.min_topic_seconds,
            max_seconds=config.max_topic_seconds,
        )
        spans = boundaries_to_topics(boundaries, start, end)

        # While recording, the trailing span is still open -- do not commit it
        # unless it is already over the maximum length.
        if not final and spans:
            last = spans[-1]
            if last.duration < config.max_topic_seconds:
                spans = spans[:-1]

        if not spans:
            return SegmentationResult([], start)

        committed: list[Topic] = []
        next_idx = self.db.next_topic_idx(session_id)
        for offset, span in enumerate(spans):
            topic = self.db.add_topic(
                session_id,
                idx=next_idx + offset,
                start=span.start,
                end=span.end,
                label=span.label or f"Topic {next_idx + offset + 1}",
                summary=span.summary,
                tags=span.tags,
                method=span.method,
                confidence=span.confidence,
            )
            committed.append(topic)
            log.info(
                "topic.committed",
                extra={
                    "session_id": session_id,
                    "idx": topic.idx,
                    "start": round(topic.start, 1),
                    "duration": round(topic.duration, 1),
                    "label": topic.label,
                    "method": topic.method,
                },
            )

        watermark = committed[-1].end
        self.db.update_session(session_id, topics_watermark=watermark)
        return SegmentationResult(
            topics=committed,
            watermark=watermark,
            semantic_count=len(semantic),
            llm_count=len(llm),
        )
