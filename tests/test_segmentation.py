"""Topic segmentation against a fixture transcript with known boundaries."""

from __future__ import annotations

import pytest

from streamclipper.config import SemanticConfig
from streamclipper.segment.embeddings import TfidfEmbedder, cosine
from streamclipper.segment.semantic import semantic_boundaries, similarity_profile
from streamclipper.segment.topics import TopicSegmenter
from streamclipper.transcribe.transcript import (
    sentences_from_utterances,
    slice_sentences,
)

TOLERANCE = 60.0  # a boundary within a minute of the truth is a hit


def nearest(times, target):
    return min((abs(t - target), t) for t in times)[1] if times else None


# --------------------------------------------------------------------------
# Sentence construction
# --------------------------------------------------------------------------


def test_sentences_are_ordered_and_timed(utterances):
    sentences = sentences_from_utterances(utterances)
    assert sentences
    for previous, current in zip(sentences, sentences[1:]):
        assert previous.start <= current.start
        assert previous.end <= current.end + 1e-6
    assert all(s.end > s.start for s in sentences)


def test_sentences_preserve_all_text(utterances):
    sentences = sentences_from_utterances(utterances)
    joined = " ".join(s.text for s in sentences)
    # Every source utterance must survive somewhere in the flattened text.
    for utterance in utterances:
        assert utterance.text.split()[0] in joined


def test_short_fragments_merge_but_complete_sentences_do_not():
    from streamclipper.state.models import Utterance

    sentences = sentences_from_utterances(
        [
            Utterance(0, 5, "This is a complete sentence with plenty of content."),
            Utterance(6, 7, "yeah"),
            Utterance(8, 14, "And here is the next complete thought about something else."),
        ]
    )
    texts = [s.text for s in sentences]
    assert len(sentences) == 2
    assert "yeah" in texts[1]  # the fragment merged forward
    assert texts[0].endswith("content.")


def test_slice_sentences_assigns_each_sentence_once(utterances):
    sentences = sentences_from_utterances(utterances)
    first = slice_sentences(sentences, 0, 600)
    second = slice_sentences(sentences, 600, 2000)
    assert len(first) + len(second) == len(sentences)
    assert not set(id(s) for s in first) & set(id(s) for s in second)


# --------------------------------------------------------------------------
# Semantic signal
# --------------------------------------------------------------------------


def test_embedder_separates_the_fixture_topics(utterances, expected_boundaries):
    embedder = TfidfEmbedder()
    first, second = expected_boundaries[0], expected_boundaries[1]
    politics = " ".join(u.text for u in utterances if u.start < first)
    gaming = " ".join(u.text for u in utterances if first <= u.start < second)
    cooking = " ".join(u.text for u in utterances if u.start >= second)

    vectors = embedder.encode([politics, gaming, cooking])
    cross = [
        cosine(vectors[0], vectors[1]),
        cosine(vectors[0], vectors[2]),
        cosine(vectors[1], vectors[2]),
    ]
    # Distinct topics must not look alike to the embedder.
    assert max(cross) < 0.35


def test_similarity_profile_dips_at_the_real_boundaries(utterances, expected_boundaries):
    sentences = sentences_from_utterances(utterances)
    embedder = TfidfEmbedder()
    gaps = similarity_profile(sentences, embedder.encode([s.text for s in sentences]), window=6)
    assert gaps
    assert any(g.full_window for g in gaps)

    usable = [g for g in gaps if g.full_window]
    for boundary in expected_boundaries:
        near = [g for g in usable if abs(g.time - boundary) <= TOLERANCE]
        assert near, f"no gap scored near boundary {boundary}"
        elsewhere = [g for g in usable if abs(g.time - boundary) > TOLERANCE * 2]
        # The dip at a real change must beat the typical gap.
        assert min(g.similarity for g in near) < (
            sum(g.similarity for g in elsewhere) / len(elsewhere)
        )


def test_edge_gaps_are_excluded_from_boundaries(utterances):
    """A lopsided window reads as low similarity for the wrong reason."""
    sentences = sentences_from_utterances(utterances)
    embedder = TfidfEmbedder()
    window = 6
    gaps = similarity_profile(sentences, embedder.encode([s.text for s in sentences]), window)
    assert not gaps[0].full_window
    assert not gaps[-1].full_window
    assert all(
        window <= g.index <= len(sentences) - window for g in gaps if g.full_window
    )


def test_normalised_similarity_is_scale_free(utterances):
    """Thresholds must mean the same thing whatever the vector scale."""
    sentences = sentences_from_utterances(utterances)
    base = TfidfEmbedder().encode([s.text for s in sentences])
    # Halving every weight changes the raw vectors but not their directions.
    scaled = [{k: v * 0.5 for k, v in vec.items()} for vec in base]

    a = similarity_profile(sentences, base, 6)
    b = similarity_profile(sentences, scaled, 6)
    for left, right in zip(a, b):
        assert left.normalised == pytest.approx(right.normalised, abs=1e-9)
        assert left.depth == pytest.approx(right.depth, abs=1e-9)


def test_semantic_boundaries_find_the_fixture_topics(utterances, expected_boundaries):
    sentences = sentences_from_utterances(utterances)
    found = semantic_boundaries(
        sentences, TfidfEmbedder(), SemanticConfig(), min_gap_seconds=180
    )
    times = [g.time for g in found]
    assert times, "no semantic boundaries detected at all"

    # Recall: every real topic change must be found, at default thresholds.
    for boundary in expected_boundaries:
        assert abs(nearest(times, boundary) - boundary) <= TOLERANCE

    # Precision: intra-topic drift produces some extra edges, which the merge
    # stage prunes -- but the detector should not be firing indiscriminately.
    assert len(times) <= 2 * (len(expected_boundaries) + 1)


def test_semantic_boundaries_are_time_ordered_and_deduped(utterances):
    sentences = sentences_from_utterances(utterances)
    found = semantic_boundaries(
        sentences, TfidfEmbedder(), SemanticConfig(), min_gap_seconds=180
    )
    times = [g.time for g in found]
    assert times == sorted(times)
    for previous, current in zip(times, times[1:]):
        assert current - previous >= 180


def test_uniform_transcript_yields_no_boundaries():
    from streamclipper.state.models import Utterance

    line = "the same words about the same subject repeated with no change at all"
    utterances = [Utterance(i * 20.0, i * 20.0 + 10.0, line) for i in range(30)]
    found = semantic_boundaries(
        sentences_from_utterances(utterances),
        TfidfEmbedder(),
        SemanticConfig(window_sentences=4),
    )
    assert found == []


def test_too_few_sentences_is_not_an_error():
    from streamclipper.state.models import Utterance

    assert semantic_boundaries(
        sentences_from_utterances([Utterance(0, 5, "only one thing said here.")]),
        TfidfEmbedder(),
        SemanticConfig(),
    ) == []


# --------------------------------------------------------------------------
# End to end through the segmenter
# --------------------------------------------------------------------------


def test_segmenter_commits_topics_covering_the_stream(
    config, db, session_with_transcript, expected_boundaries, stream_duration
):
    config.segment.min_topic_seconds = 180
    config.segment.max_topic_seconds = 1800

    result = TopicSegmenter(config, db).segment_session(
        session_with_transcript.id, final=True
    )
    assert result.topics

    topics = db.list_topics(session_with_transcript.id)
    assert topics[0].start == pytest.approx(0.0, abs=1.0)
    # Topics tile the timeline with no gaps and no overlaps.
    for previous, current in zip(topics, topics[1:]):
        assert current.start == pytest.approx(previous.end, abs=1e-6)
    assert topics[-1].end == pytest.approx(stream_duration, abs=stream_duration * 0.05)

    starts = [t.start for t in topics[1:]]
    for boundary in expected_boundaries:
        assert abs(nearest(starts, boundary) - boundary) <= TOLERANCE


def test_segmenter_respects_length_bounds(config, db, session_with_transcript):
    config.segment.min_topic_seconds = 200
    config.segment.max_topic_seconds = 500

    TopicSegmenter(config, db).segment_session(session_with_transcript.id, final=True)
    topics = db.list_topics(session_with_transcript.id)
    assert topics
    for topic in topics:
        assert topic.duration <= config.segment.max_topic_seconds + 1e-6
    # Only the final topic may fall short, since the stream just ended there.
    for topic in topics[:-1]:
        assert topic.duration >= config.segment.min_topic_seconds - 1e-6


def test_segmenter_advances_the_watermark_and_is_idempotent(
    config, db, session_with_transcript
):
    segmenter = TopicSegmenter(config, db)
    first = segmenter.segment_session(session_with_transcript.id, final=True)
    assert first.topics
    watermark = db.get_session(session_with_transcript.id).topics_watermark
    assert watermark == pytest.approx(first.topics[-1].end)

    # Re-running with no new transcript must not duplicate topics.
    second = segmenter.segment_session(session_with_transcript.id, final=True)
    assert second.topics == []
    assert len(db.list_topics(session_with_transcript.id)) == len(first.topics)


def test_segmenter_holds_back_the_open_topic_while_recording(
    config, db, session_with_transcript
):
    """A live sweep must not commit the topic still in progress."""
    config.segment.settle_seconds = 120
    live = TopicSegmenter(config, db).segment_session(
        session_with_transcript.id, final=False
    )
    if live.topics:
        settled = db.transcribed_seconds(session_with_transcript.id) - 120
        assert live.topics[-1].end <= settled + 1e-6


def test_segmenter_does_nothing_before_min_topic_length(config, db):
    from streamclipper.state.models import SegmentStatus, Utterance

    session = db.create_session("test", "offline", 0.0)
    segment = db.add_segment(session.id, 0, "/fake/0.ts", 0.0, 60.0)
    db.finish_segment(segment.id, 60.0, 10)
    db.set_segment_status(segment.id, SegmentStatus.TRANSCRIBED.value)
    db.add_transcript(session.id, segment.id, [Utterance(0, 30, "not enough content yet.")])

    result = TopicSegmenter(config, db).segment_session(session.id, final=False)
    assert result.topics == []
