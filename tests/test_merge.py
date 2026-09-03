"""Boundary reconciliation: clustering, length bounds, span construction."""

from __future__ import annotations


from streamclipper.segment.merge import (
    SOURCE_BOTH,
    SOURCE_LLM,
    SOURCE_SEMANTIC,
    SOURCE_SPLIT,
    SOURCE_START,
    Boundary,
    boundaries_to_topics,
    cluster_boundaries,
    enforce_max_length,
    enforce_min_length,
    merge_boundaries,
)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def test_nearby_boundaries_from_both_signals_become_one():
    merged = cluster_boundaries(
        [
            Boundary(300.0, SOURCE_SEMANTIC, 0.6),
            Boundary(320.0, SOURCE_LLM, 0.8, label="Election recap", tags=["politics"]),
        ],
        tolerance=45.0,
    )
    assert len(merged) == 1
    assert merged[0].source == SOURCE_BOTH
    # The LLM side supplies the time and the labelling.
    assert merged[0].time == 320.0
    assert merged[0].label == "Election recap"
    assert merged[0].tags == ["politics"]
    # Agreement raises confidence, but never to certainty.
    assert 0.8 < merged[0].confidence <= 0.95


def test_distant_boundaries_stay_separate():
    merged = cluster_boundaries(
        [Boundary(300.0, SOURCE_SEMANTIC), Boundary(500.0, SOURCE_LLM)], tolerance=45.0
    )
    assert [b.time for b in merged] == [300.0, 500.0]


def test_clustering_is_order_independent():
    pair = [
        Boundary(500.0, SOURCE_LLM, 0.7, label="Later"),
        Boundary(480.0, SOURCE_SEMANTIC, 0.5),
    ]
    forward = cluster_boundaries(pair, 45.0)
    backward = cluster_boundaries(list(reversed(pair)), 45.0)
    assert [b.time for b in forward] == [b.time for b in backward]
    assert forward[0].label == backward[0].label == "Later"


def test_clustering_empty_input():
    assert cluster_boundaries([], 45.0) == []


# --------------------------------------------------------------------------
# Minimum length
# --------------------------------------------------------------------------


def test_min_length_drops_a_boundary_that_would_make_a_runt():
    kept, dropped = enforce_min_length(
        [Boundary(300.0, SOURCE_SEMANTIC, 0.5), Boundary(350.0, SOURCE_SEMANTIC, 0.4)],
        start=0.0,
        end=1000.0,
        min_seconds=180.0,
    )
    assert [b.time for b in kept] == [300.0]
    assert [b.time for b in dropped] == [350.0]


def test_min_length_applies_to_the_opening_topic_too():
    """A boundary too close to the session start makes a runt as surely as one
    too close to its predecessor."""
    kept, dropped = enforce_min_length(
        [Boundary(100.0, SOURCE_SEMANTIC, 0.5)],
        start=0.0,
        end=1000.0,
        min_seconds=180.0,
    )
    assert kept == []
    assert [b.time for b in dropped] == [100.0]


def test_min_length_keeps_the_stronger_of_two_close_boundaries():
    kept, dropped = enforce_min_length(
        [Boundary(200.0, SOURCE_SEMANTIC, 0.4), Boundary(260.0, SOURCE_LLM, 0.9)],
        start=0.0,
        end=1000.0,
        min_seconds=180.0,
    )
    assert [b.time for b in kept] == [260.0]
    assert [b.time for b in dropped] == [200.0]


def test_min_length_drops_a_short_trailing_topic():
    kept, _ = enforce_min_length(
        [Boundary(300.0, SOURCE_SEMANTIC, 0.6), Boundary(950.0, SOURCE_SEMANTIC, 0.6)],
        start=0.0,
        end=1000.0,
        min_seconds=180.0,
    )
    assert [b.time for b in kept] == [300.0]


def test_min_length_ignores_boundaries_outside_the_span():
    kept, dropped = enforce_min_length(
        [Boundary(-50.0, SOURCE_LLM), Boundary(500.0, SOURCE_LLM), Boundary(2000.0, SOURCE_LLM)],
        start=0.0,
        end=1000.0,
        min_seconds=180.0,
    )
    assert [b.time for b in kept] == [500.0]
    assert len(dropped) == 2


def test_every_kept_boundary_satisfies_the_minimum():
    boundaries = [Boundary(float(t), SOURCE_SEMANTIC, 0.5) for t in range(50, 1000, 50)]
    kept, _ = enforce_min_length(boundaries, 0.0, 1000.0, min_seconds=180.0)
    marks = [0.0, *(b.time for b in kept), 1000.0]
    for previous, current in zip(marks, marks[1:]):
        assert current - previous >= 180.0


# --------------------------------------------------------------------------
# Maximum length
# --------------------------------------------------------------------------


def test_max_length_splits_an_overlong_span():
    result = enforce_max_length(
        [], start=0.0, end=3000.0, max_seconds=1000.0, min_seconds=180.0
    )
    marks = [0.0, *(b.time for b in result), 3000.0]
    for previous, current in zip(marks, marks[1:]):
        assert current - previous <= 1000.0 + 1e-6
    assert all(b.source == SOURCE_SPLIT for b in result)


def test_max_length_prefers_a_dropped_boundary_over_an_arbitrary_cut():
    spare = Boundary(900.0, SOURCE_SEMANTIC, 0.7)
    result = enforce_max_length(
        [], start=0.0, end=1800.0, max_seconds=1000.0, min_seconds=180.0,
        fallbacks=[spare],
    )
    assert [b.time for b in result] == [900.0]
    assert result[0].source == SOURCE_SPLIT


def test_max_length_ignores_a_fallback_that_would_break_the_minimum():
    # 50s in is a legal split point by length, but leaves a 50s first topic.
    result = enforce_max_length(
        [], start=0.0, end=1800.0, max_seconds=1000.0, min_seconds=180.0,
        fallbacks=[Boundary(50.0, SOURCE_SEMANTIC, 0.9)],
    )
    assert all(b.time >= 180.0 for b in result)


def test_max_length_leaves_compliant_spans_alone():
    existing = [Boundary(500.0, SOURCE_LLM, 0.8, label="Second")]
    result = enforce_max_length(
        existing, start=0.0, end=900.0, max_seconds=1000.0, min_seconds=180.0
    )
    assert [b.time for b in result] == [500.0]
    assert result[0].label == "Second"


# --------------------------------------------------------------------------
# Full merge
# --------------------------------------------------------------------------


def test_merge_produces_bounded_ordered_boundaries():
    semantic = [Boundary(float(t), SOURCE_SEMANTIC, 0.5) for t in (300, 905, 1500, 1520)]
    llm = [
        Boundary(310.0, SOURCE_LLM, 0.8, label="Election recap"),
        Boundary(2000.0, SOURCE_LLM, 0.9, label="Game patch"),
    ]
    result = merge_boundaries(
        semantic, llm, start=0.0, end=3600.0,
        tolerance=45.0, min_seconds=180.0, max_seconds=1200.0,
    )
    times = [b.time for b in result]
    assert times == sorted(times)

    marks = [0.0, *times, 3600.0]
    for previous, current in zip(marks, marks[1:]):
        assert 180.0 - 1e-6 <= current - previous <= 1200.0 + 1e-6

    labelled = {b.time: b.label for b in result if b.label}
    assert labelled.get(310.0) == "Election recap"
    assert labelled.get(2000.0) == "Game patch"


def test_merge_with_no_signals_still_bounds_length():
    result = merge_boundaries(
        [], [], start=0.0, end=5000.0,
        tolerance=45.0, min_seconds=180.0, max_seconds=1200.0,
    )
    marks = [0.0, *(b.time for b in result), 5000.0]
    assert all(c - p <= 1200.0 + 1e-6 for p, c in zip(marks, marks[1:]))


def test_merge_of_an_empty_span_is_empty():
    assert merge_boundaries([], [], 100.0, 100.0, 45.0, 180.0, 1200.0) == []
    assert merge_boundaries([], [], 200.0, 100.0, 45.0, 180.0, 1200.0) == []


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def test_topics_tile_the_timeline_without_gaps():
    boundaries = [
        Boundary(300.0, SOURCE_LLM, 0.8, label="Second", summary="s", tags=["a"]),
        Boundary(900.0, SOURCE_SEMANTIC, 0.5),
    ]
    topics = boundaries_to_topics(boundaries, 0.0, 1500.0)

    assert len(topics) == 3
    assert topics[0].start == 0.0
    assert topics[-1].end == 1500.0
    for previous, current in zip(topics, topics[1:]):
        assert current.start == previous.end

    # A boundary labels the topic it opens, not the one it closes.
    assert topics[0].method == SOURCE_START
    assert topics[1].label == "Second"
    assert topics[1].tags == ["a"]
    assert topics[2].label == "Untitled topic"


def test_topics_from_no_boundaries_is_one_span():
    topics = boundaries_to_topics([], 0.0, 600.0)
    assert len(topics) == 1
    assert (topics[0].start, topics[0].end) == (0.0, 600.0)


def test_topics_of_an_empty_span_is_empty():
    assert boundaries_to_topics([], 500.0, 500.0) == []
