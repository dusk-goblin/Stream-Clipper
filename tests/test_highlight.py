"""Highlight ranking: chat signals, window sweep, selection."""

from __future__ import annotations

import pytest

from streamclipper.highlight.chat_signals import (
    build_profile,
    emote_rate,
    message_rate,
    squash,
    zscores,
)
from streamclipper.highlight.llm_score import shortlist
from streamclipper.highlight.rank import (
    Candidate,
    combine,
    rank_topic,
    score_candidates,
    select,
    snap_to_speech,
    sweep_windows,
)
from streamclipper.state.models import ChatMessage, Utterance
from streamclipper.transcribe.transcript import sentences_from_utterances

# Where the chat fixture's hype spikes are.
SPIKES = [(300.0, 340.0), (700.0, 730.0)]


# --------------------------------------------------------------------------
# Chat signals
# --------------------------------------------------------------------------


def test_profile_indexes_only_the_requested_range(chat_messages):
    profile = build_profile(chat_messages, 200.0, 400.0, ["KEKW"])
    assert profile.total_messages > 0
    assert all(200.0 <= t < 400.0 for t in profile.times)


def test_message_rate_rises_during_a_hype_spike(chat_messages):
    profile = build_profile(chat_messages, 0.0, 1200.0, ["KEKW", "OMEGALUL"])
    spike = message_rate(profile, *SPIKES[0])
    quiet = message_rate(profile, 100.0, 200.0)
    assert spike > quiet * 3


def test_emote_rate_rises_during_a_hype_spike(chat_messages):
    watch = ["KEKW", "OMEGALUL", "LULW", "PepeLaugh", "Pog"]
    profile = build_profile(chat_messages, 0.0, 1200.0, watch)
    assert emote_rate(profile, *SPIKES[0]) > emote_rate(profile, 100.0, 200.0)


def test_emote_watchlist_filters(chat_messages):
    everything = build_profile(chat_messages, 0.0, 1200.0, [])
    only_kekw = build_profile(chat_messages, 0.0, 1200.0, ["KEKW"])
    assert len(only_kekw.emote_times) < len(everything.emote_times)


def test_repeated_emotes_in_one_message_each_count():
    messages = [ChatMessage(ts=1.0, user="u", text="KEKW KEKW KEKW", emotes=["KEKW"] * 3)]
    profile = build_profile(messages, 0.0, 10.0, ["KEKW"])
    assert profile.total_messages == 1
    assert len(profile.emote_times) == 3


def test_counting_is_half_open_so_windows_do_not_double_count():
    messages = [ChatMessage(ts=float(t), user="u", text="x") for t in range(10)]
    profile = build_profile(messages, 0.0, 10.0)
    assert profile.count_between(0.0, 5.0) + profile.count_between(5.0, 10.0) == 10


def test_zscores_of_identical_values_are_zero():
    assert zscores([4.0] * 5) == [0.0] * 5
    assert zscores([]) == []


def test_squash_is_monotonic_and_bounded():
    values = [squash(z) for z in (-4, -1, 0, 1, 4)]
    assert values == sorted(values)
    assert all(0.0 < v < 1.0 for v in values)
    assert squash(0.0) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Window sweep
# --------------------------------------------------------------------------


def test_sweep_respects_the_length_bounds(config):
    windows = sweep_windows(0.0, 600.0, config.highlight)
    assert windows
    for start, end in windows:
        assert config.highlight.clip_min_seconds <= end - start <= config.highlight.clip_max_seconds
        assert 0.0 <= start and end <= 600.0


def test_sweep_of_a_topic_shorter_than_a_clip_is_empty(config):
    assert sweep_windows(0.0, 10.0, config.highlight) == []


def test_sweep_offers_several_clip_lengths(config):
    lengths = {round(e - s, 3) for s, e in sweep_windows(0.0, 600.0, config.highlight)}
    assert len(lengths) > 1


# --------------------------------------------------------------------------
# Scoring and selection
# --------------------------------------------------------------------------


def test_ranking_finds_the_chat_spike(config, chat_messages):
    profile = build_profile(chat_messages, 0.0, 600.0, config.highlight.emotes)
    picks = rank_topic(0.0, 600.0, [], profile, config.highlight)
    assert picks
    best = max(picks, key=lambda c: c.score)
    spike_start, spike_end = SPIKES[0]
    # The best clip must overlap the burst it is supposed to have found.
    assert best.start < spike_end and best.end > spike_start


def test_selected_clips_do_not_overlap(config, chat_messages):
    config.highlight.per_topic = 4
    profile = build_profile(chat_messages, 0.0, 1200.0, config.highlight.emotes)
    picks = rank_topic(0.0, 1200.0, [], profile, config.highlight)
    for a, b in zip(picks, picks[1:]):
        assert not a.overlaps(b)
        assert a.start <= b.start


def test_per_topic_limit_is_honoured(config, chat_messages):
    config.highlight.per_topic = 1
    profile = build_profile(chat_messages, 0.0, 1200.0, config.highlight.emotes)
    assert len(rank_topic(0.0, 1200.0, [], profile, config.highlight)) <= 1


def test_candidates_below_min_score_are_rejected(config):
    config.highlight.min_score = 0.99
    profile = build_profile(
        [ChatMessage(ts=float(t), user="u", text="x") for t in range(600)],
        0.0, 600.0,
    )
    assert rank_topic(0.0, 600.0, [], profile, config.highlight) == []


def test_flat_chat_still_produces_a_candidate_when_the_bar_is_low(config):
    """A quiet topic is not a broken topic."""
    config.highlight.min_score = 0.1
    profile = build_profile(
        [ChatMessage(ts=float(t), user="u", text="x") for t in range(600)],
        0.0, 600.0,
    )
    assert rank_topic(0.0, 600.0, [], profile, config.highlight)


def test_scoring_renormalises_when_the_llm_signal_is_missing(config):
    """Without the LLM weight redistributed, everything would fall below min_score."""
    candidate = Candidate(start=0.0, end=60.0, chat_score=1.0, emote_score=1.0)
    without = combine(candidate, config.highlight, has_llm=False)
    with_llm = combine(candidate, config.highlight, has_llm=True)
    assert without == pytest.approx(1.0)
    assert with_llm < without


def test_llm_score_moves_the_ranking(config, chat_messages):
    profile = build_profile(chat_messages, 0.0, 1200.0, config.highlight.emotes)
    windows = sweep_windows(0.0, 1200.0, config.highlight)
    scored = score_candidates(windows, profile, config.highlight)

    # Give the last window a perfect rating and everything else nothing.
    llm = {i: (1.0 if i == len(scored) - 1 else 0.0) for i in range(len(scored))}
    config.highlight.per_topic = 1
    config.highlight.min_score = 0.0
    with_llm = rank_topic(
        0.0, 1200.0, [], profile, config.highlight, llm_scores=llm
    )
    without = rank_topic(0.0, 1200.0, [], profile, config.highlight)
    assert with_llm[0].start != without[0].start


def test_select_returns_highest_scoring_first_then_time_ordered(config):
    candidates = [
        Candidate(start=0.0, end=30.0, score=0.4),
        Candidate(start=100.0, end=130.0, score=0.9),
        Candidate(start=200.0, end=230.0, score=0.6),
    ]
    config.highlight.per_topic = 2
    config.highlight.min_score = 0.5
    chosen = select(candidates, config.highlight)
    assert [c.start for c in chosen] == [100.0, 200.0]


# --------------------------------------------------------------------------
# Speech snapping and shortlisting
# --------------------------------------------------------------------------


def test_snapping_moves_edges_onto_sentence_boundaries(config):
    sentences = sentences_from_utterances(
        [
            Utterance(0.0, 12.0, "The first complete thought lands right here for us."),
            Utterance(13.0, 48.0, "A second, considerably longer thought follows it now."),
            Utterance(49.0, 62.0, "And then a third one closes out this whole passage."),
        ]
    )
    start, end = snap_to_speech(1.5, 47.0, sentences, config.highlight)
    assert start == pytest.approx(0.0)
    assert config.highlight.clip_min_seconds <= end - start <= config.highlight.clip_max_seconds


def test_snapping_refuses_to_break_the_length_bounds(config):
    sentences = sentences_from_utterances(
        [Utterance(0.0, 3.0, "A very short line indeed here.")]
    )
    # Snapping the end back to 3.0 would leave a 3-second clip.
    start, end = snap_to_speech(0.0, 40.0, sentences, config.highlight)
    assert (start, end) == (0.0, 40.0)


def test_shortlist_spreads_candidates_out_in_time():
    candidates = [
        Candidate(start=float(i * 5), end=float(i * 5 + 30), chat_score=1.0 - i * 0.01)
        for i in range(40)
    ]
    picked = shortlist(candidates, limit=5, min_separation=60.0)
    times = sorted(candidates[i].start for i in picked)
    assert len(picked) <= 5
    for a, b in zip(times, times[1:]):
        assert b - a >= 60.0
