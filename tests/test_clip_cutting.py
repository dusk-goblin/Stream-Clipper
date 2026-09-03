"""Clip cutting: keyframe snapping, cut planning, ffmpeg argument assembly."""

from __future__ import annotations

from pathlib import Path

import pytest

from streamclipper.clips.cutter import (
    build_ffmpeg_args,
    plan_cut,
    snap_to_keyframe,
)
from streamclipper.clips.subtitles import build_cues, render_srt
from streamclipper.clips.vertical import vertical_filter
from streamclipper.config import ClipsConfig, VerticalConfig
from streamclipper.errors import ClipCutError
from streamclipper.state.models import Segment, SegmentStatus, Word

# A 2-second GOP, which is what Twitch delivers.
KEYFRAMES = [float(t) for t in range(0, 1200, 2)]


def segments(count: int = 4, length: float = 300.0) -> list[Segment]:
    return [
        Segment(
            id=i + 1, session_id=1, seq=i, path=f"/rec/seg_{i:05d}.ts",
            start=i * length, duration=length, status=SegmentStatus.READY.value,
        )
        for i in range(count)
    ]


@pytest.fixture
def clips_config() -> ClipsConfig:
    return ClipsConfig(
        mode="auto", snap_tolerance=1.5, pad_before=2.0, pad_after=1.5,
        vertical=VerticalConfig(),
    )


# --------------------------------------------------------------------------
# Keyframe snapping
# --------------------------------------------------------------------------


def test_snap_moves_backwards_to_the_preceding_keyframe():
    snapped, drift = snap_to_keyframe(KEYFRAMES, 101.3, tolerance=1.5)
    assert snapped == 100.0
    assert drift == pytest.approx(1.3)


def test_snap_never_moves_forwards():
    """Cutting late loses the opening words; cutting early only adds lead-in."""
    for target in (10.1, 10.9, 11.99):
        snapped, _ = snap_to_keyframe(KEYFRAMES, target, tolerance=5.0)
        assert snapped <= target


def test_snap_on_an_exact_keyframe_has_no_drift():
    snapped, drift = snap_to_keyframe(KEYFRAMES, 100.0, tolerance=1.5)
    assert snapped == 100.0
    assert drift == pytest.approx(0.0)


def test_snap_reports_infinite_drift_without_keyframes():
    """Unknown keyframes must not silently produce a non-keyframe copy cut."""
    snapped, drift = snap_to_keyframe([], 50.0, tolerance=1.5)
    assert snapped == 50.0
    assert drift == float("inf")


def test_snap_before_the_first_keyframe():
    snapped, drift = snap_to_keyframe([10.0, 20.0], 9.5, tolerance=1.5)
    assert snapped == 10.0
    assert drift == pytest.approx(0.5)

    _, far_drift = snap_to_keyframe([10.0, 20.0], 2.0, tolerance=1.5)
    assert far_drift == float("inf")


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_plan_applies_padding_on_both_sides(clips_config):
    clips_config.mode = "reencode"
    plan = plan_cut(segments(), 100.0, 160.0, clips_config)
    assert plan.actual_start == pytest.approx(98.0)
    assert plan.actual_end == pytest.approx(161.5)
    assert plan.duration == pytest.approx(63.5)
    assert plan.requested_start == 100.0 and plan.requested_end == 160.0


def test_plan_selects_only_the_segments_it_needs(clips_config):
    plan = plan_cut(segments(), 100.0, 160.0, clips_config, KEYFRAMES)
    assert plan.sources == ["/rec/seg_00000.ts"]
    assert not plan.needs_concat


def test_plan_spans_a_segment_boundary(clips_config):
    plan = plan_cut(segments(), 290.0, 320.0, clips_config, KEYFRAMES)
    assert plan.sources == ["/rec/seg_00000.ts", "/rec/seg_00001.ts"]
    assert plan.needs_concat
    # Seek is relative to the start of the concatenated inputs.
    assert plan.timeline_start == 0.0
    assert plan.seek == pytest.approx(288.0)


def test_plan_seek_is_relative_to_the_first_source_not_the_session(clips_config):
    plan = plan_cut(segments(), 910.0, 950.0, clips_config, KEYFRAMES)
    assert plan.sources == ["/rec/seg_00003.ts"]
    assert plan.timeline_start == 900.0
    assert plan.seek == pytest.approx(plan.actual_start - 900.0)
    assert 0.0 <= plan.seek < 300.0


def test_auto_mode_copies_when_a_keyframe_is_close(clips_config):
    plan = plan_cut(segments(), 100.0, 160.0, clips_config, KEYFRAMES)
    assert plan.mode == "copy"
    assert plan.snap_drift <= clips_config.snap_tolerance
    # A copy must begin exactly on a keyframe.
    assert plan.actual_start in KEYFRAMES


def test_auto_mode_reencodes_when_the_snap_would_drift_too_far(clips_config):
    sparse = [0.0, 200.0, 400.0]
    plan = plan_cut(segments(), 150.0, 200.0, clips_config, sparse)
    assert plan.mode == "reencode"
    # Re-encoding is frame accurate, so no snapping happened.
    assert plan.actual_start == pytest.approx(148.0)
    assert plan.snap_drift == 0.0


def test_auto_mode_reencodes_when_keyframes_are_unknown(clips_config):
    plan = plan_cut(segments(), 150.0, 200.0, clips_config, [])
    assert plan.mode == "reencode"


def test_copy_mode_snaps_even_when_drift_is_large(clips_config):
    """An explicit `copy` is a promise about speed, not accuracy."""
    clips_config.mode = "copy"
    plan = plan_cut(segments(), 150.0, 200.0, clips_config, [0.0, 200.0])
    assert plan.mode == "copy"
    assert plan.actual_start == 0.0


def test_plan_clamps_to_the_recorded_footage(clips_config):
    clips_config.mode = "reencode"
    available = segments(count=1)          # only 0-300s exists
    plan = plan_cut(available, 1.0, 299.0, clips_config)
    assert plan.actual_start >= 0.0
    assert plan.actual_end <= 300.0


def test_plan_fails_when_the_footage_is_gone(clips_config):
    with pytest.raises(ClipCutError, match="No recorded segment"):
        plan_cut(segments(count=1), 5000.0, 5060.0, clips_config)


def test_plan_rejects_an_inverted_range(clips_config):
    with pytest.raises(ClipCutError, match="must be after"):
        plan_cut(segments(), 200.0, 100.0, clips_config)


def test_plan_duration_is_never_zero(clips_config):
    clips_config.pad_before = 0.0
    clips_config.pad_after = 0.0
    clips_config.mode = "reencode"
    plan = plan_cut(segments(), 100.0, 100.05, clips_config)
    assert plan.duration > 0.0


# --------------------------------------------------------------------------
# ffmpeg arguments
# --------------------------------------------------------------------------


def args_for(plan, config, **kwargs) -> list[str]:
    """Build args with the binary lookup stubbed, so no ffmpeg is required."""
    import streamclipper.clips.cutter as cutter

    original = cutter.require_binary
    cutter.require_binary = lambda name: f"/usr/bin/{name}"
    try:
        return build_ffmpeg_args(plan, Path("/out/clip.mp4"), config, **kwargs)
    finally:
        cutter.require_binary = original


def test_copy_plan_streams_without_reencoding(clips_config):
    plan = plan_cut(segments(), 100.0, 160.0, clips_config, KEYFRAMES)
    args = args_for(plan, clips_config)
    assert "-c" in args and args[args.index("-c") + 1] == "copy"
    assert "-crf" not in args
    # Input-side seek: -ss must come before -i to be fast.
    assert args.index("-ss") < args.index("-i")


def test_reencode_plan_carries_codec_settings(clips_config):
    clips_config.mode = "reencode"
    plan = plan_cut(segments(), 100.0, 160.0, clips_config)
    args = args_for(plan, clips_config)
    assert args[args.index("-c:v") + 1] == clips_config.video_codec
    assert args[args.index("-crf") + 1] == str(clips_config.crf)
    assert "-pix_fmt" in args


def test_concat_plan_uses_the_concat_demuxer(clips_config):
    plan = plan_cut(segments(), 290.0, 320.0, clips_config, KEYFRAMES)
    args = args_for(plan, clips_config, concat_file=Path("/tmp/list.txt"))
    assert args[args.index("-f") + 1] == "concat"
    assert "/tmp/list.txt" in args


def test_concat_plan_without_a_list_file_is_an_error(clips_config):
    plan = plan_cut(segments(), 290.0, 320.0, clips_config, KEYFRAMES)
    with pytest.raises(ClipCutError, match="concat list"):
        args_for(plan, clips_config)


def test_any_filter_forces_a_reencode(clips_config):
    """A stream copy cannot have its pixels touched."""
    plan = plan_cut(segments(), 100.0, 160.0, clips_config, KEYFRAMES)
    assert plan.mode == "copy"
    args = args_for(plan, clips_config, vertical=True)
    assert "-vf" in args
    assert "-c:v" in args
    assert not (("-c" in args) and args[args.index("-c") + 1] == "copy")


def test_subtitle_burn_in_escapes_the_path(clips_config):
    clips_config.mode = "reencode"
    plan = plan_cut(segments(), 100.0, 160.0, clips_config)
    args = args_for(plan, clips_config, subtitle_file=Path("/out/a b/clip.srt"))
    filters = args[args.index("-vf") + 1]
    assert "subtitles=" in filters
    assert r"\:" in filters or ":" not in filters.split("subtitles='")[1].split("'")[0]


def test_vertical_filter_targets_the_configured_size():
    chain = vertical_filter(VerticalConfig(enabled=True, width=1080, height=1920))
    assert chain.endswith("crop=1080:1920")
    assert "scale=1080:1920" in chain


def test_vertical_focus_shifts_the_crop_window():
    left = vertical_filter(VerticalConfig(focus_x=0.0))
    right = vertical_filter(VerticalConfig(focus_x=1.0))
    assert left != right
    assert "0.0000" in left and "1.0000" in right


# --------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------


def words_for(text: str, start: float = 0.0, rate: float = 0.4) -> list[Word]:
    return [
        Word(start=start + i * rate, end=start + (i + 1) * rate, text=w)
        for i, w in enumerate(text.split())
    ]


def test_cues_are_rebased_to_the_clip_timeline():
    words = words_for("a clip cut from the middle of a long recording", start=500.0)
    cues = build_cues(words, clip_start=500.0, clip_end=520.0)
    assert cues
    assert cues[0].start == pytest.approx(0.0, abs=0.01)
    assert all(c.start >= 0.0 for c in cues)


def test_cues_are_split_for_readability():
    words = words_for(" ".join(["word"] * 60))
    cues = build_cues(words, 0.0, 100.0, max_chars=42, max_seconds=3.0)
    assert len(cues) > 1
    assert all(len(c.text) <= 45 for c in cues)


def test_cues_never_overlap():
    cues = build_cues(words_for(" ".join(["word"] * 40)), 0.0, 100.0)
    for previous, current in zip(cues, cues[1:]):
        assert previous.end <= current.start + 1e-9


def test_cues_exclude_words_outside_the_clip():
    words = words_for(" ".join(["inside"] * 10)) + words_for(
        " ".join(["outside"] * 10), start=100.0
    )
    cues = build_cues(words, 0.0, 20.0)
    assert cues
    assert all("outside" not in c.text for c in cues)


def test_srt_render_is_well_formed():
    srt = render_srt(build_cues(words_for("hello there chat this is a caption"), 0.0, 30.0))
    lines = srt.strip().splitlines()
    assert lines[0] == "1"
    assert " --> " in lines[1]
    assert "," in lines[1]  # SRT uses a comma decimal separator


def test_no_words_produces_no_cues():
    assert build_cues([], 0.0, 30.0) == []
