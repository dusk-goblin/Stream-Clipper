"""Capture-side parsing: IRC messages, emotes, and the segment list."""

from __future__ import annotations

import json

import pytest

from streamclipper.capture.chat import (
    _PRIVMSG,
    ChatLogger,
    extract_emotes,
    ingest_chat_file,
    load_chat_jsonl,
    parse_tags,
)
from streamclipper.capture.recorder import read_segment_list, seq_from_filename
from streamclipper.capture.twitch import StreamInfo
from streamclipper.util.timefmt import hhmmss, parse_hhmmss, srt_time, vod_offset

LINE = (
    "@badge-info=;display-name=Bob;emotes=25:0-4;tmi-sent-ts=1700000000000 "
    ":bob!bob@bob.tmi.twitch.tv PRIVMSG #hasanabi :Kappa that was insane KEKW KEKW"
)


# --------------------------------------------------------------------------
# IRC parsing
# --------------------------------------------------------------------------


def test_privmsg_parses():
    match = _PRIVMSG.match(LINE)
    assert match is not None
    assert match.group("nick") == "bob"
    assert match.group("channel") == "hasanabi"
    assert match.group("text").startswith("Kappa")


def test_non_privmsg_lines_are_ignored():
    assert _PRIVMSG.match(":tmi.twitch.tv 001 justinfan1 :Welcome") is None
    assert _PRIVMSG.match("PING :tmi.twitch.tv") is None


def test_tag_escapes_are_undone():
    tags = parse_tags(r"display-name=A\sB;system-msg=hi\sthere;empty=")
    assert tags["display-name"] == "A B"
    assert tags["system-msg"] == "hi there"
    assert tags["empty"] == ""


def test_native_emotes_come_from_tag_positions():
    match = _PRIVMSG.match(LINE)
    emotes = extract_emotes(match.group("text"), parse_tags(match.group("tags")), [])
    assert emotes == ["Kappa"]


def test_third_party_emotes_come_from_the_watchlist():
    match = _PRIVMSG.match(LINE)
    emotes = extract_emotes(
        match.group("text"), parse_tags(match.group("tags")), ["KEKW", "OMEGALUL"]
    )
    assert "Kappa" in emotes      # native, from the tag
    assert "KEKW" in emotes       # third party, from the watchlist


def test_emotes_are_deduplicated_and_order_preserved():
    emotes = extract_emotes("KEKW KEKW OMEGALUL KEKW", {}, ["KEKW", "OMEGALUL"])
    assert emotes == ["KEKW", "OMEGALUL"]


def test_watchlist_matches_whole_tokens_only():
    assert extract_emotes("KEKWAIT is different", {}, ["KEKW"]) == []


def test_malformed_emote_tag_does_not_raise():
    assert extract_emotes("hello", {"emotes": "garbage"}, []) == []
    assert extract_emotes("hi", {"emotes": "25:99-200"}, []) == []


def test_message_timestamp_uses_the_server_clock(tmp_path, db):
    from streamclipper.config import ChatConfig

    session_start = 1_700_000_000.0 - 60.0
    logger = ChatLogger(
        ChatConfig(), "hasanabi", db, 1, session_start, tmp_path / "c.jsonl", ["KEKW"]
    )
    message = logger._parse(LINE)
    assert message is not None
    # tmi-sent-ts is 1700000000000ms, i.e. 60s after this session started.
    assert message.ts == pytest.approx(60.0)
    assert message.user == "Bob"


# --------------------------------------------------------------------------
# Chat JSONL
# --------------------------------------------------------------------------


def test_chat_jsonl_round_trips(tmp_path, db):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"ts": t, "wall_ts": 1000.0 + t, "user": "u", "text": "hi", "emotes": ["KEKW"]})
            for t in (3.0, 1.0, 2.0)
        )
    )
    messages = load_chat_jsonl(path)
    assert [m.ts for m in messages] == [1.0, 2.0, 3.0]   # sorted on load
    assert messages[0].emotes == ["KEKW"]


def test_chat_jsonl_skips_corrupt_lines(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('{"ts": 1.0, "user": "u", "text": "ok"}\nnot json\n\n')
    assert len(load_chat_jsonl(path)) == 1


def test_chat_jsonl_missing_file_is_empty(tmp_path):
    assert load_chat_jsonl(tmp_path / "absent.jsonl") == []


def test_ingest_loads_chat_into_a_session(tmp_path, db):
    session = db.create_session("c", "offline", 0.0)
    path = tmp_path / "chat.jsonl"
    path.write_text(
        "\n".join(json.dumps({"ts": float(t), "user": "u", "text": "hi"}) for t in range(5))
    )
    assert ingest_chat_file(db, session.id, path) == 5
    assert db.chat_count(session.id) == 5


# --------------------------------------------------------------------------
# Segment list
# --------------------------------------------------------------------------


def test_segment_list_parses_completed_rows(tmp_path):
    path = tmp_path / "segments.csv"
    path.write_text("seg_00000.ts,0.000000,300.120000\nseg_00001.ts,300.120000,600.240000\n")
    assert read_segment_list(path) == [
        ("seg_00000.ts", 0.0, 300.12),
        ("seg_00001.ts", 300.12, 600.24),
    ]


def test_a_partially_written_row_is_skipped(tmp_path):
    """ffmpeg may be mid-flush; a half-written row must not become a segment."""
    path = tmp_path / "segments.csv"
    path.write_text("seg_00000.ts,0.0,300.0\nseg_00001.ts,300.0")
    assert len(read_segment_list(path)) == 1


def test_segment_list_missing_file_is_empty(tmp_path):
    assert read_segment_list(tmp_path / "nope.csv") == []


def test_sequence_number_comes_from_the_filename():
    assert seq_from_filename("seg_00042.ts") == 42
    assert seq_from_filename("seg_00042.mp4") == 42
    assert seq_from_filename("unexpected.ts") is None


# --------------------------------------------------------------------------
# Helix payloads and time formatting
# --------------------------------------------------------------------------


def test_stream_info_tolerates_a_sparse_payload():
    info = StreamInfo.from_json({"id": 123, "user_login": "hasanabi"})
    assert info.id == "123"
    assert info.title == ""
    assert info.viewer_count == 0


def test_timestamp_formatting():
    assert hhmmss(3723.456) == "01:02:03"
    assert hhmmss(3723.456, millis=True) == "01:02:03.456"
    assert srt_time(3723.456) == "01:02:03,456"
    assert hhmmss(-5) == "00:00:00"


def test_millisecond_rounding_does_not_produce_1000ms():
    assert hhmmss(1.9999, millis=True) == "00:00:02.000"


def test_timestamp_parsing():
    assert parse_hhmmss("1:02:03.5") == pytest.approx(3723.5)
    assert parse_hhmmss("02:03") == pytest.approx(123.0)
    assert parse_hhmmss("45") == pytest.approx(45.0)
    with pytest.raises(ValueError):
        parse_hhmmss("")
    with pytest.raises(ValueError):
        parse_hhmmss("1:2:3:4")


def test_vod_offsets_match_twitch_url_format():
    assert vod_offset(3723) == "1h2m3s"
    assert vod_offset(125) == "2m5s"
    assert vod_offset(9) == "9s"
