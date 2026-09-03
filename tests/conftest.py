"""Shared fixtures.

The transcript and chat fixtures describe a synthetic stream with three
clearly distinct topics and two chat hype spikes, so the segmentation and
ranking tests can assert against known-good answers rather than snapshots.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from streamclipper.capture.chat import load_chat_jsonl  # noqa: E402
from streamclipper.config import Config, load_config  # noqa: E402
from streamclipper.state import Database  # noqa: E402
from streamclipper.state.models import ChatMessage, Utterance, Word  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_data() -> dict[str, Any]:
    return json.loads((FIXTURES / "transcript_topics.json").read_text())


@pytest.fixture(scope="session")
def utterances(fixture_data: dict[str, Any]) -> list[Utterance]:
    return [
        Utterance(
            start=entry["start"],
            end=entry["end"],
            text=entry["text"],
            words=[
                Word(
                    start=w["start"], end=w["end"], text=w["text"],
                    probability=w.get("probability", 1.0),
                )
                for w in entry.get("words", [])
            ],
        )
        for entry in fixture_data["utterances"]
    ]


@pytest.fixture(scope="session")
def expected_boundaries(fixture_data: dict[str, Any]) -> list[float]:
    return list(fixture_data["expected_boundaries"])


@pytest.fixture(scope="session")
def stream_duration(fixture_data: dict[str, Any]) -> float:
    return float(fixture_data["total_duration"])


@pytest.fixture(scope="session")
def chat_messages() -> list[ChatMessage]:
    return load_chat_jsonl(FIXTURES / "chat.jsonl")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Default config pointed at a temp data dir, with the offline backends.

    Tests must not reach for a GPU, a model download or the network, so the
    embedding backend is pinned to the dependency-free one and the LLM passes
    are off.
    """
    return load_config(
        overrides={
            "paths": {"data_dir": str(tmp_path / "data")},
            "segment": {
                "embeddings": {"backend": "tfidf"},
                "llm": {"enabled": False},
            },
            "highlight": {"llm": {"enabled": False}},
            "logging": {"level": "WARNING", "format": "text"},
        }
    )


@pytest.fixture
def db(config: Config) -> Database:
    config.paths.ensure()
    database = Database(config.paths.db)
    yield database
    database.close()


@pytest.fixture
def session_with_transcript(
    db: Database, utterances: list[Utterance], chat_messages: list[ChatMessage],
    stream_duration: float,
):
    """A session preloaded with the fixture transcript, chat and segments."""
    from streamclipper.state.models import SegmentStatus

    session = db.create_session("hasanabi", "offline", started_at=0.0, source="fixture")
    # Five-minute segments covering the whole fixture timeline.
    seq = 0
    start = 0.0
    while start < stream_duration:
        duration = min(300.0, stream_duration - start)
        segment = db.add_segment(
            session.id, seq, f"/fake/seg_{seq:05d}.ts", start, duration,
            status=SegmentStatus.READY.value,
        )
        db.finish_segment(segment.id, duration, 1024)
        db.set_segment_status(segment.id, SegmentStatus.TRANSCRIBED.value)
        start += duration
        seq += 1

    db.add_transcript(session.id, None, utterances)
    db.add_chat(session.id, chat_messages)
    db.update_session(session.id, duration=stream_duration)
    refreshed = db.get_session(session.id)
    assert refreshed is not None
    return refreshed
