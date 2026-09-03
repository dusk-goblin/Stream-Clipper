"""Row objects.

Plain dataclasses that mirror the SQLite tables. ``from_row`` takes an
``sqlite3.Row`` so callers never index by column position.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    RECORDING = "recording"
    INTERRUPTED = "interrupted"   # stream dropped, inside the resume window
    PROCESSING = "processing"     # capture done, jobs still running
    COMPLETE = "complete"
    FAILED = "failed"


class SegmentStatus(str, Enum):
    RECORDING = "recording"       # streamlink is still writing this file
    READY = "ready"               # complete on disk, awaiting transcription
    TRANSCRIBED = "transcribed"
    FAILED = "failed"
    DELETED = "deleted"           # raw file reclaimed, transcript retained


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _loads(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


@dataclass
class Session:
    id: int
    channel: str
    mode: str                      # "live" | "offline"
    status: str
    started_at: float              # wall clock of stream second 0
    ended_at: float | None = None
    source: str | None = None      # VOD url / local path for offline mode
    twitch_stream_id: str | None = None
    title: str | None = None
    game: str | None = None
    vod_url: str | None = None
    # Stream seconds already committed to topics. Segmentation resumes here.
    topics_watermark: float = 0.0
    # Total recorded stream seconds, advanced as segments complete.
    duration: float = 0.0

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Session":
        return Session(
            id=row["id"],
            channel=row["channel"],
            mode=row["mode"],
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            source=row["source"],
            twitch_stream_id=row["twitch_stream_id"],
            title=row["title"],
            game=row["game"],
            vod_url=row["vod_url"],
            topics_watermark=row["topics_watermark"] or 0.0,
            duration=row["duration"] or 0.0,
        )


@dataclass
class Segment:
    id: int
    session_id: int
    seq: int
    path: str
    start: float                   # stream seconds at the first frame
    duration: float
    status: str
    bytes: int = 0
    created_at: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    def overlaps(self, start: float, end: float) -> bool:
        return self.start < end and self.end > start

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Segment":
        return Segment(
            id=row["id"],
            session_id=row["session_id"],
            seq=row["seq"],
            path=row["path"],
            start=row["start"],
            duration=row["duration"],
            status=row["status"],
            bytes=row["bytes"] or 0,
            created_at=row["created_at"] or 0.0,
        )


@dataclass
class Word:
    """One transcribed word at absolute stream time."""

    start: float
    end: float
    text: str
    probability: float = 1.0

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Word":
        return Word(
            start=row["start"],
            end=row["end"],
            text=row["text"],
            probability=row["probability"] if row["probability"] is not None else 1.0,
        )


@dataclass
class Utterance:
    """A whisper segment: a sentence-ish run of speech at stream time."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Utterance":
        return Utterance(start=row["start"], end=row["end"], text=row["text"])


@dataclass
class ChatMessage:
    ts: float                      # stream seconds
    user: str
    text: str
    wall_ts: float = 0.0
    emotes: list[str] = field(default_factory=list)

    @staticmethod
    def from_row(row: sqlite3.Row) -> "ChatMessage":
        return ChatMessage(
            ts=row["ts"],
            user=row["user"],
            text=row["text"],
            wall_ts=row["wall_ts"] or 0.0,
            emotes=_loads(row["emotes"], []),
        )


@dataclass
class Topic:
    id: int
    session_id: int
    idx: int
    start: float
    end: float
    label: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    # Which signals produced the opening boundary: "semantic", "llm", "both",
    # "session-start", "max-length".
    method: str = ""
    confidence: float = 0.0
    ranked: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Topic":
        return Topic(
            id=row["id"],
            session_id=row["session_id"],
            idx=row["idx"],
            start=row["start"],
            end=row["end"],
            label=row["label"],
            summary=row["summary"] or "",
            tags=_loads(row["tags"], []),
            method=row["method"] or "",
            confidence=row["confidence"] or 0.0,
            ranked=row["ranked"] or 0,
        )


@dataclass
class Clip:
    id: int
    session_id: int
    topic_id: int | None
    start: float
    end: float
    score: float
    status: str = "pending"
    path: str | None = None
    vertical_path: str | None = None
    subtitle_path: str | None = None
    excerpt: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    created_at: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Clip":
        return Clip(
            id=row["id"],
            session_id=row["session_id"],
            topic_id=row["topic_id"],
            start=row["start"],
            end=row["end"],
            score=row["score"],
            status=row["status"],
            path=row["path"],
            vertical_path=row["vertical_path"],
            subtitle_path=row["subtitle_path"],
            excerpt=row["excerpt"] or "",
            scores=_loads(row["scores"], {}),
            created_at=row["created_at"] or 0.0,
        )


@dataclass
class Job:
    id: int
    session_id: int
    kind: str
    payload: dict[str, Any]
    status: str
    attempts: int = 0
    priority: int = 100
    last_error: str | None = None
    lease_expires: float | None = None

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Job":
        return Job(
            id=row["id"],
            session_id=row["session_id"],
            kind=row["kind"],
            payload=_loads(row["payload"], {}),
            status=row["status"],
            attempts=row["attempts"] or 0,
            priority=row["priority"] or 100,
            last_error=row["last_error"],
            lease_expires=row["lease_expires"],
        )
