"""Job kinds and the enqueue helpers.

The queue itself lives in SQLite (see ``state.db``), which is what makes the
pipeline resumable: a job survives the process that queued it, and a job
claimed by a worker that died is reclaimed when its lease expires.

Priorities are ordered so the pipeline drains front to back -- transcription
first, because everything downstream is blocked on it, and finalize last.
"""

from __future__ import annotations

from ..logging_setup import get_logger
from ..state import Database

log = get_logger(__name__)


class JobKind:
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    RANK = "rank"
    CUT = "cut"
    FINALIZE = "finalize"

    ALL = (TRANSCRIBE, SEGMENT, RANK, CUT, FINALIZE)
    # Transcription is pinned to its own worker set so a single GPU is not
    # contended, and so ffmpeg cutting never waits behind a whisper run.
    HEAVY = (TRANSCRIBE,)
    LIGHT = (SEGMENT, RANK, CUT, FINALIZE)


PRIORITY = {
    JobKind.TRANSCRIBE: 10,
    JobKind.SEGMENT: 20,
    JobKind.RANK: 30,
    JobKind.CUT: 40,
    JobKind.FINALIZE: 90,
}


def enqueue_transcribe(db: Database, session_id: int, segment_id: int) -> int | None:
    return db.enqueue(
        session_id,
        JobKind.TRANSCRIBE,
        {"segment_id": segment_id},
        priority=PRIORITY[JobKind.TRANSCRIBE],
        dedupe_key=f"segment:{segment_id}",
    )


def enqueue_segment(db: Database, session_id: int, final: bool = False) -> int | None:
    # Sweeps are deduped per generation: while one is queued there is no point
    # queueing another, but a *final* sweep must always get through.
    key = "final" if final else f"sweep:{db.next_topic_idx(session_id)}"
    return db.enqueue(
        session_id,
        JobKind.SEGMENT,
        {"final": final},
        priority=PRIORITY[JobKind.SEGMENT],
        dedupe_key=key,
    )


def enqueue_rank(db: Database, session_id: int, topic_id: int) -> int | None:
    return db.enqueue(
        session_id,
        JobKind.RANK,
        {"topic_id": topic_id},
        priority=PRIORITY[JobKind.RANK],
        dedupe_key=f"topic:{topic_id}",
    )


def enqueue_cut(db: Database, session_id: int, clip_id: int) -> int | None:
    return db.enqueue(
        session_id,
        JobKind.CUT,
        {"clip_id": clip_id},
        priority=PRIORITY[JobKind.CUT],
        dedupe_key=f"clip:{clip_id}",
    )


def enqueue_finalize(db: Database, session_id: int) -> int | None:
    return db.enqueue(
        session_id,
        JobKind.FINALIZE,
        {},
        priority=PRIORITY[JobKind.FINALIZE],
        dedupe_key="finalize",
    )
