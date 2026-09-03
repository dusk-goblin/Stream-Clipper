"""SQLite state store.

Everything a crash could otherwise lose lives here: which segments exist and
how far they are through the pipeline, the transcript, chat, topics, clips,
and the job queue. Restarting the process re-reads this and picks up where it
left off.

Thread-safety: one connection per thread, handed out by ``Database.conn``.
WAL mode lets the recorder's writes and a worker's reads proceed concurrently.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from ..logging_setup import get_logger
from ..util.timefmt import now_ts
from .models import (
    ChatMessage,
    Clip,
    Job,
    JobStatus,
    Segment,
    SegmentStatus,
    Session,
    SessionStatus,
    Topic,
    Utterance,
    Word,
)

log = get_logger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    channel          TEXT    NOT NULL,
    mode             TEXT    NOT NULL DEFAULT 'live',
    status           TEXT    NOT NULL,
    started_at       REAL    NOT NULL,
    ended_at         REAL,
    source           TEXT,
    twitch_stream_id TEXT,
    title            TEXT,
    game             TEXT,
    vod_url          TEXT,
    topics_watermark REAL    NOT NULL DEFAULT 0,
    duration         REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);

CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    path       TEXT    NOT NULL,
    start      REAL    NOT NULL,
    duration   REAL    NOT NULL DEFAULT 0,
    status     TEXT    NOT NULL,
    bytes      INTEGER NOT NULL DEFAULT 0,
    created_at REAL    NOT NULL DEFAULT 0,
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id, start);
CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);

-- Whisper segments: the unit we build sentences from.
CREATE TABLE IF NOT EXISTS utterances (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    segment_id INTEGER REFERENCES segments(id) ON DELETE SET NULL,
    start      REAL    NOT NULL,
    end        REAL    NOT NULL,
    text       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utterances_session ON utterances(session_id, start);

CREATE TABLE IF NOT EXISTS words (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    segment_id  INTEGER REFERENCES segments(id) ON DELETE SET NULL,
    start       REAL    NOT NULL,
    end         REAL    NOT NULL,
    text        TEXT    NOT NULL,
    probability REAL
);
CREATE INDEX IF NOT EXISTS idx_words_session ON words(session_id, start);

CREATE TABLE IF NOT EXISTS chat (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts         REAL    NOT NULL,
    wall_ts    REAL    NOT NULL DEFAULT 0,
    user       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    emotes     TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat(session_id, ts);

CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    start      REAL    NOT NULL,
    end        REAL    NOT NULL,
    label      TEXT    NOT NULL,
    summary    TEXT,
    tags       TEXT,
    method     TEXT,
    confidence REAL    NOT NULL DEFAULT 0,
    ranked     INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, idx)
);
CREATE INDEX IF NOT EXISTS idx_topics_session ON topics(session_id, start);

CREATE TABLE IF NOT EXISTS clips (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    topic_id      INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    start         REAL    NOT NULL,
    end           REAL    NOT NULL,
    score         REAL    NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'pending',
    path          TEXT,
    vertical_path TEXT,
    subtitle_path TEXT,
    excerpt       TEXT,
    scores        TEXT,
    created_at    REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_clips_session ON clips(session_id, start);
CREATE INDEX IF NOT EXISTS idx_clips_topic ON clips(topic_id);

CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind          TEXT    NOT NULL,
    payload       TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    priority      INTEGER NOT NULL DEFAULT 100,
    last_error    TEXT,
    lease_expires REAL,
    dedupe_key    TEXT,
    created_at    REAL    NOT NULL DEFAULT 0,
    updated_at    REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, priority, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedupe
    ON jobs(session_id, kind, dedupe_key) WHERE dedupe_key IS NOT NULL;
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a tuned connection with row access by column name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path), timeout=30.0, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


class Database:
    """Typed accessors over the schema, with a per-thread connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._migrate()

    # -- connection management --------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self.path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _migrate(self) -> None:
        conn = self.conn
        with self._write_lock:
            conn.executescript(SCHEMA)
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database at {self.path} was written by a newer stream-clipper "
                    f"(schema v{row['value']}, this build understands v{SCHEMA_VERSION})."
                )

    def _write(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            return self.conn.execute(sql, params)

    def _writemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        batch = list(rows)
        if not batch:
            return
        with self._write_lock:
            self.conn.executemany(sql, batch)

    # -- sessions ----------------------------------------------------------

    def create_session(
        self,
        channel: str,
        mode: str,
        started_at: float,
        source: str | None = None,
        twitch_stream_id: str | None = None,
        title: str | None = None,
        game: str | None = None,
        vod_url: str | None = None,
    ) -> Session:
        cur = self._write(
            """INSERT INTO sessions
               (channel, mode, status, started_at, source, twitch_stream_id,
                title, game, vod_url)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                channel,
                mode,
                SessionStatus.RECORDING.value,
                started_at,
                source,
                twitch_stream_id,
                title,
                game,
                vod_url,
            ),
        )
        session = self.get_session(int(cur.lastrowid))
        assert session is not None
        log.info(
            "session.created",
            extra={"session_id": session.id, "channel": channel, "mode": mode},
        )
        return session

    def get_session(self, session_id: int) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return Session.from_row(row) if row else None

    def latest_session(self, channel: str | None = None) -> Session | None:
        if channel:
            row = self.conn.execute(
                "SELECT * FROM sessions WHERE channel = ? ORDER BY id DESC LIMIT 1",
                (channel,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return Session.from_row(row) if row else None

    def resumable_session(self, channel: str, within: float) -> Session | None:
        """The most recent session for ``channel`` that a reconnect may rejoin.

        A session qualifies while it is still marked recording/interrupted and
        its last activity is inside the resume window.
        """
        row = self.conn.execute(
            """SELECT * FROM sessions
               WHERE channel = ? AND mode = 'live' AND status IN (?, ?)
               ORDER BY id DESC LIMIT 1""",
            (channel, SessionStatus.RECORDING.value, SessionStatus.INTERRUPTED.value),
        ).fetchone()
        if row is None:
            return None
        session = Session.from_row(row)
        last_seen = (session.ended_at or 0.0) or (session.started_at + session.duration)
        if now_ts() - last_seen > within:
            return None
        return session

    def list_sessions(self, limit: int = 50) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [Session.from_row(r) for r in rows]

    def update_session(self, session_id: int, **fields: Any) -> None:
        allowed = {
            "status", "ended_at", "title", "game", "twitch_stream_id",
            "vod_url", "topics_watermark", "duration", "source",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        self._write(
            f"UPDATE sessions SET {assignments} WHERE id = ?",
            (*updates.values(), session_id),
        )

    # -- segments ----------------------------------------------------------

    def add_segment(
        self, session_id: int, seq: int, path: str, start: float,
        duration: float = 0.0, status: str = SegmentStatus.RECORDING.value,
    ) -> Segment:
        self._write(
            """INSERT OR REPLACE INTO segments
               (session_id, seq, path, start, duration, status, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, seq, path, start, duration, status, now_ts()),
        )
        row = self.conn.execute(
            "SELECT * FROM segments WHERE session_id = ? AND seq = ?", (session_id, seq)
        ).fetchone()
        return Segment.from_row(row)

    def finish_segment(
        self, segment_id: int, duration: float, size_bytes: int
    ) -> None:
        self._write(
            """UPDATE segments SET duration = ?, bytes = ?, status = ?
               WHERE id = ?""",
            (duration, size_bytes, SegmentStatus.READY.value, segment_id),
        )

    def set_segment_status(self, segment_id: int, status: str) -> None:
        self._write("UPDATE segments SET status = ? WHERE id = ?", (status, segment_id))

    def get_segment(self, segment_id: int) -> Segment | None:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        return Segment.from_row(row) if row else None

    def list_segments(
        self, session_id: int, statuses: Sequence[str] | None = None
    ) -> list[Segment]:
        sql = "SELECT * FROM segments WHERE session_id = ?"
        params: list[Any] = [session_id]
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params.extend(statuses)
        sql += " ORDER BY seq"
        return [Segment.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def segments_covering(
        self, session_id: int, start: float, end: float
    ) -> list[Segment]:
        """Segments whose recorded span intersects ``[start, end)``, in order."""
        rows = self.conn.execute(
            """SELECT * FROM segments
               WHERE session_id = ? AND status != ?
                 AND start < ? AND (start + duration) > ?
               ORDER BY seq""",
            (session_id, SegmentStatus.DELETED.value, end, start),
        ).fetchall()
        return [Segment.from_row(r) for r in rows]

    def next_segment_seq(self, session_id: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) AS s FROM segments WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["s"]) + 1

    def recorded_seconds(self, session_id: int) -> float:
        """Stream time covered by segments -- where the next one starts."""
        row = self.conn.execute(
            """SELECT COALESCE(MAX(start + duration), 0) AS e
               FROM segments WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        return float(row["e"] or 0.0)

    def transcribed_seconds(self, session_id: int) -> float:
        """Stream time with a finished transcript -- the settle watermark."""
        row = self.conn.execute(
            """SELECT COALESCE(MAX(start + duration), 0) AS e
               FROM segments
               WHERE session_id = ? AND status IN (?, ?)""",
            (session_id, SegmentStatus.TRANSCRIBED.value, SegmentStatus.DELETED.value),
        ).fetchone()
        return float(row["e"] or 0.0)

    # -- transcript --------------------------------------------------------

    def add_transcript(
        self,
        session_id: int,
        segment_id: int | None,
        utterances: Sequence[Utterance],
    ) -> None:
        """Store one segment's transcription. Idempotent per segment."""
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                if segment_id is not None:
                    conn.execute(
                        "DELETE FROM utterances WHERE segment_id = ?", (segment_id,)
                    )
                    conn.execute(
                        "DELETE FROM words WHERE segment_id = ?", (segment_id,)
                    )
                conn.executemany(
                    """INSERT INTO utterances (session_id, segment_id, start, end, text)
                       VALUES (?,?,?,?,?)""",
                    [
                        (session_id, segment_id, u.start, u.end, u.text)
                        for u in utterances
                    ],
                )
                conn.executemany(
                    """INSERT INTO words
                       (session_id, segment_id, start, end, text, probability)
                       VALUES (?,?,?,?,?,?)""",
                    [
                        (session_id, segment_id, w.start, w.end, w.text, w.probability)
                        for u in utterances
                        for w in u.words
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def utterances(
        self, session_id: int, start: float = 0.0, end: float | None = None
    ) -> list[Utterance]:
        sql = "SELECT * FROM utterances WHERE session_id = ? AND end > ?"
        params: list[Any] = [session_id, start]
        if end is not None:
            sql += " AND start < ?"
            params.append(end)
        sql += " ORDER BY start"
        return [Utterance.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def words(
        self, session_id: int, start: float = 0.0, end: float | None = None
    ) -> list[Word]:
        sql = "SELECT * FROM words WHERE session_id = ? AND end > ?"
        params: list[Any] = [session_id, start]
        if end is not None:
            sql += " AND start < ?"
            params.append(end)
        sql += " ORDER BY start"
        return [Word.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def transcript_text(
        self, session_id: int, start: float = 0.0, end: float | None = None
    ) -> str:
        return " ".join(
            u.text.strip() for u in self.utterances(session_id, start, end) if u.text.strip()
        )

    # -- chat --------------------------------------------------------------

    def add_chat(self, session_id: int, messages: Sequence[ChatMessage]) -> None:
        self._writemany(
            """INSERT INTO chat (session_id, ts, wall_ts, user, text, emotes)
               VALUES (?,?,?,?,?,?)""",
            [
                (
                    session_id,
                    m.ts,
                    m.wall_ts,
                    m.user,
                    m.text,
                    json.dumps(m.emotes) if m.emotes else None,
                )
                for m in messages
            ],
        )

    def chat_between(
        self, session_id: int, start: float, end: float
    ) -> list[ChatMessage]:
        rows = self.conn.execute(
            """SELECT * FROM chat
               WHERE session_id = ? AND ts >= ? AND ts < ? ORDER BY ts""",
            (session_id, start, end),
        ).fetchall()
        return [ChatMessage.from_row(r) for r in rows]

    def chat_count(self, session_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chat WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row["c"])

    # -- topics ------------------------------------------------------------

    def add_topic(
        self,
        session_id: int,
        idx: int,
        start: float,
        end: float,
        label: str,
        summary: str = "",
        tags: Sequence[str] = (),
        method: str = "",
        confidence: float = 0.0,
    ) -> Topic:
        cur = self._write(
            """INSERT OR REPLACE INTO topics
               (session_id, idx, start, end, label, summary, tags, method, confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                session_id, idx, start, end, label, summary,
                json.dumps(list(tags)), method, confidence,
            ),
        )
        topic = self.get_topic(int(cur.lastrowid))
        assert topic is not None
        return topic

    def get_topic(self, topic_id: int) -> Topic | None:
        row = self.conn.execute(
            "SELECT * FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        return Topic.from_row(row) if row else None

    def list_topics(self, session_id: int) -> list[Topic]:
        rows = self.conn.execute(
            "SELECT * FROM topics WHERE session_id = ? ORDER BY idx", (session_id,)
        ).fetchall()
        return [Topic.from_row(r) for r in rows]

    def next_topic_idx(self, session_id: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(idx), -1) AS i FROM topics WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["i"]) + 1

    def mark_topic_ranked(self, topic_id: int) -> None:
        self._write("UPDATE topics SET ranked = 1 WHERE id = ?", (topic_id,))

    # -- clips -------------------------------------------------------------

    def add_clip(
        self,
        session_id: int,
        topic_id: int | None,
        start: float,
        end: float,
        score: float,
        excerpt: str = "",
        scores: dict[str, float] | None = None,
    ) -> Clip:
        cur = self._write(
            """INSERT INTO clips
               (session_id, topic_id, start, end, score, excerpt, scores, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                session_id, topic_id, start, end, score, excerpt,
                json.dumps(scores or {}), now_ts(),
            ),
        )
        clip = self.get_clip(int(cur.lastrowid))
        assert clip is not None
        return clip

    def get_clip(self, clip_id: int) -> Clip | None:
        row = self.conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        return Clip.from_row(row) if row else None

    def update_clip(self, clip_id: int, **fields: Any) -> None:
        allowed = {"status", "path", "vertical_path", "subtitle_path", "score", "excerpt"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        self._write(
            f"UPDATE clips SET {assignments} WHERE id = ?", (*updates.values(), clip_id)
        )

    def list_clips(
        self, session_id: int | None = None, status: str | None = None
    ) -> list[Clip]:
        sql = "SELECT * FROM clips WHERE 1=1"
        params: list[Any] = []
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY session_id, start"
        return [Clip.from_row(r) for r in self.conn.execute(sql, params).fetchall()]

    def clips_overlapping(self, session_id: int, start: float, end: float) -> list[Clip]:
        rows = self.conn.execute(
            """SELECT * FROM clips
               WHERE session_id = ? AND start < ? AND end > ? ORDER BY start""",
            (session_id, end, start),
        ).fetchall()
        return [Clip.from_row(r) for r in rows]

    # -- jobs --------------------------------------------------------------

    def enqueue(
        self,
        session_id: int,
        kind: str,
        payload: dict[str, Any] | None = None,
        priority: int = 100,
        dedupe_key: str | None = None,
    ) -> int | None:
        """Add a job. Returns None if ``dedupe_key`` already has a live job."""
        ts = now_ts()
        try:
            cur = self._write(
                """INSERT INTO jobs
                   (session_id, kind, payload, status, priority, dedupe_key,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    session_id, kind, json.dumps(payload or {}),
                    JobStatus.PENDING.value, priority, dedupe_key, ts, ts,
                ),
            )
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            # dedupe_key collision: an identical job is already queued or ran.
            return None

    def claim_job(
        self, kinds: Sequence[str], lease_seconds: float, max_attempts: int
    ) -> Job | None:
        """Atomically take the next runnable job of one of ``kinds``.

        Also reclaims jobs whose lease expired -- that is how a job survives
        the worker holding it being killed.
        """
        if not kinds:
            return None
        ts = now_ts()
        placeholders = ",".join("?" * len(kinds))
        with self._write_lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    f"""SELECT * FROM jobs
                        WHERE kind IN ({placeholders})
                          AND attempts < ?
                          AND (status = 'pending'
                               OR (status = 'running' AND lease_expires < ?))
                        ORDER BY priority, id LIMIT 1""",
                    (*kinds, max_attempts, ts),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                conn.execute(
                    """UPDATE jobs
                       SET status = ?, attempts = attempts + 1,
                           lease_expires = ?, updated_at = ?
                       WHERE id = ?""",
                    (JobStatus.RUNNING.value, ts + lease_seconds, ts, row["id"]),
                )
                claimed = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (row["id"],)
                ).fetchone()
                conn.execute("COMMIT")
                return Job.from_row(claimed)
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def finish_job(self, job_id: int) -> None:
        self._write(
            "UPDATE jobs SET status = ?, lease_expires = NULL, updated_at = ? WHERE id = ?",
            (JobStatus.DONE.value, now_ts(), job_id),
        )

    def fail_job(self, job_id: int, error: str, max_attempts: int) -> None:
        """Return a job to the queue, or bury it once it is out of attempts."""
        row = self.conn.execute(
            "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        attempts = int(row["attempts"]) if row else max_attempts
        status = (
            JobStatus.FAILED.value if attempts >= max_attempts else JobStatus.PENDING.value
        )
        self._write(
            """UPDATE jobs SET status = ?, last_error = ?, lease_expires = NULL,
               updated_at = ? WHERE id = ?""",
            (status, error[:2000], now_ts(), job_id),
        )

    def release_stale_jobs(self) -> int:
        """Reset jobs left 'running' by a process that died. Called at startup."""
        cur = self._write(
            """UPDATE jobs SET status = 'pending', lease_expires = NULL
               WHERE status = 'running'"""
        )
        return cur.rowcount or 0

    def pending_job_count(self, session_id: int | None = None) -> int:
        sql = "SELECT COUNT(*) AS c FROM jobs WHERE status IN ('pending','running')"
        params: list[Any] = []
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        return int(self.conn.execute(sql, params).fetchone()["c"])

    def jobs_for_session(self, session_id: int) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [Job.from_row(r) for r in rows]

    def iter_all_clips(self) -> Iterator[Clip]:
        for row in self.conn.execute("SELECT * FROM clips ORDER BY id"):
            yield Clip.from_row(row)
