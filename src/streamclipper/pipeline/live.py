"""Live recording orchestration.

Wires the monitor, the worker pools and the periodic topic sweep together on
one event loop, and shuts all of it down cleanly on SIGINT.

Shutdown is two-phase on purpose. The first Ctrl-C stops capture and lets the
queue drain, so segments already on disk still become clips. A second Ctrl-C
stops immediately -- the queue is durable, so `record` or `process` picks the
work back up next run.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from ..capture.monitor import StreamMonitor
from ..config import Config
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import SegmentStatus, Session, SessionStatus
from .queue import (
    JobKind,
    enqueue_finalize,
    enqueue_segment,
    enqueue_transcribe,
)
from .workers import JobRunner, WorkerPool

log = get_logger(__name__)

# How often to check whether enough new transcript has settled to segment.
_SWEEP_INTERVAL = 120.0


class LivePipeline:
    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db
        self.runner = JobRunner(config, db)
        self.stop = asyncio.Event()
        self.drain = asyncio.Event()
        self._pools: list[WorkerPool] = []
        self._active_sessions: set[int] = set()

    # -- lifecycle --------------------------------------------------------

    def _start_workers(self) -> None:
        runtime = self.config.runtime
        self._pools = [
            WorkerPool(
                self.config, self.db, self.runner, JobKind.HEAVY,
                size=max(1, self.config.transcribe.workers), name="transcribe",
            ),
            WorkerPool(
                self.config, self.db, self.runner, JobKind.LIGHT,
                size=max(1, runtime.workers), name="process",
            ),
        ]
        for pool in self._pools:
            pool.start()

    def _stop_workers(self) -> None:
        for pool in self._pools:
            pool.stop()
        self._pools.clear()

    def _install_signals(self, loop: asyncio.AbstractEventLoop) -> None:
        def handle() -> None:
            if not self.stop.is_set():
                log.info("shutdown.requested")
                print(
                    "\nStopping capture. Draining the queue -- Ctrl-C again to exit now.",
                    flush=True,
                )
                self.stop.set()
            else:
                log.info("shutdown.forced")
                self.drain.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, handle)

    # -- callbacks --------------------------------------------------------

    async def _on_segment(self, session_id: int, segment_id: int) -> None:
        self._active_sessions.add(session_id)
        if self.config.stages.transcribe:
            enqueue_transcribe(self.db, session_id, segment_id)

    async def _on_session_end(self, session: Session) -> None:
        log.info("pipeline.session_ended", extra={"session_id": session.id})
        self._active_sessions.add(session.id)

    # -- periodic ---------------------------------------------------------

    async def _sweep_loop(self) -> None:
        """Queue a topic sweep whenever enough transcript has settled."""
        while not self.stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.stop.wait(), timeout=_SWEEP_INTERVAL)
            if not self.config.stages.segment:
                continue
            for session_id in list(self._active_sessions):
                session = self.db.get_session(session_id)
                if session is None:
                    continue
                settled = (
                    self.db.transcribed_seconds(session_id)
                    - self.config.segment.settle_seconds
                )
                if settled - session.topics_watermark >= self.config.segment.min_topic_seconds:
                    enqueue_segment(self.db, session_id, final=False)

    # -- run --------------------------------------------------------------

    async def run(self, once: bool = False) -> None:
        loop = asyncio.get_running_loop()
        self._install_signals(loop)

        reclaimed = self.db.release_stale_jobs()
        if reclaimed:
            log.info("queue.reclaimed", extra={"jobs": reclaimed})
        self._resume_unfinished()

        self._start_workers()
        monitor = StreamMonitor(
            self.config,
            self.db,
            on_segment=self._on_segment,
            on_session_end=self._on_session_end,
        )
        sweeper = asyncio.create_task(self._sweep_loop(), name="topic-sweep")

        try:
            if self.config.stages.capture:
                await monitor.run(self.stop, once=once)
            else:
                log.info("capture.disabled")
                await self.stop.wait()
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            await self._finish()

    def _resume_unfinished(self) -> None:
        """Re-queue work for sessions a previous run left mid-flight."""
        for session in self.db.list_sessions(limit=20):
            if session.status in {
                SessionStatus.COMPLETE.value,
                SessionStatus.FAILED.value,
            }:
                continue
            self._active_sessions.add(session.id)
            pending = 0
            for segment in self.db.list_segments(
                session.id, statuses=[SegmentStatus.READY.value]
            ):
                if enqueue_transcribe(self.db, session.id, segment.id) is not None:
                    pending += 1
            if pending:
                log.info(
                    "session.resumed_jobs",
                    extra={"session_id": session.id, "segments": pending},
                )

    async def _finish(self) -> None:
        """Close out sessions, then let the queue drain before stopping."""
        for session_id in self._active_sessions:
            session = self.db.get_session(session_id)
            if session is None:
                continue
            if session.status not in {
                SessionStatus.COMPLETE.value,
                SessionStatus.FAILED.value,
            }:
                self.db.update_session(session_id, status=SessionStatus.PROCESSING.value)
            if self.config.stages.segment:
                enqueue_segment(self.db, session_id, final=True)
            enqueue_finalize(self.db, session_id)

        await self._drain()
        self._stop_workers()

    async def _drain(self, poll: float = 2.0) -> None:
        remaining = self.db.pending_job_count()
        if remaining:
            log.info("queue.draining", extra={"jobs": remaining})
            print(f"Draining {remaining} queued job(s)...", flush=True)
        while not self.drain.is_set():
            pending = self.db.pending_job_count()
            busy = any(pool.busy for pool in self._pools)
            if pending == 0 and not busy:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.drain.wait(), timeout=poll)
        log.info(
            "queue.drain_abandoned",
            extra={"remaining": self.db.pending_job_count()},
        )


async def run_live(config: Config, db: Database, once: bool = False) -> None:
    await LivePipeline(config, db).run(once=once)
