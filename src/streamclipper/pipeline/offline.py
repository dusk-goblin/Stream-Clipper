"""Offline processing of an existing VOD or local file.

Runs exactly the same stages as live capture -- the only difference is where
segments come from. A local file is split by ffmpeg into the same fixed-length
segments the recorder would have produced; a Twitch VOD URL is pulled by
streamlink first. From the segment table onward the two modes are
indistinguishable, which is what makes this a usable test path for the live
pipeline.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..capture.chat import ingest_chat_file
from ..capture.recorder import read_segment_list, seq_from_filename
from ..config import Config
from ..errors import StreamClipperError
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import SegmentStatus, Session, SessionStatus
from ..util.proc import require_binary, run
from ..util.timefmt import now_ts
from .queue import (
    JobKind,
    enqueue_finalize,
    enqueue_segment,
    enqueue_transcribe,
)
from .workers import JobRunner, WorkerPool

log = get_logger(__name__)

_TWITCH_VOD = re.compile(r"^https?://(?:www\.)?twitch\.tv/videos/(\d+)", re.I)
_URL = re.compile(r"^https?://", re.I)


def is_url(source: str) -> bool:
    return bool(_URL.match(source))


def vod_id(source: str) -> str | None:
    match = _TWITCH_VOD.match(source)
    return match.group(1) if match else None


class OfflinePipeline:
    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db
        self.runner = JobRunner(config, db)

    # -- ingest ------------------------------------------------------------

    def _create_session(self, source: str) -> Session:
        vod = vod_id(source)
        return self.db.create_session(
            channel=self.config.channel,
            mode="offline",
            # Offline has no real wall clock; stream second 0 is file second 0.
            started_at=now_ts(),
            source=source,
            vod_url=source if vod else None,
            title=Path(source).name if not is_url(source) else source,
        )

    def _download_vod(self, url: str, destination: Path) -> Path:
        """Pull a VOD to disk with streamlink before segmenting it."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        log.info("offline.downloading", extra={"url": url, "to": str(destination)})
        args = [
            require_binary("streamlink"),
            *self.config.capture.streamlink_args,
            "--force",
            "-o", str(destination),
            url,
            self.config.capture.quality,
        ]
        try:
            run(args, timeout=None)
        except subprocess.CalledProcessError as exc:
            raise StreamClipperError(
                f"streamlink could not download {url}: {(exc.stderr or '')[-600:]}"
            ) from exc
        if not destination.exists() or destination.stat().st_size == 0:
            raise StreamClipperError(f"streamlink produced no output for {url}")
        return destination

    def _split(self, media: Path, session: Session) -> int:
        """Split a media file into segments and register them."""
        session_dir = self.config.paths.segments / f"session_{session.id:05d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        ext = self.config.capture.container
        pattern = str(session_dir / f"seg_%05d.{ext}")
        list_path = session_dir / "segments.csv"

        log.info(
            "offline.splitting",
            extra={
                "session_id": session.id,
                "file": media.name,
                "segment_seconds": self.config.capture.segment_seconds,
            },
        )
        args = [
            require_binary("ffmpeg"),
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(media),
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.config.capture.segment_seconds),
            "-reset_timestamps", "1",
            "-segment_list", str(list_path),
            "-segment_list_type", "csv",
            pattern,
        ]
        try:
            run(args, timeout=None)
        except subprocess.CalledProcessError as exc:
            raise StreamClipperError(
                f"ffmpeg could not split {media.name}: {(exc.stderr or '')[-600:]}"
            ) from exc

        count = 0
        for filename, start, end in read_segment_list(list_path):
            path = session_dir / filename
            if not path.exists():
                continue
            seq = seq_from_filename(filename)
            if seq is None:
                continue
            segment = self.db.add_segment(
                session.id,
                seq=seq,
                path=str(path),
                start=start,
                duration=max(0.0, end - start),
                status=SegmentStatus.READY.value,
            )
            self.db.finish_segment(segment.id, segment.duration, path.stat().st_size)
            count += 1

        self.db.update_session(
            session.id, duration=self.db.recorded_seconds(session.id)
        )
        log.info("offline.split_done", extra={"session_id": session.id, "segments": count})
        return count

    # -- run ---------------------------------------------------------------

    def run(self, source: str, chat_file: Path | None = None) -> Session:
        """Process ``source`` end to end and return the finished session."""
        paths = self.config.paths
        paths.ensure()

        reclaimed = self.db.release_stale_jobs()
        if reclaimed:
            log.info("queue.reclaimed", extra={"jobs": reclaimed})

        session = self._create_session(source)

        if is_url(source):
            media = paths.root / "vods" / f"session_{session.id:05d}.mp4"
            media = self._download_vod(source, media)
        else:
            media = Path(source).expanduser().resolve()
            if not media.exists():
                self.db.update_session(session.id, status=SessionStatus.FAILED.value)
                raise StreamClipperError(f"Input file not found: {media}")

        segments = self._split(media, session)
        if segments == 0:
            self.db.update_session(session.id, status=SessionStatus.FAILED.value)
            raise StreamClipperError(
                f"No segments were produced from {media.name}. "
                "Check that it is a media file ffmpeg can read."
            )

        if chat_file is not None:
            count = ingest_chat_file(self.db, session.id, chat_file)
            log.info(
                "offline.chat_loaded",
                extra={"session_id": session.id, "messages": count},
            )

        if self.config.stages.transcribe:
            for segment in self.db.list_segments(
                session.id, statuses=[SegmentStatus.READY.value]
            ):
                enqueue_transcribe(self.db, session.id, segment.id)

        self.db.update_session(session.id, status=SessionStatus.PROCESSING.value)
        self._drain(session)

        # Segmentation, ranking and cutting only become runnable once the
        # transcript exists, so queue them after the first drain and drain
        # again. Each stage enqueues the next.
        if self.config.stages.segment:
            enqueue_segment(self.db, session.id, final=True)
        enqueue_finalize(self.db, session.id)
        self._drain(session)

        finished = self.db.get_session(session.id)
        assert finished is not None
        return finished

    def _drain(self, session: Session) -> None:
        """Run every queued job for this session to completion."""
        pools = [
            WorkerPool(
                self.config, self.db, self.runner, JobKind.HEAVY,
                size=max(1, self.config.transcribe.workers), name="transcribe",
            ),
            WorkerPool(
                self.config, self.db, self.runner, JobKind.LIGHT,
                size=max(1, self.config.runtime.workers), name="process",
            ),
        ]
        for pool in pools:
            pool.start()
        try:
            import time  # noqa: PLC0415

            while True:
                pending = self.db.pending_job_count(session.id)
                busy = any(pool.busy for pool in pools)
                if pending == 0 and not busy:
                    return
                time.sleep(self.config.runtime.poll_interval)
        finally:
            for pool in pools:
                pool.stop()


def run_offline(
    config: Config, db: Database, source: str, chat_file: Path | None = None
) -> Session:
    return OfflinePipeline(config, db).run(source, chat_file)
