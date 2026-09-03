"""Segmented capture of a live HLS stream.

streamlink pulls the HLS stream and writes it to stdout; ffmpeg's segment
muxer slices that byte stream into fixed-length files without re-encoding::

    streamlink --stdout twitch.tv/<ch> best | ffmpeg -i pipe:0 -c copy \\
        -f segment -segment_time 300 -segment_list <csv> out_%05d.ts

The segment muxer appends one CSV row per *completed* file, so tailing that
list is how we learn a segment is closed and safe to transcribe. Nothing
downstream ever touches the file ffmpeg is currently writing.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ..config import Config
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import SegmentStatus
from ..util.proc import require_binary

log = get_logger(__name__)

# How often the segment-list tailer wakes up.
_TAIL_INTERVAL = 2.0

SegmentCallback = Callable[[int, int], Awaitable[None]]
"""Called as ``(session_id, segment_id)`` once a segment file is complete."""


@dataclass
class RecorderResult:
    segments_written: int
    seconds_recorded: float
    exit_code: int
    error: str = ""


def read_segment_list(list_path: Path) -> list[tuple[str, float, float]]:
    """Parse the segment muxer's CSV: ``filename,start,end`` per row.

    The last row can be a partial write while ffmpeg is mid-flush, so
    malformed rows are skipped rather than raised. Offline mode reuses this
    to read the list its own split produced.
    """
    try:
        text = list_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[tuple[str, float, float]] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 3:
            continue
        try:
            rows.append((row[0].strip(), float(row[1]), float(row[2])))
        except ValueError:
            continue
    return rows


def seq_from_filename(filename: str) -> int | None:
    """``seg_00042.ts`` -> 42. None when the name does not carry a number."""
    _, _, digits = Path(filename).stem.rpartition("_")
    return int(digits) if digits.isdigit() else None


class Recorder:
    """One streamlink+ffmpeg capture run.

    A run ends when the stream drops. The caller decides whether to start
    another run against the same session (resume) or finalize it.
    """

    def __init__(
        self,
        config: Config,
        db: Database,
        session_id: int,
        on_segment: SegmentCallback | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.session_id = session_id
        self.on_segment = on_segment
        self._streamlink: asyncio.subprocess.Process | None = None
        self._ffmpeg: asyncio.subprocess.Process | None = None

    # -- command construction (pure, so it is testable without binaries) ---

    def streamlink_command(self, url: str) -> list[str]:
        return [
            require_binary("streamlink"),
            *self.config.capture.streamlink_args,
            "--stdout",
            url,
            self.config.capture.quality,
        ]

    def ffmpeg_command(self, out_pattern: str, list_path: Path, start_number: int) -> list[str]:
        return [
            require_binary("ffmpeg"),
            "-hide_banner",
            "-loglevel", "warning",
            "-i", "pipe:0",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.config.capture.segment_seconds),
            "-segment_start_number", str(start_number),
            "-reset_timestamps", "1",
            # +live keeps the list flushed as segments close, rather than at exit.
            "-segment_list", str(list_path),
            "-segment_list_type", "csv",
            "-segment_list_flags", "+live",
            out_pattern,
        ]

    # -- run ---------------------------------------------------------------

    async def run(self, url: str, stop: asyncio.Event) -> RecorderResult:
        """Capture until the stream ends or ``stop`` is set."""
        paths = self.config.paths
        session_dir = paths.segments / f"session_{self.session_id:05d}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Resume-safe: continue numbering and stream time where we left off.
        start_seq = self.db.next_segment_seq(self.session_id)
        time_offset = self.db.recorded_seconds(self.session_id)
        ext = self.config.capture.container
        pattern = str(session_dir / f"seg_%05d.{ext}")
        list_path = session_dir / f"segments_{start_seq:05d}.csv"

        log.info(
            "recorder.start",
            extra={
                "session_id": self.session_id,
                "start_seq": start_seq,
                "time_offset": round(time_offset, 2),
                "quality": self.config.capture.quality,
            },
        )

        self._streamlink = await asyncio.create_subprocess_exec(
            *self.streamlink_command(url),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._ffmpeg = await asyncio.create_subprocess_exec(
            *self.ffmpeg_command(pattern, list_path, start_seq),
            stdin=self._streamlink.stdout,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # The child owns the pipe now; drop our handle so ffmpeg sees EOF when
        # streamlink exits.
        if self._streamlink.stdout is not None:
            self._streamlink.stdout.close()

        tailer = asyncio.create_task(
            self._tail_segments(list_path, session_dir, time_offset, stop),
            name=f"segment-tail-{self.session_id}",
        )
        stderr_task = asyncio.create_task(
            self._drain_stderr(self._ffmpeg.stderr, "ffmpeg"), name="ffmpeg-stderr"
        )
        sl_stderr_task = asyncio.create_task(
            self._drain_stderr(self._streamlink.stderr, "streamlink"), name="sl-stderr"
        )

        waiter = asyncio.create_task(self._ffmpeg.wait(), name="ffmpeg-wait")
        stopper = asyncio.create_task(stop.wait(), name="stop-wait")
        try:
            await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                await self._terminate()
            exit_code = await waiter
        finally:
            stopper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stopper
            with contextlib.suppress(Exception):
                await self._streamlink.wait()
            # ffmpeg flushes the final CSV row on exit -- read it before stopping.
            tailer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tailer
            written = await self._scan_segments(list_path, session_dir, time_offset)
            for task in (stderr_task, sl_stderr_task):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        recorded = self.db.recorded_seconds(self.session_id)
        self.db.update_session(self.session_id, duration=recorded)
        log.info(
            "recorder.stop",
            extra={
                "session_id": self.session_id,
                "exit_code": exit_code,
                "segments": written,
                "recorded": round(recorded, 1),
            },
        )
        return RecorderResult(
            segments_written=written, seconds_recorded=recorded, exit_code=exit_code
        )

    async def _terminate(self) -> None:
        """Stop streamlink first so ffmpeg sees EOF and closes cleanly.

        Killing ffmpeg outright would leave the in-flight segment truncated
        and unlisted.
        """
        for proc, grace in ((self._streamlink, 5.0), (self._ffmpeg, 15.0)):
            if proc is None or proc.returncode is not None:
                continue
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace)
            except asyncio.TimeoutError:
                log.warning("recorder.kill", extra={"pid": proc.pid})
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

    async def _drain_stderr(self, stream: asyncio.StreamReader | None, who: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if text:
                log.debug("capture.stderr", extra={"proc": who, "line": text[:400]})

    async def _tail_segments(
        self, list_path: Path, session_dir: Path, time_offset: float, stop: asyncio.Event
    ) -> None:
        while True:
            try:
                await self._scan_segments(list_path, session_dir, time_offset)
            except Exception:  # a bad row must not take the recorder down
                log.exception("recorder.tail_failed", extra={"session_id": self.session_id})
            try:
                await asyncio.wait_for(stop.wait(), timeout=_TAIL_INTERVAL)
                return
            except asyncio.TimeoutError:
                continue

    async def _scan_segments(
        self, list_path: Path, session_dir: Path, time_offset: float
    ) -> int:
        """Register any newly listed segments. Returns how many are known now."""
        if not list_path.exists():
            return 0
        known = {seg.seq for seg in self.db.list_segments(self.session_id)}
        rows = read_segment_list(list_path)
        registered = 0
        for filename, seg_start, seg_end in rows:
            path = session_dir / filename
            seq = seq_from_filename(filename)
            if seq is None or seq in known:
                registered += 1 if seq in known else 0
                continue
            if not path.exists():
                continue
            duration = max(0.0, seg_end - seg_start)
            segment = self.db.add_segment(
                self.session_id,
                seq=seq,
                path=str(path),
                start=time_offset + seg_start,
                duration=duration,
                status=SegmentStatus.READY.value,
            )
            self.db.finish_segment(segment.id, duration, path.stat().st_size)
            self.db.update_session(
                self.session_id, duration=self.db.recorded_seconds(self.session_id)
            )
            log.info(
                "segment.ready",
                extra={
                    "session_id": self.session_id,
                    "seq": seq,
                    "start": round(segment.start, 1),
                    "duration": round(duration, 1),
                },
            )
            known.add(seq)
            registered += 1
            if self.on_segment is not None:
                await self.on_segment(self.session_id, segment.id)
        return registered
