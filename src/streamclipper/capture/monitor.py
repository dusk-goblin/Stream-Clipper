"""Live-status polling and session lifecycle.

Polls Helix for the channel's stream status. When it goes live we open (or
re-open) a recording session and supervise the recorder. A drop does not end
the session immediately: the monitor keeps polling for ``resume_window``
seconds and, if the stream returns, resumes capture into the *same* session
so the transcript, chat log and topic timeline stay continuous.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Awaitable, Callable

from ..config import Config
from ..errors import TwitchAPIError
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import Session, SessionStatus
from ..util.retry import sleep_unless_stopped
from ..util.timefmt import now_ts
from .chat import ChatLogger
from .recorder import Recorder
from .twitch import StreamInfo, TwitchClient

log = get_logger(__name__)

SegmentCallback = Callable[[int, int], Awaitable[None]]
SessionCallback = Callable[[Session], Awaitable[None]]


class StreamMonitor:
    """Watches one channel and keeps a recording session alive across drops."""

    def __init__(
        self,
        config: Config,
        db: Database,
        on_segment: SegmentCallback | None = None,
        on_session_end: SessionCallback | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.on_segment = on_segment
        self.on_session_end = on_session_end
        self.channel = config.channel.lower()
        self._chat_task: asyncio.Task[object] | None = None
        self._chat_stop: asyncio.Event | None = None

    async def run(self, stop: asyncio.Event, once: bool = False) -> None:
        """Poll until ``stop``. With ``once``, return after one session ends."""
        async with TwitchClient(self.config.twitch) as twitch:
            while not stop.is_set():
                try:
                    stream = await twitch.get_stream(self.channel)
                except TwitchAPIError:
                    log.exception("monitor.poll_failed", extra={"channel": self.channel})
                    stream = None
                except Exception:
                    log.exception("monitor.poll_error", extra={"channel": self.channel})
                    stream = None

                if stream is None:
                    log.debug("monitor.offline", extra={"channel": self.channel})
                    if await sleep_unless_stopped(
                        self.config.twitch.poll_interval, stop
                    ):
                        return
                    continue

                await self._record_session(twitch, stream, stop)
                if once or stop.is_set():
                    return

    # -- session ----------------------------------------------------------

    async def _record_session(
        self, twitch: TwitchClient, stream: StreamInfo, stop: asyncio.Event
    ) -> None:
        session = self._open_session(stream)
        url = f"https://twitch.tv/{self.channel}"

        if self.config.stages.chat:
            await self._start_chat(session)

        vod_task = asyncio.create_task(
            self._resolve_vod(twitch, session), name="vod-lookup"
        )
        try:
            restarts = 0
            while not stop.is_set():
                recorder = Recorder(
                    self.config, self.db, session.id, on_segment=self.on_segment
                )
                result = await recorder.run(url, stop)
                if stop.is_set():
                    break

                if result.segments_written == 0:
                    restarts += 1
                    if restarts >= self.config.capture.max_restarts:
                        log.error(
                            "monitor.giving_up",
                            extra={"session_id": session.id, "restarts": restarts},
                        )
                        break
                else:
                    restarts = 0

                # Capture ended. Wait out the resume window, checking whether
                # the stream comes back before we close the session.
                self.db.update_session(
                    session.id,
                    status=SessionStatus.INTERRUPTED.value,
                    ended_at=now_ts(),
                )
                if not await self._await_resume(twitch, session, stop):
                    break
                self.db.update_session(
                    session.id, status=SessionStatus.RECORDING.value, ended_at=None
                )
                log.info("monitor.resumed", extra={"session_id": session.id})
        finally:
            vod_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await vod_task
            await self._stop_chat()
            await self._close_session(session)

    def _open_session(self, stream: StreamInfo) -> Session:
        """Reuse an interrupted session for the same stream id, else start one."""
        existing = self.db.resumable_session(
            self.channel, within=self.config.twitch.resume_window
        )
        if existing is not None and (
            not existing.twitch_stream_id or existing.twitch_stream_id == stream.id
        ):
            self.db.update_session(
                existing.id,
                status=SessionStatus.RECORDING.value,
                ended_at=None,
                title=stream.title or existing.title,
                game=stream.game_name or existing.game,
                twitch_stream_id=stream.id,
            )
            log.info(
                "session.resumed",
                extra={"session_id": existing.id, "stream_id": stream.id},
            )
            refreshed = self.db.get_session(existing.id)
            assert refreshed is not None
            return refreshed

        return self.db.create_session(
            channel=self.channel,
            mode="live",
            started_at=now_ts(),
            twitch_stream_id=stream.id,
            title=stream.title,
            game=stream.game_name,
        )

    async def _await_resume(
        self, twitch: TwitchClient, session: Session, stop: asyncio.Event
    ) -> bool:
        """Poll through the resume window. True if the stream came back."""
        deadline = now_ts() + self.config.twitch.resume_window
        log.info(
            "monitor.awaiting_resume",
            extra={
                "session_id": session.id,
                "window": self.config.twitch.resume_window,
            },
        )
        while now_ts() < deadline:
            if await sleep_unless_stopped(self.config.twitch.live_poll_interval, stop):
                return False
            try:
                stream = await twitch.get_stream(self.channel)
            except Exception:
                log.exception("monitor.resume_poll_failed")
                continue
            if stream is None:
                continue
            if session.twitch_stream_id and stream.id != session.twitch_stream_id:
                # A genuinely new broadcast -- it deserves its own session.
                log.info(
                    "monitor.new_broadcast",
                    extra={"old": session.twitch_stream_id, "new": stream.id},
                )
                return False
            return True
        return False

    async def _close_session(self, session: Session) -> None:
        self.db.update_session(
            session.id,
            status=SessionStatus.PROCESSING.value,
            ended_at=now_ts(),
            duration=self.db.recorded_seconds(session.id),
        )
        refreshed = self.db.get_session(session.id)
        log.info(
            "session.capture_complete",
            extra={
                "session_id": session.id,
                "duration": round(refreshed.duration if refreshed else 0.0, 1),
            },
        )
        if self.on_session_end and refreshed is not None:
            await self.on_session_end(refreshed)

    # -- side tasks -------------------------------------------------------

    async def _start_chat(self, session: Session) -> None:
        if self._chat_task is not None and not self._chat_task.done():
            return
        self._chat_stop = asyncio.Event()
        logger = ChatLogger(
            self.config.chat,
            self.channel,
            self.db,
            session.id,
            session.started_at,
            self.config.paths.chat / f"session_{session.id:05d}.jsonl",
            emote_watchlist=self.config.highlight.emotes,
        )
        self._chat_task = asyncio.create_task(
            logger.run(self._chat_stop), name=f"chat-{session.id}"
        )

    async def _stop_chat(self) -> None:
        if self._chat_stop is not None:
            self._chat_stop.set()
        if self._chat_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._chat_task, timeout=15)
        self._chat_task = None
        self._chat_stop = None

    async def _resolve_vod(self, twitch: TwitchClient, session: Session) -> None:
        """Twitch publishes the archive a little after going live -- poll for it."""
        for _ in range(20):
            await asyncio.sleep(30)
            try:
                vod = await twitch.latest_vod(self.channel, session.twitch_stream_id)
            except Exception:
                continue
            if vod and vod.url:
                self.db.update_session(session.id, vod_url=vod.url)
                log.info("session.vod", extra={"session_id": session.id, "vod": vod.url})
                return
