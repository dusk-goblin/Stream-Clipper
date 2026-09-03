"""Twitch IRC chat logger.

Connects read-only (anonymous ``justinfan`` login needs no credentials),
requests the tags capability so native emote spans are available, and writes
every PRIVMSG to JSONL *and* SQLite.

Timestamps are stream seconds: ``wall_clock - session.started_at``. That is
the same clock the transcript uses, so a chat burst can be lined up against
what was being said.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import ChatConfig
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import ChatMessage
from ..util.retry import backoff_delays
from ..util.timefmt import now_ts

log = get_logger(__name__)

# :nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :message text
_PRIVMSG = re.compile(
    r"^(?:@(?P<tags>[^ ]*) )?:(?P<nick>[^!]+)![^ ]+ PRIVMSG #(?P<channel>[^ ]+) :(?P<text>.*)$"
)


def parse_tags(raw: str) -> dict[str, str]:
    """IRCv3 tag string -> dict, with the standard escapes undone."""
    tags: dict[str, str] = {}
    for pair in raw.split(";"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        tags[key] = (
            value.replace(r"\s", " ")
            .replace(r"\:", ";")
            .replace(r"\\", "\\")
            .replace(r"\r", "\r")
            .replace(r"\n", "\n")
        )
    return tags


def extract_emotes(text: str, tags: dict[str, str], watchlist: Sequence[str]) -> list[str]:
    """Emote names in one message.

    Two sources, because Twitch only knows about its own:
      * native emotes, whose *positions* are in the ``emotes`` tag -- we slice
        the name back out of the message text;
      * third-party emotes (BTTV/FFZ/7TV -- KEKW, OMEGALUL and friends), which
        are plain words, matched against the configured watchlist.
    """
    found: list[str] = []

    raw = tags.get("emotes") or ""
    for chunk in raw.split("/"):
        if not chunk or ":" not in chunk:
            continue
        _, _, spans = chunk.partition(":")
        first = spans.split(",")[0]
        start_str, _, end_str = first.partition("-")
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            continue
        if 0 <= start <= end < len(text):
            found.append(text[start : end + 1])

    if watchlist:
        tokens = set(text.split())
        found.extend(name for name in watchlist if name in tokens)

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return [e for e in found if not (e in seen or seen.add(e))]


@dataclass
class ChatStats:
    messages: int = 0
    reconnects: int = 0


class ChatLogger:
    """Reconnecting IRC reader for one channel and one recording session."""

    def __init__(
        self,
        config: ChatConfig,
        channel: str,
        db: Database,
        session_id: int,
        session_start: float,
        jsonl_path: Path,
        emote_watchlist: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.channel = channel.lower()
        self.db = db
        self.session_id = session_id
        self.session_start = session_start
        self.jsonl_path = jsonl_path
        self.emote_watchlist = list(emote_watchlist)
        self.stats = ChatStats()
        self._buffer: list[ChatMessage] = []
        self._buffer_lock = asyncio.Lock()

    async def run(self, stop: asyncio.Event) -> ChatStats:
        """Stay connected until ``stop``, reconnecting with backoff."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        flusher = asyncio.create_task(self._flush_loop(stop), name="chat-flush")
        attempt = 0
        try:
            while not stop.is_set():
                try:
                    await self._session(stop)
                    attempt = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if stop.is_set():
                        break
                    self.stats.reconnects += 1
                    delays = backoff_delays(8, base=2.0, cap=60.0)
                    delay = delays[min(attempt, len(delays) - 1)] if delays else 5.0
                    attempt += 1
                    log.warning(
                        "chat.reconnect",
                        extra={
                            "session_id": self.session_id,
                            "delay": round(delay, 1),
                            "reason": str(exc)[:200],
                        },
                    )
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=delay)
        finally:
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flusher
            await self._flush()
        return self.stats

    async def _session(self, stop: asyncio.Event) -> None:
        reader, writer = await asyncio.open_connection(self.config.server, self.config.port)
        try:
            token = self.config.resolve_token()
            nick = self.config.nick or f"justinfan{random.randint(10_000, 99_999)}"
            lines = ["CAP REQ :twitch.tv/tags twitch.tv/commands"]
            if token:
                bearer = token if token.startswith("oauth:") else f"oauth:{token}"
                lines.append(f"PASS {bearer}")
            lines += [f"NICK {nick}", f"JOIN #{self.channel}"]
            writer.write(("\r\n".join(lines) + "\r\n").encode())
            await writer.drain()
            log.info(
                "chat.connected",
                extra={"session_id": self.session_id, "channel": self.channel, "nick": nick},
            )

            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(reader.readline(), timeout=300)
                except asyncio.TimeoutError:
                    # Five minutes of silence on a busy channel means a dead
                    # socket, not a quiet chat.
                    raise ConnectionError("no data for 300s")
                if not raw:
                    raise ConnectionError("connection closed by server")
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("PING"):
                    writer.write(f"PONG {line[5:]}\r\n".encode())
                    await writer.drain()
                    continue
                message = self._parse(line)
                if message is not None:
                    async with self._buffer_lock:
                        self._buffer.append(message)
                    self.stats.messages += 1
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _parse(self, line: str) -> ChatMessage | None:
        match = _PRIVMSG.match(line)
        if not match:
            return None
        tags = parse_tags(match.group("tags") or "")
        text = match.group("text")
        wall = now_ts()
        # Prefer the server's own send time; it is immune to our read lag.
        if tags.get("tmi-sent-ts", "").isdigit():
            wall = int(tags["tmi-sent-ts"]) / 1000.0
        return ChatMessage(
            ts=wall - self.session_start,
            user=tags.get("display-name") or match.group("nick"),
            text=text,
            wall_ts=wall,
            emotes=extract_emotes(text, tags, self.emote_watchlist),
        )

    async def _flush_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self.config.flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        async with self._buffer_lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return
        await asyncio.to_thread(self._write_batch, batch)

    def _write_batch(self, batch: Sequence[ChatMessage]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for message in batch:
                handle.write(
                    json.dumps(
                        {
                            "ts": round(message.ts, 3),
                            "wall_ts": message.wall_ts,
                            "user": message.user,
                            "text": message.text,
                            "emotes": message.emotes,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        self.db.add_chat(self.session_id, batch)


def load_chat_jsonl(path: Path, session_start: float = 0.0) -> list[ChatMessage]:
    """Read a chat JSONL back in -- used by offline mode and `clips export`."""
    messages: list[ChatMessage] = []
    if not path.exists():
        return messages
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            wall = float(entry.get("wall_ts", 0.0) or 0.0)
            ts = entry.get("ts")
            if ts is None:
                ts = wall - session_start if wall else 0.0
            messages.append(
                ChatMessage(
                    ts=float(ts),
                    user=str(entry.get("user", "")),
                    text=str(entry.get("text", "")),
                    wall_ts=wall,
                    emotes=list(entry.get("emotes") or []),
                )
            )
    messages.sort(key=lambda m: m.ts)
    return messages


def ingest_chat_file(
    db: Database, session_id: int, path: Path, session_start: float = 0.0
) -> int:
    """Bulk-load a chat JSONL into a session. Returns the message count."""
    messages = load_chat_jsonl(path, session_start)
    db.add_chat(session_id, messages)
    return len(messages)
