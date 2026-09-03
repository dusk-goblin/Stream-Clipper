"""Twitch Helix client.

Only what the pipeline needs: an app access token, stream-live status, and
the VOD that a live stream is being archived to (so clip manifests can carry
a real VOD offset link).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import TwitchConfig
from ..errors import RetryableError, TwitchAPIError
from ..logging_setup import get_logger
from ..util.retry import async_retry
from ..util.timefmt import now_ts

log = get_logger(__name__)


@dataclass
class StreamInfo:
    """A live stream, as Helix reports it."""

    id: str
    user_login: str
    user_name: str
    title: str
    game_name: str
    started_at: str
    viewer_count: int

    @staticmethod
    def from_json(data: dict[str, Any]) -> "StreamInfo":
        return StreamInfo(
            id=str(data.get("id", "")),
            user_login=data.get("user_login", ""),
            user_name=data.get("user_name", ""),
            title=data.get("title", ""),
            game_name=data.get("game_name", ""),
            started_at=data.get("started_at", ""),
            viewer_count=int(data.get("viewer_count", 0) or 0),
        )


@dataclass
class VodInfo:
    id: str
    url: str
    created_at: str
    stream_id: str | None


class TwitchClient:
    """Async Helix client with client-credentials auth and retry on 5xx/429."""

    def __init__(self, config: TwitchConfig) -> None:
        self.config = config
        self._client_id, self._secret = config.resolve_credentials()
        self._token: str | None = None
        self._token_expires: float = 0.0
        self._http = httpx.AsyncClient(timeout=20.0)
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "TwitchClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- auth --------------------------------------------------------------

    async def _ensure_token(self, force: bool = False) -> str:
        async with self._token_lock:
            # 60s of slack so a token cannot expire mid-flight.
            if not force and self._token and now_ts() < self._token_expires - 60:
                return self._token
            response = await self._http.post(
                f"{self.config.auth_base}/token",
                data={
                    "client_id": self._client_id,
                    "client_secret": self._secret,
                    "grant_type": "client_credentials",
                },
            )
            if response.status_code != 200:
                raise TwitchAPIError(
                    f"Token request failed ({response.status_code}): {response.text[:200]}. "
                    "Check TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET."
                )
            payload = response.json()
            self._token = payload["access_token"]
            self._token_expires = now_ts() + float(payload.get("expires_in", 3600))
            log.debug("twitch.token.refreshed")
            return self._token

    @async_retry(attempts=5, base=2.0, cap=60.0, exceptions=(RetryableError, httpx.HTTPError))
    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        token = await self._ensure_token()
        response = await self._http.get(
            f"{self.config.api_base}{path}",
            params=params,
            headers={"Client-Id": self._client_id, "Authorization": f"Bearer {token}"},
        )
        if response.status_code == 401:
            # Token revoked or rotated -- force a refresh and let retry re-run.
            await self._ensure_token(force=True)
            raise RetryableError("Helix returned 401; refreshed token")
        if response.status_code == 429:
            reset = response.headers.get("Ratelimit-Reset")
            raise RetryableError(f"Helix rate limited (reset={reset})")
        if response.status_code >= 500:
            raise RetryableError(f"Helix {response.status_code}")
        if response.status_code != 200:
            raise TwitchAPIError(
                f"Helix {path} -> {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    # -- endpoints ---------------------------------------------------------

    async def get_stream(self, channel: str) -> StreamInfo | None:
        """The channel's live stream, or None when it is offline."""
        data = await self._get("/streams", {"user_login": channel.lower()})
        entries = data.get("data") or []
        if not entries:
            return None
        return StreamInfo.from_json(entries[0])

    async def get_user_id(self, channel: str) -> str | None:
        data = await self._get("/users", {"login": channel.lower()})
        entries = data.get("data") or []
        return str(entries[0]["id"]) if entries else None

    async def latest_vod(self, channel: str, stream_id: str | None = None) -> VodInfo | None:
        """The channel's newest archive VOD, optionally matching a stream id.

        Twitch publishes the archive a little after a stream starts, so this
        is polled rather than read once.
        """
        user_id = await self.get_user_id(channel)
        if not user_id:
            return None
        data = await self._get(
            "/videos", {"user_id": user_id, "type": "archive", "first": 5}
        )
        for entry in data.get("data") or []:
            if stream_id and str(entry.get("stream_id") or "") != stream_id:
                continue
            return VodInfo(
                id=str(entry["id"]),
                url=entry.get("url", ""),
                created_at=entry.get("created_at", ""),
                stream_id=str(entry.get("stream_id") or "") or None,
            )
        return None
