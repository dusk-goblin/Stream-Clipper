"""Chat-derived hype signals.

Two measurements per candidate window:

* **message rate** -- messages per second. Chat speeds up when something
  happens.
* **emote spike** -- rate of reaction emotes (KEKW, OMEGALUL, ...). A burst of
  these is a more specific signal than raw volume, which also rises during
  ordinary busy stretches.

Both are turned into z-scores against the *topic's own* baseline rather than
absolute thresholds, because a channel's chat rate varies by an order of
magnitude across a broadcast and between channels.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Sequence

from ..state.models import ChatMessage


@dataclass
class ChatProfile:
    """Chat for one topic, indexed so window queries are cheap."""

    times: list[float] = field(default_factory=list)
    emote_times: list[float] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        return max(1e-6, self.end - self.start)

    @property
    def total_messages(self) -> int:
        return len(self.times)

    def count_between(self, start: float, end: float) -> int:
        return bisect_left(self.times, end) - bisect_left(self.times, start)

    def emotes_between(self, start: float, end: float) -> int:
        return bisect_left(self.emote_times, end) - bisect_left(self.emote_times, start)


def build_profile(
    messages: Sequence[ChatMessage],
    start: float,
    end: float,
    watchlist: Sequence[str] = (),
) -> ChatProfile:
    """Index a topic's chat.

    A message contributes one emote event per matching emote, so ten KEKW in
    one message counts as ten -- spam-walling an emote *is* the reaction.
    """
    watch = {str(e).casefold() for e in watchlist}
    times: list[float] = []
    emote_times: list[float] = []
    for message in messages:
        if not (start <= message.ts < end):
            continue
        times.append(message.ts)
        for emote in message.emotes:
            if not watch or emote.casefold() in watch:
                emote_times.append(message.ts)
    times.sort()
    emote_times.sort()
    return ChatProfile(times=times, emote_times=emote_times, start=start, end=end)


def message_rate(profile: ChatProfile, start: float, end: float) -> float:
    span = max(1e-6, end - start)
    return profile.count_between(start, end) / span


def emote_rate(profile: ChatProfile, start: float, end: float) -> float:
    span = max(1e-6, end - start)
    return profile.emotes_between(start, end) / span


def zscores(values: Sequence[float]) -> list[float]:
    """Standard scores. All-equal input scores zero rather than dividing by zero."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(variance)
    if sd <= 1e-9:
        return [0.0] * n
    return [(v - mean) / sd for v in values]


def squash(z: float, scale: float = 2.0) -> float:
    """Map a z-score into 0..1, saturating rather than clipping.

    A window two standard deviations above its topic's baseline lands near
    0.75; beyond that the curve flattens, so one enormous outlier cannot
    dominate the ranking for the whole topic.
    """
    return 1.0 / (1.0 + math.exp(-z / scale * 2.0))
