"""Retry helpers with exponential backoff and jitter.

Used for every network boundary: Helix polling, IRC connects, LLM calls.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from typing import Any, Callable, Iterable, TypeVar

from ..logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def backoff_delays(
    attempts: int, base: float = 1.0, factor: float = 2.0, cap: float = 60.0,
    jitter: float = 0.25,
) -> list[float]:
    """Delays to sleep *between* ``attempts`` tries, so ``attempts - 1`` of them.

    Jitter is proportional and full-range-symmetric, which keeps several
    workers that failed at the same instant from retrying in lockstep.
    """
    delays: list[float] = []
    for i in range(max(0, attempts - 1)):
        raw = min(cap, base * (factor**i))
        spread = raw * jitter
        delays.append(max(0.0, raw + random.uniform(-spread, spread)))
    return delays


def retry(
    attempts: int = 4,
    base: float = 1.0,
    factor: float = 2.0,
    cap: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a sync callable with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delays = backoff_delays(attempts, base, factor, cap)
            last: BaseException | None = None
            for attempt in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if attempt >= attempts - 1:
                        break
                    delay = delays[attempt]
                    if on_retry:
                        on_retry(attempt + 1, exc, delay)
                    else:
                        log.warning(
                            "retry",
                            extra={
                                "func": func.__name__,
                                "attempt": attempt + 1,
                                "of": attempts,
                                "delay": round(delay, 2),
                                "reason": str(exc),
                            },
                        )
                    time.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator


def async_retry(
    attempts: int = 4,
    base: float = 1.0,
    factor: float = 2.0,
    cap: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an async callable with exponential backoff.

    ``asyncio.CancelledError`` is never retried -- a cancel is a shutdown
    request, not a transient failure.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delays = backoff_delays(attempts, base, factor, cap)
            last: BaseException | None = None
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except asyncio.CancelledError:
                    raise
                except exceptions as exc:
                    last = exc
                    if attempt >= attempts - 1:
                        break
                    delay = delays[attempt]
                    log.warning(
                        "retry",
                        extra={
                            "func": func.__name__,
                            "attempt": attempt + 1,
                            "of": attempts,
                            "delay": round(delay, 2),
                            "reason": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator


async def sleep_unless_stopped(seconds: float, stop: asyncio.Event) -> bool:
    """Sleep, waking early if ``stop`` is set. Returns True if it was set."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


def first(iterable: Iterable[T], default: T | None = None) -> T | None:
    for item in iterable:
        return item
    return default
