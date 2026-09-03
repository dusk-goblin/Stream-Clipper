"""Structured logging.

Every log record carries the fields the pipeline cares about (session, segment,
job) so a JSON line is enough to reconstruct what a worker was doing. Attach
them with the ``extra=`` kwarg::

    log.info("segment.recorded", extra={"session_id": 4, "seq": 12})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Attributes LogRecord always has; anything else was passed via extra=.
_STANDARD = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


class JSONFormatter(logging.Formatter):
    """One JSON object per line, with extras flattened into the object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable single line, with extras appended as key=value."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-28s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = " ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _STANDARD and not k.startswith("_")
        )
        return f"{base} {extras}" if extras else base


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def setup_logging(
    level: str = "INFO", fmt: str = "json", file: str | Path | None = None
) -> None:
    """Configure the root logger. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter: logging.Formatter = (
        JSONFormatter() if fmt.lower() == "json" else TextFormatter()
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if file:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # These are chatty at DEBUG and never tell us anything we need.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
