"""Subprocess helpers for the external binaries we shell out to."""

from __future__ import annotations

import shutil
import subprocess
from typing import Sequence

from ..errors import MissingBinary
from ..logging_setup import get_logger

log = get_logger(__name__)

_HINTS = {
    "ffmpeg": "Install it from https://ffmpeg.org/download.html (or `apt install ffmpeg` / `brew install ffmpeg`).",
    "ffprobe": "It ships with ffmpeg -- installing ffmpeg gives you both.",
    "streamlink": 'Install with: pip install "stream-clipper[capture]"',
}


def require_binary(name: str) -> str:
    """Absolute path to ``name``, or raise MissingBinary with an install hint."""
    path = shutil.which(name)
    if not path:
        raise MissingBinary(name, _HINTS.get(name, ""))
    return path


def has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def run(
    args: Sequence[str], timeout: float | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a command to completion, capturing output.

    On failure the raised CalledProcessError carries stderr, which is where
    ffmpeg puts everything worth reading.
    """
    log.debug("proc.run", extra={"cmd": list(args)})
    result = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, list(args), output=result.stdout, stderr=result.stderr
        )
    return result
