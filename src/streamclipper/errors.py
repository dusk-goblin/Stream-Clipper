"""Exception hierarchy shared across the pipeline."""

from __future__ import annotations


class StreamClipperError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigError(StreamClipperError):
    """The YAML config is missing something required, or holds a bad value."""


class MissingDependency(StreamClipperError):
    """An optional extra is needed for the stage the user asked for.

    Carries the pip extra so the CLI can tell the user exactly what to install.
    """

    def __init__(self, package: str, extra: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} needs the '{package}' package. "
            f'Install it with: pip install "stream-clipper[{extra}]"'
        )
        self.package = package
        self.extra = extra


class MissingBinary(StreamClipperError):
    """An external binary (ffmpeg, ffprobe, streamlink) is not on PATH."""

    def __init__(self, binary: str, hint: str = "") -> None:
        msg = f"'{binary}' was not found on PATH."
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)
        self.binary = binary


class TwitchAPIError(StreamClipperError):
    """Helix returned something we cannot act on."""


class TranscriptionError(StreamClipperError):
    """A segment could not be transcribed."""


class ClipCutError(StreamClipperError):
    """ffmpeg failed to produce a clip."""


class LLMError(StreamClipperError):
    """The LLM call failed or returned an unusable response."""


class RetryableError(StreamClipperError):
    """A transient failure. Worth retrying with backoff."""
