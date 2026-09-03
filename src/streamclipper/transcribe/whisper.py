"""faster-whisper wrapper.

The model is loaded lazily and held for the process lifetime -- reloading it
per segment would dominate runtime. Transcribing a segment yields utterances
with word-level timings, shifted from file-relative time into absolute stream
time by the segment's own start offset.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

from ..config import TranscribeConfig
from ..errors import MissingDependency, TranscriptionError
from ..logging_setup import get_logger
from ..state.models import Utterance, Word

log = get_logger(__name__)


def resolve_device(device: str) -> str:
    """'auto' -> cuda when a GPU is actually usable, else cpu."""
    if device != "auto":
        return device
    try:
        import torch  # noqa: PLC0415 -- optional, only present with a GPU install

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


class Transcriber:
    """Thread-safe front end to a single faster-whisper model."""

    def __init__(self, config: TranscribeConfig) -> None:
        self.config = config
        self._model: Any = None
        self._lock = threading.Lock()
        self.device = resolve_device(config.device)
        self.compute_type = resolve_compute_type(config.compute_type, self.device)

    def load(self) -> Any:
        """Load the model on first use. Safe to call from several threads."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # noqa: PLC0415
            except ImportError as exc:
                raise MissingDependency(
                    "faster-whisper", "whisper", "Transcription"
                ) from exc
            log.info(
                "whisper.loading",
                extra={
                    "model": self.config.model,
                    "device": self.device,
                    "compute_type": self.compute_type,
                },
            )
            self._model = WhisperModel(
                self.config.model,
                device=self.device,
                compute_type=self.compute_type,
            )
            return self._model

    def transcribe_file(self, path: str | Path, time_offset: float = 0.0) -> list[Utterance]:
        """Transcribe one media file into stream-time utterances."""
        media = Path(path)
        if not media.exists():
            raise TranscriptionError(f"Media file is missing: {media}")

        model = self.load()
        try:
            segments, info = model.transcribe(
                str(media),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                word_timestamps=True,
            )
            utterances = [
                self._to_utterance(segment, time_offset)
                for segment in segments  # generator: this is where work happens
            ]
        except Exception as exc:
            raise TranscriptionError(f"Transcribing {media.name} failed: {exc}") from exc

        spoken = sum(u.end - u.start for u in utterances)
        log.info(
            "whisper.done",
            extra={
                "file": media.name,
                "utterances": len(utterances),
                "spoken_seconds": round(spoken, 1),
                "language": getattr(info, "language", None),
            },
        )
        return [u for u in utterances if u.text.strip()]

    @staticmethod
    def _to_utterance(segment: Any, offset: float) -> Utterance:
        words: list[Word] = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                Word(
                    start=float(word.start) + offset,
                    end=float(word.end) + offset,
                    text=str(word.word).strip(),
                    probability=float(getattr(word, "probability", 1.0) or 1.0),
                )
            )
        return Utterance(
            start=float(segment.start) + offset,
            end=float(segment.end) + offset,
            text=str(segment.text).strip(),
            words=words,
        )


def utterances_from_words(words: Sequence[Word], max_gap: float = 0.8) -> list[Utterance]:
    """Rebuild utterances from loose words, splitting on pauses.

    Only needed when a transcript arrives word-first (an imported transcript,
    say) rather than from whisper's own segmentation.
    """
    groups: list[list[Word]] = []
    for word in words:
        if groups and word.start - groups[-1][-1].end <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [
        Utterance(
            start=group[0].start,
            end=group[-1].end,
            text=" ".join(w.text for w in group).strip(),
            words=list(group),
        )
        for group in groups
    ]
