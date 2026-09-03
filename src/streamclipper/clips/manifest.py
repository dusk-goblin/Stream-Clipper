"""The clips manifest.

One JSON file per session listing every clip with the context needed to use
it without re-running the pipeline: topic label, summary, tags, timestamps,
VOD offset, transcript excerpt and the score breakdown that selected it.

Written atomically -- a manifest is regenerated after every clip, and a
half-written file read by a concurrent `clips list` would be worse than a
stale one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ..logging_setup import get_logger
from ..state.models import Clip, Session, Topic
from ..util.timefmt import hhmmss, iso, vod_offset

log = get_logger(__name__)

MANIFEST_VERSION = 1


def clip_entry(
    clip: Clip, topic: Topic | None, session: Session, output_dir: Path
) -> dict[str, Any]:
    """One manifest record."""
    entry: dict[str, Any] = {
        "id": clip.id,
        "topic": {
            "id": topic.id if topic else None,
            "index": topic.idx if topic else None,
            "label": topic.label if topic else "",
            "summary": topic.summary if topic else "",
            "tags": list(topic.tags) if topic else [],
            "start": round(topic.start, 3) if topic else None,
            "end": round(topic.end, 3) if topic else None,
            "method": topic.method if topic else "",
        },
        "start": round(clip.start, 3),
        "end": round(clip.end, 3),
        "duration": round(clip.duration, 3),
        "start_hms": hhmmss(clip.start),
        "end_hms": hhmmss(clip.end),
        "hype_score": round(clip.score, 4),
        "scores": clip.scores,
        "transcript": clip.excerpt,
        "status": clip.status,
        "created_at": iso(clip.created_at) if clip.created_at else None,
        "vod": {
            "offset_seconds": round(clip.start, 3),
            "offset": vod_offset(clip.start),
            "url": (
                f"{session.vod_url}?t={vod_offset(clip.start)}" if session.vod_url else None
            ),
        },
        "files": {},
    }
    for key, value in (
        ("video", clip.path),
        ("vertical", clip.vertical_path),
        ("subtitles", clip.subtitle_path),
    ):
        if value:
            path = Path(value)
            try:
                relative: str = str(path.relative_to(output_dir))
            except ValueError:
                relative = str(path)
            entry["files"][key] = relative
    return entry


class ManifestWriter:
    """Builds and writes a session's manifest."""

    def __init__(self, output_dir: Path, name: str = "manifest.json") -> None:
        self.output_dir = Path(output_dir)
        self.name = name

    @property
    def path(self) -> Path:
        return self.output_dir / self.name

    def build(
        self, session: Session, topics: Sequence[Topic], clips: Sequence[Clip]
    ) -> dict[str, Any]:
        by_id = {topic.id: topic for topic in topics}
        entries = [
            clip_entry(clip, by_id.get(clip.topic_id or -1), session, self.output_dir)
            for clip in sorted(clips, key=lambda c: c.start)
        ]
        return {
            "manifest_version": MANIFEST_VERSION,
            "session": {
                "id": session.id,
                "channel": session.channel,
                "mode": session.mode,
                "title": session.title,
                "game": session.game,
                "started_at": iso(session.started_at),
                "ended_at": iso(session.ended_at) if session.ended_at else None,
                "duration": round(session.duration, 3),
                "source": session.source,
                "vod_url": session.vod_url,
            },
            "topics": [
                {
                    "id": topic.id,
                    "index": topic.idx,
                    "label": topic.label,
                    "summary": topic.summary,
                    "tags": list(topic.tags),
                    "start": round(topic.start, 3),
                    "end": round(topic.end, 3),
                    "start_hms": hhmmss(topic.start),
                    "duration": round(topic.duration, 3),
                    "method": topic.method,
                    "confidence": round(topic.confidence, 4),
                }
                for topic in sorted(topics, key=lambda t: t.idx)
            ],
            "clips": entries,
            "clip_count": len(entries),
        }

    def write(
        self, session: Session, topics: Sequence[Topic], clips: Sequence[Clip]
    ) -> Path:
        payload = self.build(session, topics, clips)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.output_dir, prefix=".manifest-", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        log.info(
            "manifest.written",
            extra={"path": str(self.path), "clips": len(payload["clips"])},
        )
        return self.path
