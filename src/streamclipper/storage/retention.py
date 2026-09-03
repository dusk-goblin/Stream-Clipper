"""Disk management.

A long broadcast at source quality is tens of gigabytes of raw segments. Once
the clips overlapping a segment have been cut, the segment itself is usually
dead weight -- but only usually, so every deletion path here checks that
nothing still needs the file first.

Transcripts, chat and topics are never deleted: they are small, and they are
what makes a session re-processable into different clips later.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..logging_setup import get_logger
from ..state import Database
from ..state.models import Clip, JobStatus, Segment, SegmentStatus
from ..util.timefmt import now_ts

log = get_logger(__name__)


class RetentionManager:
    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db

    # -- predicates -------------------------------------------------------

    def _pending_clip_needs(self, segment: Segment) -> bool:
        """Whether an uncut clip still needs this segment's footage."""
        for clip in self.db.clips_overlapping(
            segment.session_id,
            segment.start - self.config.clips.pad_before - 1,
            segment.end + self.config.clips.pad_after + 1,
        ):
            if clip.status != "done" and clip.status != "unavailable":
                return True
        return False

    def _pending_job_needs(self, segment: Segment) -> bool:
        """Whether queued work would still read this file.

        Cut jobs are *not* checked here, deliberately: every queued cut has a
        clip row that is not yet done, so ``_pending_clip_needs`` already
        covers them -- and checking cut jobs would make a segment undeletable
        from inside the very cut that finished with it, which is exactly when
        ``after_clip`` runs.

        Segmentation and ranking are different: they have not decided which
        clips exist yet, so any pending one can still stake a claim on
        footage anywhere in the session.
        """
        if segment.status != SegmentStatus.TRANSCRIBED.value:
            return True  # not transcribed yet: the transcript is not banked
        live = {JobStatus.PENDING.value, JobStatus.RUNNING.value}
        for job in self.db.jobs_for_session(segment.session_id):
            if job.status not in live:
                continue
            if job.kind == "transcribe" and job.payload.get("segment_id") == segment.id:
                return True
            if job.kind in {"segment", "rank"}:
                return True
        return False

    def can_delete(self, segment: Segment) -> bool:
        if segment.status == SegmentStatus.DELETED.value:
            return False
        return not self._pending_job_needs(segment) and not self._pending_clip_needs(segment)

    # -- actions ----------------------------------------------------------

    def delete_segment(self, segment: Segment) -> bool:
        """Remove the raw file, keeping its database row and transcript."""
        path = Path(segment.path)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning(
                "retention.delete_failed",
                extra={"path": str(path), "reason": str(exc)},
            )
            return False
        self.db.set_segment_status(segment.id, SegmentStatus.DELETED.value)
        log.info(
            "retention.deleted",
            extra={
                "session_id": segment.session_id,
                "seq": segment.seq,
                "freed_mb": round(segment.bytes / 1_048_576, 1),
            },
        )
        return True

    def after_clip(self, clip: Clip) -> None:
        """Called once a clip is cut, if the policy reclaims raw footage."""
        if not self.config.retention.delete_segments_after_clip:
            return
        for segment in self.db.segments_covering(
            clip.session_id, clip.start - 60, clip.end + 60
        ):
            if self.can_delete(segment):
                self.delete_segment(segment)

    def sweep(self, session_id: int | None = None) -> int:
        """Apply age and size policies. Returns segments deleted."""
        policy = self.config.retention
        deleted = 0

        sessions = (
            [session_id]
            if session_id is not None
            else [s.id for s in self.db.list_sessions(limit=200)]
        )

        if policy.delete_segments_after_clip:
            for sid in sessions:
                for segment in self.db.list_segments(sid):
                    if (
                        segment.status != SegmentStatus.DELETED.value
                        and self.can_delete(segment)
                    ):
                        deleted += int(self.delete_segment(segment))

        if policy.raw_max_age_hours > 0:
            cutoff = now_ts() - policy.raw_max_age_hours * 3600
            for sid in sessions:
                for segment in self.db.list_segments(sid):
                    # created_at <= 0 means the row predates timestamping;
                    # an unknown age is not an old age, so leave it alone.
                    if (
                        segment.status != SegmentStatus.DELETED.value
                        and segment.created_at > 0
                        and segment.created_at < cutoff
                        and self.can_delete(segment)
                    ):
                        deleted += int(self.delete_segment(segment))

        if policy.max_disk_gb > 0:
            deleted += self._enforce_size(policy.max_disk_gb)

        if deleted:
            log.info("retention.sweep", extra={"deleted": deleted})
        return deleted

    def _enforce_size(self, max_gb: float) -> int:
        """Delete oldest deletable segments until under the size budget."""
        budget = max_gb * 1_073_741_824
        candidates: list[Segment] = []
        total = 0
        for session in self.db.list_sessions(limit=200):
            for segment in self.db.list_segments(session.id):
                if segment.status == SegmentStatus.DELETED.value:
                    continue
                total += segment.bytes
                candidates.append(segment)

        if total <= budget:
            return 0

        deleted = 0
        # Oldest first: the newest footage is the likeliest to still be wanted.
        for segment in sorted(candidates, key=lambda s: (s.created_at, s.id)):
            if total <= budget:
                break
            if not self.can_delete(segment):
                continue
            if self.delete_segment(segment):
                total -= segment.bytes
                deleted += 1
        return deleted

    def disk_usage(self) -> dict[str, float]:
        """Megabytes held under each managed directory."""

        def size_of(directory: Path) -> float:
            if not directory.exists():
                return 0.0
            total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            return round(total / 1_048_576, 1)

        paths = self.config.paths
        return {
            "segments_mb": size_of(paths.segments),
            "clips_mb": size_of(paths.output),
            "chat_mb": size_of(paths.chat),
        }
