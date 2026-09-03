"""Worker pool and job handlers.

Workers are threads, not tasks: every job here is blocking work in a C
extension (faster-whisper) or a subprocess (ffmpeg), both of which release
the GIL. Keeping them off the event loop is what stops transcription from
stalling the recorder.

Two pools claim from disjoint job-kind sets, so a long whisper run cannot
block clip cutting and vice versa.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Callable, Sequence

from ..clips.cutter import ClipCutter, plan_cut
from ..clips.ffprobe import keyframe_times
from ..clips.manifest import ManifestWriter
from ..clips.subtitles import write_srt
from ..config import Config
from ..errors import ClipCutError, MissingBinary, MissingDependency, StreamClipperError
from ..highlight.chat_signals import build_profile
from ..highlight.llm_score import attach_llm_scores
from ..highlight.rank import rank_topic, score_candidates, sweep_windows
from ..logging_setup import get_logger
from ..segment.llm import LLMClient
from ..segment.topics import TopicSegmenter
from ..state import Database
from ..state.models import Job, SegmentStatus, SessionStatus
from ..storage.retention import RetentionManager
from ..transcribe.transcript import excerpt_for, sentences_from_utterances
from ..transcribe.whisper import Transcriber
from .queue import JobKind, enqueue_cut, enqueue_rank

log = get_logger(__name__)


class JobRunner:
    """Executes one job at a time. Shared, thread-safe, holds the heavy models."""

    def __init__(self, config: Config, db: Database) -> None:
        self.config = config
        self.db = db
        self.transcriber = Transcriber(config.transcribe)
        self.llm = LLMClient(config.llm)
        self.segmenter = TopicSegmenter(config, db, llm=self.llm)
        self.cutter = ClipCutter(config.clips, config.paths.output)
        self.retention = RetentionManager(config, db)
        self._manifest_lock = threading.Lock()

    # -- dispatch ---------------------------------------------------------

    def run(self, job: Job) -> None:
        handlers: dict[str, Callable[[Job], None]] = {
            JobKind.TRANSCRIBE: self.handle_transcribe,
            JobKind.SEGMENT: self.handle_segment,
            JobKind.RANK: self.handle_rank,
            JobKind.CUT: self.handle_cut,
            JobKind.FINALIZE: self.handle_finalize,
        }
        handler = handlers.get(job.kind)
        if handler is None:
            raise StreamClipperError(f"Unknown job kind: {job.kind}")
        handler(job)

    # -- handlers ---------------------------------------------------------

    def handle_transcribe(self, job: Job) -> None:
        if not self.config.stages.transcribe:
            return
        segment = self.db.get_segment(int(job.payload["segment_id"]))
        if segment is None:
            log.warning("transcribe.missing_segment", extra={"job_id": job.id})
            return
        if segment.status in {SegmentStatus.TRANSCRIBED.value, SegmentStatus.DELETED.value}:
            return

        utterances = self.transcriber.transcribe_file(segment.path, time_offset=segment.start)
        self.db.add_transcript(segment.session_id, segment.id, utterances)
        self.db.set_segment_status(segment.id, SegmentStatus.TRANSCRIBED.value)
        log.info(
            "transcribe.done",
            extra={
                "session_id": segment.session_id,
                "seq": segment.seq,
                "utterances": len(utterances),
            },
        )

    def handle_segment(self, job: Job) -> None:
        if not self.config.stages.segment:
            return
        final = bool(job.payload.get("final"))
        result = self.segmenter.segment_session(job.session_id, final=final)
        if self.config.stages.rank:
            for topic in result.topics:
                enqueue_rank(self.db, job.session_id, topic.id)

    def handle_rank(self, job: Job) -> None:
        if not self.config.stages.rank:
            return
        topic = self.db.get_topic(int(job.payload["topic_id"]))
        if topic is None:
            return

        config = self.config.highlight
        sentences = sentences_from_utterances(
            self.db.utterances(topic.session_id, topic.start, topic.end)
        )
        profile = build_profile(
            self.db.chat_between(topic.session_id, topic.start, topic.end),
            topic.start,
            topic.end,
            config.emotes,
        )

        # Score on chat first, so the LLM only rates a shortlist.
        windows = sweep_windows(topic.start, topic.end, config)
        prescored = score_candidates(windows, profile, config)
        scores, titles, reasons = attach_llm_scores(
            prescored, sentences, topic.label, self.llm, config
        )

        picks = rank_topic(
            topic.start,
            topic.end,
            sentences,
            profile,
            config,
            llm_scores=scores or None,
            llm_titles=titles or None,
            llm_reasons=reasons or None,
        )
        self.db.mark_topic_ranked(topic.id)

        for candidate in picks:
            # Do not re-cut ground an existing clip already covers.
            if self.db.clips_overlapping(topic.session_id, candidate.start, candidate.end):
                continue
            breakdown = candidate.breakdown()
            if candidate.title:
                breakdown["has_title"] = 1.0
            clip = self.db.add_clip(
                topic.session_id,
                topic.id,
                candidate.start,
                candidate.end,
                candidate.score,
                excerpt=candidate.text
                or excerpt_for(
                    self.db.utterances(topic.session_id, candidate.start, candidate.end),
                    candidate.start,
                    candidate.end,
                ),
                scores=breakdown,
            )
            log.info(
                "clip.selected",
                extra={
                    "session_id": topic.session_id,
                    "clip_id": clip.id,
                    "topic": topic.label,
                    "start": round(clip.start, 1),
                    "duration": round(clip.duration, 1),
                    "score": round(clip.score, 3),
                },
            )
            if self.config.stages.cut:
                enqueue_cut(self.db, topic.session_id, clip.id)

    def handle_cut(self, job: Job) -> None:
        if not self.config.stages.cut:
            return
        clip = self.db.get_clip(int(job.payload["clip_id"]))
        if clip is None or clip.status == "done":
            return

        session = self.db.get_session(clip.session_id)
        topic = self.db.get_topic(clip.topic_id) if clip.topic_id else None
        if session is None:
            return

        segments = self.db.segments_covering(
            clip.session_id,
            clip.start - self.config.clips.pad_before - 1,
            clip.end + self.config.clips.pad_after + 1,
        )
        if not segments:
            self.db.update_clip(clip.id, status="unavailable")
            log.warning(
                "clip.no_source",
                extra={"clip_id": clip.id, "start": round(clip.start, 1)},
            )
            return

        # Keyframes come from the segment the cut starts in; ffprobe reports
        # them file-relative, so shift into stream time.
        first = segments[0]
        window_start = max(0.0, clip.start - self.config.clips.pad_before - first.start - 15)
        window_end = clip.start - first.start + 15
        keyframes = [
            first.start + t
            for t in keyframe_times(first.path, window_start, window_end)
        ]

        plan = plan_cut(segments, clip.start, clip.end, self.config.clips, keyframes)

        session_dir = self.config.paths.output / f"session_{session.id:05d}"
        stem = self._clip_stem(clip.id, topic.label if topic else "", clip.start)
        output = session_dir / f"{stem}.mp4"

        subtitle_path: Path | None = None
        if self.config.clips.burn_subtitles or self.config.clips.vertical.enabled:
            words = self.db.words(clip.session_id, plan.actual_start, plan.actual_end)
            subtitle_path = write_srt(
                session_dir / f"{stem}.srt", words, plan.actual_start, plan.actual_end
            )

        self.cutter.cut(
            plan,
            output,
            subtitle_file=subtitle_path if self.config.clips.burn_subtitles else None,
        )
        self.db.update_clip(
            clip.id,
            status="done",
            path=str(output),
            subtitle_path=str(subtitle_path) if subtitle_path else None,
        )

        if self.config.clips.vertical.enabled:
            try:
                vertical = session_dir / f"{stem}_vertical.mp4"
                self.cutter.cut(
                    plan,
                    vertical,
                    subtitle_file=subtitle_path if self.config.clips.burn_subtitles else None,
                    vertical=True,
                )
                self.db.update_clip(clip.id, vertical_path=str(vertical))
            except ClipCutError:
                # The landscape clip already succeeded; losing the vertical
                # variant is not worth failing and retrying the whole job.
                log.exception("clip.vertical_failed", extra={"clip_id": clip.id})

        self.write_manifest(clip.session_id)
        self.retention.after_clip(clip)

    def handle_finalize(self, job: Job) -> None:
        session = self.db.get_session(job.session_id)
        if session is None:
            return
        self.write_manifest(job.session_id)
        self.retention.sweep(job.session_id)
        self.db.update_session(job.session_id, status=SessionStatus.COMPLETE.value)
        log.info(
            "session.complete",
            extra={
                "session_id": job.session_id,
                "clips": len(self.db.list_clips(job.session_id, status="done")),
                "topics": len(self.db.list_topics(job.session_id)),
            },
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _clip_stem(clip_id: int, label: str, start: float) -> str:
        slug = "".join(
            c if c.isalnum() else "-" for c in label.lower()
        ).strip("-")
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug[:48].strip("-") or "clip"
        return f"{int(start):06d}_{slug}_{clip_id}"

    def write_manifest(self, session_id: int) -> Path | None:
        session = self.db.get_session(session_id)
        if session is None:
            return None
        writer = ManifestWriter(
            self.config.paths.output / f"session_{session_id:05d}",
            self.config.clips.manifest_name,
        )
        with self._manifest_lock:
            return writer.write(
                session, self.db.list_topics(session_id), self.db.list_clips(session_id)
            )


class WorkerPool:
    """A set of threads claiming jobs of given kinds until stopped."""

    def __init__(
        self,
        config: Config,
        db: Database,
        runner: JobRunner,
        kinds: Sequence[str],
        size: int,
        name: str = "worker",
    ) -> None:
        self.config = config
        self.db = db
        self.runner = runner
        self.kinds = list(kinds)
        self.size = max(1, size)
        self.name = name
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._idle = threading.Event()
        self._active = 0
        self._active_lock = threading.Lock()

    def start(self) -> None:
        for index in range(self.size):
            thread = threading.Thread(
                target=self._loop, name=f"{self.name}-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        log.info("workers.started", extra={"pool": self.name, "size": self.size})

    def stop(self, timeout: float = 30.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        log.info("workers.stopped", extra={"pool": self.name})

    @property
    def busy(self) -> bool:
        with self._active_lock:
            return self._active > 0

    def _loop(self) -> None:
        runtime = self.config.runtime
        while not self._stop.is_set():
            try:
                job = self.db.claim_job(
                    self.kinds, runtime.job_lease_seconds, runtime.max_job_attempts
                )
            except Exception:
                log.exception("worker.claim_failed", extra={"pool": self.name})
                job = None

            if job is None:
                self._stop.wait(timeout=runtime.poll_interval)
                continue

            with self._active_lock:
                self._active += 1
            try:
                self.runner.run(job)
                self.db.finish_job(job.id)
            except (MissingDependency, MissingBinary) as exc:
                # A missing tool will not fix itself on retry -- bury it now
                # so the queue does not spin.
                log.error(
                    "job.unrunnable",
                    extra={"job_id": job.id, "kind": job.kind, "reason": str(exc)},
                )
                self.db.fail_job(job.id, str(exc), max_attempts=0)
            except Exception as exc:
                log.exception(
                    "job.failed",
                    extra={
                        "job_id": job.id,
                        "kind": job.kind,
                        "attempt": job.attempts,
                        "session_id": job.session_id,
                    },
                )
                self.db.fail_job(
                    job.id,
                    f"{exc}\n{traceback.format_exc(limit=5)}",
                    self.config.runtime.max_job_attempts,
                )
            finally:
                with self._active_lock:
                    self._active -= 1
