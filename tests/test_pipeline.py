"""End-to-end wiring: segment -> rank -> cut -> manifest, and retention.

ffmpeg and ffprobe are stubbed, so this exercises the orchestration and the
data that flows between stages rather than the media tooling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streamclipper.clips.manifest import ManifestWriter
from streamclipper.pipeline.queue import JobKind, enqueue_segment
from streamclipper.pipeline.workers import JobRunner, WorkerPool
from streamclipper.state.models import SegmentStatus
from streamclipper.storage.retention import RetentionManager


@pytest.fixture
def runner(config, db, monkeypatch) -> JobRunner:
    """A JobRunner whose media calls are replaced by fakes."""
    import streamclipper.pipeline.workers as workers

    # A dense keyframe grid, so cuts plan as stream copies.
    monkeypatch.setattr(
        workers, "keyframe_times", lambda path, a=0.0, b=0.0: tuple(float(t) for t in range(0, 4000, 2))
    )

    instance = JobRunner(config, db)

    def fake_cut(plan, output, subtitle_file=None, vertical=False):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-media")
        return output

    monkeypatch.setattr(instance.cutter, "cut", fake_cut)
    return instance


def run_queue(db, runner, session_id: int, limit: int = 200) -> int:
    """Drain the queue synchronously, so assertions are deterministic."""
    processed = 0
    while processed < limit:
        job = db.claim_job(list(JobKind.ALL), 600.0, 3)
        if job is None:
            return processed
        runner.run(job)
        db.finish_job(job.id)
        processed += 1
    raise AssertionError("queue did not drain -- a handler is enqueueing forever")


# --------------------------------------------------------------------------
# Whole pipeline
# --------------------------------------------------------------------------


def test_segment_rank_and_cut_chain_through_the_queue(
    config, db, runner, session_with_transcript
):
    session_id = session_with_transcript.id
    enqueue_segment(db, session_id, final=True)
    assert run_queue(db, runner, session_id) > 0

    topics = db.list_topics(session_id)
    assert topics, "segmentation produced nothing"
    assert all(t.ranked for t in topics), "every committed topic should be ranked"

    clips = db.list_clips(session_id, status="done")
    assert clips, "ranking produced no clips"
    for clip in clips:
        assert Path(clip.path).exists()
        assert config.highlight.clip_min_seconds - 1 <= clip.duration <= config.highlight.clip_max_seconds + 1
        assert clip.excerpt, "a clip should carry its transcript excerpt"
        assert clip.topic_id in {t.id for t in topics}


def test_clips_do_not_overlap_each_other(config, db, runner, session_with_transcript):
    enqueue_segment(db, session_with_transcript.id, final=True)
    run_queue(db, runner, session_with_transcript.id)

    clips = sorted(db.list_clips(session_with_transcript.id), key=lambda c: c.start)
    for a, b in zip(clips, clips[1:]):
        assert a.end <= b.start + 1e-6


def test_manifest_describes_every_clip(config, db, runner, session_with_transcript):
    session_id = session_with_transcript.id
    enqueue_segment(db, session_id, final=True)
    run_queue(db, runner, session_id)

    manifest_path = config.paths.output / f"session_{session_id:05d}" / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())

    assert payload["session"]["id"] == session_id
    assert payload["topics"]
    assert payload["clip_count"] == len(payload["clips"])

    for entry in payload["clips"]:
        # Everything the deliverable promised a manifest record would carry.
        assert entry["topic"]["label"]
        assert "summary" in entry["topic"]
        assert isinstance(entry["topic"]["tags"], list)
        assert entry["end"] > entry["start"]
        assert entry["start_hms"].count(":") == 2
        assert isinstance(entry["hype_score"], float)
        assert "chat" in entry["scores"]
        assert entry["vod"]["offset_seconds"] == pytest.approx(entry["start"])
        assert entry["vod"]["offset"].endswith("s")
        assert "transcript" in entry


def test_manifest_paths_are_relative_to_the_output_directory(
    config, db, runner, session_with_transcript
):
    session_id = session_with_transcript.id
    enqueue_segment(db, session_id, final=True)
    run_queue(db, runner, session_id)

    payload = json.loads(
        (config.paths.output / f"session_{session_id:05d}" / "manifest.json").read_text()
    )
    for entry in payload["clips"]:
        video = entry["files"].get("video")
        if video:
            assert not Path(video).is_absolute()


def test_manifest_write_is_atomic(config, db, session_with_transcript, tmp_path):
    writer = ManifestWriter(tmp_path / "out")
    first = writer.write(session_with_transcript, [], [])
    second = writer.write(session_with_transcript, [], [])
    assert first == second
    # No temp files left behind on either pass.
    assert not list((tmp_path / "out").glob(".manifest-*"))


def test_disabled_stages_stop_the_chain(config, db, runner, session_with_transcript):
    config.stages.cut = False
    enqueue_segment(db, session_with_transcript.id, final=True)
    run_queue(db, runner, session_with_transcript.id)

    assert db.list_topics(session_with_transcript.id)
    assert db.list_clips(session_with_transcript.id, status="done") == []


def test_finalize_marks_the_session_complete(config, db, runner, session_with_transcript):
    from streamclipper.pipeline.queue import enqueue_finalize
    from streamclipper.state.models import SessionStatus

    enqueue_segment(db, session_with_transcript.id, final=True)
    run_queue(db, runner, session_with_transcript.id)
    enqueue_finalize(db, session_with_transcript.id)
    run_queue(db, runner, session_with_transcript.id)

    assert db.get_session(session_with_transcript.id).status == SessionStatus.COMPLETE.value


def test_a_cut_with_no_surviving_footage_is_marked_unavailable(
    config, db, runner, session_with_transcript
):
    from streamclipper.pipeline.queue import enqueue_cut

    clip = db.add_clip(session_with_transcript.id, None, 99_000.0, 99_040.0, 0.9)
    enqueue_cut(db, session_with_transcript.id, clip.id)
    run_queue(db, runner, session_with_transcript.id)
    assert db.get_clip(clip.id).status == "unavailable"


def test_worker_pool_buries_a_permanently_broken_job(config, db, session_with_transcript):
    """A missing binary will not fix itself, so it must not spin the queue."""
    from streamclipper.errors import MissingBinary
    from streamclipper.state.models import JobStatus

    class ExplodingRunner(JobRunner):
        def run(self, job):
            raise MissingBinary("ffmpeg")

    db.enqueue(session_with_transcript.id, JobKind.CUT, {}, dedupe_key="boom")
    pool = WorkerPool(
        config, db, ExplodingRunner(config, db), JobKind.LIGHT, size=1, name="t"
    )
    pool.start()
    try:
        import time

        deadline = time.time() + 10
        while time.time() < deadline:
            jobs = db.jobs_for_session(session_with_transcript.id)
            if jobs and jobs[0].status == JobStatus.FAILED.value:
                break
            time.sleep(0.1)
    finally:
        pool.stop(timeout=5)

    job = db.jobs_for_session(session_with_transcript.id)[0]
    assert job.status == JobStatus.FAILED.value
    assert job.attempts == 1          # buried immediately, not retried


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------


def test_retention_keeps_a_segment_a_pending_clip_still_needs(
    config, db, session_with_transcript
):
    manager = RetentionManager(config, db)
    segment = db.list_segments(session_with_transcript.id)[0]
    db.add_clip(session_with_transcript.id, None, segment.start + 10, segment.start + 50, 0.9)
    assert not manager.can_delete(segment)


def test_retention_keeps_an_untranscribed_segment(config, db, session_with_transcript):
    manager = RetentionManager(config, db)
    segment = db.list_segments(session_with_transcript.id)[0]
    db.set_segment_status(segment.id, SegmentStatus.READY.value)
    assert not manager.can_delete(db.get_segment(segment.id))


def test_retention_deletes_the_file_but_keeps_the_transcript(
    config, db, session_with_transcript, tmp_path
):
    manager = RetentionManager(config, db)
    segment = db.list_segments(session_with_transcript.id)[0]

    real_file = tmp_path / "seg.ts"
    real_file.write_bytes(b"x" * 1024)
    db._write("UPDATE segments SET path = ? WHERE id = ?", (str(real_file), segment.id))
    segment = db.get_segment(segment.id)

    assert manager.can_delete(segment)
    assert manager.delete_segment(segment)
    assert not real_file.exists()
    assert db.get_segment(segment.id).status == SegmentStatus.DELETED.value
    # The transcript is what makes a session re-processable -- it must survive.
    assert db.utterances(session_with_transcript.id)


def test_retention_age_policy_only_touches_old_segments(
    config, db, session_with_transcript, tmp_path
):
    config.retention.raw_max_age_hours = 1.0
    manager = RetentionManager(config, db)

    import time

    now = time.time()
    segments = db.list_segments(session_with_transcript.id)
    for index, segment in enumerate(segments[:2]):
        path = tmp_path / f"s{index}.ts"
        path.write_bytes(b"x")
        # Only the first is older than the one-hour policy.
        created = now - 7200 if index == 0 else now
        db._write(
            "UPDATE segments SET path = ?, created_at = ? WHERE id = ?",
            (str(path), created, segment.id),
        )

    assert manager.sweep(session_with_transcript.id) == 1
    assert (tmp_path / "s1.ts").exists()


def test_retention_ignores_segments_with_an_unknown_age(
    config, db, session_with_transcript, tmp_path
):
    """An unknown creation time is not an old one."""
    config.retention.raw_max_age_hours = 1.0
    segment = db.list_segments(session_with_transcript.id)[0]
    path = tmp_path / "unknown.ts"
    path.write_bytes(b"x")
    db._write(
        "UPDATE segments SET path = ?, created_at = 0 WHERE id = ?",
        (str(path), segment.id),
    )
    assert RetentionManager(config, db).sweep(session_with_transcript.id) == 0
    assert path.exists()


def test_retention_is_a_noop_when_disabled(config, db, session_with_transcript):
    manager = RetentionManager(config, db)
    assert manager.sweep(session_with_transcript.id) == 0
    assert all(
        s.status != SegmentStatus.DELETED.value
        for s in db.list_segments(session_with_transcript.id)
    )


def test_retention_reclaims_footage_once_the_clips_are_cut(
    config, db, runner, session_with_transcript, tmp_path
):
    """With the policy on, the raw segments go once nothing needs them."""
    config.retention.delete_segments_after_clip = True

    # Give the segments real files so deletion has something to remove.
    for index, segment in enumerate(db.list_segments(session_with_transcript.id)):
        path = tmp_path / f"raw_{index}.ts"
        path.write_bytes(b"x" * 512)
        db._write("UPDATE segments SET path = ? WHERE id = ?", (str(path), segment.id))

    from streamclipper.pipeline.queue import enqueue_finalize

    enqueue_segment(db, session_with_transcript.id, final=True)
    run_queue(db, runner, session_with_transcript.id)
    enqueue_finalize(db, session_with_transcript.id)
    run_queue(db, runner, session_with_transcript.id)

    segments = db.list_segments(session_with_transcript.id)
    assert any(s.status == SegmentStatus.DELETED.value for s in segments)
    assert not any(Path(s.path).exists() for s in segments)
    # The transcript and the clips outlive the footage.
    assert db.utterances(session_with_transcript.id)
    assert db.list_clips(session_with_transcript.id, status="done")


def test_retention_holds_footage_while_ranking_is_still_queued(
    config, db, session_with_transcript
):
    """A pending rank job can still stake a claim on footage anywhere."""
    config.retention.delete_segments_after_clip = True
    topic = db.add_topic(session_with_transcript.id, 0, 0.0, 600.0, "T")
    db.enqueue(
        session_with_transcript.id, JobKind.RANK, {"topic_id": topic.id}, dedupe_key="r"
    )
    manager = RetentionManager(config, db)
    assert not manager.can_delete(db.list_segments(session_with_transcript.id)[0])
    assert manager.sweep(session_with_transcript.id) == 0
