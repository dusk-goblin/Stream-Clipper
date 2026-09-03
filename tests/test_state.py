"""SQLite state: the job queue's atomicity and the resumability guarantees."""

from __future__ import annotations

import threading


from streamclipper.state.models import (
    ChatMessage,
    JobStatus,
    SegmentStatus,
    SessionStatus,
    Utterance,
    Word,
)
from streamclipper.util.timefmt import now_ts


def test_session_round_trip(db):
    session = db.create_session("hasanabi", "live", 1000.0, title="Stream title")
    loaded = db.get_session(session.id)
    assert loaded is not None
    assert (loaded.channel, loaded.title) == ("hasanabi", "Stream title")
    assert loaded.status == SessionStatus.RECORDING.value


def test_segment_timeline_accounting(db):
    session = db.create_session("c", "live", 0.0)
    for seq in range(3):
        segment = db.add_segment(session.id, seq, f"/s/{seq}.ts", seq * 300.0, 300.0)
        db.finish_segment(segment.id, 300.0, 1000)

    assert db.recorded_seconds(session.id) == 900.0
    assert db.transcribed_seconds(session.id) == 0.0

    first = db.list_segments(session.id)[0]
    db.set_segment_status(first.id, SegmentStatus.TRANSCRIBED.value)
    assert db.transcribed_seconds(session.id) == 300.0


def test_next_segment_seq_continues_after_a_resume(db):
    session = db.create_session("c", "live", 0.0)
    assert db.next_segment_seq(session.id) == 0
    db.add_segment(session.id, 0, "/s/0.ts", 0.0, 300.0)
    db.add_segment(session.id, 1, "/s/1.ts", 300.0, 300.0)
    assert db.next_segment_seq(session.id) == 2


def test_segments_covering_selects_only_overlapping_files(db):
    session = db.create_session("c", "live", 0.0)
    for seq in range(4):
        segment = db.add_segment(session.id, seq, f"/s/{seq}.ts", seq * 300.0, 300.0)
        db.finish_segment(segment.id, 300.0, 1)

    assert [s.seq for s in db.segments_covering(session.id, 290.0, 320.0)] == [0, 1]
    assert [s.seq for s in db.segments_covering(session.id, 10.0, 20.0)] == [0]
    assert db.segments_covering(session.id, 5000.0, 5100.0) == []


def test_deleted_segments_are_excluded_from_coverage(db):
    session = db.create_session("c", "live", 0.0)
    segment = db.add_segment(session.id, 0, "/s/0.ts", 0.0, 300.0)
    db.finish_segment(segment.id, 300.0, 1)
    db.set_segment_status(segment.id, SegmentStatus.DELETED.value)
    assert db.segments_covering(session.id, 10.0, 20.0) == []


def test_transcript_write_is_idempotent_per_segment(db):
    """A retried transcription job must not duplicate the transcript."""
    session = db.create_session("c", "live", 0.0)
    segment = db.add_segment(session.id, 0, "/s/0.ts", 0.0, 300.0)
    utterances = [Utterance(0.0, 5.0, "hello there", [Word(0.0, 1.0, "hello")])]

    db.add_transcript(session.id, segment.id, utterances)
    db.add_transcript(session.id, segment.id, utterances)

    assert len(db.utterances(session.id)) == 1
    assert len(db.words(session.id)) == 1


def test_transcript_queries_are_time_windowed(db):
    session = db.create_session("c", "live", 0.0)
    db.add_transcript(
        session.id, None,
        [Utterance(float(i * 10), float(i * 10 + 5), f"line {i}") for i in range(10)],
    )
    windowed = db.utterances(session.id, 20.0, 50.0)
    assert [u.text for u in windowed] == ["line 2", "line 3", "line 4"]


def test_chat_round_trip_preserves_emotes(db):
    session = db.create_session("c", "live", 0.0)
    db.add_chat(
        session.id,
        [ChatMessage(ts=5.0, user="bob", text="KEKW", wall_ts=1.0, emotes=["KEKW"])],
    )
    loaded = db.chat_between(session.id, 0.0, 10.0)
    assert loaded[0].emotes == ["KEKW"]
    assert loaded[0].user == "bob"


# --------------------------------------------------------------------------
# Job queue
# --------------------------------------------------------------------------


def test_dedupe_key_prevents_duplicate_jobs(db):
    session = db.create_session("c", "live", 0.0)
    assert db.enqueue(session.id, "transcribe", {"segment_id": 1}, dedupe_key="seg:1")
    assert db.enqueue(session.id, "transcribe", {"segment_id": 1}, dedupe_key="seg:1") is None
    assert db.pending_job_count(session.id) == 1


def test_jobs_are_claimed_in_priority_order(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "cut", {}, priority=40, dedupe_key="a")
    db.enqueue(session.id, "transcribe", {}, priority=10, dedupe_key="b")
    claimed = db.claim_job(["cut", "transcribe"], 60.0, 3)
    assert claimed is not None and claimed.kind == "transcribe"


def test_a_job_is_claimed_by_exactly_one_worker(db):
    """The claim must be atomic or two workers transcribe the same segment."""
    session = db.create_session("c", "live", 0.0)
    for i in range(20):
        db.enqueue(session.id, "transcribe", {"n": i}, dedupe_key=f"seg:{i}")

    claimed: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            job = db.claim_job(["transcribe"], 60.0, 3)
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == 20
    assert len(set(claimed)) == 20


def test_claiming_respects_the_kind_filter(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "cut", {}, dedupe_key="a")
    assert db.claim_job(["transcribe"], 60.0, 3) is None
    assert db.claim_job(["cut"], 60.0, 3) is not None


def test_an_expired_lease_lets_the_job_be_reclaimed(db):
    """This is how a job survives the worker holding it being killed."""
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "transcribe", {}, dedupe_key="a")

    first = db.claim_job(["transcribe"], lease_seconds=-1.0, max_attempts=5)
    assert first is not None
    second = db.claim_job(["transcribe"], lease_seconds=60.0, max_attempts=5)
    assert second is not None and second.id == first.id
    assert second.attempts == 2


def test_a_live_lease_is_not_stolen(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "transcribe", {}, dedupe_key="a")
    assert db.claim_job(["transcribe"], 600.0, 5) is not None
    assert db.claim_job(["transcribe"], 600.0, 5) is None


def test_a_failed_job_retries_then_is_buried(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "transcribe", {}, dedupe_key="a")

    job = db.claim_job(["transcribe"], 60.0, 3)
    db.fail_job(job.id, "boom", max_attempts=3)
    assert db.pending_job_count(session.id) == 1      # retryable

    for _ in range(2):
        retry = db.claim_job(["transcribe"], 60.0, 3)
        assert retry is not None
        db.fail_job(retry.id, "boom", max_attempts=3)

    assert db.claim_job(["transcribe"], 60.0, 3) is None
    assert db.pending_job_count(session.id) == 0
    assert db.jobs_for_session(session.id)[0].status == JobStatus.FAILED.value
    assert "boom" in (db.jobs_for_session(session.id)[0].last_error or "")


def test_release_stale_jobs_recovers_a_crashed_run(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "transcribe", {}, dedupe_key="a")
    db.claim_job(["transcribe"], 3600.0, 3)      # a long lease, then "crash"

    assert db.release_stale_jobs() == 1
    assert db.claim_job(["transcribe"], 60.0, 3) is not None


def test_finished_jobs_leave_the_queue(db):
    session = db.create_session("c", "live", 0.0)
    db.enqueue(session.id, "transcribe", {}, dedupe_key="a")
    job = db.claim_job(["transcribe"], 60.0, 3)
    db.finish_job(job.id)
    assert db.pending_job_count(session.id) == 0
    assert db.claim_job(["transcribe"], 60.0, 3) is None


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_a_recent_interrupted_session_is_resumable(db):
    session = db.create_session("hasanabi", "live", now_ts())
    db.update_session(
        session.id, status=SessionStatus.INTERRUPTED.value, ended_at=now_ts()
    )
    assert db.resumable_session("hasanabi", within=900.0) is not None


def test_a_stale_session_is_not_resumable(db):
    session = db.create_session("hasanabi", "live", now_ts() - 10_000)
    db.update_session(
        session.id, status=SessionStatus.INTERRUPTED.value, ended_at=now_ts() - 10_000
    )
    assert db.resumable_session("hasanabi", within=900.0) is None


def test_a_completed_session_is_not_resumable(db):
    session = db.create_session("hasanabi", "live", now_ts())
    db.update_session(
        session.id, status=SessionStatus.COMPLETE.value, ended_at=now_ts()
    )
    assert db.resumable_session("hasanabi", within=900.0) is None


def test_resume_is_scoped_to_the_channel(db):
    session = db.create_session("someone_else", "live", now_ts())
    db.update_session(session.id, status=SessionStatus.INTERRUPTED.value, ended_at=now_ts())
    assert db.resumable_session("hasanabi", within=900.0) is None


def test_topic_watermark_persists(db):
    session = db.create_session("c", "live", 0.0)
    db.update_session(session.id, topics_watermark=1234.5)
    assert db.get_session(session.id).topics_watermark == 1234.5


def test_topics_and_clips_link_up(db):
    session = db.create_session("c", "live", 0.0)
    topic = db.add_topic(session.id, 0, 0.0, 600.0, "Label", "Summary", ["a", "b"], "both", 0.9)
    clip = db.add_clip(session.id, topic.id, 10.0, 50.0, 0.8, "text", {"chat": 0.5})

    assert db.get_topic(topic.id).tags == ["a", "b"]
    assert db.get_clip(clip.id).scores == {"chat": 0.5}
    assert db.next_topic_idx(session.id) == 1

    db.update_clip(clip.id, status="done", path="/out/clip.mp4")
    assert db.list_clips(session.id, status="done")[0].path == "/out/clip.mp4"


def test_clips_overlapping_finds_conflicts(db):
    session = db.create_session("c", "live", 0.0)
    db.add_clip(session.id, None, 100.0, 160.0, 0.5)
    assert db.clips_overlapping(session.id, 150.0, 200.0)
    assert not db.clips_overlapping(session.id, 200.0, 260.0)
