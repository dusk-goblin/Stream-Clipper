# Stream-Clipper

Records a Twitch live stream and cuts it into clips, segmented by topic.

The pipeline watches a channel, captures the stream in fixed-length segments,
transcribes them, works out where the streamer moves from one subject to
another, ranks the moments inside each topic by how hard chat reacted and how
well they stand alone, and cuts the winners with ffmpeg — writing a JSON
manifest with a label, summary, tags, timestamps, VOD offset, transcript
excerpt and hype score for every clip.

Default channel is `hasanabi`; it works on any channel.

---

## How it works

```
 monitor ──▶ recorder ──▶ segments/*.ts ──┐
 (Helix)     (streamlink|ffmpeg)          │
                                          ▼
 chat logger ──▶ chat/*.jsonl        ┌─────────┐
 (IRC, anon)                         │ SQLite  │  ← job queue + all state
                                     │  queue  │
                                     └────┬────┘
                                          │
   transcribe ──▶ segment ──▶ rank ──▶ cut ──▶ clips/*.mp4 + manifest.json
   (whisper)      (topics)    (hype)   (ffmpeg)
```

Capture and processing are decoupled by a SQLite-backed job queue. The
recorder only enqueues "this segment is complete"; transcription, segmentation
and cutting are pulled off the queue by worker threads. Transcription can fall
arbitrarily far behind without the recorder dropping a frame.

**Topic segmentation** combines two signals:

- *Semantic drift* — sentence embeddings over a sliding window; a boundary is
  a gap where the block before and the block after stop pointing the same way,
  and where that dip is a genuine valley relative to the surrounding
  similarity, not just a low absolute number.
- *LLM pass* — transcript windows go to Claude, which returns boundary
  timestamps plus a topic label, a one-line summary and 3–8 tags.

The two are merged: boundaries the signals agree on are treated as one (and
gain confidence), then minimum and maximum topic lengths are enforced —
splitting an overlong topic at a real boundary that was dropped for being too
close, rather than at an arbitrary midpoint.

**Highlight ranking** inside each topic scores candidate windows on chat
message rate, reaction-emote spikes (KEKW, OMEGALUL, …) and LLM-rated
clippability. Chat signals are z-scored against the topic's own baseline, so a
quiet topic can still yield its best moment and a busy one is not entirely
clips.

---

## Setup

### 1. System dependencies

**ffmpeg** (required — segmenting, clip cutting, keyframe detection):

```bash
sudo apt install ffmpeg        # Debian/Ubuntu
brew install ffmpeg            # macOS
winget install ffmpeg          # Windows
```

Verify with `ffmpeg -version` and `ffprobe -version`; both must be on PATH.

### 2. Install

```bash
git clone <this repo> && cd Stream-Clipper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

The core install is deliberately light. `[all]` pulls in everything; install
only what you need instead:

| Extra | Brings in | Needed for |
|---|---|---|
| `capture` | `streamlink` | live recording, VOD download |
| `whisper` | `faster-whisper` | transcription |
| `embeddings` | `sentence-transformers` | semantic topic boundaries |
| `llm` | `anthropic` | topic labels, summaries, tags, clippability |

Every stage degrades rather than crashing when its extra is missing: without
`embeddings` a dependency-free TF-IDF backend takes over; without `llm` topics
are segmented on drift alone and clips are ranked on chat alone.

**GPU transcription.** `faster-whisper` needs a CUDA-enabled build of its
runtime. Install a CUDA torch for your platform (see
[pytorch.org](https://pytorch.org/get-started/locally/)) and leave
`transcribe.device: auto` — it picks `cuda` when a GPU is visible and `cpu`
otherwise.

### 3. Twitch app credentials

Live recording polls the Helix API, which needs an application's client
credentials (no user login, no scopes):

1. Go to <https://dev.twitch.tv/console/apps> and **Register Your Application**.
2. Name it anything; OAuth Redirect URL `http://localhost`; Category `Application Integration`.
3. Copy the **Client ID**, then **New Secret** and copy the secret.

```bash
export TWITCH_CLIENT_ID=your_client_id
export TWITCH_CLIENT_SECRET=your_client_secret
```

Or put them in your YAML under `twitch.client_id` / `twitch.client_secret`.

Chat logging needs no credentials — it connects anonymously and read-only.

### 4. Anthropic API key

For topic labels, summaries, tags and clippability scoring:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

An `ant auth login` profile is picked up automatically if you have one.

### 5. Check it over

```bash
stream-clipper doctor
```

Reports which binaries, extras and credentials are present, and where files
will be written.

---

## Usage

```bash
# Watch the default channel and record when it goes live. Ctrl-C to stop.
stream-clipper record

# A different channel, recording one broadcast then processing and exiting.
stream-clipper --channel someone record --once

# Run the same pipeline on a VOD or a local file -- no waiting for a stream.
stream-clipper process https://twitch.tv/videos/123456789
stream-clipper process ./recording.mp4 --chat ./chat.jsonl

# What came out.
stream-clipper clips list
stream-clipper clips list --session 3 --json

# Copy the good ones somewhere, with a metadata index alongside.
stream-clipper clips export ./out --min-score 0.6 --limit 20
stream-clipper clips export ./shorts --vertical
```

`process` is the way to try the pipeline out: it runs every stage the live
path runs, just reading from a file instead of an HLS pull.

### Shutdown

The first `Ctrl-C` stops capture and drains the queue, so segments already on
disk still become clips. A second `Ctrl-C` exits immediately — the queue is
durable, and the next `record` or `process` picks the work back up.

---

## Configuration

Everything is configurable from YAML. Only the keys you want to change need to
be present; the rest fall back to the packaged defaults in
[`src/streamclipper/resources/default.yaml`](src/streamclipper/resources/default.yaml),
which documents every option. [`config/example.yaml`](config/example.yaml) is
a trimmed starting point.

```bash
stream-clipper -c config/example.yaml record
```

The knobs you are most likely to touch:

```yaml
channel: hasanabi

capture:
  quality: 720p60          # streamlink selector; "best" is source quality
  segment_seconds: 300     # recorded segment length

transcribe:
  model: large-v3          # tiny | base | small | medium | large-v3 | distil-large-v3
  device: auto             # auto -> cuda when available

segment:
  min_topic_seconds: 180   # topic length bounds
  max_topic_seconds: 2700
  merge_tolerance: 45      # boundaries this close are the same boundary
  semantic:
    similarity_threshold: 0.55   # see note below
    depth_threshold: 0.12

highlight:
  clip_min_seconds: 30     # clip length bounds
  clip_max_seconds: 90
  per_topic: 2             # max clips per topic
  min_score: 0.35
  weights:                 # relative; normalised automatically
    chat_rate: 0.35
    emote_spike: 0.25
    llm: 0.40

clips:
  mode: auto               # copy | reencode | auto
  pad_before: 2.0
  pad_after: 1.5
  burn_subtitles: false
  vertical:
    enabled: false         # 9:16 crop for shorts
    focus_x: 0.5           # 0 = left edge, 1 = right edge

stages:                    # turn individual stages off
  capture: true
  chat: true
  transcribe: true
  segment: true
  rank: true
  cut: true

retention:
  delete_segments_after_clip: false
  max_disk_gb: 0           # 0 = no limit
```

**On the semantic thresholds.** Absolute cosine values are not comparable
across embedding backends — within-topic similarity sits around 0.6 for
sentence-transformer vectors and around 0.2 for TF-IDF. So the similarity
profile is normalised to the range that stream actually exhibited, and both
thresholds are read against that. `similarity_threshold: 0.55` means "in the
lower 55% of this stream's own similarity range", which behaves the same
whichever backend produced the vectors.

**On `clips.mode`.** A stream copy is near-instant but can only begin on a
keyframe, so the cut is snapped *backwards* to the nearest one — starting
slightly early keeps the opening words, where starting late would lose them.
`auto` copies unless that snap would drift more than `snap_tolerance` seconds
(or the keyframes cannot be read), and re-encodes for frame accuracy when it
would. `copy` and `reencode` force one or the other.

---

## Output

```
data/
├── state.db                     SQLite: sessions, segments, transcript, chat,
│                                topics, clips, job queue
├── segments/session_00001/      raw recorded segments
├── chat/session_00001.jsonl     chat log, timestamped to stream time
└── clips/session_00001/
    ├── 001430_election-night-results_7.mp4
    ├── 001430_election-night-results_7_vertical.mp4
    ├── 001430_election-night-results_7.srt
    └── manifest.json
```

A manifest entry:

```json
{
  "id": 7,
  "topic": {
    "index": 2,
    "label": "Election night results",
    "summary": "Going through swing-state returns as they come in.",
    "tags": ["politics", "election", "swing states"],
    "start": 1200.0, "end": 2400.0,
    "method": "both"
  },
  "start": 1430.5, "end": 1490.5, "duration": 60.0,
  "start_hms": "00:23:50",
  "hype_score": 0.8412,
  "scores": {
    "chat": 0.91, "emote": 0.88, "llm": 0.77, "total": 0.8412,
    "msgs_per_sec": 14.2, "emotes_per_sec": 5.1
  },
  "transcript": "The margin in the county that always predicts the winner…",
  "vod": {
    "offset_seconds": 1430.5,
    "offset": "23m50s",
    "url": "https://twitch.tv/videos/123456789?t=23m50s"
  },
  "files": {
    "video": "001430_election-night-results_7.mp4",
    "vertical": "001430_election-night-results_7_vertical.mp4",
    "subtitles": "001430_election-night-results_7.srt"
  }
}
```

`topic.method` records which signals produced the topic's opening boundary:
`both`, `llm`, `semantic`, `session-start`, or `max-length`.

---

## Resilience

**Stream drops.** A drop does not end the session. The monitor keeps polling
for `twitch.resume_window` seconds and, if the stream returns as the same
broadcast, resumes capture into the same session — segment numbering and
stream time continue where they left off, so the transcript, chat log and
topic timeline stay continuous. A genuinely new broadcast gets its own session.

**Crashes.** Everything that a crash could lose is in SQLite: which segments
exist, how far each is through the pipeline, the transcript, and the queue
itself. Completed segments survive; a job claimed by a worker that died is
reclaimed when its lease expires. Restarting re-queues the unfinished work.

**Network.** Helix polling, IRC connects and LLM calls all retry with
exponential backoff and jitter. A 401 refreshes the token and retries; a
permanent failure (bad model, bad key) disables that stage rather than
retrying forever.

**Disk.** Optional retention reclaims raw segments — after the clips
overlapping them are cut, past an age limit, or to stay under a size budget.
Nothing is deleted while a pending job or an uncut clip still needs it, and
transcripts are always kept, so a session can be re-processed into different
clips later.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

The suite runs on the core install alone — no GPU, no model downloads, no
network, no ffmpeg. Tests that need an optional extra skip themselves when it
is absent.

Coverage is concentrated where the logic is, per the design:

| File | Covers |
|---|---|
| `tests/test_segmentation.py` | sentence building, semantic drift, the segmenter against a fixture transcript with known boundaries |
| `tests/test_merge.py` | boundary clustering, min/max length enforcement, span construction |
| `tests/test_clip_cutting.py` | keyframe snapping, cut planning, ffmpeg argument assembly, subtitles |
| `tests/test_highlight.py` | chat signals, window sweep, ranking and selection |
| `tests/test_pipeline.py` | segment → rank → cut → manifest end to end, retention |
| `tests/test_state.py` | queue atomicity, lease expiry, resumability |
| `tests/test_llm.py` | request shape and response parsing against a stand-in server |
| `tests/test_config.py` | YAML merge and validation |
| `tests/test_capture.py` | IRC parsing, emote extraction, segment-list parsing |

`tests/fixtures/transcript_topics.json` is a synthetic transcript with three
deliberately distinct topics and known boundary times;
`tests/fixtures/chat.jsonl` is a matching chat log with two emote-heavy hype
spikes. Between them the segmentation and ranking tests assert against
known-good answers rather than snapshots.

### Layout

```
src/streamclipper/
  cli.py              record | process | clips list | clips export | doctor
  config.py           YAML -> dataclasses, merge, validation
  logging_setup.py    structured JSON logging
  capture/            twitch.py (Helix), monitor.py (live + resume),
                      recorder.py (streamlink|ffmpeg), chat.py (IRC -> JSONL)
  transcribe/         whisper.py, transcript.py (sentences, stream time)
  segment/            embeddings.py, semantic.py, llm.py, merge.py, topics.py
  highlight/          chat_signals.py, llm_score.py, rank.py
  clips/              ffprobe.py, cutter.py, subtitles.py, vertical.py, manifest.py
  pipeline/           queue.py, workers.py, live.py, offline.py
  storage/            retention.py
  state/              db.py (schema + queue), models.py
```

---

## Limitations

- Twitch mid-roll ads can appear in captured footage. `--twitch-disable-ads`
  is passed to streamlink by default, which helps but is not a guarantee.
- Topic boundaries land at pauses in speech, not at scene changes. A streamer
  who switches subject mid-sentence will get a boundary a beat late.
- Chat hype and clippability are correlated but not identical to "good clip".
  The score is a ranking aid, not a verdict; `clips list` and the manifest
  exist so you can review before publishing.
- Without the `embeddings` extra the TF-IDF fallback is lexical only, so it
  misses a topic change phrased in the same vocabulary.

## License

MIT
