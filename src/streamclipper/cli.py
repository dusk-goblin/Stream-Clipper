"""Command line interface.

    stream-clipper record                    watch the channel and record when live
    stream-clipper process <vod-or-file>     run the same pipeline offline
    stream-clipper clips list                what has been produced
    stream-clipper clips export <dest>       copy clips out with their metadata
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Config, load_config
from .errors import StreamClipperError
from .logging_setup import get_logger, setup_logging
from .state import Database
from .util.proc import has_binary
from .util.timefmt import hhmmss, iso

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stream-clipper",
        description="Record a Twitch stream and cut it into topic-segmented clips.",
    )
    parser.add_argument("--version", action="version", version=f"stream-clipper {__version__}")
    parser.add_argument("-c", "--config", metavar="PATH", help="YAML config file")
    parser.add_argument("--channel", help="Override the configured channel")
    parser.add_argument("--data-dir", help="Override paths.data_dir")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Override log level"
    )
    parser.add_argument(
        "--log-format", choices=["json", "text"], help="Override log format"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Watch the channel and record when it goes live")
    record.add_argument(
        "--once",
        action="store_true",
        help="Record a single broadcast, then process it and exit",
    )
    record.add_argument(
        "--no-chat", action="store_true", help="Skip IRC chat logging"
    )

    process = subparsers.add_parser(
        "process", help="Run the pipeline on an existing VOD URL or local media file"
    )
    process.add_argument("source", help="Twitch VOD URL or path to a local media file")
    process.add_argument(
        "--chat", metavar="JSONL", help="Chat log to load alongside the media"
    )

    clips = subparsers.add_parser("clips", help="Inspect and export produced clips")
    clip_subparsers = clips.add_subparsers(dest="clips_command", required=True)

    listing = clip_subparsers.add_parser("list", help="List clips")
    listing.add_argument("--session", type=int, help="Only this session id")
    listing.add_argument("--json", action="store_true", help="Emit JSON")
    listing.add_argument(
        "--all", action="store_true", help="Include clips that were never cut"
    )

    export = clip_subparsers.add_parser("export", help="Copy clips to a directory")
    export.add_argument("dest", help="Destination directory")
    export.add_argument("--session", type=int, help="Only this session id")
    export.add_argument(
        "--min-score", type=float, default=0.0, help="Only clips scoring at least this"
    )
    export.add_argument("--limit", type=int, help="Export at most this many clips")
    export.add_argument(
        "--vertical", action="store_true", help="Export the 9:16 variant where present"
    )
    export.add_argument(
        "--move", action="store_true", help="Move instead of copy"
    )

    subparsers.add_parser("doctor", help="Check that external tools and extras are present")

    return parser


def config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.channel:
        overrides["channel"] = args.channel
    if args.data_dir:
        overrides["paths"] = {"data_dir": args.data_dir}
    logging_overrides: dict[str, Any] = {}
    if args.log_level:
        logging_overrides["level"] = args.log_level
    if args.log_format:
        logging_overrides["format"] = args.log_format
    if logging_overrides:
        overrides["logging"] = logging_overrides
    if getattr(args, "no_chat", False):
        overrides["stages"] = {"chat": False}
    return overrides


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_record(config: Config, db: Database, args: argparse.Namespace) -> int:
    from .pipeline.live import run_live

    # Fail before we start watching rather than at the first segment.
    config.twitch.resolve_credentials()
    if config.stages.capture and not has_binary("streamlink"):
        raise StreamClipperError(
            'streamlink is required to record. Install it with: pip install "stream-clipper[capture]"'
        )
    if config.stages.capture and not has_binary("ffmpeg"):
        raise StreamClipperError("ffmpeg is required to record. See the README for install steps.")

    print(f"Watching twitch.tv/{config.channel} -- Ctrl-C to stop.", flush=True)
    asyncio.run(run_live(config, db, once=args.once))
    return 0


def cmd_process(config: Config, db: Database, args: argparse.Namespace) -> int:
    from .pipeline.offline import run_offline

    chat_file = Path(args.chat).expanduser() if args.chat else None
    if chat_file is not None and not chat_file.exists():
        raise StreamClipperError(f"Chat log not found: {chat_file}")

    session = run_offline(config, db, args.source, chat_file)
    topics = db.list_topics(session.id)
    clips = db.list_clips(session.id, status="done")
    print(
        f"\nSession {session.id}: {len(topics)} topic(s), {len(clips)} clip(s) -> "
        f"{config.paths.output / f'session_{session.id:05d}'}",
        flush=True,
    )
    return 0


def cmd_clips_list(config: Config, db: Database, args: argparse.Namespace) -> int:
    status = None if args.all else "done"
    clips = db.list_clips(args.session, status=status)

    if args.json:
        topics = {t.id: t for session in db.list_sessions() for t in db.list_topics(session.id)}
        print(
            json.dumps(
                [
                    {
                        "id": clip.id,
                        "session_id": clip.session_id,
                        "topic": topics[clip.topic_id].label if clip.topic_id in topics else "",
                        "tags": topics[clip.topic_id].tags if clip.topic_id in topics else [],
                        "start": clip.start,
                        "end": clip.end,
                        "duration": round(clip.duration, 2),
                        "score": round(clip.score, 4),
                        "status": clip.status,
                        "path": clip.path,
                        "excerpt": clip.excerpt,
                    }
                    for clip in clips
                ],
                indent=2,
            )
        )
        return 0

    if not clips:
        print("No clips yet." if not args.session else f"No clips for session {args.session}.")
        return 0

    topic_labels: dict[int, str] = {}
    for session_id in {clip.session_id for clip in clips}:
        for topic in db.list_topics(session_id):
            topic_labels[topic.id] = topic.label

    print(f"{'ID':>5}  {'SESS':>4}  {'START':>9}  {'LEN':>6}  {'SCORE':>6}  TOPIC")
    print("-" * 78)
    for clip in clips:
        label = topic_labels.get(clip.topic_id or -1, "")
        marker = "" if clip.status == "done" else f" [{clip.status}]"
        print(
            f"{clip.id:>5}  {clip.session_id:>4}  {hhmmss(clip.start):>9}  "
            f"{clip.duration:>5.1f}s  {clip.score:>6.3f}  {label[:40]}{marker}"
        )
    print(f"\n{len(clips)} clip(s).")
    return 0


def cmd_clips_export(config: Config, db: Database, args: argparse.Namespace) -> int:
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    clips = [
        clip
        for clip in db.list_clips(args.session, status="done")
        if clip.score >= args.min_score
    ]
    clips.sort(key=lambda c: c.score, reverse=True)
    if args.limit:
        clips = clips[: args.limit]

    if not clips:
        print("Nothing to export.")
        return 0

    topics = {
        topic.id: topic
        for session_id in {clip.session_id for clip in clips}
        for topic in db.list_topics(session_id)
    }

    exported: list[dict[str, Any]] = []
    for clip in clips:
        source_path = clip.vertical_path if args.vertical and clip.vertical_path else clip.path
        if not source_path:
            continue
        source = Path(source_path)
        if not source.exists():
            log.warning("export.missing", extra={"clip_id": clip.id, "path": str(source)})
            continue

        target = dest / source.name
        counter = 1
        while target.exists():
            target = dest / f"{source.stem}_{counter}{source.suffix}"
            counter += 1

        if args.move:
            shutil.move(str(source), target)
            db.update_clip(
                clip.id,
                **(
                    {"vertical_path": str(target)}
                    if args.vertical and clip.vertical_path
                    else {"path": str(target)}
                ),
            )
        else:
            shutil.copy2(source, target)

        topic = topics.get(clip.topic_id or -1)
        exported.append(
            {
                "file": target.name,
                "clip_id": clip.id,
                "session_id": clip.session_id,
                "topic": topic.label if topic else "",
                "summary": topic.summary if topic else "",
                "tags": topic.tags if topic else [],
                "start": round(clip.start, 3),
                "end": round(clip.end, 3),
                "start_hms": hhmmss(clip.start),
                "duration": round(clip.duration, 3),
                "hype_score": round(clip.score, 4),
                "scores": clip.scores,
                "transcript": clip.excerpt,
            }
        )
        print(f"{'Moved' if args.move else 'Copied'} {target.name}")

    index = dest / "clips.json"
    index.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(exported)} clip(s) -> {dest}  (index: {index.name})")
    return 0


def cmd_doctor(config: Config, db: Database, args: argparse.Namespace) -> int:
    print(f"stream-clipper {__version__}\n")

    print("External binaries")
    for binary, purpose in (
        ("ffmpeg", "segmenting and clip cutting (required)"),
        ("ffprobe", "keyframe detection (required)"),
        ("streamlink", "live capture and VOD download"),
    ):
        mark = "OK " if has_binary(binary) else "-- "
        print(f"  [{mark}] {binary:<12} {purpose}")

    print("\nOptional Python extras")
    for module, extra, purpose in (
        ("faster_whisper", "whisper", "transcription"),
        ("sentence_transformers", "embeddings", "semantic topic boundaries"),
        ("anthropic", "llm", "topic labels and clippability scoring"),
    ):
        try:
            __import__(module)
            print(f"  [OK ] {module:<22} {purpose}")
        except ImportError:
            print(f"  [-- ] {module:<22} {purpose}  -> pip install \"stream-clipper[{extra}]\"")

    print("\nCredentials")
    try:
        config.twitch.resolve_credentials()
        print("  [OK ] Twitch app credentials")
    except StreamClipperError:
        print("  [-- ] Twitch app credentials  -> set TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET")
    print(
        "  [OK ] Anthropic API key"
        if config.llm.resolve_api_key()
        else "  [-- ] Anthropic API key  -> set ANTHROPIC_API_KEY (or run `ant auth login`)"
    )

    print("\nPaths")
    for name, path in (
        ("data", config.paths.root),
        ("segments", config.paths.segments),
        ("clips", config.paths.output),
        ("database", config.paths.db),
    ):
        print(f"  {name:<10} {path}{'' if path.exists() else '  (will be created)'}")

    sessions = db.list_sessions(limit=5)
    if sessions:
        print("\nRecent sessions")
        for session in sessions:
            print(
                f"  #{session.id:<4} {session.channel:<18} {session.status:<12} "
                f"{iso(session.started_at)}  {hhmmss(session.duration)}"
            )
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, config_overrides(args))
    except StreamClipperError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.logging.level, config.logging.format, config.logging.file)
    config.paths.ensure()
    db = Database(config.paths.db)

    handlers = {
        ("record", None): cmd_record,
        ("process", None): cmd_process,
        ("doctor", None): cmd_doctor,
        ("clips", "list"): cmd_clips_list,
        ("clips", "export"): cmd_clips_export,
    }
    key = (args.command, getattr(args, "clips_command", None))
    handler = handlers.get(key)
    if handler is None:
        parser.error(f"Unknown command: {' '.join(k for k in key if k)}")

    try:
        return handler(config, db, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except StreamClipperError as exc:
        log.error("command.failed", extra={"command": args.command, "reason": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
