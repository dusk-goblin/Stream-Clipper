"""YAML configuration -> typed dataclasses.

User config is merged recursively over ``config/default.yaml`` so a partial
file only needs the keys it changes. Secrets fall back to environment
variables rather than living in the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml

from .errors import ConfigError

# Shipped as package data so an installed wheel has the same defaults as a
# source checkout.
_DEFAULTS_FILE = Path(__file__).resolve().parent / "resources" / "default.yaml"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    segments_dir: str | None = None
    output_dir: str | None = None
    db_path: str | None = None

    @property
    def root(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def segments(self) -> Path:
        return Path(self.segments_dir).expanduser().resolve() if self.segments_dir else self.root / "segments"

    @property
    def output(self) -> Path:
        return Path(self.output_dir).expanduser().resolve() if self.output_dir else self.root / "clips"

    @property
    def db(self) -> Path:
        return Path(self.db_path).expanduser().resolve() if self.db_path else self.root / "state.db"

    @property
    def chat(self) -> Path:
        return self.root / "chat"

    def ensure(self) -> None:
        for directory in (self.root, self.segments, self.output, self.chat):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class StagesConfig:
    capture: bool = True
    chat: bool = True
    transcribe: bool = True
    segment: bool = True
    rank: bool = True
    cut: bool = True


@dataclass
class TwitchConfig:
    client_id: str | None = None
    client_secret: str | None = None
    poll_interval: float = 60.0
    live_poll_interval: float = 30.0
    resume_window: float = 900.0
    api_base: str = "https://api.twitch.tv/helix"
    auth_base: str = "https://id.twitch.tv/oauth2"

    def resolve_credentials(self) -> tuple[str, str]:
        """Credentials from the YAML, else the environment. Raises if absent."""
        client_id = self.client_id or os.environ.get("TWITCH_CLIENT_ID")
        secret = self.client_secret or os.environ.get("TWITCH_CLIENT_SECRET")
        if not client_id or not secret:
            raise ConfigError(
                "Twitch app credentials are required for live recording. Set "
                "twitch.client_id / twitch.client_secret in the config, or export "
                "TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET. Create an app at "
                "https://dev.twitch.tv/console/apps."
            )
        return client_id, secret


@dataclass
class CaptureConfig:
    quality: str = "best"
    segment_seconds: int = 300
    container: str = "ts"
    streamlink_args: list[str] = field(
        default_factory=lambda: ["--twitch-disable-ads", "--hls-live-restart"]
    )
    max_restarts: int = 10


@dataclass
class ChatConfig:
    server: str = "irc.chat.twitch.tv"
    port: int = 6667
    nick: str | None = None
    oauth_token: str | None = None
    flush_interval: float = 5.0

    def resolve_token(self) -> str | None:
        return self.oauth_token or os.environ.get("CHAT_OAUTH_TOKEN")


@dataclass
class TranscribeConfig:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = "en"
    beam_size: int = 5
    vad_filter: bool = True
    workers: int = 1


@dataclass
class EmbeddingsConfig:
    backend: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32


@dataclass
class SemanticConfig:
    window_sentences: int = 6
    similarity_threshold: float = 0.55
    depth_threshold: float = 0.12


@dataclass
class SegmentLLMConfig:
    enabled: bool = True
    window_seconds: float = 1800.0
    overlap_seconds: float = 180.0


@dataclass
class SegmentConfig:
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    llm: SegmentLLMConfig = field(default_factory=SegmentLLMConfig)
    merge_tolerance: float = 45.0
    min_topic_seconds: float = 180.0
    max_topic_seconds: float = 2700.0
    settle_seconds: float = 120.0

    def validate(self) -> None:
        if self.min_topic_seconds <= 0:
            raise ConfigError("segment.min_topic_seconds must be > 0")
        if self.max_topic_seconds <= self.min_topic_seconds:
            raise ConfigError(
                "segment.max_topic_seconds must be greater than min_topic_seconds"
            )
        if not 0.0 <= self.semantic.similarity_threshold <= 1.0:
            raise ConfigError(
                "segment.semantic.similarity_threshold must be between 0 and 1"
            )


DEFAULT_EMOTES = (
    "KEKW", "OMEGALUL", "LULW", "LUL", "Pog", "PogU", "POGGERS", "PogChamp",
    "Sadge", "PepeLaugh", "COPIUM", "Clueless", "monkaS", "monkaW", "EZ", "D:",
    "WeirdChamp", "Pepega",
)


@dataclass
class HighlightWeights:
    chat_rate: float = 0.35
    emote_spike: float = 0.25
    llm: float = 0.40

    def normalised(self) -> "HighlightWeights":
        total = self.chat_rate + self.emote_spike + self.llm
        if total <= 0:
            raise ConfigError("highlight.weights must sum to a positive number")
        return HighlightWeights(
            chat_rate=self.chat_rate / total,
            emote_spike=self.emote_spike / total,
            llm=self.llm / total,
        )


@dataclass
class HighlightLLMConfig:
    enabled: bool = True
    max_candidates: int = 12


@dataclass
class HighlightConfig:
    clip_min_seconds: float = 30.0
    clip_max_seconds: float = 90.0
    stride_seconds: float = 5.0
    per_topic: int = 2
    min_score: float = 0.35
    weights: HighlightWeights = field(default_factory=HighlightWeights)
    emotes: list[str] = field(default_factory=lambda: list(DEFAULT_EMOTES))
    llm: HighlightLLMConfig = field(default_factory=HighlightLLMConfig)

    def validate(self) -> None:
        if self.clip_min_seconds <= 0:
            raise ConfigError("highlight.clip_min_seconds must be > 0")
        if self.clip_max_seconds < self.clip_min_seconds:
            raise ConfigError(
                "highlight.clip_max_seconds must be >= clip_min_seconds"
            )
        if self.stride_seconds <= 0:
            raise ConfigError("highlight.stride_seconds must be > 0")


@dataclass
class VerticalConfig:
    enabled: bool = False
    width: int = 1080
    height: int = 1920
    focus_x: float = 0.5


@dataclass
class ClipsConfig:
    mode: str = "auto"
    snap_tolerance: float = 1.5
    pad_before: float = 2.0
    pad_after: float = 1.5
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "veryfast"
    burn_subtitles: bool = False
    subtitle_style: str = ""
    vertical: VerticalConfig = field(default_factory=VerticalConfig)
    manifest_name: str = "manifest.json"

    def validate(self) -> None:
        if self.mode not in {"copy", "reencode", "auto"}:
            raise ConfigError("clips.mode must be one of: copy, reencode, auto")
        if self.pad_before < 0 or self.pad_after < 0:
            raise ConfigError("clips padding must be >= 0")


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    effort: str = "medium"
    max_tokens: int = 16000
    api_key: str | None = None
    max_retries: int = 4
    timeout: float = 120.0
    refusal_fallbacks: bool = True

    def resolve_api_key(self) -> str | None:
        """Explicit key, else the env var.

        ``None`` is not necessarily fatal: the Anthropic SDK also picks up an
        ``ant auth login`` profile, and the pipeline degrades gracefully when
        no credentials resolve at all.
        """
        return self.api_key or os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class RetentionConfig:
    delete_segments_after_clip: bool = False
    raw_max_age_hours: float = 0.0
    max_disk_gb: float = 0.0
    keep_transcripts: bool = True


@dataclass
class RuntimeConfig:
    workers: int = 2
    job_lease_seconds: float = 1800.0
    max_job_attempts: int = 3
    poll_interval: float = 2.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    file: str | None = None


@dataclass
class Config:
    channel: str = "hasanabi"
    paths: PathsConfig = field(default_factory=PathsConfig)
    stages: StagesConfig = field(default_factory=StagesConfig)
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    clips: ClipsConfig = field(default_factory=ClipsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        if not self.channel:
            raise ConfigError("channel must be set")
        if self.capture.segment_seconds < 10:
            raise ConfigError("capture.segment_seconds must be at least 10")
        if self.capture.container not in {"ts", "mp4"}:
            raise ConfigError("capture.container must be 'ts' or 'mp4'")
        self.segment.validate()
        self.highlight.validate()
        self.clips.validate()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Mappings merge key by key; every other type (lists included) replaces
    wholesale, so overriding ``highlight.emotes`` gives you exactly your list
    rather than yours appended to the defaults.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _build(cls: type, data: Any, path: str = "") -> Any:
    """Instantiate the dataclass ``cls`` from a plain dict, recursively."""
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path or 'config'} must be a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = f" under '{path}'" if path else ""
        raise ConfigError(
            f"Unknown config key(s){where}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}"
        )

    # `from __future__ import annotations` leaves f.type as a string, so resolve
    # the real types before deciding what needs recursion.
    hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        child_path = f"{path}.{name}" if path else name
        value = data[name]
        target = hints.get(name)
        # Unwrap Optional[X] so nested dataclasses declared as X | None still
        # get built.
        if get_origin(target) is not None:
            args = [a for a in get_args(target) if a is not type(None)]
            target = args[0] if len(args) == 1 else target
        if isinstance(target, type) and is_dataclass(target):
            kwargs[name] = _build(target, value, child_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_defaults() -> dict[str, Any]:
    """The packaged defaults as a plain dict."""
    if not _DEFAULTS_FILE.exists():
        return {}
    with _DEFAULTS_FILE.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(
    path: str | Path | None = None, overrides: Mapping[str, Any] | None = None
) -> Config:
    """Load defaults, merge a user file and CLI overrides, and validate."""
    data = load_defaults()

    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            user = yaml.safe_load(handle) or {}
        if not isinstance(user, Mapping):
            raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
        data = deep_merge(data, user)

    if overrides:
        data = deep_merge(data, overrides)

    config: Config = _build(Config, data)
    config.validate()
    return config
