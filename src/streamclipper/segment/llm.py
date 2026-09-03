"""LLM passes: topic boundaries with labels, and clippability scoring.

Both calls use the Messages API with ``output_config.format`` set to a JSON
schema, so responses are schema-valid JSON rather than prose we have to
salvage. Adaptive thinking is left on -- deciding where a rant about one
subject turns into another is exactly the kind of judgement it helps with --
with ``effort`` configurable for cost.

Every entry point degrades: if the ``llm`` extra is not installed, no
credentials resolve, or the API declines, the caller gets an empty result and
the pipeline continues on the semantic and chat signals alone.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from ..config import LLMConfig
from ..errors import LLMError
from ..logging_setup import get_logger
from ..transcribe.transcript import Sentence, format_timestamped
from ..util.retry import backoff_delays
from ..util.timefmt import hhmmss, parse_hhmmss

log = get_logger(__name__)

# Beta flag gating the `fallbacks: "default"` scalar form. Anthropic re-runs a
# policy-declined request on its recommended model rather than handing us a
# refusal -- worth having when the input is unfiltered livestream speech.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

BOUNDARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Start timestamp, HH:MM:SS, copied from a transcript line.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Topic name, at most 8 words.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One sentence describing what is discussed.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 8,
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["start", "label", "summary", "tags", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}

CLIPPABILITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {
                        "type": "number",
                        "description": "Clippability from 0.0 to 1.0.",
                    },
                    "reason": {"type": "string"},
                    "title": {
                        "type": "string",
                        "description": "Short title for the clip, at most 10 words.",
                    },
                },
                "required": ["id", "score", "reason", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

SEGMENT_SYSTEM = """You segment livestream transcripts into topics.

You are given a timestamped transcript window from a single live broadcast. \
Identify where the streamer moves from one subject to a genuinely different \
one, and label each resulting topic.

Rules:
- A topic is a sustained subject, not every passing remark. Reading a chat \
message, a short tangent, or a joke inside a longer discussion is not a new \
topic.
- Return a start timestamp for every topic in the window, including the one \
already in progress at the start of the window.
- Copy start timestamps verbatim from a transcript line's [HH:MM:SS] marker. \
Never invent a time that does not appear in the transcript.
- Timestamps must increase and must not repeat.
- The label names the subject ("Reacting to the debate clip"), not the format \
("talking about something").
- Tags are lowercase topical keywords: 3 to 8 of them.
- confidence is 0.0 to 1.0: how sure you are that a real topic change starts \
at that timestamp. Use a low value for the topic already in progress at the \
window start.

Transcripts are automatic speech recognition of unscripted live speech: \
expect disfluency, profanity and mistranscription. Describe what is discussed \
neutrally; you are indexing content, not endorsing or judging it."""

CLIPPABILITY_SYSTEM = """You rate excerpts from a livestream for how well they \
would work as standalone short clips.

Score each candidate from 0.0 to 1.0:
- 1.0  A self-contained moment with a clear hook: a strong reaction, a punchline \
that lands, a surprising claim, a story with a payoff.
- 0.5  Interesting but needs context, or the payoff sits outside the excerpt.
- 0.0  Filler: setup with no payoff, reading text aloud, dead air, technical chatter.

Judge only what the excerpt itself contains. An excerpt that starts or ends \
mid-thought should score lower, because a viewer sees exactly this and no more. \
Reward moments that resolve inside the window.

Give every candidate a title that would work as the clip's name.

Transcripts are automatic speech recognition of unscripted live speech: expect \
disfluency, profanity and mistranscription."""


@dataclass
class LLMTopic:
    start: float
    label: str
    summary: str
    tags: list[str]
    confidence: float


@dataclass
class LLMScore:
    id: int
    score: float
    reason: str
    title: str


class LLMClient:
    """Anthropic Messages API client for the two structured passes."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None
        self._lock = threading.Lock()
        self._unavailable_reason: str | None = None

    # -- availability -----------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether LLM calls can be attempted at all."""
        if self._unavailable_reason is not None:
            return False
        return self._ensure_client() is not None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._unavailable_reason is not None:
            return None
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                import anthropic  # noqa: PLC0415
            except ImportError:
                self._disable(
                    'the anthropic package is not installed '
                    '(pip install "stream-clipper[llm]")'
                )
                return None
            try:
                api_key = self.config.resolve_api_key()
                kwargs: dict[str, Any] = {
                    "timeout": self.config.timeout,
                    # We do our own backoff around the whole call so a refusal
                    # or a schema miss is retried the same way a 429 is.
                    "max_retries": 0,
                }
                if api_key:
                    kwargs["api_key"] = api_key
                # With no explicit key the SDK still resolves an `ant auth
                # login` profile; only a genuine failure here disables us.
                self._client = anthropic.Anthropic(**kwargs)
            except Exception as exc:
                self._disable(f"client init failed: {exc}")
                return None
            return self._client

    def _disable(self, reason: str) -> None:
        self._unavailable_reason = reason
        log.warning("llm.disabled", extra={"reason": reason})

    # -- transport --------------------------------------------------------

    def _complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """One structured call, with backoff. None if it could not be served."""
        client = self._ensure_client()
        if client is None:
            return None

        import anthropic  # noqa: PLC0415 -- guaranteed importable past this point

        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            "thinking": {"type": "adaptive"},
        }

        attempts = max(1, self.config.max_retries)
        delays = backoff_delays(attempts, base=2.0, cap=45.0)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                if self.config.refusal_fallbacks:
                    response = client.beta.messages.create(
                        betas=[_FALLBACK_BETA], fallbacks="default", **request
                    )
                else:
                    response = client.messages.create(**request)

                # A refusal arrives as HTTP 200 -- check before reading content.
                if getattr(response, "stop_reason", None) == "refusal":
                    details = getattr(response, "stop_details", None)
                    log.warning(
                        "llm.refused",
                        extra={"category": getattr(details, "category", None)},
                    )
                    return None
                if getattr(response, "stop_reason", None) == "max_tokens":
                    raise LLMError(
                        "response hit max_tokens; raise llm.max_tokens or shrink "
                        "segment.llm.window_seconds"
                    )

                text = next(
                    (b.text for b in response.content if getattr(b, "type", "") == "text"),
                    None,
                )
                if not text:
                    raise LLMError("no text block in response")
                return json.loads(text)

            except anthropic.NotFoundError as exc:
                self._disable(f"model {self.config.model!r} not available: {exc}")
                return None
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
                self._disable(f"authentication failed: {exc}")
                return None
            except anthropic.BadRequestError as exc:
                # A malformed request will fail identically on every retry.
                log.error("llm.bad_request", extra={"reason": str(exc)[:400]})
                return None
            except (
                anthropic.RateLimitError,
                anthropic.APIStatusError,
                anthropic.APIConnectionError,
                LLMError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    delay = delays[attempt]
                    log.warning(
                        "llm.retry",
                        extra={
                            "attempt": attempt + 1,
                            "of": attempts,
                            "delay": round(delay, 1),
                            "reason": str(exc)[:200],
                        },
                    )
                    import time  # noqa: PLC0415

                    time.sleep(delay)

        log.error("llm.failed", extra={"reason": str(last_error)[:400]})
        return None

    # -- passes -----------------------------------------------------------

    def topic_boundaries(
        self, sentences: Sequence[Sentence], window_start: float, window_end: float
    ) -> list[LLMTopic]:
        """Topic starts, labels, summaries and tags for one transcript window."""
        if not sentences:
            return []

        transcript = format_timestamped(sentences)
        user = (
            f"Transcript window: {hhmmss(window_start)} to {hhmmss(window_end)}.\n"
            f"Return every topic that starts within this window.\n\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )
        payload = self._complete_json(SEGMENT_SYSTEM, user, BOUNDARY_SCHEMA)
        if not payload:
            return []

        topics: list[LLMTopic] = []
        for entry in payload.get("topics") or []:
            try:
                start = parse_hhmmss(str(entry["start"]))
            except (KeyError, ValueError):
                log.debug("llm.bad_timestamp", extra={"entry": str(entry)[:200]})
                continue
            # The model occasionally answers just outside the window it was
            # given; clamping is safer than discarding a real boundary.
            if not (window_start - 60 <= start <= window_end + 60):
                continue
            start = min(max(start, window_start), window_end)
            topics.append(
                LLMTopic(
                    start=start,
                    label=str(entry.get("label", "")).strip(),
                    summary=str(entry.get("summary", "")).strip(),
                    tags=[str(t).strip() for t in (entry.get("tags") or []) if str(t).strip()],
                    confidence=_clamp(entry.get("confidence", 0.5)),
                )
            )
        topics.sort(key=lambda t: t.start)
        log.info(
            "llm.topics",
            extra={
                "window": f"{hhmmss(window_start)}-{hhmmss(window_end)}",
                "found": len(topics),
            },
        )
        return topics

    def score_clippability(
        self, candidates: Sequence[tuple[int, float, float, str]], topic_label: str
    ) -> dict[int, LLMScore]:
        """Rate excerpts. ``candidates`` is ``(id, start, end, text)``."""
        if not candidates:
            return {}

        blocks = "\n\n".join(
            f'<candidate id="{cid}" start="{hhmmss(start)}" '
            f'duration="{end - start:.0f}s">\n{text}\n</candidate>'
            for cid, start, end, text in candidates
        )
        user = (
            f"These excerpts all come from one topic: {topic_label or 'unlabelled'}.\n"
            f"Score every candidate. Return one entry per candidate id.\n\n{blocks}"
        )
        payload = self._complete_json(CLIPPABILITY_SYSTEM, user, CLIPPABILITY_SCHEMA)
        if not payload:
            return {}

        valid_ids = {cid for cid, _, _, _ in candidates}
        scores: dict[int, LLMScore] = {}
        for entry in payload.get("candidates") or []:
            try:
                cid = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if cid not in valid_ids:
                continue
            scores[cid] = LLMScore(
                id=cid,
                score=_clamp(entry.get("score", 0.0)),
                reason=str(entry.get("reason", "")).strip(),
                title=str(entry.get("title", "")).strip(),
            )
        log.info(
            "llm.clippability",
            extra={"topic": topic_label[:60], "scored": len(scores), "of": len(candidates)},
        )
        return scores


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low
