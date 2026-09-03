"""LLM pass: request shape, response parsing, and graceful degradation.

The transport tests run against a local stand-in server rather than the real
API, so they assert the exact request this code sends without spending money
or needing credentials.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading

import pytest

from streamclipper.config import HighlightConfig, LLMConfig
from streamclipper.highlight.llm_score import NEUTRAL, attach_llm_scores
from streamclipper.highlight.rank import Candidate
from streamclipper.segment.llm import LLMClient
from streamclipper.transcribe.transcript import Sentence

anthropic = pytest.importorskip("anthropic", reason="needs the [llm] extra")


def message(content_text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


class StubAPI:
    """A local server standing in for the Messages API."""

    def __init__(self, response: dict, status: int = 200) -> None:
        self.response = response
        self.status = status
        self.requests: list[dict] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                outer.requests.append(
                    {
                        "path": self.path,
                        "body": json.loads(self.rfile.read(length) or b"{}"),
                        "headers": dict(self.headers),
                    }
                )
                payload = json.dumps(outer.response).encode()
                self.send_response(outer.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "StubAPI":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def body(self) -> dict:
        return self.requests[0]["body"]


@pytest.fixture
def llm_config(monkeypatch):
    def build(url: str, **kwargs) -> LLMConfig:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        return LLMConfig(api_key="sk-ant-test", max_retries=1, timeout=10.0, **kwargs)

    return build


SENTENCES = [
    Sentence(380.0, 386.0, "Alright, the new patch dropped and the balance changes are insane."),
    Sentence(390.0, 396.0, "They nerfed the shotgun damage falloff by thirty percent."),
]

TOPIC_RESPONSE = json.dumps(
    {
        "topics": [
            {
                "start": "00:06:20",
                "label": "Game patch balance changes",
                "summary": "Going through the new patch notes.",
                "tags": ["gaming", "patch", "balance"],
                "confidence": 0.86,
            }
        ]
    }
)


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


def test_request_uses_the_configured_model_and_structured_output(llm_config):
    with StubAPI(message(TOPIC_RESPONSE)) as api:
        client = LLMClient(llm_config(api.url, model="claude-opus-5", effort="medium"))
        client.topic_boundaries(SENTENCES, 0.0, 1800.0)

    body = api.body
    assert body["model"] == "claude-opus-5"
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"
    # Structured output is what makes the response parseable rather than prose.
    assert body["output_config"]["format"]["type"] == "json_schema"
    schema = body["output_config"]["format"]["schema"]
    assert schema["properties"]["topics"]["items"]["required"] == [
        "start", "label", "summary", "tags", "confidence"
    ]
    assert schema["additionalProperties"] is False


def test_refusal_fallbacks_are_enabled_by_default(llm_config):
    with StubAPI(message(TOPIC_RESPONSE)) as api:
        LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0)
    assert api.body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in api.requests[0]["headers"]["anthropic-beta"]


def test_refusal_fallbacks_can_be_turned_off(llm_config):
    with StubAPI(message(TOPIC_RESPONSE)) as api:
        client = LLMClient(llm_config(api.url, refusal_fallbacks=False))
        client.topic_boundaries(SENTENCES, 0.0, 1800.0)
    assert "fallbacks" not in api.body


def test_transcript_is_sent_with_absolute_timestamps(llm_config):
    with StubAPI(message(TOPIC_RESPONSE)) as api:
        LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0)
    user = api.body["messages"][0]["content"]
    assert "[00:06:20]" in user
    assert "the new patch dropped" in user


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def test_topics_parse_timestamps_labels_and_tags(llm_config):
    with StubAPI(message(TOPIC_RESPONSE)) as api:
        topics = LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0)

    assert len(topics) == 1
    assert topics[0].start == pytest.approx(380.0)
    assert topics[0].label == "Game patch balance changes"
    assert topics[0].tags == ["gaming", "patch", "balance"]
    assert topics[0].confidence == pytest.approx(0.86)


def test_unparseable_timestamps_are_dropped_not_fatal(llm_config):
    payload = json.dumps(
        {
            "topics": [
                {"start": "sometime later", "label": "Bad", "summary": "", "tags": [], "confidence": 0.5},
                {"start": "00:06:20", "label": "Good", "summary": "", "tags": [], "confidence": 0.5},
            ]
        }
    )
    with StubAPI(message(payload)) as api:
        topics = LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0)
    assert [t.label for t in topics] == ["Good"]


def test_out_of_window_timestamps_are_discarded(llm_config):
    payload = json.dumps(
        {
            "topics": [
                {"start": "05:00:00", "label": "Way past", "summary": "", "tags": [], "confidence": 0.9}
            ]
        }
    )
    with StubAPI(message(payload)) as api:
        assert LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0) == []


def test_confidence_is_clamped(llm_config):
    payload = json.dumps(
        {
            "topics": [
                {"start": "00:06:20", "label": "X", "summary": "", "tags": [], "confidence": 7.5}
            ]
        }
    )
    with StubAPI(message(payload)) as api:
        topics = LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0)
    assert topics[0].confidence == 1.0


def test_a_refusal_yields_no_topics_rather_than_raising(llm_config):
    """stop_reason 'refusal' arrives as HTTP 200 -- reading content would break."""
    refusal = message("", stop_reason="refusal")
    refusal["content"] = []
    with StubAPI(refusal) as api:
        assert LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0) == []


def test_clippability_scores_map_back_to_candidate_ids(llm_config):
    payload = json.dumps(
        {
            "candidates": [
                {"id": 0, "score": 0.9, "reason": "clear payoff", "title": "The shotgun nerf"},
                {"id": 99, "score": 0.5, "reason": "not a real id", "title": "Ignored"},
            ]
        }
    )
    with StubAPI(message(payload)) as api:
        scored = LLMClient(llm_config(api.url)).score_clippability(
            [(0, 100.0, 140.0, "some transcript text")], "Game patch"
        )
    assert set(scored) == {0}
    assert scored[0].score == pytest.approx(0.9)
    assert scored[0].title == "The shotgun nerf"


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


def test_a_server_error_degrades_to_no_result(llm_config):
    with StubAPI({"error": "boom"}, status=500) as api:
        assert LLMClient(llm_config(api.url)).topic_boundaries(SENTENCES, 0.0, 1800.0) == []


def test_authentication_failure_disables_the_client(llm_config):
    with StubAPI({"error": {"type": "authentication_error"}}, status=401) as api:
        client = LLMClient(llm_config(api.url))
        assert client.topic_boundaries(SENTENCES, 0.0, 1800.0) == []
        # Further calls must not keep hammering a key that will never work.
        assert client.available is False


def test_scoring_shortlist_leaves_unrated_candidates_neutral(llm_config):
    candidates = [
        Candidate(start=float(i * 60), end=float(i * 60 + 40), chat_score=0.5)
        for i in range(6)
    ]
    payload = json.dumps(
        {"candidates": [{"id": 0, "score": 0.9, "reason": "r", "title": "t"}]}
    )
    with StubAPI(message(payload)) as api:
        scores, titles, _ = attach_llm_scores(
            candidates,
            [Sentence(0.0, 400.0, "text covering all of the candidates here")],
            "Topic",
            LLMClient(llm_config(api.url)),
            HighlightConfig(),
        )
    assert scores[0] == pytest.approx(0.9)
    assert all(scores[i] == NEUTRAL for i in range(1, 6))
    assert titles[0] == "t"


def test_scoring_is_skipped_when_disabled():
    config = HighlightConfig()
    config.llm.enabled = False
    scores, _, _ = attach_llm_scores(
        [Candidate(start=0.0, end=40.0)], [], "Topic", LLMClient(LLMConfig()), config
    )
    assert scores == {}
