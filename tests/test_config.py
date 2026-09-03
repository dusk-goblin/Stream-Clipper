"""Config loading, merging and validation."""

from __future__ import annotations

import pytest
import yaml

from streamclipper.config import deep_merge, load_config
from streamclipper.errors import ConfigError


def test_defaults_load_and_nest():
    config = load_config()
    assert config.channel == "hasanabi"
    assert isinstance(config.segment.semantic.window_sentences, int)
    assert isinstance(config.clips.vertical.height, int)
    assert config.highlight.emotes


def test_emote_names_with_punctuation_survive_yaml():
    """`D:` is a real emote and a YAML mapping key -- it must stay a string."""
    emotes = load_config().highlight.emotes
    assert all(isinstance(e, str) for e in emotes)
    assert "D:" in emotes


def test_user_config_merges_over_defaults(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(
        yaml.safe_dump({"channel": "someone", "clips": {"pad_before": 5.0}})
    )
    config = load_config(path)
    assert config.channel == "someone"
    assert config.clips.pad_before == 5.0
    # Untouched keys keep their defaults.
    assert config.clips.pad_after == 1.5
    assert config.clips.crf == 20


def test_cli_overrides_beat_the_file(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump({"channel": "fromfile"}))
    config = load_config(path, {"channel": "fromcli"})
    assert config.channel == "fromcli"


def test_lists_replace_rather_than_append(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump({"highlight": {"emotes": ["OnlyThis"]}}))
    assert load_config(path).highlight.emotes == ["OnlyThis"]


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"b": 9}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_unknown_keys_are_rejected_with_a_useful_message(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump({"clips": {"padd_before": 1.0}}))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "padd_before" in str(exc.value)
    assert "clips" in str(exc.value)


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"channel": ""}, "channel"),
        ({"capture": {"segment_seconds": 1}}, "segment_seconds"),
        ({"capture": {"container": "avi"}}, "container"),
        ({"segment": {"min_topic_seconds": 0}}, "min_topic_seconds"),
        ({"segment": {"min_topic_seconds": 900, "max_topic_seconds": 300}}, "max_topic_seconds"),
        ({"highlight": {"clip_min_seconds": 90, "clip_max_seconds": 30}}, "clip_max_seconds"),
        ({"highlight": {"stride_seconds": 0}}, "stride_seconds"),
        ({"clips": {"mode": "fast"}}, "clips.mode"),
        ({"clips": {"pad_before": -1}}, "padding"),
    ],
)
def test_invalid_values_are_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides)


def test_paths_derive_from_the_data_dir(tmp_path):
    config = load_config(overrides={"paths": {"data_dir": str(tmp_path)}})
    assert config.paths.segments == tmp_path / "segments"
    assert config.paths.output == tmp_path / "clips"
    assert config.paths.db == tmp_path / "state.db"

    config.paths.ensure()
    assert config.paths.segments.is_dir()
    assert config.paths.chat.is_dir()


def test_explicit_paths_override_the_derived_ones(tmp_path):
    config = load_config(
        overrides={
            "paths": {"data_dir": str(tmp_path), "output_dir": str(tmp_path / "elsewhere")}
        }
    )
    assert config.paths.output == tmp_path / "elsewhere"
    assert config.paths.segments == tmp_path / "segments"


def test_weights_normalise_to_one():
    weights = load_config(
        overrides={"highlight": {"weights": {"chat_rate": 2, "emote_spike": 1, "llm": 1}}}
    ).highlight.weights.normalised()
    total = weights.chat_rate + weights.emote_spike + weights.llm
    assert total == pytest.approx(1.0)
    assert weights.chat_rate == pytest.approx(0.5)


def test_zero_weights_are_rejected():
    weights = load_config(
        overrides={"highlight": {"weights": {"chat_rate": 0, "emote_spike": 0, "llm": 0}}}
    ).highlight.weights
    with pytest.raises(ConfigError, match="positive"):
        weights.normalised()


def test_twitch_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TWITCH_CLIENT_ID", "abc")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "def")
    assert load_config().twitch.resolve_credentials() == ("abc", "def")


def test_missing_twitch_credentials_explain_themselves(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    with pytest.raises(ConfigError, match="dev.twitch.tv"):
        load_config().twitch.resolve_credentials()


def test_stages_can_be_disabled(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text(yaml.safe_dump({"stages": {"cut": False, "rank": False}}))
    config = load_config(path)
    assert config.stages.cut is False
    assert config.stages.transcribe is True
