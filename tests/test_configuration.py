"""Configuration contracts: no silent fallback, no lost state, no side effects."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rai.cli import cli
from rai.configuration.storage import (
    ConfigurationError,
    configuration_path,
    load_settings,
    load_runtime_state,
    save_settings,
    selected_path,
    state_path,
)
from rai.configuration.resolution import effective_model_settings
from rai import config_manager


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("RAI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("RAI_CONFIG_FILE", raising=False)
    return tmp_path


@pytest.mark.parametrize("extension", ["json", "yaml", "yml"])
def test_formats_preserve_actions_and_do_not_write_on_read(settings_dir, extension):
    path = settings_dir / f"config.{extension}"
    data = {
        "assistant": {"backend": "lemonade", "enable_thinking": False},
        "actions": {"allowed_file_roots": ["/tmp/rai-documents"]},
    }
    path.write_text(json.dumps(data) if extension == "json" else yaml.safe_dump(data))
    before = path.read_bytes()
    assert load_settings(str(path)).assistant.backend == "lemonade"
    assert path.read_bytes() == before
    save_settings(load_settings(str(path)).runtime_mapping(), str(path))
    assert load_settings(str(path)).actions.allowed_file_roots == ["/tmp/rai-documents"]
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "content", ["{bad", "[]", "null", '{"assistant":{},"assistant":{}}']
)
def test_invalid_json_fails_without_overwriting(settings_dir, content):
    path = settings_dir / "config.json"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_settings(str(path))
    assert path.read_text() == content


@pytest.mark.parametrize(
    "content",
    [
        "assistant: []",
        'assistant:\n  enable_thinking: "false"',
        "assistant:\n  backent: lemonade",
        'actions:\n  allowed_file_roots: ["/tmp/.."]',
        "assistant: {}\nassistant: {}",
        '!!python/object/apply:os.system ["false"]',
    ],
)
def test_yaml_rejects_bad_types_unknown_keys_duplicates_and_tags(settings_dir, content):
    path = settings_dir / "config.yaml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        load_settings(str(path))


def test_custom_config_uses_sibling_profiles_not_global(settings_dir):
    (settings_dir / "agents.yaml").write_text("global: {model: wrong}")
    nested = settings_dir / "custom"
    nested.mkdir()
    (nested / "agents.yaml").write_text("work: {model: right}")
    path = nested / "config.json"
    path.write_text("{}")
    assert set(load_settings(str(path)).agents) == {"work"}
    assert set(settings_dir.iterdir()) == {nested, settings_dir / "agents.yaml"}


def test_ambiguous_default_requires_explicit_choice(settings_dir):
    for name in ("config.yaml", "config.json"):
        (settings_dir / name).write_text("{}")
    with pytest.raises(ConfigurationError):
        selected_path()
    assert load_settings(str(settings_dir / "config.yaml"))


def test_missing_explicit_file_is_an_error(settings_dir):
    with pytest.raises(ConfigurationError):
        load_settings(str(settings_dir / "missing.yaml"))
    assert load_settings().active_agent == "default"
    assert list(settings_dir.iterdir()) == []


def test_config_roundtrip_preserves_session_mapping_outside_settings(settings_dir):
    path = settings_dir / "config.json"
    path.write_text(
        json.dumps(
            {"session_conversation_ids": {"work": "abc"}, "active_session_id": "abc"}
        )
    )
    loaded = config_manager.load_config(str(path))
    config_manager.save_config(loaded, str(path))
    assert "session_conversation_ids" not in json.loads(path.read_text())
    assert load_runtime_state(str(path))["session_conversation_ids"] == {"work": "abc"}
    assert state_path(str(path)).exists()


def test_nested_config_context_restores_selection(settings_dir):
    with configuration_path(str(settings_dir / "a.yaml")):
        with configuration_path(str(settings_dir / "b.json")):
            assert selected_path().name == "b.json"
        assert selected_path().name == "a.yaml"
    assert selected_path().name == "config.yaml"


def test_precedence_preserves_explicit_false_and_cli_over_env():
    values, sources = effective_model_settings(
        {
            "agents": {"default": {"model": "profile"}},
            "local_ai": {"model": "local", "enable_thinking": True},
            "assistant": {"model": "assistant", "enable_thinking": False},
        },
        environment={"RAI_ASSISTANT_MODEL": "env"},
        overrides={"model": "cli"},
    )
    assert values["model"] == "cli" and sources["model"] == "CLI"
    assert values["enable_thinking"] is False


def test_migration_preserves_sources_and_state_and_refuses_overwrite(settings_dir):
    source = settings_dir / "config.json"
    source.write_text(
        '{"sessions":{"work":{"backend":"lemonade"}},"session_conversation_ids":{"work":"abc"}}'
    )
    output = settings_dir / "new.yaml"
    before = source.read_bytes()
    runner = CliRunner()
    args = ["config", "migrate", str(source), "--output", str(output)]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert source.read_bytes() == before
    assert load_settings(str(output)).agents["work"].backend == "lemonade"
    assert load_runtime_state(str(output))["session_conversation_ids"] == {
        "work": "abc"
    }
    assert runner.invoke(cli, args).exit_code != 0


def test_effective_config_redacts_keys_and_set_uses_types(settings_dir):
    path = settings_dir / "config.yaml"
    path.write_text("assistant:\n  lemonade_api_key: do-not-print\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "show", "--effective"])
    assert result.exit_code == 0, result.output
    assert "do-not-print" not in result.output and "<redacted>" in result.output
    assert (
        runner.invoke(
            cli, ["config", "set", "assistant.enable_thinking", "false"]
        ).exit_code
        == 0
    )
    assert load_settings().assistant.enable_thinking is False
    assert (
        runner.invoke(cli, ["config", "set", "assistant.backent", "oops"]).exit_code
        != 0
    )
    assert "backent" not in path.read_text()


def test_cli_invalid_config_reports_safe_error(settings_dir):
    path = settings_dir / "config.yaml"
    path.write_text("assistant:\n  lemonade_api_key: [do-not-print]\n")
    result = CliRunner().invoke(cli, ["config", "validate"])
    assert result.exit_code != 0
    assert "lemonade_api_key" in result.output
    assert "do-not-print" not in result.output


def test_failed_atomic_replace_preserves_config(settings_dir, monkeypatch):
    from rai.configuration import storage

    path = settings_dir / "config.yaml"
    path.write_text("assistant: {backend: lemonade}")
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("simulated disk error")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError):
        save_settings({"assistant": {"backend": "llama"}}, str(path))
    assert path.read_bytes() == before
    assert list(settings_dir.iterdir()) == [path]


def test_chat_config_uses_same_effective_settings(settings_dir):
    path = settings_dir / "config.yaml"
    path.write_text("assistant: {backend: deterministic, enable_thinking: false}\n")
    runner = CliRunner()
    shown = runner.invoke(cli, ["config", "show", "--effective"])
    chat = runner.invoke(cli, ["assistant", "chat"], input="/config\n/q\n")
    assert chat.exit_code == 0, chat.output
    values = yaml.safe_load(shown.output)["effective_assistant"]
    assert f'"backend": "{values["backend"]}"' in chat.output
    assert '"enable_thinking": false' in chat.output
