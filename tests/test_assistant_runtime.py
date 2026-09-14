"""Runtime composition and CLI regression tests for the local assistant MVP."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
import pytest

from rai.assistant.runtime import (
    AssistantConfigurationError,
    resolve_assistant_config,
)
from rai.cli import cli

MINIMUM_GUILE_OCCURRENCES = 2


def test_runtime_auto_discovers_chat_gguf(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model = model_dir / "tiny-chat.Q4_K_M.gguf"
    model.write_bytes(b"gguf")

    resolved = resolve_assistant_config({}, search_root=tmp_path)

    assert resolved.backend == "llama"
    assert resolved.model == str(model)


def test_runtime_fails_without_model_instead_of_using_stub(tmp_path: Path) -> None:
    with pytest.raises(AssistantConfigurationError, match="No local assistant model"):
        resolve_assistant_config({}, search_root=tmp_path)


def test_runtime_allows_explicit_deterministic_conformance_backend() -> None:
    resolved = resolve_assistant_config({"assistant": {"backend": "deterministic"}})
    assert resolved.backend == "deterministic"
    assert resolved.model == "deterministic-conformance"


def test_runtime_uses_active_profile_for_backend_model_and_memory_scope() -> None:
    resolved = resolve_assistant_config(
        {
            "active_agent": "work",
            "agents": {
                "work": {
                    "backend": "ollama",
                    "model": "qwen3.5:2b",
                    "ollama_host": "http://127.0.0.1:11434",
                    "system": "Answer concisely.",
                }
            },
        }
    )

    assert resolved.backend == "ollama"
    assert resolved.model == "qwen3.5:2b"
    assert resolved.profile_scope == "work"
    assert resolved.system_instruction == "Answer concisely."


def test_cli_chat_persists_memory_through_interactive_path(tmp_path: Path) -> None:
    runner = CliRunner()
    environment = {"RAI_DATA_DIR": str(tmp_path / "data")}
    config = {"assistant": {"backend": "deterministic"}}

    with patch("rai.cli_commands._assistant_config", return_value=config):
        result = runner.invoke(
            cli,
            ["assistant", "chat", "--backend", "deterministic", "--show-context"],
            input=(
                "Zapamiętaj, że w przykładach kodu preferuję Guile.\n"
                "W jakim języku powinieneś pokazywać mi przykłady kodu?\n"
                "/q\n"
            ),
            env=environment,
        )

    assert result.exit_code == 0, result.output
    assert result.output.count("Guile") >= MINIMUM_GUILE_OCCURRENCES
    assert "durable_memory_ids" in result.output


def test_cli_ask_allows_explicit_conformance_backend(tmp_path: Path) -> None:
    runner = CliRunner()
    config = {"assistant": {"backend": "deterministic"}}
    with patch("rai.cli_commands._assistant_config", return_value=config):
        result = runner.invoke(
            cli,
            ["assistant", "ask", "hello", "--backend", "deterministic"],
            env={"RAI_DATA_DIR": str(tmp_path / "data")},
        )

    assert result.exit_code == 0
    assert "Rozumiem." in result.output


def test_cli_remembers_name_across_process_like_invocations(tmp_path: Path) -> None:
    runner = CliRunner()
    config = {"assistant": {"backend": "deterministic"}}
    environment = {"RAI_DATA_DIR": str(tmp_path / "data")}

    with patch("rai.cli_commands._assistant_config", return_value=config):
        admission = runner.invoke(
            cli,
            ["assistant", "ask", "Jestem Tomek, a Ty?", "--backend", "deterministic"],
            env=environment,
        )
        recall = runner.invoke(
            cli,
            ["assistant", "ask", "Jak mam na imię?", "--backend", "deterministic"],
            env=environment,
        )

    assert admission.exit_code == 0, admission.output
    assert "Zapamiętałem Twoje imię" in admission.output
    assert recall.exit_code == 0, recall.output
    assert "Masz na imię Tomek" in recall.output
