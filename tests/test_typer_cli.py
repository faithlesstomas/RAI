"""Public Typer contracts replacing removed compatibility CLI tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Failure, Success
from typer.testing import CliRunner

from rai.cli import cli
from rai.commands import approval
from rai.commands.dialog import _run_assistant_ask, _run_assistant_chat
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, ProducerIdentity
from rai.assistant.records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--help"],
        ["assistant", "--help"],
        ["capability", "--help"],
        ["neural", "--help"],
        ["server", "--help"],
        ["config", "--help"],
        ["profile", "--help"],
    ],
)
def test_modular_cli_help(args):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code in (0, 2), result.output
    assert "Usage:" in result.output


def test_cli_accepts_native_remote_backend_and_classification():
    with patch("rai.commands.assistant._run_assistant_ask") as run:
        result = CliRunner().invoke(
            cli,
            [
                "assistant",
                "ask",
                "hello",
                "--backend",
                "antigravity",
                "--data-class",
                "PUBLIC",
            ],
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.args[2] == "antigravity"
    assert run.call_args.kwargs["data_class"] == DataClass.PUBLIC


def test_unknown_backend_reaches_runtime_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("RAI_CONFIG_DIR", str(tmp_path))
    result = CliRunner().invoke(
        cli, ["assistant", "ask", "hello", "--backend", "bogus", "--model", "dummy"]
    )
    assert result.exit_code != 0
    assert "Unsupported assistant backend" in result.output


def test_cli_overrides_environment(tmp_path, monkeypatch):
    from rai.commands.settings import _assistant_config
    from rai.assistant.runtime import resolve_assistant_config

    monkeypatch.setenv("RAI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("RAI_ASSISTANT_BACKEND", "ollama")
    monkeypatch.setenv("RAI_ASSISTANT_MODEL", "environment-model")
    runtime = resolve_assistant_config(_assistant_config("lemonade", "cli-model"))
    assert runtime.backend == "lemonade" and runtime.model == "cli-model"


@pytest.mark.asyncio
async def test_noninteractive_approval_denies(monkeypatch):
    monkeypatch.setattr(approval.sys.stdin, "isatty", lambda: False)
    assert not await approval.confirm("approve?", CancellationToken())


def package(data_class):
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    manifest = AssistantContextManifest(
        producer=producer,
        session_id="session",
        turn_id="turn",
        items=(
            AssistantContextManifestItem(
                source_id="turn",
                source_type="turn",
                layer="recent",
                data_class=data_class,
            ),
        ),
    )
    return AssistantContextPackage(
        producer=producer,
        session_id="session",
        turn_id="turn",
        manifest=manifest,
        content={},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "classification", [DataClass.LOCAL, DataClass.SECRET, DataClass.BLOCKED]
)
async def test_egress_never_approves_local_or_forbidden_data(
    monkeypatch, classification
):
    async def unexpected(*args):
        pytest.fail("Must reject before prompting")

    monkeypatch.setattr(approval, "confirm", unexpected)
    assert isinstance(
        await approval.approve_egress(package(classification), CancellationToken()),
        Failure,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_egress_approval_changes_only_exact_manifest(monkeypatch, accepted):
    async def answer(*args):
        return accepted

    monkeypatch.setattr(approval, "confirm", answer)
    original = package(DataClass.PRIVATE)
    result = await approval.approve_egress(original, CancellationToken())
    assert not original.manifest.approved
    if accepted:
        assert isinstance(result, Success)
        assert result.unwrap().manifest.approved
        assert result.unwrap().manifest.items == original.manifest.items
    else:
        assert isinstance(result, Failure)


@pytest.mark.asyncio
async def test_terminal_prompt_cancellation_reaps_input(monkeypatch):
    import asyncio
    import prompt_toolkit

    started = asyncio.Event()
    reaped = asyncio.Event()

    class Prompt:
        async def prompt_async(self, text):
            started.set()
            try:
                await asyncio.Future()
            finally:
                reaped.set()

    monkeypatch.setattr(prompt_toolkit, "PromptSession", Prompt)
    monkeypatch.setattr(approval.sys.stdin, "isatty", lambda: True)
    token = CancellationToken()
    task = asyncio.create_task(approval.confirm("approve?", token))
    await started.wait()
    token.cancel()
    assert not await task
    assert reaped.is_set()


@pytest.mark.parametrize("layers", [[], ["--layer", "2", "--layer", "4"]])
def test_neural_layer_options_preserve_tuple_contract(tmp_path, layers):
    from types import SimpleNamespace

    source = tmp_path / "prompts.txt"
    source.write_text("hello")
    args = [
        "neural",
        "fit-lens",
        "--model",
        "model",
        "--revision",
        "rev",
        "--lens-id",
        "lens",
        "--lens-revision",
        "rev",
        "--prompts",
        str(source),
        "--output",
        str(tmp_path / "out"),
        "--corpus-id",
        "test",
        "--corpus-license",
        "test",
        *layers,
    ]
    with patch(
        "rai.neural.fitting.fit_lens_artifact",
        return_value=SimpleNamespace(lens_id="lens", lens_revision="rev", layers=[]),
    ) as fit:
        result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert fit.call_args.args[0].layers == ((2, 4) if layers else ())


@pytest.mark.parametrize("command", ["ask", "chat"])
def test_cli_accepts_token_and_context_limits(command):
    with patch(f"rai.commands.assistant._run_assistant_{command}") as run:
        result = CliRunner().invoke(
            cli,
            [
                "assistant",
                command,
                *(["hello"] if command == "ask" else []),
                "--max-output-tokens",
                "4096",
                "--context-window",
                "8192",
            ],
        )
    assert result.exit_code == 0, result.output
    assert run.call_args.kwargs["max_output_tokens"] == 4096
    assert run.call_args.kwargs["context_window"] == 8192


@pytest.mark.parametrize("command", ["ask", "chat"])
@pytest.mark.parametrize("option,value", [
    ("--max-output-tokens", "0"),
    ("--context-window", "-1"),
    ("--thinking-budget", "-1"),
])
def test_cli_rejects_invalid_budgets_before_runtime(command, option, value):
    with patch(f"rai.commands.assistant._run_assistant_{command}") as run:
        result = CliRunner().invoke(
            cli, ["assistant", command, *(["hello"] if command == "ask" else []), option, value]
        )
    assert result.exit_code == 2
    run.assert_not_called()


@pytest.mark.parametrize("backend", ["lemonade", "deterministic"])
def test_budget_defaults_and_cli_overrides_match_runtime(tmp_path, monkeypatch, backend):
    from rai.commands.settings import _assistant_config
    from rai.assistant.runtime import resolve_assistant_config
    from rai.configuration.resolution import effective_model_settings

    monkeypatch.setenv("RAI_CONFIG_DIR", str(tmp_path))
    config = _assistant_config(backend, "test-model")
    effective, _ = effective_model_settings(config)
    runtime = resolve_assistant_config(config)
    for name, expected in {
        "max_output_tokens": 4096,
        "context_window": 8192,
        "max_context_characters": 32000,
    }.items():
        assert effective[name] == getattr(runtime, name) == expected
    config = _assistant_config(
        backend, "test-model", max_output_tokens=512, context_window=4096,
        thinking_budget=0,
    )
    effective, sources = effective_model_settings(config)
    runtime = resolve_assistant_config(config)
    assert effective["max_output_tokens"] == runtime.max_output_tokens == 512
    assert effective["context_window"] == runtime.context_window == 4096
    assert sources["max_output_tokens"] == sources["context_window"] == "CLI"
    if backend != "deterministic":
        assert runtime.thinking_budget == 0


@pytest.mark.parametrize("mode", ["ask", "chat"])
@pytest.mark.parametrize("data_class", [DataClass.LOCAL, DataClass.PUBLIC, DataClass.PRIVATE])
def test_dialog_preserves_classification_for_remote_backend(
    mode: str, data_class: DataClass,
) -> None:
    mock_container = MagicMock()
    mock_service = MagicMock()
    mock_service.backend.is_remote = True
    mock_candidate = MagicMock(
        text="ok",
        reasoning_content=None,
        admitted_memory_ids=(),
        manifest_id="m-1",
    )
    mock_service.accept_turn = AsyncMock(return_value=Success(mock_candidate))
    mock_service.start = AsyncMock(return_value=Success(None))
    mock_service.get_recent_turns = AsyncMock(return_value=Success(()))
    mock_service.store.get_latest_manifest_for_session = AsyncMock(return_value=Success(None))
    mock_container.assistant_service = mock_service
    mock_container.close = AsyncMock()
    with patch("rai.container.ApplicationContainer", return_value=mock_container):
        with patch("rai.commands.approval.configure_approvals"):
            if mode == "ask":
                _run_assistant_ask(
                    "hello", None, "antigravity", "gemini-3.8-flash", False,
                    data_class=data_class,
                )
            else:
                with patch("rai.commands.dialog.click.prompt", side_effect=["hello", "/exit"]):
                    _run_assistant_chat(
                        None, "antigravity", "gemini-3.8-flash", False,
                        data_class=data_class,
                    )
    turn = mock_service.accept_turn.call_args[0][0]
    assert turn.data_class == data_class
