"""Assistant commands."""

from __future__ import annotations
from pathlib import Path
from rai.assistant.service import AssistantService
import click
import typer
from rai.commands import completion
from rai.kernel.records import DataClass
from .memory import (
    _echo_context_manifest,
    _echo_memories,
    _echo_history,
    _echo_sessions,
    _echo_memory_operations,
    _echo_memory_diagnostics,
    _run_assistant_read,
)
from .dialog import _run_assistant_ask, _run_assistant_chat
from .benchmarks import (
    _run_assistant_benchmark,
    _run_assistant_dialog_benchmark,
    _run_assistant_context_rot_benchmark,
)

assistant = typer.Typer(help="Assistant operations.", no_args_is_help=True)


@assistant.command(name="ask")
def ask_command(  # noqa: PLR0913
    prompt: str = typer.Argument(...),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Session ID for the conversation turn."
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        autocompletion=completion.backends,
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        autocompletion=completion.models,
        help="GGUF path or local Ollama/Lemonade model name.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
    system: str | None = typer.Option(
        None, "--system", help="Override the profile system instruction."
    ),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the context manifest after the reply."
    ),
    thinking: bool | None = typer.Option(
        None,
        "--thinking/--no-thinking",
        help="Enable or disable model chain-of-thought thinking.",
    ),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Print model reasoning content in the output."
    ),
    thinking_budget: int | None = typer.Option(
        None,
        "--thinking-budget",
        min=0,
        help="Token budget for model thinking (llama.cpp engine).",
    ),
    max_output_tokens: int | None = typer.Option(
        None,
        "--max-output-tokens",
        min=1,
        help="Maximum generated token limit for model responses.",
    ),
    context_window: int | None = typer.Option(
        None,
        "--context-window",
        min=1,
        help="Context window size in tokens.",
    ),
    data_class: DataClass = typer.Option(
        DataClass.LOCAL,
        "--data-class",
        help="Classification of new turns; does not reclassify stored context.",
    ),
) -> None:
    """Send a single prompt to the assistant."""
    _run_assistant_ask(
        prompt,
        session_id,
        backend,
        model,
        show_context,
        profile,
        system,
        thinking=thinking,
        show_thinking=show_thinking,
        thinking_budget=thinking_budget,
        max_output_tokens=max_output_tokens,
        context_window=context_window,
        data_class=data_class,
    )


@assistant.command(name="chat")
def chat_command(  # noqa: PLR0913
    session_id: str | None = typer.Option(
        None, "--session-id", help="Session ID for the conversation."
    ),
    backend: str = typer.Option(
        "auto",
        "--backend",
        autocompletion=completion.backends,
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        autocompletion=completion.models,
        help="GGUF path or local Ollama/Lemonade model name.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
    system: str | None = typer.Option(
        None, "--system", help="Override the profile system instruction."
    ),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print each context manifest."
    ),
    thinking: bool | None = typer.Option(
        None,
        "--thinking/--no-thinking",
        help="Enable or disable model chain-of-thought thinking.",
    ),
    show_thinking: bool = typer.Option(
        False, "--show-thinking", help="Print model reasoning content in the output."
    ),
    thinking_budget: int | None = typer.Option(
        None,
        "--thinking-budget",
        min=0,
        help="Token budget for model thinking (llama.cpp engine).",
    ),
    max_output_tokens: int | None = typer.Option(
        None,
        "--max-output-tokens",
        min=1,
        help="Maximum generated token limit for model responses.",
    ),
    context_window: int | None = typer.Option(
        None,
        "--context-window",
        min=1,
        help="Context window size in tokens.",
    ),
    data_class: DataClass = typer.Option(
        DataClass.LOCAL,
        "--data-class",
        help="Classification of new turns; does not reclassify stored context.",
    ),
) -> None:
    """Start an interactive chat session with the assistant."""
    _run_assistant_chat(
        session_id,
        backend,
        model,
        show_context,
        profile,
        system,
        thinking=thinking,
        show_thinking=show_thinking,
        thinking_budget=thinking_budget,
        max_output_tokens=max_output_tokens,
        context_window=context_window,
        data_class=data_class,
    )


@assistant.command(name="memories")
def memories_command(
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
) -> None:
    """List active durable memories without loading a model."""
    _run_assistant_read(profile, _echo_memories)


@assistant.command(name="operations")
def operations_command(
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
    limit: int = typer.Option(20, "--limit", show_default=True, min=1, max=200),
) -> None:
    """Inspect the append-only memory operation audit trail."""

    async def show(service: AssistantService) -> None:
        await _echo_memory_operations(service, limit=limit)

    _run_assistant_read(profile, show)


@assistant.command(name="diagnostics")
def diagnostics_command(
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
) -> None:
    """Check memory extraction, admission, storage, update and retrieval state."""
    _run_assistant_read(profile, _echo_memory_diagnostics)


@assistant.command(name="benchmark-memory")
def benchmark_memory_command(  # noqa: PLR0913
    backend: str = typer.Option(
        "deterministic",
        "--backend",
        autocompletion=completion.backends,
        show_default=True,
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        autocompletion=completion.models,
        help="GGUF path or local Ollama model name.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile configuration.",
    ),
    corpus: Path = typer.Option(
        Path("tests/fixtures/assistant/v1/retrieval-floor.corpus.json"),
        "--corpus",
        show_default=True,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("assistant-retrieval-benchmark.json"),
        "--output",
        show_default=True,
        dir_okay=False,
    ),
    energy: str = typer.Option(
        "auto",
        "--energy",
        show_default=True,
        help="Measure readable Linux RAPL/hwmon sensors or disable measurement.",
    ),
    require_energy: bool = typer.Option(
        False,
        "--require-energy",
        help="Fail unless every evaluated answer has an energy measurement.",
    ),
    max_input_tokens: int = typer.Option(
        4096, "--max-input-tokens", show_default=True, min=128
    ),
    max_output_tokens: int = typer.Option(
        256, "--max-output-tokens", show_default=True, min=1
    ),
    max_latency_seconds: float = typer.Option(
        60.0, "--max-latency-seconds", show_default=True, min=0.1
    ),
    trials: int = typer.Option(
        3,
        "--trials",
        show_default=True,
        help="Repeat every answer/routing case to expose model variance.",
        min=1,
    ),
) -> None:
    """Compare lexical, summary, dense, RRF and graph memory channels."""
    if energy not in {"auto", "none"}:
        raise typer.BadParameter("energy must be auto or none")
    _run_assistant_benchmark(
        backend,
        model,
        profile,
        corpus,
        output,
        energy,
        require_energy,
        max_input_tokens,
        max_output_tokens,
        max_latency_seconds,
        trials,
    )


@assistant.command(name="benchmark-dialog")
def benchmark_dialog_command(  # noqa: PLR0913
    backend: str = typer.Option(
        "deterministic",
        "--backend",
        autocompletion=completion.backends,
        show_default=True,
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        autocompletion=completion.models,
        help="GGUF path or local Ollama/Lemonade model name.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile configuration.",
    ),
    corpus: Path = typer.Option(
        Path("tests/fixtures/assistant/v2/conversational-dialog.corpus.json"),
        "--corpus",
        show_default=True,
        exists=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("assistant-dialog-benchmark.json"),
        "--output",
        show_default=True,
        dir_okay=False,
    ),
    max_output_tokens: int = typer.Option(
        256, "--max-output-tokens", show_default=True, min=1
    ),
) -> None:
    """Run the multi-turn conversational benchmark and verify dialog coherence."""
    _run_assistant_dialog_benchmark(
        backend, model, profile, corpus, output, max_output_tokens
    )


@assistant.command(name="benchmark-context-rot")
def benchmark_context_rot_command(  # noqa: PLR0913
    backend: str = typer.Option(
        "lemonade", "--backend", autocompletion=completion.backends, show_default=True
    ),
    model: str | None = typer.Option(
        None, "--model", autocompletion=completion.models, help="Model name override."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile to evaluate.",
    ),
    token_steps: str = typer.Option(
        "512,2048,8192,32768",
        "--token-steps",
        show_default=True,
        help="Comma-separated target token saturation steps.",
    ),
    depths: str = typer.Option(
        "start,middle,end",
        "--depths",
        show_default=True,
        help="Comma-separated needle depths (start, middle, end).",
    ),
    trials: int = typer.Option(
        3, "--trials", show_default=True, help="Trials per configuration.", min=1
    ),
    corpus: Path = typer.Option(
        Path("tests/fixtures/assistant/v2/context-rot.corpus.json"),
        "--corpus",
        show_default=True,
        dir_okay=False,
    ),
    output: Path = typer.Option(
        Path("assistant-context-rot-benchmark.json"),
        "--output",
        show_default=True,
        dir_okay=False,
    ),
    server_context_window: int | None = typer.Option(
        None,
        "--server-context-window",
        help="Context window configured in the inference server. Required for auditable Lemonade/Ollama runs because RAI cannot discover it reliably.",
        min=1,
    ),
    max_output_tokens: int = typer.Option(
        256,
        "--max-output-tokens",
        show_default=True,
        help="Maximum generation output tokens per query (use 512+ for deep reasoning models).",
        min=1,
    ),
    max_latency_seconds: float = typer.Option(
        90.0, "--max-latency-seconds", show_default=True, min=1.0
    ),
) -> None:
    """Run the long-context needle-in-a-haystack & context rot benchmark."""
    _run_assistant_context_rot_benchmark(
        backend=backend,
        model=model,
        profile=profile,
        token_steps_str=token_steps,
        depths_str=depths,
        trials=trials,
        corpus=corpus,
        output=output,
        server_context_window=server_context_window,
        max_output_tokens=max_output_tokens,
        max_latency_seconds=max_latency_seconds,
    )


@assistant.command(name="sessions")
def sessions_command(
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
    limit: int = typer.Option(20, "--limit", show_default=True, min=1, max=200),
) -> None:
    """List local conversation sessions without loading a model."""

    async def show(service: AssistantService) -> None:
        await _echo_sessions(service, limit=limit)

    _run_assistant_read(profile, show)


@assistant.command(name="history")
def history_command(
    session_id: str = typer.Option(
        ..., "--session-id", help="Conversation session ID."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
    limit: int = typer.Option(20, "--limit", show_default=True, min=1, max=200),
) -> None:
    """Show persisted user and assistant turns for one session."""

    async def show(service: AssistantService) -> None:
        await _echo_history(service, session_id, limit=limit)

    _run_assistant_read(profile, show)


@assistant.command(name="context")
def context_command(
    manifest_id: str | None = typer.Option(
        None, "--manifest-id", help="Exact context manifest ID."
    ),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Use the latest context in a session."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        autocompletion=completion.profiles,
        help="Assistant profile and memory scope.",
    ),
) -> None:
    """Show the exact persisted context window used for a response."""
    if bool(manifest_id) == bool(session_id):
        raise click.UsageError("provide exactly one of --manifest-id or --session-id")

    async def show(service: AssistantService) -> None:
        selected = manifest_id
        if selected is None and session_id is not None:
            result = await service.store.get_latest_manifest_for_session(session_id)
            from returns.result import Success  # noqa: PLC0415

            if not isinstance(result, Success) or result.unwrap() is None:
                click.echo("No context manifest found for this session.", err=True)
                return
            selected = result.unwrap().record_id
        if selected is not None:
            await _echo_context_manifest(service, selected)

    _run_assistant_read(profile, show)
