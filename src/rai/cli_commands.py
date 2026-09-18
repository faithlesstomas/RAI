"""Stage 1 command definitions, separate from transport and rendering."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from .cli_rendering import render_capabilities, render_envelope
from .cli_transport import invoke_local, list_local_capabilities
from .kernel.records import CapabilityRequest


def register_kernel_commands(root: click.Group) -> None:
    """Attach kernel commands to the compatibility root CLI."""

    @root.group(name="capability")
    def capability() -> None:
        """Discover and invoke policy-controlled capabilities."""

    @capability.command(name="list")
    def list_capability_command() -> None:
        click.echo(render_capabilities(list_local_capabilities()))

    @capability.command(name="invoke")
    @click.argument("request_json")
    def invoke_capability_command(request_json: str) -> None:
        """Invoke a complete versioned CapabilityRequest JSON record."""
        try:
            request = CapabilityRequest.model_validate(json.loads(request_json))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise click.ClickException(f"invalid CapabilityRequest: {exc}") from exc
        envelope = asyncio.run(invoke_local(request))
        click.echo(render_envelope(envelope))


def _assistant_config(
    backend: str | None,
    model: str | None,
    profile: str | None = None,
    system: str | None = None,
) -> dict[str, object]:
    from . import config_manager  # noqa: PLC0415

    config = config_manager.load_config()
    if profile:
        profiles = config.get("agents", {})
        if not isinstance(profiles, dict) or profile not in profiles:
            raise click.ClickException(f"Unknown assistant profile: {profile}")
        config["active_agent"] = profile
    assistant = dict(config.get("assistant", {}))
    if backend and backend != "auto":
        assistant["backend"] = backend
        profiles = config.get("agents", {})
        active_profile = str(config.get("active_agent") or "default")
        profile_config = (
            profiles.get(active_profile, {}) if isinstance(profiles, dict) else {}
        )
        if (
            not model
            and isinstance(profile_config, dict)
            and profile_config.get("backend") != backend
        ):
            copied_profiles = dict(profiles)
            copied_profile = dict(profile_config)
            copied_profile.pop("model", None)
            copied_profiles[active_profile] = copied_profile
            config["agents"] = copied_profiles
    if model:
        assistant["model"] = model
    if system:
        assistant["system"] = system
    config["assistant"] = assistant
    return config


async def _echo_context_manifest(service: Any, manifest_id: str) -> None:  # noqa: ANN401
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.get_manifest(manifest_id)
    if isinstance(result, Success) and result.unwrap() is not None:
        manifest = result.unwrap()
        summary = {
            "manifest_id": manifest.record_id,
            "session_id": manifest.session_id,
            "recent_turn_ids": manifest.recent_turn_ids,
            "durable_memory_ids": manifest.durable_memory_ids,
            "exclusions": manifest.exclusions,
            "redactions": manifest.redactions,
            "backend": manifest.backend_name,
            "model": manifest.model_name,
            "model_artifact_version": manifest.model_artifact_version,
            "prompt_template_version": manifest.prompt_template_version,
            "estimated_tokens": manifest.actual_tokens,
        }
        entries = await service.audit_ledger.list_for_session(manifest.session_id)
        run = next(
            (
                entry
                for entry in reversed(entries)
                if entry.manifest_id == manifest.record_id
            ),
            None,
        )
        if run is not None:
            summary["run"] = {
                "latency_ms": round(run.latency_ms, 2),
                "tokens": run.tokens,
                "generation": run.generation_metadata,
            }
        package_result = await service.store.get_context_package(manifest.record_id)
        if isinstance(package_result, Success) and package_result.unwrap() is not None:
            summary["context_window"] = package_result.unwrap().content
        click.echo(f"\nContext manifest:\n{json.dumps(summary, indent=2)}", err=True)
        return
    click.echo("Context manifest not found.", err=True)


async def _echo_memories(service: Any) -> None:  # noqa: ANN401
    """Show active memories in the current local profile scope."""
    from returns.result import Success  # noqa: PLC0415
    from .kernel.records import DataClass  # noqa: PLC0415

    result = await service.store.retrieve_relevant_memories(
        profile_scope=service.profile_scope,
        data_classes=(DataClass.PUBLIC, DataClass.LOCAL, DataClass.PRIVATE),
        limit=20,
    )
    if not isinstance(result, Success):
        click.echo(f"Could not read memories: {result.failure().message}", err=True)
        return
    memories = result.unwrap()
    if not memories:
        click.echo("No active durable memories for this profile.")
        return
    click.echo("Active durable memories:")
    for memory, _reason in memories:
        click.echo(
            f"- [{memory.kind}] {memory.topic}: {json.dumps(memory.content, ensure_ascii=False)}"
        )


async def _echo_history(service: Any, session_id: str, limit: int = 20) -> None:  # noqa: ANN401
    """Show the persisted conversation window for a session."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.get_recent_turns(session_id, limit=limit)
    if not isinstance(result, Success):
        click.echo(f"Could not read chat history: {result.failure().message}", err=True)
        return
    turns = result.unwrap()
    if not turns:
        click.echo(f"No completed turns for session {session_id}.")
        return
    click.echo(f"Chat history for {session_id} ({len(turns)} turns):")
    for turn in turns:
        timestamp = turn.timestamp.astimezone().isoformat(timespec="seconds")
        click.echo(f"- {timestamp} {turn.role}: {turn.text}")


async def _echo_sessions(service: Any, limit: int = 20) -> None:  # noqa: ANN401
    """Show discoverable local conversation sessions."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.list_sessions(limit=limit)
    if not isinstance(result, Success):
        click.echo(f"Could not list sessions: {result.failure().message}", err=True)
        return
    sessions = result.unwrap()
    if not sessions:
        click.echo("No local assistant sessions found.")
        return
    click.echo("Assistant sessions:")
    for session in sessions:
        timestamp = session.updated_at.astimezone().isoformat(timespec="seconds")
        click.echo(
            f"- {session.session_id} | {session.turn_count} turns | {timestamp} | "
            f"{session.last_role}: {session.preview}"
        )


async def _echo_memory_operations(service: Any, limit: int = 20) -> None:  # noqa: ANN401
    """Show the append-only memory operation trace for the active profile."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.list_memory_operations(
        profile_scope=service.profile_scope, limit=limit
    )
    if not isinstance(result, Success):
        click.echo(
            f"Could not list memory operations: {result.failure().message}", err=True
        )
        return
    operations = result.unwrap()
    if not operations:
        click.echo("No memory operations for this profile.")
        return
    click.echo("Memory operations:")
    for operation in operations:
        click.echo(
            f"- {operation.timestamp.astimezone().isoformat(timespec='seconds')} "
            f"{operation.operation} {operation.status} "
            f"targets={','.join(operation.target_memory_ids) or '-'} "
            f"results={','.join(operation.result_memory_ids) or '-'}"
        )


async def _echo_memory_diagnostics(service: Any) -> None:  # noqa: ANN401
    """Show stage-specific memory integrity diagnostics."""
    from returns.result import Success  # noqa: PLC0415

    from .assistant.diagnostics import diagnose_memory  # noqa: PLC0415

    result = await diagnose_memory(service.store, profile_scope=service.profile_scope)
    if not isinstance(result, Success):
        click.echo(f"Could not diagnose memory: {result.failure().message}", err=True)
        return
    report = result.unwrap()
    click.echo(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _run_assistant_read(
    profile: str | None,
    action: Callable[[Any], Awaitable[None]],
) -> None:
    """Run a model-free assistant inspection command against durable local state."""
    from returns.result import Success  # noqa: PLC0415

    from .assistant.service import AssistantService  # noqa: PLC0415
    from .assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415

    config = _assistant_config(None, None, profile)
    scope = str(config.get("active_agent") or "default")
    service = AssistantService(
        store=SQLiteMemoryGraphStore(),
        profile_scope=scope,
    )

    async def _run() -> None:
        start_res = await service.start()
        if not isinstance(start_res, Success):
            error = start_res.failure()
            raise click.ClickException(f"[{error.code}] {error.message}")
        try:
            await action(service)
        finally:
            await service.stop()

    asyncio.run(_run())


def _run_assistant_ask(  # noqa: PLR0913
    prompt: str,
    session_id: str | None,
    backend: str | None,
    model: str | None,
    show_context: bool,
    profile: str | None = None,
    system: str | None = None,
) -> None:
    from returns.result import Success  # noqa: PLC0415

    from .assistant.records import ConversationTurn  # noqa: PLC0415
    from .container import ApplicationContainer  # noqa: PLC0415
    from .kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    try:
        container = ApplicationContainer(
            config=_assistant_config(backend, model, profile, system)
        )
        service = container.assistant_service
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    sid = session_id or _new_id()
    turn = ConversationTurn(
        record_id=_new_id(),
        producer=ProducerIdentity(producer_id="cli-ask", kind="user", version="1.0.0"),
        session_id=sid,
        role="user",
        text=prompt,
    )

    async def _run() -> None:
        try:
            start_res = await service.start()
            if not isinstance(start_res, Success):
                err = start_res.failure()
                raise click.ClickException(f"[{err.code}] {err.message}")
            res = await service.accept_turn(turn)
            if isinstance(res, Success):
                response = res.unwrap()
                click.echo(response.text)
                if show_context:
                    if response.admitted_memory_ids:
                        click.echo(
                            "\nAdmitted memories: "
                            + ", ".join(response.admitted_memory_ids),
                            err=True,
                        )
                    await _echo_context_manifest(service, response.manifest_id)
            else:
                err = res.failure()
                raise click.ClickException(f"[{err.code}] {err.message}")
        finally:
            await container.close()

    asyncio.run(_run())


def _run_assistant_chat(  # noqa: PLR0913, PLR0915
    session_id: str | None,
    backend: str | None,
    model: str | None,
    show_context: bool,
    profile: str | None = None,
    system: str | None = None,
) -> None:
    from returns.result import Success  # noqa: PLC0415

    from .assistant.records import ConversationTurn  # noqa: PLC0415
    from .container import ApplicationContainer  # noqa: PLC0415
    from .kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    try:
        container = ApplicationContainer(
            config=_assistant_config(backend, model, profile, system)
        )
        service = container.assistant_service
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    sid = session_id or _new_id()
    click.echo(f"Assistant profile: {service.profile_scope}")
    click.echo(f"Session ID: {sid}")
    click.echo(
        f"Backend: {getattr(service.backend, 'backend_name', 'unknown')} | "
        f"Model: {getattr(service.backend, 'model_name', 'unknown')}"
    )
    click.echo("Type /help for commands, /exit or /q to quit.\n")

    async def _run_loop() -> None:  # noqa: PLR0912, PLR0915
        reply_to_turn_id: str | None = None
        last_manifest_id: str | None = None
        try:
            start_res = await service.start()
            if not isinstance(start_res, Success):
                err = start_res.failure()
                raise click.ClickException(f"[{err.code}] {err.message}")
            recent_res = await service.get_recent_turns(sid, limit=1)
            if isinstance(recent_res, Success) and recent_res.unwrap():
                reply_to_turn_id = recent_res.unwrap()[-1].record_id
            latest_manifest_res = await service.store.get_latest_manifest_for_session(
                sid
            )
            if (
                isinstance(latest_manifest_res, Success)
                and latest_manifest_res.unwrap() is not None
            ):
                last_manifest_id = latest_manifest_res.unwrap().record_id
            while True:
                try:
                    user_input = click.prompt("You", prompt_suffix="> ")
                except (EOFError, KeyboardInterrupt):
                    click.echo("\nExiting session.")
                    break
                stripped = user_input.strip()
                if stripped in ("/exit", "/quit", "/q"):
                    click.echo("Exiting session.")
                    break
                if stripped == "/help":
                    click.echo(
                        "/memories  active durable memories\n"
                        "/operations memory operation audit trail\n"
                        "/diagnostics stage-specific memory integrity report\n"
                        "/history [N] persisted turns in this session\n"
                        "/context   exact context window for the latest reply\n"
                        "/remember TEXT explicitly save a fact\n"
                        "/forget TEXT remove matching memory; use 'all' for everything\n"
                        "/session   current profile and session ID\n"
                        "/exit      leave the chat"
                    )
                    continue
                if stripped == "/memories":
                    await _echo_memories(service)
                    continue
                if stripped == "/operations":
                    await _echo_memory_operations(service)
                    continue
                if stripped == "/diagnostics":
                    await _echo_memory_diagnostics(service)
                    continue
                if stripped.startswith("/history"):
                    parts = stripped.split()
                    try:
                        limit = int(parts[1]) if len(parts) > 1 else 20
                    except ValueError:
                        click.echo("Usage: /history [number-of-turns]", err=True)
                        continue
                    await _echo_history(service, sid, max(1, min(limit, 200)))
                    continue
                if stripped == "/context":
                    if last_manifest_id is None:
                        click.echo("No response manifest exists in this chat yet.")
                    else:
                        await _echo_context_manifest(service, last_manifest_id)
                    continue
                if stripped == "/session":
                    click.echo(f"Profile: {service.profile_scope} | Session ID: {sid}")
                    continue
                if not stripped:
                    continue

                if stripped.startswith("/remember "):
                    stripped = (
                        f"Remember that {stripped.removeprefix('/remember ').strip()}"
                    )
                elif stripped.startswith("/forget "):
                    target = stripped.removeprefix("/forget ").strip()
                    stripped = (
                        "Forget everything from memory"
                        if target.casefold() == "all"
                        else f"Forget about {target}"
                    )

                turn = ConversationTurn(
                    record_id=_new_id(),
                    producer=ProducerIdentity(
                        producer_id="cli-chat", kind="user", version="1.0.0"
                    ),
                    session_id=sid,
                    role="user",
                    text=stripped,
                    reply_to_turn_id=reply_to_turn_id,
                )
                result = await service.accept_turn(turn)
                if isinstance(result, Success):
                    response = result.unwrap()
                    click.echo(f"Assistant: {response.text}")
                    reply_to_turn_id = response.turn_id
                    last_manifest_id = response.manifest_id
                    if show_context:
                        if response.admitted_memory_ids:
                            click.echo(
                                "Admitted memories: "
                                + ", ".join(response.admitted_memory_ids),
                                err=True,
                            )
                        await _echo_context_manifest(service, response.manifest_id)
                else:
                    err = result.failure()
                    click.echo(f"[Error: {err.code}] {err.message}", err=True)
        finally:
            await container.close()

    asyncio.run(_run_loop())


def _run_assistant_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    corpus: Path,
    output: Path,
    energy: str,
    require_energy: bool,
    max_input_tokens: int,
    max_output_tokens: int,
    max_latency_seconds: float,
    trials: int,
) -> None:
    """Run the isolated equal-budget memory benchmark and persist its manifest."""
    from returns.result import Success  # noqa: PLC0415

    from .assistant.benchmark import run_retrieval_benchmark  # noqa: PLC0415
    from .assistant.energy import LinuxEnergyMeter  # noqa: PLC0415
    from .assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    meter = LinuxEnergyMeter() if energy == "auto" else None

    async def _run() -> None:
        result = await run_retrieval_benchmark(
            model_backend,
            corpus_path=corpus,
            output_path=output,
            energy_meter=meter,
            require_energy=require_energy,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
            trials=trials,
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        artifact = result.unwrap()
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": artifact.backend_name,
                    "model": artifact.model_name,
                    "corpus": artifact.run.corpus_version,
                    "channels": [
                        aggregate.model_dump(mode="json")
                        for aggregate in artifact.aggregates
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    asyncio.run(_run())


def _run_assistant_dialog_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    corpus: Path,
    output: Path,
    max_output_tokens: int,
) -> None:
    """Run the multi-turn conversational benchmark and persist its report."""
    from returns.result import Success  # noqa: PLC0415

    from .assistant.conversational_evaluation import (  # noqa: PLC0415
        run_conversational_benchmark,
    )
    from .assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    async def _run() -> None:
        result = await run_conversational_benchmark(
            model_backend,
            corpus_path=corpus,
            output_path=output,
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        report = result.unwrap()
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": report.backend_name,
                    "model": report.model_name,
                    "corpus": report.corpus_version,
                    "scenarios": report.scenario_count,
                    "total_turns": report.total_turns,
                    "mean_coherence": report.mean_coherence,
                    "parroting_rate": report.parroting_rate,
                    "repetition_rate": report.repetition_rate,
                    "role_confusion_rate": report.role_confusion_rate,
                    "mean_latency_ms": report.mean_latency_ms,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    asyncio.run(_run())


def _run_assistant_context_rot_benchmark(  # noqa: PLR0913
    backend: str,
    model: str | None,
    profile: str | None,
    token_steps_str: str,
    depths_str: str,
    trials: int,
    corpus: Path,
    output: Path,
    max_output_tokens: int,
    max_latency_seconds: float,
) -> None:
    """Run the context rot A/B benchmark and format summary output."""
    from returns.result import Success  # noqa: PLC0415

    from .assistant.context_rot_evaluation import (  # noqa: PLC0415
        evaluate_context_rot,
        format_context_rot_summary_table,
        load_context_rot_corpus,
        save_context_rot_report,
    )
    from .assistant.runtime import (  # noqa: PLC0415
        build_assistant_backend,
        resolve_assistant_config,
    )

    try:
        config = _assistant_config(backend, model, profile)
        assistant_config = dict(config.get("assistant", {}))
        assistant_config["max_output_tokens"] = max_output_tokens
        config["assistant"] = assistant_config
        runtime = resolve_assistant_config(config)
        model_backend = build_assistant_backend(runtime)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        corpus_data = load_context_rot_corpus(corpus)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"Failed to load context rot corpus: {exc}") from exc

    try:
        token_steps = tuple(
            int(s.strip()) for s in token_steps_str.split(",") if s.strip()
        )
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid token-steps format: {token_steps_str}"
        ) from exc

    depths = tuple(d.strip() for d in depths_str.split(",") if d.strip())
    for d in depths:
        if d not in ("start", "middle", "end"):
            raise click.ClickException(
                f"Invalid depth: {d}. Expected start, middle, or end."
            )

    async def _run() -> None:
        result = await evaluate_context_rot(
            model_backend,
            corpus=corpus_data,
            token_steps=token_steps,
            depths=depths,  # type: ignore[arg-type]
            trials=trials,
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
        )
        if not isinstance(result, Success):
            failure = result.failure()
            raise click.ClickException(f"[{failure.code}] {failure.message}")
        report = result.unwrap()
        save_context_rot_report(report, output)
        click.echo(format_context_rot_summary_table(report))
        click.echo(
            json.dumps(
                {
                    "output": str(output),
                    "backend": report.backend_name,
                    "model": report.model_name,
                    "token_steps": list(report.token_steps),
                    "total_cases": report.total_cases,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    asyncio.run(_run())


def register_assistant_commands(root: click.Group) -> None:
    """Attach assistant commands to the root CLI."""

    @root.group(name="assistant")
    def assistant() -> None:
        """Interact with the Rich Assistant using durable graph memory."""

    @assistant.command(name="ask")
    @click.argument("prompt")
    @click.option(
        "--session-id", default=None, help="Session ID for the conversation turn."
    )
    @click.option(
        "--backend",
        type=click.Choice(["auto", "llama", "ollama", "lemonade", "deterministic"]),
        default="auto",
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    )
    @click.option("--model", default=None, help="GGUF path or local Ollama/Lemonade model name.")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    @click.option(
        "--system", default=None, help="Override the profile system instruction."
    )
    @click.option(
        "--show-context",
        is_flag=True,
        help="Print the context manifest after the reply.",
    )
    def ask_command(  # noqa: PLR0913
        prompt: str,
        session_id: str | None,
        backend: str,
        model: str | None,
        profile: str | None,
        system: str | None,
        show_context: bool,
    ) -> None:
        """Send a single prompt to the assistant."""
        _run_assistant_ask(
            prompt, session_id, backend, model, show_context, profile, system
        )

    @assistant.command(name="chat")
    @click.option("--session-id", default=None, help="Session ID for the conversation.")
    @click.option(
        "--backend",
        type=click.Choice(["auto", "llama", "ollama", "lemonade", "deterministic"]),
        default="auto",
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    )
    @click.option("--model", default=None, help="GGUF path or local Ollama/Lemonade model name.")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    @click.option(
        "--system", default=None, help="Override the profile system instruction."
    )
    @click.option("--show-context", is_flag=True, help="Print each context manifest.")
    def chat_command(  # noqa: PLR0913
        session_id: str | None,
        backend: str,
        model: str | None,
        profile: str | None,
        system: str | None,
        show_context: bool,
    ) -> None:
        """Start an interactive chat session with the assistant."""
        _run_assistant_chat(session_id, backend, model, show_context, profile, system)

    @assistant.command(name="memories")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    def memories_command(profile: str | None) -> None:
        """List active durable memories without loading a model."""
        _run_assistant_read(profile, _echo_memories)

    @assistant.command(name="operations")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    @click.option("--limit", default=20, type=click.IntRange(1, 200), show_default=True)
    def operations_command(profile: str | None, limit: int) -> None:
        """Inspect the append-only memory operation audit trail."""

        async def show(service: Any) -> None:  # noqa: ANN401
            await _echo_memory_operations(service, limit=limit)

        _run_assistant_read(profile, show)

    @assistant.command(name="diagnostics")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    def diagnostics_command(profile: str | None) -> None:
        """Check memory extraction, admission, storage, update and retrieval state."""
        _run_assistant_read(profile, _echo_memory_diagnostics)

    @assistant.command(name="benchmark-memory")
    @click.option(
        "--backend",
        type=click.Choice(["llama", "ollama", "lemonade", "deterministic"]),
        default="deterministic",
        show_default=True,
    )
    @click.option("--model", default=None, help="GGUF path or local Ollama model name.")
    @click.option("--profile", default=None, help="Assistant profile configuration.")
    @click.option(
        "--corpus",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("tests/fixtures/assistant/v1/retrieval-floor.corpus.json"),
        show_default=True,
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path, dir_okay=False),
        default=Path("assistant-retrieval-benchmark.json"),
        show_default=True,
    )
    @click.option(
        "--energy",
        type=click.Choice(["auto", "none"]),
        default="auto",
        show_default=True,
        help="Measure readable Linux RAPL/hwmon sensors or disable measurement.",
    )
    @click.option(
        "--require-energy",
        is_flag=True,
        help="Fail unless every evaluated answer has an energy measurement.",
    )
    @click.option(
        "--max-input-tokens",
        type=click.IntRange(min=128),
        default=4096,
        show_default=True,
    )
    @click.option(
        "--max-output-tokens",
        type=click.IntRange(min=1),
        default=256,
        show_default=True,
    )
    @click.option(
        "--max-latency-seconds",
        type=click.FloatRange(min=0.1),
        default=60.0,
        show_default=True,
    )
    @click.option(
        "--trials",
        type=click.IntRange(min=1),
        default=3,
        show_default=True,
        help="Repeat every answer/routing case to expose model variance.",
    )
    def benchmark_memory_command(  # noqa: PLR0913
        backend: str,
        model: str | None,
        profile: str | None,
        corpus: Path,
        output: Path,
        energy: str,
        require_energy: bool,
        max_input_tokens: int,
        max_output_tokens: int,
        max_latency_seconds: float,
        trials: int,
    ) -> None:
        """Compare lexical, summary, dense, RRF and graph memory channels."""
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
    @click.option(
        "--backend",
        type=click.Choice(["llama", "ollama", "lemonade", "deterministic"]),
        default="deterministic",
        show_default=True,
    )
    @click.option(
        "--model", default=None, help="GGUF path or local Ollama/Lemonade model name."
    )
    @click.option("--profile", default=None, help="Assistant profile configuration.")
    @click.option(
        "--corpus",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=Path("tests/fixtures/assistant/v1/conversational-dialog.corpus.json"),
        show_default=True,
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path, dir_okay=False),
        default=Path("assistant-dialog-benchmark.json"),
        show_default=True,
    )
    @click.option(
        "--max-output-tokens",
        type=click.IntRange(min=1),
        default=256,
        show_default=True,
    )
    def benchmark_dialog_command(  # noqa: PLR0913
        backend: str,
        model: str | None,
        profile: str | None,
        corpus: Path,
        output: Path,
        max_output_tokens: int,
    ) -> None:
        """Run the multi-turn conversational benchmark and verify dialog coherence."""
        _run_assistant_dialog_benchmark(
            backend,
            model,
            profile,
            corpus,
            output,
            max_output_tokens,
        )

    @assistant.command(name="benchmark-context-rot")
    @click.option(
        "--backend",
        type=click.Choice(["lemonade", "ollama", "llama", "deterministic", "local"]),
        default="lemonade",
        show_default=True,
    )
    @click.option("--model", default=None, help="Model name override.")
    @click.option("--profile", default=None, help="Assistant profile to evaluate.")
    @click.option(
        "--token-steps",
        default="1024,2048,4096,7500",
        show_default=True,
        help="Comma-separated target token saturation steps.",
    )
    @click.option(
        "--depths",
        default="start,middle,end",
        show_default=True,
        help="Comma-separated needle depths (start, middle, end).",
    )
    @click.option(
        "--trials",
        type=click.IntRange(min=1),
        default=1,
        show_default=True,
        help="Trials per configuration.",
    )
    @click.option(
        "--corpus",
        type=click.Path(path_type=Path, dir_okay=False),
        default=Path("tests/fixtures/assistant/v1/context-rot.corpus.json"),
        show_default=True,
    )
    @click.option(
        "--output",
        type=click.Path(path_type=Path, dir_okay=False),
        default=Path("assistant-context-rot-benchmark.json"),
        show_default=True,
    )
    @click.option(
        "--max-output-tokens",
        type=click.IntRange(min=1),
        default=256,
        show_default=True,
        help="Maximum generation output tokens per query (use 512+ for deep reasoning models).",
    )
    @click.option(
        "--max-latency-seconds",
        type=click.FloatRange(min=1.0),
        default=90.0,
        show_default=True,
    )
    def benchmark_context_rot_command(  # noqa: PLR0913
        backend: str,
        model: str | None,
        profile: str | None,
        token_steps: str,
        depths: str,
        trials: int,
        corpus: Path,
        output: Path,
        max_output_tokens: int,
        max_latency_seconds: float,
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
            max_output_tokens=max_output_tokens,
            max_latency_seconds=max_latency_seconds,
        )

    @assistant.command(name="sessions")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    @click.option("--limit", default=20, type=click.IntRange(1, 200), show_default=True)
    def sessions_command(profile: str | None, limit: int) -> None:
        """List local conversation sessions without loading a model."""

        async def show(service: Any) -> None:  # noqa: ANN401
            await _echo_sessions(service, limit=limit)

        _run_assistant_read(profile, show)

    @assistant.command(name="history")
    @click.option("--session-id", required=True, help="Conversation session ID.")
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    @click.option("--limit", default=20, type=click.IntRange(1, 200), show_default=True)
    def history_command(session_id: str, profile: str | None, limit: int) -> None:
        """Show persisted user and assistant turns for one session."""

        async def show(service: Any) -> None:  # noqa: ANN401
            await _echo_history(service, session_id, limit=limit)

        _run_assistant_read(profile, show)

    @assistant.command(name="context")
    @click.option("--manifest-id", default=None, help="Exact context manifest ID.")
    @click.option(
        "--session-id", default=None, help="Use the latest context in a session."
    )
    @click.option("--profile", default=None, help="Assistant profile and memory scope.")
    def context_command(
        manifest_id: str | None, session_id: str | None, profile: str | None
    ) -> None:
        """Show the exact persisted context window used for a response."""
        if bool(manifest_id) == bool(session_id):
            raise click.UsageError(
                "provide exactly one of --manifest-id or --session-id"
            )

        async def show(service: Any) -> None:  # noqa: ANN401
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
