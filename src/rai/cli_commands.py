"""Stage 1 command definitions, separate from transport and rendering."""

from __future__ import annotations

import asyncio
import json
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
        click.echo(f"\nContext manifest:\n{json.dumps(summary, indent=2)}", err=True)


async def _echo_memories(service: Any) -> None:  # noqa: ANN401
    """Show active memories in the current local profile scope."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.retrieve_relevant_memories(
        profile_scope=service.profile_scope,
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
                        "/context   manifest for the latest reply\n"
                        "/session   current profile and session ID\n"
                        "/exit      leave the chat"
                    )
                    continue
                if stripped == "/memories":
                    await _echo_memories(service)
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
        type=click.Choice(["auto", "llama", "ollama", "deterministic"]),
        default="auto",
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    )
    @click.option("--model", default=None, help="GGUF path or local Ollama model name.")
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
        type=click.Choice(["auto", "llama", "ollama", "deterministic"]),
        default="auto",
        show_default=True,
        help="Local inference backend. Deterministic is only for conformance tests.",
    )
    @click.option("--model", default=None, help="GGUF path or local Ollama model name.")
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
