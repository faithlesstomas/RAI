"""Assistant dialog lifecycle and interactive rendering."""

from __future__ import annotations
import asyncio
import json
import click
from rai.kernel.records import DataClass
from .settings import _assistant_config
from .memory import (
    _echo_context_manifest,
    _echo_memories,
    _echo_history,
    _echo_memory_operations,
    _echo_memory_diagnostics,
)


def _run_assistant_ask(  # noqa: PLR0913
    prompt: str,
    session_id: str | None,
    backend: str | None,
    model: str | None,
    show_context: bool,
    profile: str | None = None,
    system: str | None = None,
    thinking: bool | None = None,
    show_thinking: bool = False,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    data_class: DataClass = DataClass.LOCAL,
) -> None:
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.records import ConversationTurn  # noqa: PLC0415
    from rai.container import ApplicationContainer  # noqa: PLC0415
    from rai.kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    try:
        container = ApplicationContainer(
            config=_assistant_config(
                backend,
                model,
                profile,
                system,
                thinking=thinking,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
            )
        )
        service = container.assistant_service
        from rai.commands.approval import configure_approvals  # noqa: PLC0415

        configure_approvals(container, service)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    sid = session_id or _new_id()
    turn = ConversationTurn(
        record_id=_new_id(),
        producer=ProducerIdentity(producer_id="cli-ask", kind="user", version="1.0.0"),
        session_id=sid,
        role="user",
        data_class=data_class,
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
                if show_thinking and response.reasoning_content:
                    click.secho(
                        f"\n[Thinking]\n{response.reasoning_content}\n[/Thinking]\n",
                        fg="cyan",
                        err=True,
                    )
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
    thinking: bool | None = None,
    show_thinking: bool = False,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    data_class: DataClass = DataClass.LOCAL,
) -> None:
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.records import ConversationTurn  # noqa: PLC0415
    from rai.container import ApplicationContainer  # noqa: PLC0415
    from rai.kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    try:
        container = ApplicationContainer(
            config=_assistant_config(
                backend,
                model,
                profile,
                system,
                thinking=thinking,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
            )
        )
        service = container.assistant_service
        from rai.commands.approval import configure_approvals  # noqa: PLC0415

        configure_approvals(container, service)
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

    current_thinking = (
        thinking
        if thinking is not None
        else getattr(service.backend, "enable_thinking", False)
    )
    current_show_thinking = show_thinking

    async def _run_loop() -> None:  # noqa: PLR0912, PLR0915
        nonlocal current_thinking, current_show_thinking
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
                        "/thinking [on|off|show|hide] toggle model thinking\n"
                        "/remember TEXT explicitly save a fact\n"
                        "/forget TEXT remove matching memory; use 'all' for everything\n"
                        "/session   current profile and session ID\n"
                        "/exit      leave the chat"
                    )
                    continue
                if stripped in {"/config", "/config show"}:
                    from rai.configuration.resolution import effective_model_settings  # noqa: PLC0415
                    from rai.configuration.storage import redact  # noqa: PLC0415

                    values, sources = effective_model_settings(container.config)
                    values["enable_thinking"] = current_thinking
                    click.echo(
                        json.dumps(
                            redact({"effective_assistant": values, "sources": sources}),
                            indent=2,
                        )
                    )
                    continue
                if stripped.startswith("/thinking"):
                    parts = stripped.split()
                    if len(parts) > 1:
                        cmd = parts[1].lower()
                        if cmd in ("on", "true", "1", "enable"):
                            current_thinking = True
                            click.echo("Thinking mode: ON")
                        elif cmd in ("off", "false", "0", "disable"):
                            current_thinking = False
                            click.echo("Thinking mode: OFF")
                        elif cmd == "show":
                            current_show_thinking = True
                            click.echo("Show thinking: ON")
                        elif cmd == "hide":
                            current_show_thinking = False
                            click.echo("Show thinking: OFF")
                        else:
                            click.echo("Usage: /thinking [on|off|show|hide]")
                    else:
                        current_thinking = not current_thinking
                        click.echo(
                            f"Thinking mode: {'ON' if current_thinking else 'OFF'}"
                        )
                    if hasattr(service.backend, "enable_thinking"):
                        service.backend.enable_thinking = current_thinking
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
                    data_class=data_class,
                    text=stripped,
                    reply_to_turn_id=reply_to_turn_id,
                )
                result = await service.accept_turn(turn)
                if isinstance(result, Success):
                    response = result.unwrap()
                    if current_show_thinking and response.reasoning_content:
                        click.secho(
                            f"\n[Thinking]\n{response.reasoning_content}\n[/Thinking]",
                            fg="cyan",
                            err=True,
                        )
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
