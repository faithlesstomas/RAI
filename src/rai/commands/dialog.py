"""Assistant dialog lifecycle and interactive rendering."""

from __future__ import annotations
import asyncio
import contextlib
import inspect
import json
import signal
import sys
import click
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass
from .settings import _assistant_config
from .memory import (
    _echo_context_manifest,
    _echo_memories,
    _echo_history,
    _echo_memory_operations,
    _echo_memory_diagnostics,
)


def _run_assistant_ask(  # noqa: PLR0913, PLR0915
    prompt: str,
    session_id: str | None,
    backend: str | None,
    model: str | None,
    show_context: bool,
    profile: str | None = None,
    system: str | None = None,
    thinking: bool | None = None,
    show_thinking: bool | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    data_class: DataClass = DataClass.LOCAL,
    timeout: float | None = None,
    controller: str | None = None,
    max_iterations: str | int | None = None,
    max_full_observations: int | None = None,
    max_context_characters: int | None = None,
) -> None:
    from returns.result import Failure, Success  # noqa: PLC0415

    from rai.assistant.records import ConversationTurn  # noqa: PLC0415
    from rai.container import ApplicationContainer  # noqa: PLC0415
    from rai.kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    resolved_thinking = thinking
    if show_thinking is True and resolved_thinking is None:
        resolved_thinking = True

    try:
        container = ApplicationContainer(
            config=_assistant_config(
                backend,
                model,
                profile,
                system,
                thinking=resolved_thinking,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
                timeout=timeout,
                controller=controller,
                max_iterations=max_iterations,
                max_full_observations=max_full_observations,
                max_context_characters=max_context_characters,
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

    async def _run() -> None:  # noqa: PLR0912, PLR0915
        try:
            start_res = await service.start()
            if not isinstance(start_res, Success):
                err = start_res.failure()
                raise click.ClickException(f"[{err.code}] {err.message}")
            turn_token = CancellationToken()
            accept_task = asyncio.create_task(
                service.accept_turn(turn, cancellation=turn_token)
            )
            loop = asyncio.get_running_loop()
            sigint_installed = False

            def _handle_sigint() -> None:
                turn_token.cancel()
                if not accept_task.done():
                    accept_task.cancel()

            try:
                loop.add_signal_handler(signal.SIGINT, _handle_sigint)
                sigint_installed = True
            except (NotImplementedError, RuntimeError, ValueError):
                sigint_installed = False

            cancelled = False
            try:
                res = await accept_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                cancelled = True
                turn_token.cancel()
                res = None
            finally:
                if sigint_installed:
                    with contextlib.suppress(Exception):
                        loop.remove_signal_handler(signal.SIGINT)

            if (
                cancelled
                or turn_token.cancelled
                or (
                    isinstance(res, Failure)
                    and res.failure().code == "CANCELLED"
                )
            ):
                if hasattr(service, "rollback_turn"):
                    rollback_fn = getattr(service, "rollback_turn")
                    rollback_res = rollback_fn(turn.record_id)
                    if asyncio.iscoroutine(rollback_res) or inspect.isawaitable(rollback_res):
                        await rollback_res
                elif hasattr(service, "delete_turn"):
                    del_fn = getattr(service, "delete_turn")
                    del_res = del_fn(turn.record_id)
                    if asyncio.iscoroutine(del_res) or inspect.isawaitable(del_res):
                        await del_res
                click.echo("\n[Generation interrupted. Turn rolled back.]", err=True)
                return

            if isinstance(res, Success):
                response = res.unwrap()
                resolved_show_thinking = (
                    show_thinking
                    if show_thinking is not None
                    else (
                        resolved_thinking
                        if resolved_thinking is not None
                        else getattr(service.backend, "enable_thinking", False)
                    )
                )
                if resolved_show_thinking and response.reasoning_content:
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
            elif res is not None:
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
    show_thinking: bool | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    data_class: DataClass = DataClass.LOCAL,
    timeout: float | None = None,
    controller: str | None = None,
    max_iterations: str | int | None = None,
    max_full_observations: int | None = None,
    max_context_characters: int | None = None,
) -> None:
    from returns.result import Failure, Success  # noqa: PLC0415

    from rai.assistant.records import ConversationTurn  # noqa: PLC0415
    from rai.container import ApplicationContainer  # noqa: PLC0415
    from rai.kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    resolved_thinking = thinking
    if show_thinking is True and resolved_thinking is None:
        resolved_thinking = True

    try:
        container = ApplicationContainer(
            config=_assistant_config(
                backend,
                model,
                profile,
                system,
                thinking=resolved_thinking,
                thinking_budget=thinking_budget,
                max_output_tokens=max_output_tokens,
                context_window=context_window,
                timeout=timeout,
                controller=controller,
                max_iterations=max_iterations,
                max_full_observations=max_full_observations,
                max_context_characters=max_context_characters,
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
        f"Model: {getattr(service.backend, 'model_name', 'unknown')} | "
        f"Controller: {type(service.controller).__name__}"
    )
    click.echo("Type /help for commands, /exit or /q to quit.\n")

    current_thinking = (
        resolved_thinking
        if resolved_thinking is not None
        else getattr(service.backend, "enable_thinking", False)
    )
    current_show_thinking = (
        show_thinking
        if show_thinking is not None
        else current_thinking
    )
    if hasattr(service.backend, "enable_thinking"):
        service.backend.enable_thinking = current_thinking

    prompt_session = None
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import PromptSession  # noqa: PLC0415
            from prompt_toolkit.completion import WordCompleter  # noqa: PLC0415
            from prompt_toolkit.history import FileHistory  # noqa: PLC0415
            from rai.paths import data_dir  # noqa: PLC0415

            slash_commands = [
                "/help",
                "/exit",
                "/quit",
                "/q",
                "/memories",
                "/operations",
                "/diagnostics",
                "/history",
                "/context",
                "/capabilities",
                "/status",
                "/thinking",
                "/thinking on",
                "/thinking off",
                "/thinking show",
                "/thinking hide",
                "/remember",
                "/forget",
                "/session",
                "/repl",
                "/repl reset",
                "/export",
                "/config",
                "/config show",
            ]
            history_dir = data_dir() / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            history_file = history_dir / "chat_input_history"
            prompt_session = PromptSession(
                history=FileHistory(str(history_file)),
                completer=WordCompleter(slash_commands, sentence=True),
            )
        except Exception:  # noqa: BLE001
            prompt_session = None

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
                    if prompt_session is not None:
                        user_input = await prompt_session.prompt_async("You> ")
                    else:
                        user_input = click.prompt("You", prompt_suffix="> ")
                except EOFError:
                    click.echo("\nExiting session.")
                    break
                except KeyboardInterrupt:
                    try:
                        if prompt_session is not None:
                            confirm_exit = await prompt_session.prompt_async("\nExit chat? [y/N] ")
                        else:
                            confirm_exit = click.prompt("\nExit chat? [y/N]", default="n", show_default=False)
                    except (EOFError, KeyboardInterrupt):
                        click.echo("\nExiting session.")
                        break
                    if confirm_exit.strip().lower() in ("y", "yes", "tak", "t"):
                        click.echo("Exiting session.")
                        break
                    continue
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
                        "/capabilities registered actions for this conversation\n"
                        "/status    local history configuration and conversation scope\n"
                        "/thinking [on|off|show|hide] toggle model thinking\n"
                        "/remember TEXT explicitly save a fact\n"
                        "/forget TEXT remove matching memory; use 'all' for everything\n"
                        "/session   current profile and session ID\n"
                        "/repl [expr] inspect active variables & values, or evaluate Python expression\n"
                        "/repl reset clear REPL session state\n"
                        "/export [file.ipynb] export conversation as Jupyter Notebook\n"
                        "/exit      leave the chat"
                    )
                    continue
                if stripped == "/capabilities":
                    click.echo(service.context_builder.system_instruction)
                    continue
                if stripped == "/status":
                    history = container.config.get("rich_history", {})
                    from rai.configuration.storage import selected_path  # noqa: PLC0415
                    from rai.paths import data_dir  # noqa: PLC0415
                    from rai.history.diagnostics import history_storage_status  # noqa: PLC0415

                    history_path = container.rich_history_path or data_dir() / "history" / "activity-v1.sqlite3"
                    storage_status = await asyncio.to_thread(history_storage_status, history_path)
                    click.echo(json.dumps({
                        "storage": storage_status.unwrap() if isinstance(storage_status, Success) else {"error": storage_status.failure()},
                        "mode": "local_cli_reads_shared_history; collectors_run_in_daemon",
                        "config_path": str(selected_path()),
                        "history_path": str(history_path),
                        "history_enabled": bool(history.get("enabled", False)),
                        "configured_collectors": sorted(history.get("collectors", {})),
                        "data_class": data_class.value,
                        "live_collector_status": "GET /api/v1/activity/collectors on the daemon",
                    }, indent=2))
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
                            current_show_thinking = True
                            click.echo("Thinking mode: ON")
                        elif cmd in ("off", "false", "0", "disable"):
                            current_thinking = False
                            current_show_thinking = False
                            click.echo("Thinking mode: OFF")
                        elif cmd == "show":
                            current_thinking = True
                            current_show_thinking = True
                            click.echo("Show thinking: ON")
                        elif cmd == "hide":
                            current_show_thinking = False
                            click.echo("Show thinking: OFF")
                        else:
                            click.echo("Usage: /thinking [on|off|show|hide]")
                    else:
                        current_thinking = not current_thinking
                        current_show_thinking = current_thinking
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
                if stripped == "/repl":
                    saved_res = await service.store.get_repl_session(
                        sid, profile_scope=service.profile_scope
                    )
                    if isinstance(saved_res, Failure):
                        click.secho(
                            f"Failed to load REPL session state: {saved_res.failure().message}",
                            fg="red",
                        )
                        continue
                    if isinstance(saved_res, Success) and saved_res.unwrap():
                        blob, vars_dict = saved_res.unwrap()
                        runner = container.monty_runner
                        detailed: dict[str, dict[str, str]] = {}
                        if blob:
                            inspected_vals = await asyncio.to_thread(
                                runner.inspect_values, blob
                            )
                            if isinstance(inspected_vals, Success):
                                detailed = inspected_vals.unwrap()
                        if detailed:
                            click.echo("Active REPL variables:")
                            for k, info in sorted(detailed.items()):
                                v_type = info.get("type", "unknown")
                                v_val = info.get("value", "")
                                if v_val:
                                    click.secho(f"  {k} ({v_type}) = {v_val}", fg="cyan")
                                else:
                                    click.secho(f"  {k}: {v_type}", fg="cyan")
                        elif vars_dict:
                            click.echo("Active REPL variables:")
                            for k, v in sorted(vars_dict.items()):
                                click.secho(f"  {k}: {v}", fg="cyan")
                        else:
                            click.echo("REPL session has no active variables.")
                    else:
                        click.echo("REPL session has no active variables.")
                    continue
                if stripped == "/repl reset":
                    from rai.tools.python import get_session_repl_lock  # noqa: PLC0415

                    lock = await get_session_repl_lock(service.profile_scope, sid)
                    async with lock:
                        clear_res = await service.store.clear_repl_session(
                            sid, profile_scope=service.profile_scope
                        )
                        if isinstance(clear_res, Failure):
                            click.secho(
                                f"Failed to reset REPL session: {clear_res.failure().message}",
                                fg="red",
                            )
                        else:
                            click.secho("REPL session reset.", fg="green")
                    continue
                if stripped.startswith("/repl "):
                    from rai.tools.python import get_session_repl_lock  # noqa: PLC0415

                    expr = stripped[len("/repl ") :].strip()
                    if expr:
                        lock = await get_session_repl_lock(service.profile_scope, sid)
                        async with lock:
                            saved_res = await service.store.get_repl_session(
                                sid, profile_scope=service.profile_scope
                            )
                            if isinstance(saved_res, Failure):
                                click.secho(
                                    f"Failed to load REPL session state: {saved_res.failure().message}",
                                    fg="red",
                                )
                                continue
                            blob = (
                                saved_res.unwrap()[0]
                                if isinstance(saved_res, Success) and saved_res.unwrap()
                                else None
                            )
                            runner = container.monty_runner
                            eval_res = await asyncio.to_thread(
                                runner.run, expr, session_blob=blob
                            )
                            if isinstance(eval_res, Success):
                                m_res = eval_res.unwrap()
                                if m_res.session_blob:
                                    save_res = await service.store.save_repl_session(
                                        sid,
                                        m_res.session_blob,
                                        m_res.variables,
                                        profile_scope=service.profile_scope,
                                    )
                                    if isinstance(save_res, Failure):
                                        click.secho(
                                            f"Failed to persist REPL session state: {save_res.failure().message}",
                                            fg="red",
                                        )
                                        continue
                                if m_res.output:
                                    click.echo(m_res.output)
                                elif m_res.variables:
                                    var_summary = ", ".join(
                                        f"{k}: {v}"
                                        for k, v in sorted(m_res.variables.items())
                                    )
                                    click.secho(
                                        f"[REPL state: {var_summary}]", fg="cyan"
                                    )
                            else:
                                click.secho(
                                    f"REPL Error: {eval_res.failure()}", fg="red"
                                )
                    continue
                if stripped.startswith("/export"):
                    parts = stripped.split(maxsplit=1)
                    target_file = parts[1] if len(parts) > 1 else f"session_{sid}.ipynb"
                    if not target_file.endswith(".ipynb"):
                        target_file += ".ipynb"
                    from pathlib import Path  # noqa: PLC0415
                    from rai.assistant.notebook import export_turns_to_notebook  # noqa: PLC0415

                    turns_res = await service.get_recent_turns(sid, limit=200)
                    if isinstance(turns_res, Success):
                        nb = export_turns_to_notebook(sid, list(turns_res.unwrap()), service.profile_scope)
                        Path(target_file).write_text(json.dumps(nb, indent=2), encoding="utf-8")
                        click.secho(f"Session exported to {target_file}", fg="green")
                    else:
                        click.echo("Failed to fetch session history for export.", err=True)
                    continue
                if not stripped:
                    continue

                if stripped == "/remember":
                    click.echo("Usage: /remember TEXT", err=True)
                    continue
                if stripped == "/forget":
                    click.echo("Usage: /forget TEXT", err=True)
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
                elif stripped.startswith("/"):
                    click.echo(
                        f"Unknown command: '{stripped}'. Type /help for available commands.",
                        err=True,
                    )
                    continue

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
                turn_token = CancellationToken()
                accept_task = asyncio.create_task(
                    service.accept_turn(turn, cancellation=turn_token)
                )

                loop = asyncio.get_running_loop()
                sigint_installed = False

                def _handle_sigint() -> None:
                    turn_token.cancel()
                    if not accept_task.done():
                        accept_task.cancel()

                try:
                    loop.add_signal_handler(signal.SIGINT, _handle_sigint)
                    sigint_installed = True
                except (NotImplementedError, RuntimeError, ValueError):
                    sigint_installed = False

                cancelled = False
                try:
                    result = await accept_task
                except (asyncio.CancelledError, KeyboardInterrupt):
                    cancelled = True
                    turn_token.cancel()
                    result = None
                finally:
                    if sigint_installed:
                        with contextlib.suppress(Exception):
                            loop.remove_signal_handler(signal.SIGINT)

                if (
                    cancelled
                    or turn_token.cancelled
                    or (
                        isinstance(result, Failure)
                        and result.failure().code == "CANCELLED"
                    )
                ):
                    if hasattr(service, "rollback_turn"):
                        rollback_fn = getattr(service, "rollback_turn")
                        rollback_res = rollback_fn(turn.record_id)
                        if asyncio.iscoroutine(rollback_res) or inspect.isawaitable(rollback_res):
                            await rollback_res
                    elif hasattr(service, "delete_turn"):
                        del_fn = getattr(service, "delete_turn")
                        del_res = del_fn(turn.record_id)
                        if asyncio.iscoroutine(del_res) or inspect.isawaitable(del_res):
                            await del_res
                    click.echo("\n[Generation interrupted. Turn rolled back.]", err=True)
                    continue

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
                elif result is not None:
                    err = result.failure()
                    click.echo(f"[Error: {err.code}] {err.message}", err=True)
        finally:
            await container.close()

    asyncio.run(_run_loop())
