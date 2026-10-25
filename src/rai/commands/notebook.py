"""Interactive terminal notebook with persistent Monty REPL and AI assistant."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from returns.result import Success

from rai.assistant.notebook import _extract_code_blocks, export_turns_to_notebook
from rai.assistant.records import ConversationTurn
from rai.container import ApplicationContainer
from rai.kernel.records import DataClass, ProducerIdentity, _new_id
from rai.paths import data_dir
from .settings import _assistant_config


def _run_assistant_notebook(  # noqa: PLR0913, PLR0915
    session_id: str | None,
    backend: str | None,
    model: str | None,
    profile: str | None = None,
    system: str | None = None,
    thinking: bool | None = None,
    show_thinking: bool = False,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    data_class: DataClass = DataClass.LOCAL,
) -> None:
    notebook_system = (
        (system or "") + "\n\n"
        "You are an interactive pair-programming notebook assistant in Rich AI (RAI). "
        "You share a persistent Python REPL session with the user. "
        "When the user asks to write code, design algorithms, create functions, or calculate results, "
        "always provide clean, executable Python code in ```python ... ``` blocks. "
        "All Python code blocks you generate are executed automatically in the active REPL session, "
        "and defined variables and functions persist across subsequent turns. "
        "You can reference and use variables and functions previously defined in the REPL session. "
        "Never pretend to execute code in memory without outputting code blocks. Always provide Python code blocks to perform calculations."
    ).strip()

    try:
        container = ApplicationContainer(
            config=_assistant_config(
                backend,
                model,
                profile,
                notebook_system,
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
    monty_runner = container.monty_runner

    click.secho("=" * 64, fg="bright_blue")
    click.secho("  RAI Assistant Interactive Notebook (Monty REPL)", bold=True, fg="bright_cyan")
    click.secho(f"  Session: {sid} | Scope: {service.profile_scope}", fg="white")
    click.secho(
        f"  Backend: {getattr(service.backend, 'backend_name', 'unknown')} | "
        f"Model: {getattr(service.backend, 'model_name', 'unknown')}",
        fg="bright_black",
    )
    click.secho("=" * 64, fg="bright_blue")
    click.echo(
        "Usage:\n"
        "  In [N]: <code>      - Run Python in persistent Monty REPL\n"
        "  ? [N]: <prompt>    - Ask AI Assistant (or prefix with '?' or 'AI:')\n"
        "  /repl              - Inspect active REPL variables\n"
        "  /repl reset        - Clear REPL session\n"
        "  /session           - Show current session and profile\n"
        "  /export [file]     - Export session as Jupyter notebook (.ipynb)\n"
        "  /history [N]       - Show recent turns\n"
        "  /memories          - Show active durable memories\n"
        "  /context           - Show context manifest for latest turn\n"
        "  /thinking [on|off] - Toggle chain-of-thought thinking\n"
        "  /exit or /q        - Quit\n"
    )

    hist_file = data_dir() / "history" / "notebook_history.txt"
    try:
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        pt_history: FileHistory | None = FileHistory(str(hist_file))
    except Exception:
        pt_history = None

    notebook_completer = WordCompleter(
        [
            "/help",
            "/repl",
            "/repl reset",
            "/session",
            "/export",
            "/history",
            "/memories",
            "/context",
            "/thinking",
            "/exit",
            "/quit",
            "/q",
        ],
        ignore_case=True,
    )
    prompt_session: PromptSession[str] | None = None
    if sys.stdin.isatty():
        prompt_session = PromptSession(history=pt_history, completer=notebook_completer)

    current_thinking = (
        thinking
        if thinking is not None
        else getattr(service.backend, "enable_thinking", False)
    )
    current_show_thinking = show_thinking

    async def _ask_prompt(prompt_label: str) -> str:
        if prompt_session is not None and sys.stdin.isatty():
            return await prompt_session.prompt_async(
                HTML(f"<b><ansigreen>{prompt_label}</ansigreen></b>: ")
            )
        return click.prompt(click.style(prompt_label, fg="green", bold=True), prompt_suffix=": ")

    async def _ask_continuation() -> str:
        if prompt_session is not None and sys.stdin.isatty():
            return await prompt_session.prompt_async(HTML("<b><ansibrightblack>   ... </ansibrightblack></b>"))
        return click.prompt("   ...", prompt_suffix=" ")

    async def _run_loop() -> None:  # noqa: PLR0912, PLR0915
        nonlocal current_thinking, current_show_thinking
        start_res = await service.start()
        if not isinstance(start_res, Success):
            err = start_res.failure()
            raise click.ClickException(f"[{err.code}] {err.message}")

        cell_index = 1
        reply_to_turn_id: str | None = None

        recent_res = await service.get_recent_turns(sid, limit=1)
        if isinstance(recent_res, Success) and recent_res.unwrap():
            reply_to_turn_id = recent_res.unwrap()[-1].record_id

        try:
            while True:
                prompt_label = f"In [{cell_index}]"
                try:
                    line = await _ask_prompt(prompt_label)
                except (EOFError, KeyboardInterrupt):
                    click.echo("\nExiting notebook.")
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                if stripped in ("/exit", "/quit", "/q"):
                    click.echo("Exiting notebook.")
                    break

                if stripped == "/help":
                    click.echo(
                        "/repl              view active variables in persistent REPL\n"
                        "/repl reset        clear REPL session\n"
                        "/session           show current profile and session ID\n"
                        "/export [file]     export session as Jupyter notebook (.ipynb)\n"
                        "/history [N]       recent session turns\n"
                        "/memories          active durable memories in profile scope\n"
                        "/context           exact context window for latest turn\n"
                        "/thinking [mode]   toggle model thinking (on, off, show, hide)\n"
                        "/exit              quit\n"
                    )
                    continue

                if stripped == "/session":
                    click.echo(f"Profile: {service.profile_scope} | Session ID: {sid}")
                    continue

                if stripped == "/memories":
                    from .memory import _echo_memories  # noqa: PLC0415
                    await _echo_memories(service)
                    continue

                if stripped == "/context":
                    latest_manifest_res = await service.store.get_latest_manifest_for_session(sid)
                    if isinstance(latest_manifest_res, Success) and latest_manifest_res.unwrap() is not None:
                        from .memory import _echo_context_manifest  # noqa: PLC0415
                        await _echo_context_manifest(service, latest_manifest_res.unwrap().record_id)
                    else:
                        click.echo("No context manifest found in this session.")
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
                        click.echo(f"Thinking mode: {'ON' if current_thinking else 'OFF'}")
                    if hasattr(service.backend, "enable_thinking"):
                        service.backend.enable_thinking = current_thinking
                    continue

                if stripped == "/repl":
                    saved_res = await service.store.get_repl_session(
                        sid, profile_scope=service.profile_scope
                    )
                    if isinstance(saved_res, Success) and saved_res.unwrap():
                        blob, vars_dict = saved_res.unwrap()
                        if not vars_dict and blob:
                            inspected = monty_runner.inspect_state(blob)
                            if isinstance(inspected, Success):
                                vars_dict = inspected.unwrap()
                        if vars_dict:
                            click.secho(f"Active REPL variables (Session {sid}):", fg="bright_cyan")
                            for k, v in sorted(vars_dict.items()):
                                click.echo(f"  {click.style(k, bold=True)}: {v}")
                        else:
                            click.echo("REPL session has no active variables.")
                    else:
                        click.echo("REPL session has no active variables.")
                    continue

                if stripped == "/repl reset":
                    await service.store.clear_repl_session(
                        sid, profile_scope=service.profile_scope
                    )
                    click.secho("REPL session reset.", fg="yellow")
                    continue

                if stripped.startswith("/history"):
                    parts = stripped.split()
                    try:
                        limit = int(parts[1]) if len(parts) > 1 else 20
                    except ValueError:
                        click.echo("Usage: /history [number-of-turns]", err=True)
                        continue
                    from .memory import _echo_history  # noqa: PLC0415

                    await _echo_history(service, sid, max(1, min(limit, 200)))
                    continue

                if stripped.startswith("/export"):
                    parts = stripped.split(maxsplit=1)
                    target_file = parts[1] if len(parts) > 1 else f"notebook_{sid}.ipynb"
                    if not target_file.endswith(".ipynb"):
                        target_file += ".ipynb"
                    turns_res = await service.get_recent_turns(sid, limit=200)
                    if isinstance(turns_res, Success):
                        nb = export_turns_to_notebook(sid, list(turns_res.unwrap()), service.profile_scope)
                        Path(target_file).write_text(json.dumps(nb, indent=2), encoding="utf-8")
                        click.secho(f"Notebook exported to {target_file}", fg="bright_green")
                    else:
                        click.echo("Failed to fetch session turns for export.", err=True)
                    continue

                if stripped.startswith("/"):
                    click.secho(f"Unknown command: '{stripped}'. Type /help for available commands.", fg="yellow")
                    continue

                # Check if this cell is an AI query
                is_ai_prompt = stripped.startswith("?") or stripped.lower().startswith("ai:")
                if is_ai_prompt:
                    query_text = stripped[1:].strip() if stripped.startswith("?") else stripped[3:].strip()
                    turn = ConversationTurn(
                        record_id=_new_id(),
                        producer=ProducerIdentity(producer_id="cli-notebook", kind="user", version="1.0.0"),
                        session_id=sid,
                        role="user",
                        data_class=data_class,
                        text=f"? {query_text}",
                        reply_to_turn_id=reply_to_turn_id,
                    )
                    res = await service.accept_turn(turn)
                    if isinstance(res, Success):
                        response = res.unwrap()
                        reply_to_turn_id = response.turn_id
                        if current_show_thinking and response.reasoning_content:
                            click.secho(f"\n[Thinking]\n{response.reasoning_content}\n[/Thinking]", fg="cyan")
                        click.secho(f"AI [{cell_index}]:", fg="bright_magenta", bold=True)
                        click.echo(response.text)

                        # Auto-execute AI generated Python code in REPL session
                        code_blocks = _extract_code_blocks(response.text)
                        py_blocks = [code for lang, code in code_blocks if lang in ("python", "py", "")]
                        if py_blocks:
                            saved_s = await service.store.get_repl_session(
                                sid, profile_scope=service.profile_scope
                            )
                            current_blob = (
                                saved_s.unwrap()[0]
                                if isinstance(saved_s, Success) and saved_s.unwrap()
                                else None
                            )
                            for block in py_blocks:
                                eval_b = monty_runner.run(block, session_blob=current_blob)
                                if isinstance(eval_b, Success):
                                    b_res = eval_b.unwrap()
                                    current_blob = b_res.session_blob
                                    if b_res.session_blob:
                                        await service.store.save_repl_session(
                                            sid,
                                            b_res.session_blob,
                                            b_res.variables,
                                            profile_scope=service.profile_scope,
                                        )
                                    if b_res.output:
                                        click.secho(f"Out [{cell_index}]: ", fg="bright_red", bold=True, nl=False)
                                        click.echo(b_res.output)
                                    if b_res.variables:
                                        var_s = ", ".join(f"{k}: {v}" for k, v in sorted(b_res.variables.items()))
                                        click.secho(f"      [REPL state updated: {var_s}]", fg="bright_cyan")
                                else:
                                    click.secho(
                                        f"Out [{cell_index}] REPL Warning: {eval_b.failure()}",
                                        fg="yellow",
                                    )
                    else:
                        err = res.failure()
                        click.secho(f"AI [{cell_index}] Error: [{err.code}] {err.message}", fg="red")
                    cell_index += 1
                    continue

                # Python code execution in persistent Monty REPL
                code_lines = [line]
                if line.endswith(":") or line.startswith(("def ", "for ", "if ", "while ", "class ", "with ", "try:")):
                    while True:
                        try:
                            sub_line = await _ask_continuation()
                        except (EOFError, KeyboardInterrupt):
                            break
                        if not sub_line.strip():
                            break
                        code_lines.append(sub_line)

                full_code = "\n".join(code_lines)

                # Fetch prior snapshot
                saved_res = await service.store.get_repl_session(
                    sid, profile_scope=service.profile_scope
                )
                session_blob = (
                    saved_res.unwrap()[0]
                    if isinstance(saved_res, Success) and saved_res.unwrap()
                    else None
                )

                eval_res = monty_runner.run(full_code, session_blob=session_blob)
                out_label = f"Out [{cell_index}]"

                if isinstance(eval_res, Success):
                    monty_res = eval_res.unwrap()
                    if monty_res.session_blob:
                        await service.store.save_repl_session(
                            sid,
                            monty_res.session_blob,
                            monty_res.variables,
                            profile_scope=service.profile_scope,
                        )

                    cell_markdown = f"```python\n{full_code}\n```"
                    if monty_res.output:
                        cell_markdown += f"\n\nOutput:\n```\n{monty_res.output}\n```"

                    code_turn = ConversationTurn(
                        record_id=_new_id(),
                        producer=ProducerIdentity(producer_id="cli-notebook", kind="user", version="1.0.0"),
                        session_id=sid,
                        role="user",
                        data_class=data_class,
                        text=cell_markdown,
                        reply_to_turn_id=reply_to_turn_id,
                    )
                    await service.store.accept_turn(code_turn)
                    reply_to_turn_id = code_turn.record_id

                    if monty_res.output:
                        click.secho(f"{out_label}: ", fg="bright_red", bold=True, nl=False)
                        click.echo(monty_res.output)
                    if monty_res.variables:
                        var_summary = ", ".join(f"{k}: {v}" for k, v in sorted(monty_res.variables.items()))
                        click.secho(f"      [REPL state: {var_summary}]", fg="bright_black")
                else:
                    click.secho(f"{out_label} Error: ", fg="red", bold=True, nl=False)
                    click.echo(eval_res.failure())

                cell_index += 1
        finally:
            await container.close()

    asyncio.run(_run_loop())
