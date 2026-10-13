"""Interactive terminal notebook with persistent Monty REPL and AI assistant."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import click
from returns.result import Success

from rai.assistant.notebook import _extract_code_blocks, export_turns_to_notebook
from rai.assistant.records import ConversationTurn
from rai.container import ApplicationContainer
from rai.kernel.records import DataClass, ProducerIdentity, _new_id
from rai.tools.security.monty import get_monty_runner
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
        "Jesteś interaktywnym asystentem w środowisku notebooka programistycznego Rich AI (RAI). "
        "Dzielisz z użytkownikiem współdzielone, persistentne środowisko sesji Python REPL (nano-sandbox Monty). "
        "Gdy użytkownik prosi o napisanie kodu, algorytmu, funkcji lub wykonanie obliczeń, "
        "zawsze podawaj czysty, gotowy do wykonania kod w blokach ```python ... ```. "
        "Wszystkie generowane przez Ciebie bloki kodu Python są automatycznie i natychmiastowo wykonywane w sesji REPL, "
        "a zdefiniowane zmienne i funkcje pozostają trwale aktywne w pamięci sesji dla kolejnych komórek i obliczeń. "
        "Możesz swobodnie odwoływać się i korzystać ze zmiennych oraz funkcji zdefiniowanych wcześniej w REPL. "
        "Nigdy nie twórz fikcyjnych deklaracji, że coś uruchomiłeś lub sprawdziłeś, bez podania kodu. "
        "Nie proponuj otwierania plików w aplikacjach desktopowych (np. Nautilus), chyba że użytkownik wyraźnie o to zapyta."
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
    monty_runner = get_monty_runner()

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
        "  /exit or /q        - Quit\n"
    )

    async def _run_loop() -> None:  # noqa: PLR0912, PLR0915
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
                    line = click.prompt(click.style(prompt_label, fg="green", bold=True), prompt_suffix=": ")
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
                        "/exit              quit\n"
                    )
                    continue

                if stripped == "/session":
                    click.echo(f"Profile: {service.profile_scope} | Session ID: {sid}")
                    continue

                if stripped == "/repl":
                    saved_res = await service.store.get_repl_session(sid)
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
                    await service.store.clear_repl_session(sid)
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
                        if show_thinking and response.reasoning_content:
                            click.secho(f"\n[Thinking]\n{response.reasoning_content}\n[/Thinking]", fg="cyan")
                        click.secho(f"AI [{cell_index}]:", fg="bright_magenta", bold=True)
                        click.echo(response.text)

                        # Auto-execute AI generated Python code in REPL session
                        code_blocks = _extract_code_blocks(response.text)
                        py_blocks = [code for lang, code in code_blocks if lang in ("python", "py", "")]
                        if py_blocks:
                            saved_s = await service.store.get_repl_session(sid)
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
                                            sid, b_res.session_blob, b_res.variables
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
                            sub_line = click.prompt("   ...", prompt_suffix=" ")
                        except (EOFError, KeyboardInterrupt):
                            break
                        if not sub_line.strip():
                            break
                        code_lines.append(sub_line)

                full_code = "\n".join(code_lines)

                # Fetch prior snapshot
                saved_res = await service.store.get_repl_session(sid)
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
                        await service.store.save_repl_session(sid, monty_res.session_blob, monty_res.variables)

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
