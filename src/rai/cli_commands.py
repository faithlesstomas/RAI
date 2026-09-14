"""Stage 1 command definitions, separate from transport and rendering."""

from __future__ import annotations

import asyncio
import json

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


def _run_assistant_ask(prompt: str, session_id: str | None) -> None:
    from returns.result import Success  # noqa: PLC0415

    from .assistant.records import ConversationTurn  # noqa: PLC0415
    from .container import ApplicationContainer  # noqa: PLC0415
    from .kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    container = ApplicationContainer(config={})
    service = container.assistant_service
    sid = session_id or _new_id()
    turn = ConversationTurn(
        record_id=_new_id(),
        producer=ProducerIdentity(producer_id="cli-ask", kind="user", version="1.0.0"),
        session_id=sid,
        role="user",
        text=prompt,
    )

    async def _run() -> None:
        await service.store.start()
        res = await service.accept_turn(turn)
        if isinstance(res, Success):
            candidate = res.unwrap()
            click.echo(candidate.text)
        else:
            err = res.failure()
            click.echo(f"Error [{err.code}]: {err.message}", err=True)

    asyncio.run(_run())


def _run_assistant_chat(session_id: str | None) -> None:
    from returns.result import Success  # noqa: PLC0415

    from .assistant.records import ConversationTurn  # noqa: PLC0415
    from .container import ApplicationContainer  # noqa: PLC0415
    from .kernel.records import ProducerIdentity, _new_id  # noqa: PLC0415

    container = ApplicationContainer(config={})
    service = container.assistant_service
    sid = session_id or _new_id()
    click.echo(f"Starting Assistant session: {sid}")
    click.echo("Type /exit or /q to quit.\n")

    async def _run_loop() -> None:
        await service.store.start()
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
            if not stripped:
                continue

            turn = ConversationTurn(
                record_id=_new_id(),
                producer=ProducerIdentity(producer_id="cli-chat", kind="user", version="1.0.0"),
                session_id=sid,
                role="user",
                text=stripped,
            )
            click.echo("Assistant: ", nl=False)
            async for chunk_res in service.accept_turn_stream(turn):
                if isinstance(chunk_res, Success):
                    click.echo(chunk_res.unwrap(), nl=False)
                else:
                    err = chunk_res.failure()
                    click.echo(f"\n[Error: {err.code}] {err.message}", err=True)
            click.echo()

    asyncio.run(_run_loop())


def register_assistant_commands(root: click.Group) -> None:
    """Attach assistant commands to the root CLI."""

    @root.group(name="assistant")
    def assistant() -> None:
        """Interact with the Rich Assistant using durable graph memory."""

    @assistant.command(name="ask")
    @click.argument("prompt")
    @click.option("--session-id", default=None, help="Session ID for the conversation turn.")
    def ask_command(prompt: str, session_id: str | None) -> None:
        """Send a single prompt to the assistant."""
        _run_assistant_ask(prompt, session_id)

    @assistant.command(name="chat")
    @click.option("--session-id", default=None, help="Session ID for the conversation.")
    def chat_command(session_id: str | None) -> None:
        """Start an interactive chat session with the assistant."""
        _run_assistant_chat(session_id)
