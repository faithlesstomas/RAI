"""Capability commands."""

import asyncio
import json
import click
import typer
from pydantic import ValidationError
from rai.kernel.records import CapabilityRequest
from rai.cli_transport import invoke_local, list_local_capabilities
from rai.cli_rendering import render_capabilities, render_envelope

capability = typer.Typer(help="Capability operations.", no_args_is_help=True)


@capability.command(name="list")
def list_capability_command() -> None:
    click.echo(render_capabilities(list_local_capabilities()))


@capability.command(name="invoke")
def invoke_capability_command(request_json: str = typer.Argument(...)) -> None:
    """Invoke a complete versioned CapabilityRequest JSON record."""
    try:
        request = CapabilityRequest.model_validate(json.loads(request_json))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise click.ClickException(f"invalid CapabilityRequest: {exc}") from exc
    envelope = asyncio.run(invoke_local(request))
    click.echo(render_envelope(envelope))
