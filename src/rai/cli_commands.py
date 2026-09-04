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
