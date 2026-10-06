"""RAI's Typer entry point. Commands depend on application services, not legacy chat."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import typer
from typer.core import TyperGroup

from rai import __version__
from rai.commands.assistant import assistant
from rai.commands.capability import capability
from rai.commands.neural import neural
from rai.configuration.storage import ConfigurationError, configuration_path


class ConfigurationGroup(TyperGroup):
    """Translate domain errors at the CLI boundary, including subcommands."""

    def invoke(self, ctx: typer.Context) -> object:
        try:
            return super().invoke(ctx)
        except (ConfigurationError, OSError, click.ClickException) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc


cli = typer.Typer(
    help="Local-first Linux assistant and capability runtime.",
    no_args_is_help=True,
    cls=ConfigurationGroup,
)
cli.add_typer(assistant, name="assistant")
cli.add_typer(capability, name="capability")
cli.add_typer(neural, name="neural")


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"rai {__version__}")
        raise typer.Exit()


@cli.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", help="Select one YAML or JSON settings file."
    ),
    trace: bool = typer.Option(False, "--trace", help="Log runtime stages and action outcomes to stderr (no content)."),
    version: bool = typer.Option(
        False, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    """Choose settings before running a subcommand."""
    from rai.diagnostics import configure_trace  # noqa: PLC0415

    configure_trace(trace)
    ctx.with_resource(configuration_path(str(config) if config else None))


def run() -> None:
    """Console boundary: actionable config errors without exposing secret values."""
    cli()


from rai.commands.config import config_app  # noqa: E402
from rai.commands.profiles import profiles
from rai.commands.server import server  # noqa: E402

cli.add_typer(config_app, name="config")
cli.add_typer(server, name="server")
cli.add_typer(profiles, name="profile")

if __name__ == "__main__":
    run()
