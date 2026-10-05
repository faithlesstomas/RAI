"""Profile management; profiles are configuration, conversations are sessions."""

import typer
import yaml

from rai.configuration.storage import load_settings, redact, save_settings

profiles = typer.Typer(
    help="Manage model configuration profiles.", no_args_is_help=True
)


@profiles.command("list")
def list_profiles() -> None:
    settings = load_settings()
    for name in settings.agents:
        typer.echo(f"{name}{' (active)' if name == settings.active_agent else ''}")


@profiles.command("show")
def show_profile(name: str) -> None:
    settings = load_settings()
    if name not in settings.agents:
        raise typer.BadParameter(f"Unknown profile: {name}")
    typer.echo(
        yaml.safe_dump(redact(settings.agents[name].model_dump(exclude_unset=True)))
    )


@profiles.command("use")
def use_profile(name: str) -> None:
    settings = load_settings().runtime_mapping()
    if name not in settings["agents"]:
        raise typer.BadParameter(f"Unknown profile: {name}")
    settings["active_agent"] = name
    save_settings(settings)
    typer.echo(f"Active profile: {name}")
