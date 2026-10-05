"""Validate, inspect and explicitly migrate persisted configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from rai.configuration.resolution import effective_model_settings
from rai.configuration.storage import (
    STATE_KEYS,
    ConfigurationError,
    atomic_write,
    load_runtime_state,
    load_settings,
    read_mapping,
    redact,
    save_settings,
    selected_path,
    validate,
)

config_app = typer.Typer(
    help="Validated YAML/JSON settings and effective values.", no_args_is_help=True
)


@config_app.command("validate")
def validate_command() -> None:
    """Validate the selected configuration, without writing or loading a model."""
    load_settings()
    typer.echo(f"Valid configuration: {selected_path()}")


@config_app.command("show")
def show(
    effective: bool = typer.Option(
        False, help="Include resolved model settings and value sources."
    ),
    profile: Optional[str] = typer.Option(None),
) -> None:
    """Show settings with credentials redacted."""
    settings = load_settings().runtime_mapping()
    if profile and profile not in settings["agents"]:
        raise typer.BadParameter(f"Unknown profile: {profile}")
    output = {"file": str(selected_path()), "settings": settings}
    if effective:
        values, sources = effective_model_settings(settings, profile=profile)
        output.update({"effective_assistant": values, "sources": sources})
    typer.echo(yaml.safe_dump(redact(output), sort_keys=False, allow_unicode=True))


@config_app.command("set")
def set_value(key: str, value: str) -> None:
    """Set a dotted settings key; VALUE uses JSON types (or unquoted plain text)."""
    settings = load_settings().runtime_mapping()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    parts = key.split(".")
    current = settings
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise typer.BadParameter("Parent setting is not a mapping")
    current[parts[-1]] = parsed
    save_settings(settings)
    typer.echo(f"Updated {key}")


@config_app.command("migrate")
def migrate(
    source: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path = typer.Option(
        ..., "--output", help="New YAML/JSON file; never overwrite an existing file."
    ),
) -> None:
    """Copy supported legacy settings to one file; preserve source and runtime state."""
    if output.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise typer.BadParameter("Output must use .yaml, .yml or .json")
    if output.exists():
        raise typer.BadParameter("Output already exists")
    raw = read_mapping(source, missing_ok=False)
    state = {k: v for k, v in raw.items() if k in STATE_KEYS}
    raw = {k: v for k, v in raw.items() if k not in STATE_KEYS}
    if "sessions" in raw:
        if "agents" in raw:
            raise ConfigurationError(
                "Both sessions and agents are present; resolve the conflict explicitly"
            )
        raw["agents"] = raw.pop("sessions")
    if "active_session" in raw:
        raw.setdefault("active_agent", raw.pop("active_session"))
    profiles = source.parent / "agents.yaml"
    if "agents" not in raw and profiles.exists():
        raw["agents"] = read_mapping(profiles)
    # Reject obsolete settings explicitly; never discard unknown keys or change models.
    settings = validate(raw).runtime_mapping()
    settings["schema_version"] = 1
    state.update(load_runtime_state(str(source)))
    if state:
        from rai.configuration.storage import save_runtime_state  # noqa: PLC0415

        save_runtime_state(state, str(output))
    atomic_write(output, settings, overwrite=False)
    typer.echo(
        f"Created {output}; source files preserved. Select the new file with --config."
    )
