"""
Centralized configuration management for the RAI application.
"""
import json
import os
from typing import Any, Dict, Optional, Tuple

from .core import console, error_console

# --- Constants ---
CONFIG_DIR = os.path.expanduser("~/.config/rai")
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_TTS_DATA_DIR = os.path.expanduser("~/.local/share/rai/piper_voices")

# --- Core Config Functions ---

def get_config_path(path: Optional[str] = None) -> str:
    """Returns the path to the configuration file."""
    return path or DEFAULT_CONFIG_FILE


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the configuration from the given path or the default config file."""
    config_file = get_config_path(path)
    if not os.path.exists(config_file):
        return {"sessions": {}, "active_session": "default"}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        error_console.print(f"[bold red]Error loading config file {config_file}: {e}[/bold red]")
        return {"sessions": {}, "active_session": "default"}

def save_config(config_data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Saves the provided configuration data to the given path or the default file."""
    config_file = get_config_path(path)
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

def get_session_config(
    session_id: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Loads the application config and returns the config for a specific session."""
    app_config = load_config(path=config_path)
    return app_config.get("sessions", {}).get(session_id, {})

def initialize_session(
    app_config: Dict[str, Any],
    session_override: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Determines the session to use, ensures it exists with default values,
    and returns its name and configuration.
    """
    session_to_use = session_override or app_config.get("active_session", "default")
    sessions = app_config.setdefault("sessions", {})

    if session_to_use not in sessions:
        sessions[session_to_use] = {
            "model": "gemma3:4b",
            "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
            "tools": [
                "CalculatorTools", "ArxivTools", "WikipediaTools",
                "DuckDuckGoTools", "WebBrowserTools", "FileTools",
                "PythonTools", "ShellTools",
            ],
        }

    if "tts" not in sessions[session_to_use]:
        sessions[session_to_use]["tts"] = {
            "data_dir": DEFAULT_TTS_DATA_DIR,
            "default_voice": "pl_PL-gosia-medium",
        }

    save_config(app_config, path=config_path)
    return session_to_use, sessions[session_to_use]

# --- CLI Logic Functions ---

def switch_session_logic(session_name: str) -> None:
    """Creates a new session or switches to an existing one."""
    app_config = load_config()
    if session_name not in app_config.get("sessions", {}):
        console.print(f"Creating new session: [bold]{session_name}[/bold]")
        app_config.setdefault("sessions", {})[session_name] = {
            "model": "gemma3:1b",
            "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
        }
    app_config["active_session"] = session_name
    save_config(app_config)
    console.print(f"Switched to session: [bold green]{session_name}[/bold green]")
    console.print("[yellow]Note: The new session will be used the next time you start rai.[/yellow]")

def list_sessions_logic() -> None:
    """Lists all available sessions."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})
    active_session = app_config.get("active_session", "default")
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    console.print("[bold]Available Sessions:[/bold]")
    for name in sessions:
        if name == active_session:
            console.print(f"- [bold green]{name} (active)[/bold green]")
        else:
            console.print(f"- {name}")

def show_session_logic(session_name: Optional[str] = None) -> None:
    """Shows configuration for a specific or active session."""
    app_config = load_config()
    target_session = session_name or app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(target_session)
    if not session_config:
        error_console.print(f"[bold red]Error: Session '{target_session}' not found.[/bold red]")
        return
    console.print(f"[bold]Configuration for session: [cyan]{target_session}[/cyan][/bold]")
    console.print(json.dumps(session_config, indent=2))

def delete_session_logic(session_name: str) -> None:
    """Deletes a specified session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    if session_name == "default":
        error_console.print("[bold red]Error: Cannot delete the default session.[/bold red]")
        return
    if session_name == active_session:
        error_console.print("[bold red]Error: Cannot delete the active session.[/bold red]")
        error_console.print(f"Switch to a different session before deleting '{session_name}'.")
        return
    if session_name not in app_config.get("sessions", {}):
        error_console.print(f"[bold red]Error: Session '{session_name}' not found.[/bold red]")
        return
    del app_config["sessions"][session_name]
    save_config(app_config)
    console.print(f"Session '[bold red]{session_name}[/bold red]' has been deleted.")

def rename_session_logic(old_name: str, new_name: str) -> None:
    """Renames a session."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})
    if old_name not in sessions:
        error_console.print(f"[bold red]Error: Session '{old_name}' not found.[/bold red]")
        return
    if new_name in sessions:
        error_console.print(f"[bold red]Error: Session name '{new_name}' already exists.[/bold red]")
        return
    sessions[new_name] = sessions.pop(old_name)
    console.print(f"Session '{old_name}' has been renamed to '[bold green]{new_name}[/bold green]'.")
    if app_config.get("active_session") == old_name:
        app_config["active_session"] = new_name
        console.print(f"Active session has been updated to '[bold green]{new_name}[/bold green]'.")
    save_config(app_config)

def show_config_logic() -> None:
    """Shows the active session's configuration."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(active_session)
    if not session_config:
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found.[/bold red]")
        return
    console.print(f"[bold]Configuration for active session: [cyan]{active_session}[/cyan][/bold]")
    console.print(json.dumps(session_config, indent=2))

def set_config_logic(key: str, value: str) -> None:
    """Sets a config value in the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    if active_session not in app_config.get("sessions", {}):
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found.[/bold red]")
        return

    # Allow nested keys like "tts.default_voice"
    keys = key.split('.')
    config_level = app_config["sessions"][active_session]

    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            if key == "tools":
                config_level[k] = [t.strip() for t in value.split(",")]
            else:
                config_level[k] = value
        else:
            config_level = config_level.setdefault(k, {})

    save_config(app_config)
    console.print(
        f"In session '[cyan]{active_session}[/cyan]', set '[bold]{key}[/bold]' to '[green]{value}[/green]'."
    )

def get_config_logic(key: str) -> None:
    """Gets a config value from the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(active_session)
    if not session_config:
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found.[/bold red]")
        return

    # Allow nested keys
    keys = key.split('.')
    value = session_config
    try:
        for k in keys:
            value = value[k]
        console.print(value)
    except (KeyError, TypeError):
        error_console.print(
            f"[bold red]Error: Key '{key}' not found in session '{active_session}'.[/bold red]"
        )
