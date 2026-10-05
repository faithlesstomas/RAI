"""Configuration adapters for remaining REST/TUI consumers.

Parsing, validation and persistence live in rai.configuration. Legacy UI helpers
below remain until #43 removes their consumers; they no longer migrate on read.
"""
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from .core import console, error_console
from .paths import config_dir, data_dir
from .configuration.storage import (
    STATE_KEYS, atomic_write, load_runtime_state, load_settings, read_mapping,
    save_runtime_state, save_settings, selected_path,
)
from .configuration.models import ProfileSettings
from pathlib import Path

CONFIG_DIR = str(config_dir())
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
DEFAULT_AGENTS_FILE = os.path.join(CONFIG_DIR, "agents.yaml")
DEFAULT_TTS_DATA_DIR = str(data_dir() / "piper_voices")


def get_config_path(path: Optional[str] = None) -> str:
    return str(selected_path(path))


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    return load_settings(path).runtime_mapping()


def save_config(config_data: Dict[str, Any], path: Optional[str] = None) -> None:
    save_settings(config_data, path)


def load_state(path: Optional[str] = None) -> Dict[str, Any]:
    state = load_runtime_state(path)
    state["active_agent"] = load_settings(path).active_agent
    return state


def save_state(state_data: Dict[str, Any], path: Optional[str] = None) -> None:
    save_runtime_state({k: v for k, v in state_data.items() if k in STATE_KEYS}, path)
    if "active_agent" in state_data:
        settings = load_config(path)
        settings["active_agent"] = state_data["active_agent"]
        save_settings(settings, path)


def load_agents(path: Optional[str] = None) -> Dict[str, Any]:
    if path is not None:
        return {k: ProfileSettings.model_validate(v).model_dump(exclude_unset=True)
                for k, v in read_mapping(Path(path)).items()}
    return load_config().get("agents", {})


def save_agents(agents_data: Dict[str, Any], path: Optional[str] = None) -> None:
    profiles = {k: ProfileSettings.model_validate(v).model_dump(exclude_unset=True)
                for k, v in agents_data.items()}
    if path is not None:
        atomic_write(Path(path), profiles)
        return
    settings = load_config()
    settings["agents"] = profiles
    save_settings(settings)


def save_agent_template(agent_data: Dict[str, Any]) -> bool:
    """Saves a new agent template."""
    try:
        agents = load_agents()
        agent_name = agent_data.get("name")
        if not agent_name:
            raise ValueError("Agent definition must have a 'name'.")
        agents[agent_name] = agent_data
        save_agents(agents)
        return True
    except Exception as e:
        error_console.print(f"[bold red]Error saving agent template: {e}[/bold red]")
        return False


def get_session_config(session_id: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration for a specific agent (historically session)."""
    agents = load_config(config_path).get("agents", {})
    return agents.get(session_id, {})


def initialize_session(
    app_config: Dict[str, Any],
    session_override: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Ensures active agent exists, hydrates fields, and returns active agent name and config.
    """
    agents = app_config.get("agents") or app_config.get("sessions", {})
    active_agent = session_override or app_config.get("active_agent") or app_config.get("active_session", "default")

    if active_agent not in agents:
        agents[active_agent] = {
            "name": active_agent,
            "model": "gemini-2.5-flash",
            "system": "You are a versatile and helpful AI assistant.",
            "tools": [
                "CalculatorTools", "ArxivTools", "WikipediaTools",
                "DuckDuckGoTools", "WebBrowserTools", "FileTools",
                "PythonTools", "ShellTools",
            ],
        }

    if "tts" not in agents[active_agent]:
        agents[active_agent]["tts"] = {
            "data_dir": DEFAULT_TTS_DATA_DIR,
            "default_voice": "pl_PL-gosia-medium",
        }

    app_config["agents"] = agents
    app_config["active_agent"] = active_agent
    save_config(app_config, path=config_path)
    return active_agent, agents[active_agent]


# --- CLI Logic Functions ---

def switch_session_logic(session_name: str) -> None:
    """Switches active agent profile."""
    agents = load_agents()
    if session_name not in agents:
        console.print(f"Creating new agent profile: [bold]{session_name}[/bold]")
        agents[session_name] = {
            "name": session_name,
            "model": "gemini-2.5-flash",
            "system": "You are a versatile and helpful AI assistant.",
        }
        save_agents(agents)

    state = load_state()
    state["active_agent"] = session_name
    save_state(state)
    console.print(f"Switched to agent profile: [bold green]{session_name}[/bold green]")


def list_sessions_logic() -> None:
    """Lists all available agent profiles."""
    agents = load_agents()
    state = load_state()
    active_agent = state.get("active_agent", "default")
    if not agents:
        console.print("[yellow]No agent profiles found.[/yellow]")
        return
    console.print("[bold]Available Agent Profiles (Templates):[/bold]")
    for name in agents:
        if name == active_agent:
            console.print(f"- [bold green]{name} (active)[/bold green]")
        else:
            console.print(f"- {name}")


def show_session_logic(session_name: Optional[str] = None) -> None:
    """Shows configuration for a specific agent profile."""
    agents = load_agents()
    state = load_state()
    target = session_name or state.get("active_agent", "default")
    agent_config = agents.get(target)
    if not agent_config:
        error_console.print(f"[bold red]Error: Agent profile '{target}' not found.[/bold red]")
        return
    console.print(f"[bold]Configuration for agent profile: [cyan]{target}[/cyan][/bold]")
    console.print(json.dumps(agent_config, indent=2))


def delete_session_logic(session_name: str) -> None:
    """Deletes a specified agent profile."""
    agents = load_agents()
    state = load_state()
    active_agent = state.get("active_agent", "default")
    if session_name == "default":
        error_console.print("[bold red]Error: Cannot delete the default agent profile.[/bold red]")
        return
    if session_name == active_agent:
        error_console.print("[bold red]Error: Cannot delete the active agent profile.[/bold red]")
        return
    if session_name not in agents:
        error_console.print(f"[bold red]Error: Agent profile '{session_name}' not found.[/bold red]")
        return
    del agents[session_name]
    save_agents(agents)
    console.print(f"Agent profile '[bold red]{session_name}[/bold red]' has been deleted.")


def rename_session_logic(old_name: str, new_name: str) -> None:
    """Renames an agent profile."""
    agents = load_agents()
    if old_name not in agents:
        error_console.print(f"[bold red]Error: Agent profile '{old_name}' not found.[/bold red]")
        return
    if new_name in agents:
        error_console.print(f"[bold red]Error: Agent profile '{new_name}' already exists.[/bold red]")
        return
    agents[new_name] = agents.pop(old_name)
    agents[new_name]["name"] = new_name
    save_agents(agents)
    console.print(f"Agent profile '{old_name}' has been renamed to '[bold green]{new_name}[/bold green]'.")

    state = load_state()
    if state.get("active_agent") == old_name:
        state["active_agent"] = new_name
        save_state(state)
        console.print(f"Active agent profile has been updated to '[bold green]{new_name}[/bold green]'.")


def switch_agent_logic(agent_name: str) -> None:
    """Switches active agent profile."""
    switch_session_logic(agent_name)


def list_agents_logic() -> None:
    """Lists all available agent profiles."""
    list_sessions_logic()


def show_agent_logic(agent_name: Optional[str] = None) -> None:
    """Shows configuration for a specific agent profile."""
    show_session_logic(agent_name)


def delete_agent_logic(agent_name: str) -> None:
    """Deletes a specified agent profile."""
    delete_session_logic(agent_name)


def rename_agent_logic(old_name: str, new_name: str) -> None:
    """Renames an agent profile."""
    rename_session_logic(old_name, new_name)


def show_config_logic() -> None:
    """Shows the active agent's configuration."""
    agents = load_agents()
    state = load_state()
    active_agent = state.get("active_agent", "default")
    agent_config = agents.get(active_agent)
    if not agent_config:
        error_console.print(f"[bold red]Error: Active agent profile '{active_agent}' not found.[/bold red]")
        return
    console.print(f"[bold]Configuration for active agent profile: [cyan]{active_agent}[/cyan][/bold]")
    console.print(json.dumps(agent_config, indent=2))


def set_config_logic(key: str, value: str) -> None:
    """Sets a config value in the active agent profile."""
    agents = load_agents()
    state = load_state()
    active_agent = state.get("active_agent", "default")
    if active_agent not in agents:
        error_console.print(f"[bold red]Error: Active agent profile '{active_agent}' not found.[/bold red]")
        return

    # Allow nested keys
    keys = key.split('.')
    config_level = agents[active_agent]

    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            if key == "tools":
                config_level[k] = [t.strip() for t in value.split(",")]
            else:
                config_level[k] = value
        else:
            config_level = config_level.setdefault(k, {})

    save_agents(agents)
    console.print(
        f"In agent profile '[cyan]{active_agent}[/cyan]', set '[bold]{key}[/bold]' to '[green]{value}[/green]'."
    )


def get_config_logic(key: str) -> None:
    """Gets a config value from the active agent profile."""
    agents = load_agents()
    state = load_state()
    active_agent = state.get("active_agent", "default")
    agent_config = agents.get(active_agent)
    if not agent_config:
        error_console.print(f"[bold red]Error: Active agent profile '{active_agent}' not found.[/bold red]")
        return

    # Allow nested keys
    keys = key.split('.')
    value = agent_config
    try:
        for k in keys:
            value = value[k]
        console.print(value)
    except (KeyError, TypeError):
        error_console.print(
            f"[bold red]Error: Key '{key}' not found in agent profile '{active_agent}'.[/bold red]"
        )


TRAJECTORY_DIR = os.path.join(CONFIG_DIR, "trajectories")


def get_conversation_id_for_session(session_name: str) -> str:
    """Returns the persistent Antigravity conversation ID associated with a session name."""
    state = load_state()
    mapping = state.get("session_conversation_ids", {})
    return mapping.get(session_name, "")


def set_conversation_id_for_session(session_name: str, conv_id: str) -> None:
    """Associates a persistent Antigravity conversation ID with a session name."""
    state = load_state()
    if "session_conversation_ids" not in state:
        state["session_conversation_ids"] = {}
    state["session_conversation_ids"][session_name] = conv_id
    save_state(state)


def clear_conversation_id_for_session(session_name: str) -> None:
    """Removes the persistent Antigravity conversation ID mapping and deletes the trajectory file."""
    state = load_state()
    mapping = state.get("session_conversation_ids", {})
    if session_name in mapping:
        conv_id = mapping.pop(session_name)
        save_state(state)

        # Also delete the trajectory file if it exists
        traj_file = os.path.join(TRAJECTORY_DIR, f"traj-{conv_id}")
        if os.path.exists(traj_file):
            try:
                os.remove(traj_file)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to remove trajectory file {traj_file}: {e}")
