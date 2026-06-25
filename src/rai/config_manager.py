"""
Centralized configuration management for the RAI application.
"""
import json
import logging
import os
import yaml
from typing import Any, Dict, Optional, Tuple

from .core import console, error_console

# --- Constants ---
CONFIG_DIR = os.path.expanduser("~/.config/rai")
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_AGENTS_FILE = os.path.join(CONFIG_DIR, "agents.yaml")
DEFAULT_TTS_DATA_DIR = os.path.expanduser("~/.local/share/rai/piper_voices")

# --- Core Helper Functions ---

def get_config_path(path: Optional[str] = None) -> str:
    """Returns the path to the state configuration file."""
    return path or DEFAULT_CONFIG_FILE


def load_state(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads active session and general state from config.json."""
    config_file = get_config_path(path)
    if not os.path.exists(config_file):
        return {"active_agent": "default", "active_session_id": ""}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"active_agent": "default", "active_session_id": ""}


def save_state(state_data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Saves active session and general state to config.json."""
    config_file = get_config_path(path)
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)


def load_agents(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads agent templates from agents.yaml with automatic backward compatibility migration."""
    agents_file = path or DEFAULT_AGENTS_FILE
    agents = {}
    loaded_from_yaml = False

    if os.path.exists(agents_file):
        try:
            with open(agents_file, "r", encoding="utf-8") as f:
                agents = yaml.safe_load(f) or {}
            loaded_from_yaml = True
        except Exception as e:
            error_console.print(f"[bold red]Error loading agents file {agents_file}: {e}[/bold red]")
            agents = {}

    # Migration path: If old config.json contains "sessions" (which were assistant configurations)
    if not loaded_from_yaml:
        config_file = DEFAULT_CONFIG_FILE
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "sessions" in data and data["sessions"]:
                    agents = data["sessions"]
                    # Save old sessions as agent profiles in agents.yaml
                    save_agents(agents, agents_file)
                    # Remove sessions key to clean up old config.json
                    data.pop("sessions", None)
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    loaded_from_yaml = True
            except Exception:
                pass

    if not agents and not loaded_from_yaml:
        # Default fallback agent template (utilizes Antigravity native capabilities)
        default_agents = {
            "default": {
                "name": "default",
                "model": "gemini-2.5-flash",
                "system": "You are a versatile and helpful AI assistant deeply integrated with Linux.",
                "tools": [
                    "CalculatorTools", "ArxivTools", "WikipediaTools",
                    "DuckDuckGoTools", "WebBrowserTools", "FileTools",
                    "PythonTools", "ShellTools",
                    "DesktopNotificationTool", "DesktopScreenshotTool", "DesktopWeatherTool"
                ],
            }
        }
        save_agents(default_agents, agents_file)
        return default_agents

    # Perform in-memory migration & clean up of legacy keys/tools
    modified = False
    tool_migration_map = {
        "GnomeNotificationTool": "DesktopNotificationTool",
        "GnomeScreenshotTool": "DesktopScreenshotTool",
        "GnomeWeatherTool": "DesktopWeatherTool"
    }

    for agent_name, agent_data in list(agents.items()):
        if not isinstance(agent_data, dict):
            continue
        
        # 1. Strip 'framework' key
        if "framework" in agent_data:
            agent_data.pop("framework")
            modified = True
            
        # 2. Migrate tools list
        if "tools" in agent_data and isinstance(agent_data["tools"], list):
            new_tools = []
            for tool in agent_data["tools"]:
                if tool in tool_migration_map:
                    new_tools.append(tool_migration_map[tool])
                    modified = True
                else:
                    new_tools.append(tool)
            # Ensure unique tools while preserving order if possible
            seen = set()
            unique_new_tools = []
            for t in new_tools:
                if t not in seen:
                    seen.add(t)
                    unique_new_tools.append(t)
            if unique_new_tools != agent_data["tools"]:
                agent_data["tools"] = unique_new_tools
                modified = True

        # 3. Migrate model from gemini-1.5-flash to gemini-2.5-flash
        if agent_data.get("model") == "gemini-1.5-flash":
            agent_data["model"] = "gemini-2.5-flash"
            modified = True

    if modified and loaded_from_yaml:
        try:
            save_agents(agents, agents_file)
        except Exception as e:
            error_console.print(f"[bold red]Error saving migrated agents to {agents_file}: {e}[/bold red]")

    return agents


def save_agents(agents_data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Saves agent templates to agents.yaml."""
    agents_file = path or DEFAULT_AGENTS_FILE
    config_dir = os.path.dirname(agents_file)
    os.makedirs(config_dir, exist_ok=True)
    with open(agents_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(agents_data, f, default_flow_style=False)


# --- Backward Compatible Core APIs ---

def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads unified configuration mapping agents.yaml into 'agents' 
    key.
    """
    agents = load_agents()
    state = load_state(path)
    
    # Check if there is an old sessions key in state and migrate it
    if "sessions" in state and not state.get("agents"):
        state["agents"] = state.pop("sessions")
        save_state(state, path)

    return {
        "agents": agents,
        "active_agent": state.get("active_agent", "default"),
        "active_session_id": state.get("active_session_id", ""),
        "tts": state.get("tts", {
            "data_dir": DEFAULT_TTS_DATA_DIR,
            "default_voice": "pl_PL-gosia-medium",
        })
    }


def save_config(config_data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Saves configuration by splitting agent profiles and general state."""
    agents_data = config_data.get("agents") or config_data.get("sessions")
    if agents_data:
        save_agents(agents_data)

    state = {
        "active_agent": config_data.get("active_agent") or config_data.get("active_session", "default"),
        "active_session_id": config_data.get("active_session_id", ""),
        "tts": config_data.get("tts", {
            "data_dir": DEFAULT_TTS_DATA_DIR,
            "default_voice": "pl_PL-gosia-medium",
        })
    }
    save_state(state, path)


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
    agents = load_agents()
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
os.makedirs(TRAJECTORY_DIR, exist_ok=True)


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



