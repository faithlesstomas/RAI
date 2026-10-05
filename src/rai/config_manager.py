"""Configuration adapters for remaining REST consumers.

Parsing, validation and persistence live in rai.configuration. Legacy provider-resume helpers remain until #43 removes their consumers.
"""
import logging
import os
from typing import Any, Dict, Optional

from .core import error_console
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
