"""
Storage management for RAI WebUI.
"""
import json
import os
from typing import Any, Dict, List, Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")

class StorageManager:
    """Manages local storage for the WebUI."""
    def __init__(self) -> None:
        self._load_data()

    def _load_data(self) -> None:
        if not os.path.exists(DATA_FILE):
            self.data = {
                "agents": [],
                "chains": [],
                "settings": {"server_url": "http://127.0.0.1:8000"},
                "chat_history": [],
                "sessions": {}
            }
            self._save_data()
        else:
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {
                    "agents": [],
                    "chains": [],
                    "settings": {"server_url": "http://127.0.0.1:8000"},
                    "chat_history": [],
                    "sessions": {}
                }

    def _save_data(self) -> None:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    # --- Agents ---
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get all agents."""
        return self.data.get("agents", [])

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an agent by name."""
        for agent in self.data.get("agents", []):
            if agent["name"] == name:
                return agent
        return None

    def save_agent(self, agent_data: Dict[str, Any]) -> None:
        """Save an agent."""
        agents = self.get_agents()
        for i, a in enumerate(agents):
            if a['name'] == agent_data['name']:
                agents[i] = agent_data
                self._save_data()
                return
        agents.append(agent_data)
        self.data['agents'] = agents
        self._save_data()

    def delete_agent(self, name: str) -> None:
        """Delete an agent by name."""
        agents = self.get_agents()
        self.data["agents"] = [a for a in agents if a["name"] != name]
        self._save_data()

    # --- Chains ---
    def get_chains(self) -> List[Dict[str, Any]]:
        """Get all chains."""
        return self.data.get('chains', [])

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a chain."""
        chains = self.get_chains()
        # Update if exists, else append
        for i, c in enumerate(chains):
            if c['name'] == chain_data['name']:
                chains[i] = chain_data
                self._save_data()
                return
        chains.append(chain_data)
        self.data['chains'] = chains
        self._save_data()

    def delete_chain(self, name: str) -> None:
        """Delete a chain by name."""
        chains = self.get_chains()
        self.data['chains'] = [c for c in chains if c['name'] != name]
        self._save_data()

    # --- Settings ---
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.data.get("settings", {})

    def save_settings(self, settings: Dict[str, Any]) -> None:
        """Save new settings."""
        self.data["settings"] = settings
        self._save_data()

    # --- Chat History (Session Aware) ---
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a specific session."""
        sessions = self.data.get("sessions", {})
        return sessions.get(session_id, [])

    def add_chat_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """Add a chat message to a specific session's history."""
        sessions = self.data.get("sessions", {})
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(message)
        self.data["sessions"] = sessions
        self._save_data()

    def clear_chat_history(self, session_id: str) -> None:
        """Clear chat history for a specific session."""
        sessions = self.data.get("sessions", {})
        if session_id in sessions:
            sessions[session_id] = []
            self._save_data()

    def get_sessions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all sessions."""
        return self.data.get("sessions", {})

storage = StorageManager()
