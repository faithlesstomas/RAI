import json
import os
from typing import Any, Dict, List, Optional

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")

class StorageManager:
    def __init__(self):
        self._load_data()

    def _load_data(self):
        if not os.path.exists(DATA_FILE):
            self.data = {"agents": [], "chains": [], "settings": {"server_url": "http://127.0.0.1:8000"}, "chat_history": [], "sessions": {}}
            self._save_data()
        else:
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {"agents": [], "chains": [], "settings": {"server_url": "http://127.0.0.1:8000"}, "chat_history": [], "sessions": {}}

    def _save_data(self):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    # --- Agents ---
    def get_agents(self) -> List[Dict[str, Any]]:
        return self.data.get("agents", [])

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        for agent in self.data.get("agents", []):
            if agent["name"] == name:
                return agent
        return None

    def save_agent(self, agent_data: Dict[str, Any]):
        agents = self.data.get("agents", [])
        # Update existing or append new
        for i, agent in enumerate(agents):
            if agent["name"] == agent_data["name"]:
                agents[i] = agent_data
                self._save_data()
                return
        agents.append(agent_data)
        self.data["agents"] = agents
        self._save_data()

    def delete_agent(self, name: str):
        agents = self.data.get("agents", [])
        self.data["agents"] = [a for a in agents if a["name"] != name]
        self._save_data()

    # --- Chains ---
    def get_chains(self) -> List[Dict[str, Any]]:
        return self.data.get("chains", [])

    def save_chain(self, chain_data: Dict[str, Any]):
        chains = self.data.get("chains", [])
        for i, chain in enumerate(chains):
            if chain["name"] == chain_data["name"]:
                chains[i] = chain_data
                self._save_data()
                return
        chains.append(chain_data)
        self.data["chains"] = chains
        self._save_data()

    def delete_chain(self, name: str):
        chains = self.data.get("chains", [])
        self.data["chains"] = [c for c in chains if c["name"] != name]
        self._save_data()

    # --- Settings ---
    def get_settings(self) -> Dict[str, Any]:
        return self.data.get("settings", {})

    def save_settings(self, settings: Dict[str, Any]):
        self.data["settings"] = settings
        self._save_data()

    # --- Chat History (Session Aware) ---
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        sessions = self.data.get("sessions", {})
        return sessions.get(session_id, [])

    def add_chat_message(self, session_id: str, message: Dict[str, Any]):
        sessions = self.data.get("sessions", {})
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(message)
        self.data["sessions"] = sessions
        self._save_data()

    def clear_chat_history(self, session_id: str):
        sessions = self.data.get("sessions", {})
        if session_id in sessions:
            sessions[session_id] = []
            self.data["sessions"] = sessions
            self._save_data()

    def get_sessions(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.data.get("sessions", {})

storage = StorageManager()
