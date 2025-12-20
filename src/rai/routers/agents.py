"""
API endpoints for managing agents.
"""
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import config_manager

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Agents"],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic Models ---

class AgentConfig(BaseModel):
    """Model for creating or updating an agent (session)."""
    model: str
    backend: str = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    system: str = "You are a helpful AI assistant."
    tools: List[str] = []

# --- Dependencies ---

def get_config() -> Dict[str, Any]:
    """Dependency to load the application configuration."""
    return config_manager.load_config()

# --- Endpoints ---

@router.get("/", response_model=Dict[str, Any])
async def list_agents(config: Dict[str, Any] = Depends(get_config)) -> Dict[str, Any]:
    """
    Lists all available agents.
    """
    return config.get("sessions", {})

@router.get("/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str, config: Dict[str, Any] = Depends(get_config)) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    """
    sessions = config.get("sessions", {})
    if agent_id not in sessions:
        raise HTTPException(status_code=404, detail="Agent not found")
    return sessions[agent_id]

@router.post("/{agent_id}")
async def create_agent(agent_id: str, agent_config: AgentConfig) -> Dict[str, Any]:
    """
    Create a new agent (session).
    """
    config = config_manager.load_config()
    sessions = config.get("sessions", {})

    if agent_id in sessions:
        raise HTTPException(status_code=400, detail="Agent already exists")

    # Convert Pydantic model to dict
    sessions[agent_id] = agent_config.model_dump()
    config["sessions"] = sessions

    config_manager.save_config(config)
    return {"status": "success", "agent_id": agent_id, "config": sessions[agent_id]}

@router.put("/{agent_id}")
async def update_agent(agent_id: str, agent_config: AgentConfig) -> Dict[str, Any]:
    """
    Update an existing agent (session).
    """
    config = config_manager.load_config()
    sessions = config.get("sessions", {})

    if agent_id not in sessions:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update existing config
    sessions[agent_id] = agent_config.model_dump()
    config["sessions"] = sessions

    config_manager.save_config(config)
    return {"status": "success", "agent_id": agent_id, "config": sessions[agent_id]}

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    Delete an agent (session).
    """
    config = config_manager.load_config()
    sessions = config.get("sessions", {})

    if agent_id not in sessions:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default agent")

    if agent_id == config.get("active_session"):
        raise HTTPException(status_code=400, detail="Cannot delete active agent")

    del sessions[agent_id]
    config["sessions"] = sessions
    config_manager.save_config(config)
    return {"status": "success", "deleted": agent_id}
