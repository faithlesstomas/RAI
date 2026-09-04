"""
API endpoints for managing agents.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from .. import config_manager
from ..kernel.defaults import create_default_capability_registry

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Agents"],
    responses={404: {"description": "Not found"}},
)

# --- Pydantic Models ---

class AgentDefinition(BaseModel):
    """Definition of an agent template."""
    name: str = Field(..., description="Unique name/ID of the agent")
    model: str = Field(..., description="Model name, e.g. gemma2:9b")
    backend: str = Field("ollama", description="Backend: ollama, gemini, openai, etc.")
    system_prompt: Optional[str] = Field(None, description="System instructions for the agent")
    tools: List[str] = Field(default_factory=list, description="List of tool names")
    description: Optional[str] = Field(None, description="Short description of the agent's specialization")

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: List[str]) -> List[str]:
        """Validates that requested tools exist in the registry."""
        known_tools = set(create_default_capability_registry().compatibility_groups()) | {
            "DesktopNotificationTool", "DesktopScreenshotTool", "DesktopWeatherTool",
            "ClientTools", "GitlabTools", "WebBrowserTools", "FileTools", "PythonTools", "ShellTools"
        }
        invalid_tools = [tool for tool in v if tool not in known_tools]
        if invalid_tools:
            raise ValueError(f"Unknown tools: {', '.join(invalid_tools)}")
        return v

class AgentConfig(BaseModel):
    """Model for creating or updating an agent (session) - Legacy/Compat."""
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

@router.post("/", status_code=201)
async def define_agent(agent: AgentDefinition) -> Dict[str, Any]:
    """
    Dynamically creates or updates an agent template in the RAI configuration.
    """
    try:
        # Convert to dict and handle mapping to internal config structure
        agent_data = agent.model_dump()
        # map system_prompt to system if needed, or keep as system_prompt
        # The current config uses 'system'
        if agent_data.get("system_prompt"):
            # Fixed indentation here
            agent_data["system"] = agent_data.pop("system_prompt")

        success = config_manager.save_agent_template(agent_data)
        if not success:
            raise HTTPException(status_code=500, detail="Error saving configuration")

        return {"message": f"Agent '{agent.name}' successfully defined.", "agent": agent_data}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/", response_model=Dict[str, Any])
async def list_agents(config: Dict[str, Any] = Depends(get_config)) -> Dict[str, Any]:
    """
    Lists all available agents.
    """
    return config.get("agents") or config.get("sessions", {})

@router.get("/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str, config: Dict[str, Any] = Depends(get_config)) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    """
    agents = config.get("agents") or config.get("sessions", {})
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents[agent_id]

@router.post("/{agent_id}")
async def create_agent(agent_id: str, agent_config: AgentConfig) -> Dict[str, Any]:
    """
    Create a new agent (session) - Legacy/Active Session creation.
    """
    config = config_manager.load_config()
    agents = config.get("agents") or config.get("sessions", {})

    if agent_id in agents:
        raise HTTPException(status_code=400, detail="Agent already exists")

    # Convert Pydantic model to dict
    agents[agent_id] = agent_config.model_dump()
    config["agents"] = agents

    config_manager.save_config(config)
    return {"status": "success", "agent_id": agent_id, "config": agents[agent_id]}

@router.put("/{agent_id}")
async def update_agent(agent_id: str, agent_config: AgentConfig) -> Dict[str, Any]:
    """
    Update an existing agent (session).
    """
    config = config_manager.load_config()
    agents = config.get("agents") or config.get("sessions", {})

    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Update existing config
    agents[agent_id] = agent_config.model_dump()
    config["agents"] = agents

    config_manager.save_config(config)
    return {"status": "success", "agent_id": agent_id, "config": agents[agent_id]}

@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    Delete an agent (session).
    """
    config = config_manager.load_config()
    agents = config.get("agents") or config.get("sessions", {})

    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default agent")

    active_agent = config.get("active_agent") or config.get("active_session")
    if agent_id == active_agent:
        raise HTTPException(status_code=400, detail="Cannot delete active agent")

    del agents[agent_id]
    config["agents"] = agents
    config_manager.save_config(config)
    return {"status": "success", "deleted": agent_id}
