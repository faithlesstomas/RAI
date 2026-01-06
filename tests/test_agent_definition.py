from unittest.mock import patch
from fastapi.testclient import TestClient
from rai.server import app

client = TestClient(app)

@patch("rai.config_manager.save_agent_template")
def test_define_agent(mock_save_template) -> None:
    """Tests the POST /api/v1/agents/ endpoint for defining a new agent."""
    mock_save_template.return_value = True
    
    agent_data = {
        "name": "python-expert",
        "model": "gemma2:2b",  # Using smaller model as requested
        "backend": "ollama",
        "system_prompt": "You are a Python expert.",
        "tools": ["PythonTools", "FileTools"],
        "description": "Expert in Python coding"
    }
    
    response = client.post("/api/v1/agents/", json=agent_data)
    assert response.status_code == 201
    assert response.json()["message"] == "Agent 'python-expert' successfully defined."
    
    # Verify save was called with correct data (system_prompt mapped to system)
    expected_data = agent_data.copy()
    expected_data["system"] = expected_data.pop("system_prompt")
    mock_save_template.assert_called_once_with(expected_data)

def test_define_agent_invalid_tool() -> None:
    """Tests defining an agent with a tool that doesn't exist."""
    agent_data = {
        "name": "bad-agent",
        "model": "gemma2:2b",
        "tools": ["NonExistentTool"]
    }
    response = client.post("/api/v1/agents/", json=agent_data)
    assert response.status_code == 422 # Pydantic validation error

@patch("rai.config_manager.save_agent_template")
def test_define_agent_save_failure(mock_save_template) -> None:
    """Tests failure during save."""
    mock_save_template.return_value = False
    
    agent_data = {
        "name": "fail-agent",
        "model": "gemma2:2b"
    }
    
    response = client.post("/api/v1/agents/", json=agent_data)
    assert response.status_code == 500
    assert response.json()["detail"] == "Error saving configuration"
