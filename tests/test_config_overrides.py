import pytest
from unittest.mock import patch
from rai.services.chat import ChatService

def test_resolve_agent_config_merging() -> None:
    """Test that ChatService._resolve_agent_config merges template config and chain_configs overrides."""
    chat_service = ChatService()

    # Base agent template config returned by load_agents
    base_agents = {
        "test-agent": {
            "name": "test-agent",
            "model": "base-model",
            "backend": "gemini",
            "system": "base system instructions",
            "tools": ["ToolA"]
        }
    }

    # Client-provided overrides
    chain_configs = [{
        "model": "override-model",
        "system": "override system instructions"
    }]

    with patch("rai.services.chat.load_agents", return_value=base_agents):
        # Resolve config with agent_id and overrides
        resolved = chat_service._resolve_agent_config(
            chain_configs=chain_configs,
            agent_id="test-agent"
        )

        # Assertions
        assert resolved["name"] == "test-agent"
        assert resolved["model"] == "override-model"  # Overridden!
        assert resolved["backend"] == "gemini"        # Kept from base!
        assert resolved["system"] == "override system instructions"  # Overridden!
        assert resolved["tools"] == ["ToolA"]         # Kept from base!


def test_resolve_agent_config_no_agent_id_merging() -> None:
    """Test that ChatService._resolve_agent_config merges active agent config and overrides when agent_id is None."""
    chat_service = ChatService()

    # Base active config returned by load_config
    app_config = {
        "agents": {
            "active-session": {
                "name": "active-session",
                "model": "base-model",
                "backend": "ollama",
                "system": "active system instructions",
                "tools": ["ToolB"]
            }
        },
        "active_agent": "active-session"
    }

    chain_configs = [{
        "model": "override-model",
        "backend": "gemini"
    }]

    with patch("rai.services.chat.load_config", return_value=app_config):
        resolved = chat_service._resolve_agent_config(
            chain_configs=chain_configs,
            agent_id=None
        )

        assert resolved["name"] == "active-session"
        assert resolved["model"] == "override-model"  # Overridden!
        assert resolved["backend"] == "gemini"        # Overridden!
        assert resolved["system"] == "active system instructions"  # Kept from base!
        assert resolved["tools"] == ["ToolB"]         # Kept from base!
