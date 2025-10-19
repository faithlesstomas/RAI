import pytest
from unittest.mock import patch, AsyncMock

from rai.adapters.pydantic_ai import PydanticAIAdapter


async def test_pydantic_ai_adapter_arun() -> None:
    """
    Tests that the PydanticAIAdapter correctly initializes and calls
    the pydantic_ai.Agent and returns the expected payload.
    """
    # 1. Define the config that will be passed to the adapter
    agent_config = {
        "backend": "ollama",
        "model": "test-model"
    }

    # 2. Mock the dependencies
    mock_agent_instance = AsyncMock()
    mock_response = AsyncMock()
    mock_response.output = "Test AI response"
    mock_agent_instance.run.return_value = mock_response

    # We patch the Agent class itself to control its instance
    with patch('rai.adapters.pydantic_ai.Agent', return_value=mock_agent_instance) as mock_agent_class:
        # 3. Instantiate our adapter with the config and run it
        adapter = PydanticAIAdapter(agent_config)
        result = await adapter.arun(prompt="Hello", session_id="test-session")

        # 4. Assert the results
        # Check that the Pydantic AI Agent was initialized correctly
        mock_agent_class.assert_called_once_with("ollama/test-model")

        # Check that the agent's run method was called
        mock_agent_instance.run.assert_called_once_with("Hello")

        # Check that our adapter returned the correct payload
        assert result == {
            "content": "Test AI response",
            "tool_calls": None
        }

