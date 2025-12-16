import pytest
from unittest.mock import AsyncMock, patch

from rai.adapters.pydantic_ai import PydanticAIAdapter


@pytest.mark.asyncio
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

    # We patch the classes that are now used in the adapter
    with patch('rai.adapters.pydantic_ai.OllamaProvider') as mock_ollama_provider_class, \
         patch('rai.adapters.pydantic_ai.OpenAIChatModel') as mock_openai_chat_model_class, \
         patch('rai.adapters.pydantic_ai.Agent', return_value=mock_agent_instance) as mock_agent_class:

        # 3. Instantiate our adapter with the config and run it
        adapter = PydanticAIAdapter(agent_config=agent_config)
        result = await adapter.arun(prompt="Hello", session_id="test-session")

        # 4. Assert the results
        # Check that the provider and model were created correctly
        mock_ollama_provider_class.assert_called_once_with(base_url='http://localhost:11434/v1')
        mock_openai_chat_model_class.assert_called_once_with(
            model_name='test-model',
            provider=mock_ollama_provider_class.return_value
        )

        # Assert that the Agent was called with the model instance
        mock_agent_class.assert_called_once_with(
            mock_openai_chat_model_class.return_value,
            tools=[]
        )

        from returns.result import Success
        
        # Assert the final result
        assert isinstance(result, Success)
        assert result.unwrap()["content"] == "Test AI response"
        mock_agent_instance.run.assert_called_once_with("Hello")

