import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from returns.result import Success
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
        "model": "test-model",
        "enable_tools": False
    }

    # 2. Mock the dependencies
    mock_agent_instance = AsyncMock()
    mock_response = MagicMock()
    mock_response.output = "Test AI response"
    # new_messages is a sync method, so we set return_value, not side_effect or async definition
    mock_response.new_messages.return_value = []
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
            tools=[],
            system_prompt="You are a helpful AI assistant."
        )



        # Assert the final result
        assert isinstance(result, Success)
        assert result.unwrap()["content"] == "Test AI response"
        mock_agent_instance.run.assert_called_once_with("Hello")


@pytest.mark.asyncio
async def test_pydantic_ai_adapter_history_injection() -> None:
    """Test that history is correctly injected into the prompt."""
    agent_config = {
        "backend": "ollama",
        "model": "test-model",
        "enable_tools": False # Vital to avoid setup_tools call
    }
    mock_agent_instance = AsyncMock()
    mock_response = MagicMock(output="Response")
    mock_response.new_messages.return_value = []
    mock_agent_instance.run = AsyncMock(return_value=mock_response)

    with patch('rai.adapters.pydantic_ai.Agent', return_value=mock_agent_instance):
        with patch('rai.adapters.pydantic_ai.OllamaProvider'), \
             patch('rai.adapters.pydantic_ai.OpenAIChatModel'):
            adapter = PydanticAIAdapter(agent_config=agent_config)
            history = [
                {"role": "user", "content": "HI"},
                {"role": "assistant", "content": "HELLO"}
            ]

            await adapter.arun(prompt="PROMPT", history=history)

            args, _ = mock_agent_instance.run.call_args
            prompt = args[0]
            assert "HI" in prompt
            assert "HELLO" in prompt
            assert "PROMPT" in prompt


@pytest.mark.asyncio
async def test_pydantic_ai_adapter_tools_conversion() -> None:
    """Test conversion of Agno tools to Pydantic AI tools."""
    # Create a mock Agno tool with a public method
    class MockAgnoTool:
        def my_tool_method(self, x: int) -> int:
            return x * 2
        def _private_method(self) -> None:
            pass

    mock_tool_instance = MockAgnoTool()

    # Patch setup_tools to return our mock
    # We must patch where it is IMPORTED in the adapter module
    with patch('rai.adapters.pydantic_ai.setup_tools', return_value=([mock_tool_instance], [])):
        with patch('pydantic_ai.Tool') as MockTool:
            # We mock Agent to avoid real init
            with patch('rai.adapters.pydantic_ai.Agent') as MockAgent:
                 with patch('rai.adapters.pydantic_ai.OllamaProvider'), \
                      patch('rai.adapters.pydantic_ai.OpenAIChatModel'):
                    config = {"tools": ["mock_tool"], "backend": "ollama", "model": "m"}
                    adapter = PydanticAIAdapter(config)

                    # Trigger agent creation
                    adapter._get_or_create_agent("s1")

                    # Verify API called with pydantic tools
                    args, kwargs = MockAgent.call_args
                    tools_arg = kwargs.get('tools')

                    # The logic extracts "my_tool_method"
                    assert len(tools_arg) == 1
                    # It should be the method itself if no conflict
                    assert tools_arg[0] == mock_tool_instance.my_tool_method


@pytest.mark.asyncio
async def test_local_backend_init() -> None:
    """Test initialization with local backend."""
    config = {"backend": "local", "model": "test.gguf", "enable_tools": False}

    with patch('rai.inference.load_local_model') as mock_load, \
         patch('rai.inference.bridges.LocalPydanticModel') as MockLocalModel, \
         patch('rai.adapters.pydantic_ai.setup_tools', return_value=([], [])), \
         patch('rai.adapters.pydantic_ai.Agent') as MockAgent:

        mock_load.return_value = Success("mock_engine")

        adapter = PydanticAIAdapter(config)
        agent = adapter._get_or_create_agent("session_local")

        mock_load.assert_called_once_with("test.gguf")
        MockLocalModel.assert_called_once()
        # Verify Agent was initialized with the local model
        MockAgent.assert_called_once()
        args, _ = MockAgent.call_args
        assert args[0] == MockLocalModel.return_value
        assert agent == MockAgent.return_value


@pytest.mark.asyncio
async def test_adapter_utilities() -> None:
    """Test reload, clear_history, astream."""
    adapter = PydanticAIAdapter({})
    adapter.agents["old"] = MagicMock()

    adapter.reload()
    assert len(adapter.agents) == 0

    adapter.agents["old"] = MagicMock()
    adapter.clear_history() # calls reload
    assert len(adapter.agents) == 0

    # Test astream (wrapper around arun)
    with patch.object(adapter, 'arun', new_callable=AsyncMock) as mock_arun:
        mock_arun.return_value = Success({"content": "streamed"})

        chunks = [c async for c in adapter.astream("hi")]
        assert chunks == ["streamed"]
