"""
Tests for the new API features: Streaming, Client Tools, and Dynamic Context.
"""
from unittest.mock import patch, AsyncMock, MagicMock
from typing import Any
import pytest
from fastapi.testclient import TestClient
from rai.server import app

from rai.adapters.agno import AgnoAdapter
from returns.result import Success

client = TestClient(app)

@pytest.mark.asyncio
async def test_stream_endpoint() -> None:
    """Test the streaming endpoint."""
    # Mock stream_chain to yield chunks
    with patch("rai.routers.execution.stream_chain") as mock_stream:
        async def mock_generator(*args: Any, **kwargs: Any) -> Any: # noqa: ANN401
            yield "Hello"
            yield " World"

        mock_stream.return_value = mock_generator()

        response = client.post(
            "/api/v1/stream",
            json={
                "chain_input": "Test input",
                "chain_configs": [{"agent_class": "AgentAgno"}]
            }
        )

        assert response.status_code == 200 # noqa: PLR2004
        content = response.content.decode("utf-8")
        assert "data: {\"content\": \"Hello\"}" in content
        assert "data: {\"content\": \" World\"}" in content
        assert "event: done" in content

@pytest.mark.asyncio
async def test_client_tool_parsing() -> None:
    """Test parsing of client tool calls."""

    # Mock Agent
    mock_agent = MagicMock()
    mock_agent.arun = AsyncMock()

    # Mock response with client tool call string
    mock_response = MagicMock()
    mock_response.content = "Some text __CLIENT_TOOL_CALL__:{\"tool\": \"eval_scheme\", \"code\": \"(+ 1 2)\"}"
    mock_response.tool_calls = None
    mock_agent.arun.return_value = mock_response

    # Initialize adapter with session_id to properly populate cache
    adapter = AgnoAdapter({"agent_class": "AgentAgno", "session_id": "test-session"})
    # Inject mock agent into the cache directly
    adapter.agents["test-session"] = mock_agent

    result = await adapter.arun("Run code")

    assert isinstance(result, Success)
    payload = result.unwrap()
    assert payload["tool_calls"] is not None
    assert len(payload["tool_calls"]) == 1
    assert payload["tool_calls"][0]["function"]["name"] == "eval_scheme"
    assert "code" in payload["tool_calls"][0]["function"]["arguments"]

@pytest.mark.asyncio
async def test_dynamic_context() -> None:
    """Test injection of dynamic context."""

    config = {
        "agent_class": "AgentAgno",
        "context": {"file_content": "some content"},
        "session_id": "test-session"
    }

    # We need to mock validate_model_env and setup_tools to avoid actual initialization
    # Also need to mock _instantiate_model since we moved it to adapter
    with patch("rai.adapters.agno.validate_model_env") as mock_validate, \
         patch("rai.adapters.agno.setup_tools") as mock_setup_tools, \
         patch("rai.adapters.agno.Agent") as mock_agent_cls, \
         patch.object(AgnoAdapter, "_instantiate_model") as mock_instantiate:

        mock_validate.return_value = ({}, [])
        mock_setup_tools.return_value = ([], [])
        mock_instantiate.return_value = MagicMock()

        adapter = AgnoAdapter(config)
        # trigger agent creation
        adapter._get_or_create_agent("test-session")

        # Check if context was added to system prompt
        # We need to inspect the call to Agent constructor
        call_args = mock_agent_cls.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "instructions" in kwargs
        assert "Context:" in kwargs["instructions"]
        assert "some content" in kwargs["instructions"]
