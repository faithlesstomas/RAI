"""
Tests for the new API features: Streaming and execution.
"""
from unittest.mock import patch
from typing import Any
import pytest
from rai.routers.execution import AgentExecutionRequest, stream_chain_endpoint

@pytest.mark.asyncio
async def test_stream_endpoint() -> None:
    """Test the streaming endpoint."""
    # Mock stream_chain to yield chunks
    with patch("rai.services.chat.ChatService.stream_chain") as mock_stream:
        async def mock_generator(*args: Any, **kwargs: Any) -> Any:
            yield "Hello"
            yield " World"

        mock_stream.return_value = mock_generator()

        # Case 1: No session_id passed (should generate one)
        response = await stream_chain_endpoint(
            AgentExecutionRequest(prompt="Test input", agent_id="default"),
            app_config={},
        )

        assert response.status_code == 200
        content = "".join([chunk async for chunk in response.body_iterator])
        assert "data: {\"content\": \"Hello\"}" in content
        assert "data: {\"content\": \" World\"}" in content
        assert "event: done" in content
        # Ensure session_id is present in the done event
        assert "data: {\"session_id\": \"" in content

        # Case 2: session_id passed (should propagate it)
        response_prop = await stream_chain_endpoint(
            AgentExecutionRequest(
                prompt="Test input",
                agent_id="default",
                session_id="custom-session-123",
            ),
            app_config={},
        )
        assert response_prop.status_code == 200
        content_prop = "".join(
            [chunk async for chunk in response_prop.body_iterator]
        )
        assert "event: done" in content_prop
        assert "data: {\"session_id\": \"custom-session-123\"}" in content_prop
