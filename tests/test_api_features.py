"""
Tests for the new API features: Streaming and execution.
"""
from unittest.mock import patch
from typing import Any
import pytest
from fastapi.testclient import TestClient
from rai.server import app

client = TestClient(app)

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
        response = client.post(
            "/api/v1/stream",
            json={
                "prompt": "Test input",
                "agent_id": "default"
            }
        )

        assert response.status_code == 200
        content = response.content.decode("utf-8")
        assert "data: {\"content\": \"Hello\"}" in content
        assert "data: {\"content\": \" World\"}" in content
        assert "event: done" in content
        # Ensure session_id is present in the done event
        assert "data: {\"session_id\": \"" in content

        # Case 2: session_id passed (should propagate it)
        response_prop = client.post(
            "/api/v1/stream",
            json={
                "prompt": "Test input",
                "agent_id": "default",
                "session_id": "custom-session-123"
            }
        )
        assert response_prop.status_code == 200
        content_prop = response_prop.content.decode("utf-8")
        assert "event: done" in content_prop
        assert "data: {\"session_id\": \"custom-session-123\"}" in content_prop

