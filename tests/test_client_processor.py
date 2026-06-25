"""Tests for ClientProcessor in rai.cli."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from returns.result import Success, Failure

from rai.cli import ClientProcessor

@pytest.fixture
def run_config() -> dict:
    return {
        "model": "test-model",
        "backend": "gemini",
        "system": "test system prompt",
        "tools": ["ToolA"]
    }

@pytest.mark.asyncio
async def test_client_processor_connect_success(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(processor.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        result = await processor.connect()
        
        assert isinstance(result, Success)
        mock_get.assert_awaited_once_with("/api/v1/models")

@pytest.mark.asyncio
async def test_client_processor_connect_failure(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    with patch.object(processor.client, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        with patch("rai.cli.error_console.print") as mock_print:
            result = await processor.connect()
            
            assert isinstance(result, Failure)
            mock_print.assert_called_once()

@pytest.mark.asyncio
async def test_client_processor_arun_success(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "status": "success",
        "payload": {
            "session_id": "new-session-id",
            "result": "response text"
        }
    }
    
    with patch.object(processor.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await processor.arun("hello")
        
        assert isinstance(result, Success)
        assert result.unwrap() == {"session_id": "new-session-id", "result": "response text"}
        assert processor.stateful_session_id == "new-session-id"
        mock_post.assert_awaited_once_with(
            "/api/v1/run",
            json={
                "prompt": "hello",
                "agent_id": "session-id",
                "session_id": None,
                "chain_configs": [run_config]
            }
        )

@pytest.mark.asyncio
async def test_client_processor_sync_config(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(processor.client, "put", new_callable=AsyncMock) as mock_put:
        mock_put.return_value = mock_response
        await processor.sync_config()
        
        mock_put.assert_awaited_once_with(
            "/api/v1/agents/session-id",
            json={
                "model": "test-model",
                "backend": "gemini",
                "system": "test system prompt",
                "tools": ["ToolA"]
            }
        )

@pytest.mark.asyncio
async def test_client_processor_reload(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    with patch.object(processor, "sync_config", new_callable=AsyncMock) as mock_sync:
        await processor.reload()
        mock_sync.assert_awaited_once()

@pytest.mark.asyncio
async def test_client_processor_close(run_config: dict) -> None:
    processor = ClientProcessor(run_config, "session-id", "http://localhost:8000")
    
    with patch.object(processor.client, "aclose", new_callable=AsyncMock) as mock_aclose:
        await processor.close()
        mock_aclose.assert_awaited_once()
