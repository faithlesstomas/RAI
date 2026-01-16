
import pytest
from unittest.mock import AsyncMock, MagicMock
from rai.adapters.ws_client import WebSocketAdapter
from returns.result import Success, Failure

class TestWebSocketAdapter:
    @pytest.mark.asyncio
    async def test_reload(self):
        """Test that reload keeps the adapter in a valid state (noop)."""
        adapter = WebSocketAdapter({})
        # Should not raise exception
        adapter.reload()

    @pytest.mark.asyncio
    async def test_astream_success(self):
        """Test that astream falls back to arun and yields content."""
        adapter = WebSocketAdapter({})
        
        # Mock arun to return Success
        adapter.arun = AsyncMock(return_value=Success({"content": "streamed content"}))
        
        chunks = []
        async for chunk in adapter.astream("test prompt"):
            chunks.append(chunk)
            
        assert chunks == ["streamed content"]
        adapter.arun.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_astream_failure(self):
        """Test that astream yields error message on failure."""
        adapter = WebSocketAdapter({})
        
        # Mock arun to return Failure
        error = Exception("Network error")
        adapter.arun = AsyncMock(return_value=Failure(error))
        
        chunks = []
        async for chunk in adapter.astream("test prompt"):
            chunks.append(chunk)
            
        assert chunks == ["Error: Network error"]

