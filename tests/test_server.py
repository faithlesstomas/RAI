"""
Tests for the FastAPI server in src/rai/server.py.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from rai.server import app, get_config

# Mock configuration dependency
def mock_get_config():
    return {"test_mode": True}

app.dependency_overrides[get_config] = mock_get_config

client = TestClient(app)


def test_health_check():
    """
    Tests the GET /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@patch("rai.engine._discover_adapters")
async def test_execute_chain_success(mock_discover_adapters):
    """
    Tests the POST /api/v1/run endpoint for a successful execution.
    """
    from returns.result import Success

    # Mock the adapter's arun method to return a Success object
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.arun = AsyncMock(return_value=Success({"result": "mocked_success"}))
    
    mock_adapter_class = MagicMock(return_value=mock_adapter_instance)
    mock_discover_adapters.return_value = {"agno": mock_adapter_class}

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "payload": {"result": "mocked_success"}
    }
    mock_adapter_instance.arun.assert_awaited_once_with(prompt="test input")


@pytest.mark.asyncio
@patch("rai.engine._discover_adapters")
async def test_execute_chain_failure(mock_discover_adapters):
    """
    Tests the POST /api/v1/run endpoint for a failed execution.
    """
    from returns.result import Failure

    # Mock the adapter's arun method to return a Failure object
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.arun = AsyncMock(return_value=Failure(Exception("Something went wrong")))
    
    mock_adapter_class = MagicMock(return_value=mock_adapter_instance)
    mock_discover_adapters.return_value = {"agno": mock_adapter_class}

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 500
    assert response.json() == {"detail": "Something went wrong"}
    mock_adapter_instance.arun.assert_awaited_once_with(prompt="test input")


@pytest.mark.asyncio
@patch("rai.engine._discover_adapters")
async def test_execute_chain_arun_exception(mock_discover_adapters):
    """
    Tests that an exception raised from the adapter's arun method is
    handled correctly and results in a 500 error.
    """
    # Mock the adapter's arun method to raise an exception
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.arun = AsyncMock(side_effect=Exception("Adapter crashed"))
    
    mock_adapter_class = MagicMock(return_value=mock_adapter_instance)
    mock_discover_adapters.return_value = {"agno": mock_adapter_class}

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 500
    assert "An error occurred during chain execution: Adapter crashed" in response.json()["detail"]
    mock_adapter_instance.arun.assert_awaited_once_with(prompt="test input")


@pytest.mark.asyncio
@patch("rai.engine._discover_adapters")
async def test_websocket_endpoint_success(mock_discover_adapters):
    """
    Tests the /ws/v1/chat WebSocket endpoint for a successful execution.
    """
    from returns.result import Success
    
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.arun = AsyncMock(return_value=Success({"result": "ws_success"}))
    
    mock_adapter_class = MagicMock(return_value=mock_adapter_instance)
    mock_discover_adapters.return_value = {"agno": mock_adapter_class}

    with client.websocket_connect("/ws/v1/chat") as websocket:
        request_payload = {
            "chain_input": "ws test",
            "chain_configs": [{"agent_class": "AgentAgno"}]
        }
        websocket.send_json(request_payload)
        response = websocket.receive_json()
        assert response == {"status": "success", "payload": {"result": "ws_success"}}

    mock_adapter_instance.arun.assert_awaited_once_with(prompt="ws test")


@pytest.mark.asyncio
@patch("rai.engine._discover_adapters")
async def test_websocket_endpoint_failure(mock_discover_adapters):
    """
    Tests the /ws/v1/chat WebSocket endpoint for a failed execution.
    """
    from returns.result import Failure
    
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.arun = AsyncMock(return_value=Failure(Exception("WS went wrong")))
    
    mock_adapter_class = MagicMock(return_value=mock_adapter_instance)
    mock_discover_adapters.return_value = {"agno": mock_adapter_class}

    with client.websocket_connect("/ws/v1/chat") as websocket:
        request_payload = {
            "chain_input": "ws test fail",
            "chain_configs": [{"agent_class": "AgentAgno"}]
        }
        websocket.send_json(request_payload)
        response = websocket.receive_json()
        assert response == {"status": "error", "detail": "WS went wrong"}
    
    mock_adapter_instance.arun.assert_awaited_once_with(prompt="ws test fail")


@pytest.mark.asyncio
async def test_websocket_endpoint_invalid_request():
    """
    Tests the /ws/v1/chat WebSocket endpoint for an invalid request.
    """
    with client.websocket_connect("/ws/v1/chat") as websocket:
        websocket.send_json({"invalid_key": "some_value"})  # Missing required fields
        response = websocket.receive_json()
        assert response["status"] == "error"
        assert "detail" in response
        assert isinstance(response["detail"], list)
