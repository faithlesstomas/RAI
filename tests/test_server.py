"""
Tests for the FastAPI server in src/rai/server.py.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from returns.result import Success, Failure

from rai.server import app
from rai.dependencies import get_config

# Mock configuration dependency
def mock_get_config() -> dict:
    return {"test_mode": True}

app.dependency_overrides[get_config] = mock_get_config

client = TestClient(app)


def test_health_check() -> None:
    """
    Tests the GET /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200 # noqa: PLR2004
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_execute_chain_success(MockChatService) -> None: # noqa: ANN001
    """
    Tests the POST /api/v1/run endpoint for a successful execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Success({"result": "mocked_success"}))

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 200 # noqa: PLR2004
    assert response.json() == {
        "status": "success",
        "payload": {"result": "mocked_success"}
    }
    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_execute_chain_failure(MockChatService) -> None: # noqa: ANN001
    """
    Tests the POST /api/v1/run endpoint for a failed execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Failure(Exception("Something went wrong")))

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 500 # noqa: PLR2004
    assert response.json() == {"detail": "Something went wrong"}
    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_execute_chain_arun_exception(MockChatService) -> None: # noqa: ANN001
    """
    Tests that a Failure from ChatService is handled correctly.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Failure(Exception("Adapter crashed")))

    request_payload = {
        "chain_input": "test input",
        "chain_configs": [{"agent_class": "AgentAgno"}]
    }

    response = client.post("/api/v1/run", json=request_payload)

    assert response.status_code == 500 # noqa: PLR2004
    assert "Adapter crashed" in response.json()["detail"]
    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_websocket_endpoint_success(MockChatService) -> None: # noqa: ANN001
    """
    Tests the /ws/v1/chat WebSocket endpoint for a successful execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Success({"result": "ws_success"}))

    with client.websocket_connect("/ws/v1/chat") as websocket:
        request_payload = {
            "chain_input": "ws test",
            "chain_configs": [{"agent_class": "AgentAgno"}]
        }
        websocket.send_json(request_payload)
        response = websocket.receive_json()
        assert response == {"type": "response", "payload": {"result": "ws_success"}}

    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_websocket_endpoint_failure(MockChatService) -> None: # noqa: ANN001
    """
    Tests the /ws/v1/chat WebSocket endpoint for a failed execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Failure(Exception("WS went wrong")))

    with client.websocket_connect("/ws/v1/chat") as websocket:
        request_payload = {
            "chain_input": "ws test fail",
            "chain_configs": [{"agent_class": "AgentAgno"}]
        }
        websocket.send_json(request_payload)
        response = websocket.receive_json()
        assert response == {"type": "error", "detail": "WS went wrong"}
    
    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_invalid_request() -> None:
    """
    Tests the /ws/v1/chat WebSocket endpoint for an invalid request.
    """
    with client.websocket_connect("/ws/v1/chat") as websocket:
        websocket.send_json({"invalid_key": "some_value"})  # Missing required fields
        response = websocket.receive_json()
        assert response["status"] == "error"
        assert "detail" in response
        assert isinstance(response["detail"], list)
