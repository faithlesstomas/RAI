"""
Tests for the FastAPI server in src/rai/server.py.
"""
import json
import pytest
from fastapi import HTTPException, WebSocketDisconnect
from unittest.mock import AsyncMock, MagicMock, patch
from returns.result import Success, Failure

from rai.server import app
from rai.dependencies import get_config
from rai.routers.execution import (
    AgentExecutionRequest,
    execute_chain,
    websocket_endpoint,
)
from rai.routers.history import clear_session_history
from conftest import ASGITestClient

# Mock configuration dependency
def mock_get_config() -> dict:
    return {"test_mode": True}

app.dependency_overrides[get_config] = mock_get_config

def test_health_check(client: ASGITestClient) -> None:
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

    response = await execute_chain(
        AgentExecutionRequest.model_validate(request_payload), app_config={}
    )

    assert response.status_code == 200 # noqa: PLR2004
    assert json.loads(response.body) == {
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

    with pytest.raises(HTTPException) as error:
        await execute_chain(
            AgentExecutionRequest.model_validate(request_payload), app_config={}
        )
    assert error.value.status_code == 500 # noqa: PLR2004
    assert error.value.detail == "Something went wrong"
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

    with pytest.raises(HTTPException) as error:
        await execute_chain(
            AgentExecutionRequest.model_validate(request_payload), app_config={}
        )
    assert error.value.status_code == 500 # noqa: PLR2004
    assert "Adapter crashed" in error.value.detail
    mock_service_instance.run_chain.assert_awaited_once()


class FakeWebSocket:
    """Minimal WebSocket double for exercising the endpoint coroutine."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.headers = {}
        self.query_params = {}
        self.client_state = "CONNECTED"
        self.sent: list[dict] = []
        self._received = False

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict:
        if self._received:
            raise WebSocketDisconnect()
        self._received = True
        return self.payload

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason
        self.client_state = "DISCONNECTED"


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_websocket_endpoint_success(MockChatService) -> None: # noqa: ANN001
    """
    Tests the /ws/v1/chat WebSocket endpoint for a successful execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Success({"result": "ws_success"}))

    websocket = FakeWebSocket({
        "chain_input": "ws test",
        "chain_configs": [{"agent_class": "AgentAgno"}],
    })
    await websocket_endpoint(websocket)
    assert websocket.sent == [
        {"type": "response", "payload": {"result": "ws_success"}}
    ]

    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
@patch("rai.routers.execution.ChatService")
async def test_websocket_endpoint_failure(MockChatService) -> None: # noqa: ANN001
    """
    Tests the /ws/v1/chat WebSocket endpoint for a failed execution.
    """
    mock_service_instance = MockChatService.return_value
    mock_service_instance.run_chain = AsyncMock(return_value=Failure(Exception("WS went wrong")))

    websocket = FakeWebSocket({
        "chain_input": "ws test fail",
        "chain_configs": [{"agent_class": "AgentAgno"}],
    })
    await websocket_endpoint(websocket)
    assert websocket.sent == [{"type": "error", "detail": "WS went wrong"}]
    
    mock_service_instance.run_chain.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_invalid_request() -> None:
    """
    Tests the /ws/v1/chat WebSocket endpoint for an invalid request.
    """
    websocket = FakeWebSocket({"invalid_key": "some_value"})
    await websocket_endpoint(websocket)
    response = websocket.sent[0]
    assert response["status"] == "error"
    assert "detail" in response
    assert isinstance(response["detail"], list)


@pytest.mark.asyncio
async def test_websocket_endpoint_rejects_missing_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "expected-token")
    websocket = FakeWebSocket({"prompt": "must not run"})

    await websocket_endpoint(websocket)

    assert websocket.client_state == "DISCONNECTED"
    assert websocket.sent == []


@pytest.mark.asyncio
async def test_clear_session_history_success() -> None:
    """
    Tests the DELETE /api/v1/history/sessions/{session_id} endpoint.
    """
    mock_history_service = MagicMock()
    mock_history_service.clear_history = AsyncMock(return_value=Success(None))
    
    response = await clear_session_history("test-session", mock_history_service)
    assert response == {
        "status": "success",
        "message": "History cleared for session 'test-session'."
    }
    mock_history_service.clear_history.assert_awaited_once_with("test-session")


@pytest.mark.asyncio
async def test_server_lifespan_closes_dependencies() -> None:
    """Test that application lifespan closes dependencies during shutdown."""
    with patch("rai.server.close_dependencies", new_callable=AsyncMock) as mock_close:
        async with app.router.lifespan_context(app):
            pass
        mock_close.assert_awaited_once()
