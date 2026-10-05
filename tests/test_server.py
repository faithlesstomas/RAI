"""Tests for the FastAPI server in src/rai/server.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from returns.result import Success

from conftest import ASGITestClient
from rai.dependencies import get_config
from rai.routers.history import clear_session_history
from rai.server import app


# Mock configuration dependency
def mock_get_config() -> dict[str, object]:
    return {"test_mode": True}


app.dependency_overrides[get_config] = mock_get_config


def test_health_check(client: ASGITestClient) -> None:
    """Tests the GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_approvals_endpoint(client: ASGITestClient) -> None:
    """Tests the GET /api/v1/approvals endpoint."""
    response = client.get("/api/v1/approvals")
    assert response.status_code == 200
    assert "approvals" in response.json()


def test_resolve_approval_not_found(client: ASGITestClient) -> None:
    """Tests resolving an unknown approval request returns 404."""
    response = client.post(
        "/api/v1/approvals/non-existent-id/resolve",
        json={"approved": True},
    )
    assert response.status_code == 404


def test_get_models_endpoint(client: ASGITestClient) -> None:
    """Tests the GET /api/v1/models endpoint."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert "models" in response.json()


def test_get_models_for_backend_endpoint(client: ASGITestClient) -> None:
    """Tests the GET /api/v1/models/{backend} endpoint."""
    response = client.get("/api/v1/models/default")
    assert response.status_code == 200
    assert "models" in response.json()


@pytest.mark.asyncio
async def test_clear_session_history_success() -> None:
    """Tests the DELETE /api/v1/history/sessions/{session_id} endpoint."""
    mock_history_service = MagicMock()
    mock_history_service.clear_history = AsyncMock(return_value=Success(None))

    response = await clear_session_history("test-session", mock_history_service)
    assert response == {
        "status": "success",
        "message": "History cleared for session 'test-session'.",
    }
    mock_history_service.clear_history.assert_awaited_once_with("test-session")


@pytest.mark.asyncio
async def test_server_lifespan_closes_dependencies() -> None:
    """Test that application lifespan closes dependencies during shutdown."""
    with patch("rai.server.close_dependencies", new_callable=AsyncMock) as mock_close:
        async with app.router.lifespan_context(app):
            pass
        mock_close.assert_awaited_once()
