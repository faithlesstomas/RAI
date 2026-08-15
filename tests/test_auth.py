"""Security boundary tests for the local control API."""

import stat
from pathlib import Path

import pytest

from rai.tools.security.auth import TOKEN_HEADER, get_api_token, is_authorized
from conftest import ASGITestClient

MINIMUM_TOKEN_LENGTH = 32
PRIVATE_FILE_MODE = 0o600


def test_api_token_is_persisted_with_private_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAI_API_TOKEN", raising=False)
    monkeypatch.setenv("RAI_RUNTIME_DIR", str(tmp_path))

    first = get_api_token()
    second = get_api_token()

    assert first == second
    assert len(first) >= MINIMUM_TOKEN_LENGTH
    token_file = tmp_path / "api-token"
    assert stat.S_IMODE(token_file.stat().st_mode) == PRIVATE_FILE_MODE


def test_authorization_uses_constant_token_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "expected-token")

    assert is_authorized({TOKEN_HEADER: "expected-token"})
    assert is_authorized({"authorization": "Bearer expected-token"})
    assert not is_authorized({TOKEN_HEADER: "wrong-token"})
    assert not is_authorized({})


def test_control_api_rejects_missing_token(
    monkeypatch: pytest.MonkeyPatch, client: ASGITestClient
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "expected-token")

    response = client.get("/api/v1/agents/")

    assert response.status_code == 401  # noqa: PLR2004
    assert response.json() == {"detail": "Unauthorized"}
