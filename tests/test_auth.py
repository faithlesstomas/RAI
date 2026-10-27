"""Security boundary tests for the local control API."""

import hashlib
import hmac
import stat
from pathlib import Path

import pytest

from rai.tools.security.auth import (
    TOKEN_HEADER,
    create_session_token,
    get_api_token,
    is_authorized,
    verify_session_token,
)
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


def test_session_token_signing_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAI_API_TOKEN", "test-secret-token")
    token = create_session_token("session-123", ttl_seconds=3600)
    assert verify_session_token(token) == "session-123"

    # Expired token
    expired_token = create_session_token("session-123", ttl_seconds=-10)
    assert verify_session_token(expired_token) is None

    # Tampered token
    parts = token.split(".")
    tampered_sig = f"{parts[0]}.{parts[1]}.bad_signature"
    assert verify_session_token(tampered_sig) is None
    tampered_session = f"session-victim.{parts[1]}.{parts[2]}"
    assert verify_session_token(tampered_session) is None

    # Invalid format
    assert verify_session_token("invalid") is None
    assert verify_session_token("") is None
    assert verify_session_token(None) is None


def test_session_token_with_dots_in_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret-token"
    monkeypatch.setenv("RAI_API_TOKEN", secret)

    # 1. 3-part token for session ID with dots
    token_dotted = create_session_token("work.v2", ttl_seconds=3600)
    assert verify_session_token(token_dotted) == "work.v2"

    token_multi_dot = create_session_token("corp.project.sub.session-1", ttl_seconds=3600)
    assert verify_session_token(token_multi_dot) == "corp.project.sub.session-1"

    # 2. Legacy 2-part token for session ID with dots
    legacy_sig = hmac.new(
        secret.encode("utf-8"), "work.v2".encode("utf-8"), hashlib.sha256
    ).hexdigest()
    legacy_token = f"work.v2.{legacy_sig}"
    assert verify_session_token(legacy_token) == "work.v2"

    # 3. Tampered token for session ID with dots
    parts = token_dotted.rsplit(".", 2)
    tampered = f"evil.session.{parts[1]}.{parts[2]}"
    assert verify_session_token(tampered) is None

    # 4. Expired token for session ID with dots
    expired = create_session_token("work.v2", ttl_seconds=-5)
    assert verify_session_token(expired) is None
