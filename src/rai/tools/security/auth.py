"""Authentication helpers for the local RAI control API."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path
from typing import Mapping, Optional

from rai.paths import runtime_dir

TOKEN_ENV = "RAI_API_TOKEN"  # noqa: S105 -- environment variable name
TOKEN_HEADER = "X-RAI-Token"  # noqa: S105 -- HTTP header name
SESSION_TOKEN_HEADER = "X-Session-Token"  # noqa: S105 -- HTTP header name
DEFAULT_SESSION_TOKEN_TTL = 86400


def auth_disabled() -> bool:
    """Return whether authentication was explicitly disabled for tests/dev."""
    return os.environ.get("RAI_DISABLE_AUTH", "").lower() in {"1", "true", "yes"}


def _token_path() -> Path:
    return runtime_dir() / "api-token"


def get_api_token() -> str:
    """Read or create the per-user token protecting RAI's control API."""
    configured = os.environ.get(TOKEN_ENV)
    if configured:
        return configured

    token_path = _token_path()
    token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        token_path.parent.chmod(0o700)
    except OSError:
        pass

    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            file_descriptor = os.open(
                token_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            token = token_path.read_text(encoding="utf-8").strip()
        else:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as token_file:
                token_file.write(token)
    if not token:
        raise RuntimeError(f"RAI API token file is empty: {token_path}")
    return token


def authorization_headers() -> dict[str, str]:
    """Return headers suitable for trusted local RAI clients."""
    return {TOKEN_HEADER: get_api_token()}


def is_authorized(
    headers: Mapping[str, str], query_token: Optional[str] = None
) -> bool:
    """Validate a request header or WebSocket query token."""
    if auth_disabled():
        return True
    supplied = headers.get(TOKEN_HEADER) or headers.get(TOKEN_HEADER.lower())
    if not supplied:
        authorization = headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
    supplied = supplied or query_token
    return bool(supplied) and hmac.compare_digest(supplied, get_api_token())


def create_session_token(
    session_id: str, ttl_seconds: int = DEFAULT_SESSION_TOKEN_TTL
) -> str:
    """Create a tamper-proof signed session token for an authorized session ID with TTL."""
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{session_id}:{expires_at}"
    secret = get_api_token().encode("utf-8")
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{session_id}.{expires_at}.{sig}"


def verify_session_token(token: str | None) -> str | None:
    """Validate a signed session token, enforce expiration, and return verified session_id or None."""
    if not token or "." not in token:
        return None
    secret = get_api_token().encode("utf-8")

    # 1. Try 3-part format: {session_id}.{expires_at}.{signature}
    # Using rsplit(".", 2) safely preserves dots in session_id (e.g. "work.v2")
    parts3 = token.rsplit(".", 2)
    if len(parts3) == 3 and parts3[1].isdigit():  # noqa: PLR2004
        session_id, exp_str, sig = parts3
        expires_at = int(exp_str)
        if time.time() > expires_at:
            return None
        payload = f"{session_id}:{exp_str}"
        expected_sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected_sig):
            return session_id

    # 2. Try legacy 2-part format: {session_id}.{signature}
    parts2 = token.rsplit(".", 1)
    if len(parts2) == 2:  # noqa: PLR2004
        session_id, sig = parts2
        expected_sig = hmac.new(secret, session_id.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected_sig):
            return session_id

    return None
