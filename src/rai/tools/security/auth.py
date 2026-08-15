"""Authentication helpers for the local RAI control API."""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from typing import Mapping, Optional

from rai.paths import runtime_dir

TOKEN_ENV = "RAI_API_TOKEN"
TOKEN_HEADER = "X-RAI-Token"


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

