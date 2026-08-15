"""Hermetic defaults shared by the RAI test suite."""

import os
import tempfile

import asyncio
from typing import Any

import httpx
import pytest
from starlette.types import ASGIApp


_TEST_ROOT = tempfile.mkdtemp(prefix="rai-tests-")
os.environ.setdefault("RAI_CONFIG_DIR", os.path.join(_TEST_ROOT, "config"))
os.environ.setdefault("RAI_CACHE_DIR", os.path.join(_TEST_ROOT, "cache"))
os.environ.setdefault("RAI_DATA_DIR", os.path.join(_TEST_ROOT, "data"))
os.environ.setdefault("RAI_RUNTIME_DIR", os.path.join(_TEST_ROOT, "run"))
os.environ.setdefault("RAI_DISABLE_AUTH", "1")

from rai.server import app as rai_app  # noqa: E402


class ASGITestClient:
    """Small synchronous client without TestClient's worker-thread portal."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def _request(
        self, method: str, path: str, **kwargs: Any  # noqa: ANN401
    ) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    def request(
        self, method: str, path: str, **kwargs: Any  # noqa: ANN401
    ) -> httpx.Response:
        return asyncio.run(self._request(method, path, **kwargs))

    def get(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        return self.request("DELETE", path, **kwargs)


@pytest.fixture
def client() -> ASGITestClient:
    """Return a deterministic in-process HTTP client for the RAI app."""
    return ASGITestClient(rai_app)
