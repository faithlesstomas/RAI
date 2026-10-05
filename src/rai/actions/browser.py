"""Stable browser results and an explicit, isolated Chromium debugging adapter."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, DataClass, RiskClass, _utc_now
from .capabilities import HANDLE_TTL, PRODUCER, failure, result
from .handles import SQLiteHandleStore
from .records import ResourceHandle

MAX_URL_CHARACTERS = 4096
MAX_PAGE_CHARACTERS = 16000
MAX_SEARCH_RESULTS = 5


def web_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                and not parsed.username and not parsed.password and len(value) <= MAX_URL_CHARACTERS)
    except ValueError:
        return False


class BrowserBackend(Protocol):
    async def search(self, query: str, token: CancellationToken) -> Result[tuple[dict, ...], str]: ...
    async def open(self, url: str, token: CancellationToken) -> Result[dict, str]: ...
    async def read(self, url: str, token: CancellationToken) -> Result[dict, str]: ...


class CdpBrowserBackend:
    """Only explicit loopback endpoints; never attach to an ambient user profile."""

    def __init__(self, endpoint: str | None = None) -> None:
        self.endpoint = endpoint
        if endpoint:
            if not isinstance(endpoint, str):
                raise ValueError("browser endpoint must be a string")
            parsed = urlsplit(endpoint)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("browser endpoint must contain only an origin")
            if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port or parsed.path not in {"", "/"}:
                raise ValueError("browser endpoint must be an explicit IPv4 loopback HTTP origin")

    async def _command(self, websocket: str, method: str, parameters: dict) -> dict:
        from websockets.asyncio.client import connect  # noqa: PLC0415

        expected = urlsplit(self.endpoint)
        actual = urlsplit(websocket)
        if actual.scheme != "ws" or actual.hostname != expected.hostname or actual.port != expected.port:
            raise ValueError("unexpected browser websocket authority")
        async with connect(websocket, max_size=1024 * 1024, open_timeout=3) as connection:
            await connection.send(json.dumps({"id": 1, "method": method, "params": parameters}))
            async def receive_result() -> dict:
                async for message in connection:
                    response = json.loads(message)
                    if response.get("id") == 1:
                        if "error" in response:
                            raise ValueError("browser command failed")
                        return response["result"]
                raise ValueError("browser disconnected")
            return await asyncio.wait_for(receive_result(), timeout=5)
        raise ValueError("browser disconnected")

    async def _targets(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            async with client.stream("GET", f"{self.endpoint}/json/list") as response:
                response.raise_for_status()
                content = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    content.extend(chunk)
                    if len(content) > 1024 * 1024:
                        raise ValueError("browser target list exceeds limit")
                return json.loads(content)

    async def search(self, query: str, token: CancellationToken) -> Result[tuple[dict, ...], str]:
        if token.cancelled:
            return Failure("CANCELLED")
        try:
            from ddgs import DDGS  # noqa: PLC0415

            def fetch() -> tuple[dict, ...]:
                with DDGS(timeout=5) as search:
                    return tuple({"name": str(item.get("title", ""))[:256], "url": item["href"]}
                                 for item in search.text(query, max_results=MAX_SEARCH_RESULTS)
                                 if web_url(item.get("href", "")))[:MAX_SEARCH_RESULTS]
            return Success(await asyncio.to_thread(fetch))
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("BROWSER_SEARCH_UNAVAILABLE")

    async def open(self, url: str, token: CancellationToken) -> Result[dict, str]:
        if not self.endpoint:
            return Failure("BROWSER_NOT_CONFIGURED")
        try:
            async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
                response = await client.put(f"{self.endpoint}/json/new?about:blank")
                response.raise_for_status()
                page = response.json()
            websocket = page["webSocketDebuggerUrl"]
            navigated = await self._command(websocket, "Page.navigate", {"url": url})
            if navigated.get("errorText"):
                return Failure("NAVIGATION_FAILED")
            for _ in range(30):
                if token.cancelled:
                    await self._command(websocket, "Page.stopLoading", {})
                    return Failure("UNKNOWN")
                tree = await self._command(websocket, "Page.getFrameTree", {})
                frame = tree["frameTree"]["frame"]
                if frame["url"] == url:
                    return Success({"target_id": page["id"], "observed_url": frame["url"], "frame_id": frame["id"]})
                await asyncio.sleep(0.1)
            return Failure("POSTCONDITION_FAILED")
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("BROWSER_UNAVAILABLE")

    async def read(self, url: str, token: CancellationToken) -> Result[dict, str]:
        if not self.endpoint:
            return Failure("BROWSER_NOT_CONFIGURED")
        try:
            pages = [page for page in await self._targets() if page.get("type") == "page" and page.get("url") == url]
            if len(pages) != 1:
                return Failure("AMBIGUOUS_OR_MISSING_TAB")
            if token.cancelled:
                return Failure("CANCELLED")
            page = pages[0]
            snapshot = await self._command(page["webSocketDebuggerUrl"], "Runtime.evaluate", {
                "expression": "JSON.stringify({url:location.href,title:document.title,text:(document.body?.innerText||'').slice(0,16000)})",
                "returnByValue": True,
            })
            content = json.loads(snapshot["result"]["value"])
            if content["url"] != url:
                return Failure("STALE_RESOURCE")
            return Success({"target_id": page["id"], "observed_url": url, "title": content["title"][:256],
                            "text": content["text"][:MAX_PAGE_CHARACTERS], "untrusted_content": True})
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("BROWSER_READ_FAILED")


class BrowserCapability:
    def __init__(self, name: str, handles: SQLiteHandleStore, backend: BrowserBackend) -> None:
        self.name, self.handles, self.backend = name, handles, backend

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:  # noqa: PLR0911
        if request.data_class != DataClass.PUBLIC:
            return Failure(failure(request, "PUBLIC_CONTEXT_REQUIRED"))
        if self.name == "browser.search":
            query = request.arguments["query"]
            if not query.strip() or len(query) > 256:  # noqa: PLR2004
                return Failure(failure(request, "INVALID_ARGUMENT"))
            found = await self.backend.search(query, cancellation)
            if isinstance(found, Failure):
                return Failure(failure(request, found.failure()))
            entries = []
            for item in found.unwrap()[:MAX_SEARCH_RESULTS]:
                if not web_url(item["url"]):
                    continue
                now = _utc_now()
                handle = ResourceHandle(producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], kind="url", target=item["url"],
                                        fingerprint=hashlib.sha256(item["url"].encode()).hexdigest(),
                                        operations=("browser.open_result", "browser.read_page"), expires_at=now + HANDLE_TTL,
                                        data_class=request.data_class)
                issued = self.handles.issue(handle)
                if isinstance(issued, Failure):
                    return Failure(failure(request, issued.failure()))
                entries.append({"name": item["name"], "url": item["url"], "handle": issued.unwrap()})
            return Success(result(request, {"results": entries, "untrusted_content": True}, {"search_response_received": True}))
        resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], operation=self.name)
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        if (handle.kind != "url" or not web_url(handle.target)
                or handle.data_class != request.data_class
                or handle.fingerprint != hashlib.sha256(handle.target.encode()).hexdigest()):
            return Failure(failure(request, "STALE_RESOURCE"))
        invoked = await (self.backend.open(handle.target, cancellation) if self.name == "browser.open_result"
                         else self.backend.read(handle.target, cancellation))
        if isinstance(invoked, Failure):
            return Failure(failure(request, invoked.failure()))
        evidence = invoked.unwrap()
        if evidence.get("observed_url") != handle.target:
            return Failure(failure(request, "POSTCONDITION_FAILED"))
        return Success(result(request, evidence, {"target_id": evidence["target_id"], "observed_url": handle.target}))


def register_browser_capabilities(registry: CapabilityRegistry, handles: SQLiteHandleStore, backend: BrowserBackend) -> None:
    for name in ("browser.search", "browser.open_result", "browser.read_page"):
        fields = ("task_id", "query" if name == "browser.search" else "handle")
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name=name, description=f"Bounded {name} in an explicitly configured browser (v1).",
            input_schema={"type": "object", "properties": {key: {"type": "string"} for key in fields},
                          "required": list(fields), "additionalProperties": False},
            risk_class=RiskClass.MODERATE, side_effects=("network", "browser") if name == "browser.open_result" else ("network",),
            isolation="host-api", verification_plan=("browser-url-identity",),
        ), implementation=BrowserCapability(name, handles, backend)))
