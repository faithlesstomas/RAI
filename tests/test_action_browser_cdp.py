"""Chromium debugging adapter and browser capabilities against hermetic CDP/HTTP doubles."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import httpx
import pytest
from returns.result import Failure, Success

import rai.actions.browser as browser
from rai.actions.browser import BrowserCapability, CdpBrowserBackend, register_browser_capabilities
from rai.actions.capabilities import PRODUCER
from rai.actions.handles import SQLiteHandleStore
from rai.actions.records import ResourceHandle
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, _utc_now
from rai.kernel.transport import normalize_request

ENDPOINT = "http://127.0.0.1:9222"
PAGE_WS = "ws://127.0.0.1:9222/devtools/page/1"
URL = "https://example.com/"
REAL_CLIENT = httpx.AsyncClient


class FakeConnection:
    def __init__(self, handler: Callable[[str, dict], dict | None]) -> None:
        self.handler, self.sent = handler, []

    async def __aenter__(self) -> FakeConnection:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def __aiter__(self) -> Any:
        async def messages() -> Any:
            request = self.sent[-1]
            yield json.dumps({"id": 99, "result": {"ignored": True}})  # unrelated events are skipped
            response = self.handler(request["method"], request["params"])
            if response is not None:
                yield json.dumps({"id": request["id"], **response})
        return messages()


def install_cdp(monkeypatch: pytest.MonkeyPatch, handler: Callable[[str, dict], dict | None],
                http: Callable[[httpx.Request], httpx.Response] | None = None) -> list[str]:
    urls: list[str] = []
    import websockets.asyncio.client as client  # noqa: PLC0415

    def connect(url: str, **_: Any) -> FakeConnection:
        urls.append(url)
        return FakeConnection(handler)

    monkeypatch.setattr(client, "connect", connect)
    transport = httpx.MockTransport(http or (lambda request: httpx.Response(404)))
    monkeypatch.setattr(browser.httpx, "AsyncClient", lambda **kwargs: REAL_CLIENT(transport=transport, **kwargs))
    return urls


def new_page(request: httpx.Request) -> httpx.Response:
    assert request.method == "PUT" and request.url.path == "/json/new"
    return httpx.Response(200, json={"id": "tab-1", "webSocketDebuggerUrl": PAGE_WS})


def tab(url: str = URL, kind: str = "page", ws: str = PAGE_WS) -> dict:
    return {"id": "tab-1", "type": kind, "url": url, "webSocketDebuggerUrl": ws}


def frame(url: str) -> dict:
    return {"result": {"frameTree": {"frame": {"url": url, "id": "frame-1"}}}}


# --- _command ---------------------------------------------------------------------------------------------------


async def test_command_rejects_foreign_websocket_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = install_cdp(monkeypatch, lambda *_: {"result": {}})
    backend = CdpBrowserBackend(ENDPOINT)
    for websocket in ("ws://example.com:9222/x", "ws://127.0.0.1:1/x", "wss://127.0.0.1:9222/x"):
        with pytest.raises(ValueError, match="authority"):
            await backend._command(websocket, "Page.enable", {})
    assert not urls


async def test_command_maps_browser_errors_and_disconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = CdpBrowserBackend(ENDPOINT)
    install_cdp(monkeypatch, lambda *_: {"result": {"ok": 1}})
    assert await backend._command(PAGE_WS, "Page.enable", {}) == {"ok": 1}
    install_cdp(monkeypatch, lambda *_: {"error": {"message": "boom"}})
    with pytest.raises(ValueError, match="failed"):
        await backend._command(PAGE_WS, "Page.enable", {})
    install_cdp(monkeypatch, lambda *_: None)
    with pytest.raises(ValueError, match="disconnected"):
        await backend._command(PAGE_WS, "Page.enable", {})


# --- open ---------------------------------------------------------------------------------------------------------


async def test_open_reports_only_observed_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cdp(monkeypatch, lambda method, _: {"result": {}} if method == "Page.navigate" else frame(URL), new_page)
    assert await CdpBrowserBackend(ENDPOINT).open(URL, CancellationToken()) == Success(
        {"target_id": "tab-1", "observed_url": URL, "frame_id": "frame-1"})


async def test_open_navigation_error_or_wrong_url_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cdp(monkeypatch, lambda *_: {"result": {"errorText": "net::ERR"}}, new_page)
    assert await CdpBrowserBackend(ENDPOINT).open(URL, CancellationToken()) == Failure("NAVIGATION_FAILED")

    real_sleep = asyncio.sleep

    async def fast_sleep(_: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(browser.asyncio, "sleep", fast_sleep)
    install_cdp(monkeypatch, lambda method, _: {"result": {}} if method == "Page.navigate" else frame("https://other/"),
                new_page)
    assert await CdpBrowserBackend(ENDPOINT).open(URL, CancellationToken()) == Failure("POSTCONDITION_FAILED")


async def test_cancelled_open_stops_loading_and_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    token = CancellationToken()
    methods: list[str] = []

    def handler(method: str, _: dict) -> dict:
        methods.append(method)
        token.cancel()
        return {"result": {}}

    install_cdp(monkeypatch, handler, new_page)
    assert await CdpBrowserBackend(ENDPOINT).open(URL, token) == Failure("UNKNOWN")
    assert methods == ["Page.navigate", "Page.stopLoading"]


async def test_open_maps_transport_failures_without_leaking_details(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cdp(monkeypatch, lambda *_: {"result": {}}, lambda _: httpx.Response(500))
    assert await CdpBrowserBackend(ENDPOINT).open(URL, CancellationToken()) == Failure("BROWSER_UNAVAILABLE")


# --- read ---------------------------------------------------------------------------------------------------------


def snapshot(url: str = URL, title: str = "T" * 300, text: str = "body") -> dict:
    return {"result": {"result": {"value": json.dumps({"url": url, "title": title, "text": text})}}}


def targets(*items: dict) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json=list(items))


async def test_read_returns_bounded_untrusted_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cdp(monkeypatch, lambda *_: snapshot(text="x" * 20000), targets(tab(), tab(kind="service_worker")))
    read = await CdpBrowserBackend(ENDPOINT).read(URL, CancellationToken())
    content = read.unwrap()
    assert content["untrusted_content"] is True and content["observed_url"] == URL
    assert len(content["title"]) == 256 and len(content["text"]) == browser.MAX_PAGE_CHARACTERS


async def test_read_requires_exactly_one_matching_current_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    install_cdp(monkeypatch, lambda *_: snapshot(), targets(tab("https://elsewhere/")))
    assert await CdpBrowserBackend(ENDPOINT).read(URL, CancellationToken()) == Failure("AMBIGUOUS_OR_MISSING_TAB")
    install_cdp(monkeypatch, lambda *_: snapshot(), targets(tab(), tab()))
    assert await CdpBrowserBackend(ENDPOINT).read(URL, CancellationToken()) == Failure("AMBIGUOUS_OR_MISSING_TAB")


async def test_read_detects_cancellation_navigation_and_oversized_target_list(monkeypatch: pytest.MonkeyPatch) -> None:
    token = CancellationToken()
    token.cancel()
    install_cdp(monkeypatch, lambda *_: snapshot(), targets(tab()))
    assert await CdpBrowserBackend(ENDPOINT).read(URL, token) == Failure("CANCELLED")
    install_cdp(monkeypatch, lambda *_: snapshot(url="https://moved/"), targets(tab()))
    assert await CdpBrowserBackend(ENDPOINT).read(URL, CancellationToken()) == Failure("STALE_RESOURCE")
    install_cdp(monkeypatch, lambda *_: snapshot(), lambda _: httpx.Response(200, content=b"[" + b" " * (1024 * 1024 + 1)))
    assert await CdpBrowserBackend(ENDPOINT).read(URL, CancellationToken()) == Failure("BROWSER_READ_FAILED")


# --- search -------------------------------------------------------------------------------------------------------


def install_ddgs(monkeypatch: pytest.MonkeyPatch, items: list[dict] | Exception) -> None:
    class DDGS:
        def __init__(self, **_: Any) -> None: ...
        def __enter__(self) -> DDGS: return self
        def __exit__(self, *_: Any) -> None: ...

        def text(self, query: str, max_results: int) -> list[dict]:
            if isinstance(items, Exception):
                raise items
            return items

    module = ModuleType("ddgs")
    module.DDGS = DDGS  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ddgs", module)


async def test_search_keeps_only_bounded_public_web_results(monkeypatch: pytest.MonkeyPatch) -> None:
    install_ddgs(monkeypatch, [{"title": "A" * 400, "href": URL}, {"title": "Local", "href": "file:///etc/passwd"},
                               {"title": "Bad"}] + [{"title": f"R{i}", "href": f"https://r{i}.example/"} for i in range(8)])
    found = (await CdpBrowserBackend().search("query", CancellationToken())).unwrap()
    assert len(found) == browser.MAX_SEARCH_RESULTS
    assert len(found[0]["name"]) == 256 and all(browser.web_url(item["url"]) for item in found)


async def test_search_failure_or_cancellation_never_returns_partial_results(monkeypatch: pytest.MonkeyPatch) -> None:
    install_ddgs(monkeypatch, RuntimeError("provider down"))
    assert await CdpBrowserBackend().search("query", CancellationToken()) == Failure("BROWSER_SEARCH_UNAVAILABLE")
    token = CancellationToken()
    token.cancel()
    assert await CdpBrowserBackend().search("query", token) == Failure("CANCELLED")


def test_web_url_rejects_malformed_and_oversized_values() -> None:
    assert browser.web_url(URL)
    assert not browser.web_url("http://[invalid/")
    assert not browser.web_url("https://example.com/" + "a" * browser.MAX_URL_CHARACTERS)
    assert not browser.web_url("https:///no-host")


# --- capability -----------------------------------------------------------------------------------------------------


class Backend:
    def __init__(self, **outcomes: Any) -> None:
        self.outcomes = outcomes

    async def search(self, query: str, token: CancellationToken) -> Any:
        return self.outcomes["search"]

    async def open(self, url: str, token: CancellationToken) -> Any:
        return self.outcomes["open"]

    async def read(self, url: str, token: CancellationToken) -> Any:
        return self.outcomes["read"]


def make(tmp_path: Path, backend: Backend) -> tuple[CapabilityRegistry, SQLiteHandleStore]:
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_browser_capabilities(registry, handles, backend)
    return registry, handles


def public(registry: CapabilityRegistry, name: str, **arguments: str) -> Any:
    return normalize_request(registry.descriptor(name), {"task_id": "t", **arguments}, data_class=DataClass.PUBLIC)


async def test_search_validates_query_and_propagates_backend_failure(tmp_path: Path) -> None:
    registry, handles = make(tmp_path, Backend(search=Failure("BROWSER_SEARCH_UNAVAILABLE")))
    capability = BrowserCapability("browser.search", handles, Backend(search=Failure("BROWSER_SEARCH_UNAVAILABLE")))
    for query in ("   ", "q" * 257):
        outcome = await capability.invoke(public(registry, "browser.search", query=query), CancellationToken())
        assert outcome.failure().code == "INVALID_ARGUMENT"
    outcome = await capability.invoke(public(registry, "browser.search", query="ok"), CancellationToken())
    assert outcome.failure().code == "BROWSER_SEARCH_UNAVAILABLE"


async def test_search_skips_backend_results_that_are_not_public_web_urls(tmp_path: Path) -> None:
    backend = Backend(search=Success(({"name": "Local", "url": "file:///etc/passwd"}, {"name": "Web", "url": URL})))
    registry, handles = make(tmp_path, backend)
    outcome = await BrowserCapability("browser.search", handles, backend).invoke(
        public(registry, "browser.search", query="ok"), CancellationToken())
    results = outcome.unwrap().output["results"]
    assert [item["url"] for item in results] == [URL]
    assert outcome.unwrap().output["untrusted_content"] is True


async def test_read_page_returns_evidence_and_propagates_backend_failures(tmp_path: Path) -> None:
    content = {"target_id": "tab-1", "observed_url": URL, "title": "T", "text": "body", "untrusted_content": True}
    backend = Backend(search=Success(({"name": "Web", "url": URL},)), read=Success(content))
    registry, handles = make(tmp_path, backend)
    searched = await BrowserCapability("browser.search", handles, backend).invoke(
        public(registry, "browser.search", query="ok"), CancellationToken())
    handle = searched.unwrap().output["results"][0]["handle"]
    reader = BrowserCapability("browser.read_page", handles, backend)
    request = public(registry, "browser.read_page", handle=handle)
    assert (await reader.invoke(request, CancellationToken())).unwrap().output == content
    backend.outcomes["read"] = Failure("BROWSER_READ_FAILED")
    assert (await reader.invoke(request, CancellationToken())).failure().code == "BROWSER_READ_FAILED"


async def test_handle_for_non_web_target_is_stale_and_never_reaches_backend(tmp_path: Path) -> None:
    backend = Backend()  # any backend call would raise KeyError
    registry, handles = make(tmp_path, backend)
    now = _utc_now()
    target = "file:///etc/passwd"
    issued = handles.issue(ResourceHandle(
        producer=PRODUCER, timestamp=now, actor_id=public(registry, "browser.search", query="x").actor.producer_id,
        task_id="t", kind="url", target=target,
        fingerprint=hashlib.sha256(target.encode()).hexdigest(), operations=("browser.read_page",),
        expires_at=now + timedelta(minutes=5), data_class=DataClass.PUBLIC,
    ))
    request = public(registry, "browser.read_page", handle=issued.unwrap())
    outcome = await BrowserCapability("browser.read_page", handles, backend).invoke(request, CancellationToken())
    assert outcome.failure().code == "STALE_RESOURCE"
