"""Stable browser results and an explicit, isolated Chromium debugging adapter."""
from __future__ import annotations

import asyncio
import hashlib
import html
import ipaddress
import json
import re
import socket
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx
from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, DataClass, RiskClass, _utc_now
from .consent import has_browser_consent
from .capabilities import HANDLE_TTL, PRODUCER, failure, result
from .handles import SQLiteHandleStore
from .records import ResourceHandle

MAX_URL_CHARACTERS = 4096
MAX_PAGE_CHARACTERS = 500_000
DEFAULT_SEARCH_RESULTS = 25
MAX_SEARCH_RESULTS = 50


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return _is_forbidden_ip(ip.ipv4_mapped)
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


def web_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        if not (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and len(value) <= MAX_URL_CHARACTERS
        ):
            return False
        host = parsed.hostname.strip("[]").lower()
        if host == "localhost" or host.endswith(
            (".localhost", ".local", ".internal", ".lan", ".arpa")
        ):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if _is_forbidden_ip(ip):
                return False
        except ValueError:
            pass
        return True
    except ValueError:
        return False


async def _resolve_and_validate_public_host(hostname: str, port: int) -> bool:
    try:
        host = hostname.strip("[]").lower()
        if host == "localhost" or host.endswith(
            (".localhost", ".local", ".internal", ".lan", ".arpa")
        ):
            return False
        try:
            ip = ipaddress.ip_address(host)
            return not _is_forbidden_ip(ip)
        except ValueError:
            pass

        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(
            hostname,
            port,
            proto=socket.IPPROTO_TCP,
        )
        if not addr_info:
            return False
        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if _is_forbidden_ip(ip):
                return False
        return True
    except (socket.gaierror, ValueError, OSError):
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
                    return tuple(
                        {
                            "name": str(item.get("title", ""))[:256],
                            "url": item["href"],
                            "snippet": str(item.get("body", ""))[:250],
                        }
                        for item in search.text(query, max_results=MAX_SEARCH_RESULTS)
                        if web_url(item.get("href", ""))
                    )[:MAX_SEARCH_RESULTS]
            return Success(await asyncio.to_thread(fetch))
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("BROWSER_SEARCH_UNAVAILABLE")

    async def open(self, url: str, token: CancellationToken) -> Result[dict, str]:  # noqa: PLR0911
        if not self.endpoint:
            return Failure("BROWSER_NOT_CONFIGURED")
        if not web_url(url):
            return Failure("STALE_RESOURCE")
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not await _resolve_and_validate_public_host(parsed.hostname or "", port):
            return Failure("STALE_RESOURCE")
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

    async def read(self, url: str, token: CancellationToken) -> Result[dict, str]:  # noqa: PLR0911
        if not self.endpoint:
            return Failure("BROWSER_NOT_CONFIGURED")
        if not web_url(url):
            return Failure("STALE_RESOURCE")
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not await _resolve_and_validate_public_host(parsed.hostname or "", port):
            return Failure("STALE_RESOURCE")
        try:
            pages = [page for page in await self._targets() if page.get("type") == "page" and page.get("url") == url]
            if len(pages) != 1:
                return Failure("AMBIGUOUS_OR_MISSING_TAB")
            if token.cancelled:
                return Failure("CANCELLED")
            page = pages[0]
            snapshot = await self._command(page["webSocketDebuggerUrl"], "Runtime.evaluate", {
                "expression": f"JSON.stringify({{url:location.href,title:document.title,text:(document.body?.innerText||'').slice(0,{MAX_PAGE_CHARACTERS})}})",
                "returnByValue": True,
            })
            content = json.loads(snapshot["result"]["value"])
            if content["url"] != url:
                return Failure("STALE_RESOURCE")
            raw_text = content.get("text", "")
            is_complete = len(raw_text) < MAX_PAGE_CHARACTERS
            return Success({
                "target_id": page["id"],
                "observed_url": url,
                "title": content["title"][:256],
                "text": raw_text[:MAX_PAGE_CHARACTERS],
                "untrusted_content": True,
                "complete": is_complete,
                "reason": None if is_complete else f"Page truncated at {MAX_PAGE_CHARACTERS} characters",
            })
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("BROWSER_READ_FAILED")


def _clean_html_to_text(raw_html: str, fallback_title: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else fallback_title
    cleaned = re.sub(
        r"<(script|style|noscript|svg|head)\b[^>]*>.*?</\1>",
        " ",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    text = re.sub(r"\s+", " ", html.unescape(cleaned)).strip()
    return title[:256], text[:MAX_PAGE_CHARACTERS]


class SafeNetworkBackend(httpcore.AnyIOBackend):
    """Network backend validating that connected TCP sockets do not point to private or loopback IPs."""

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,  # noqa: ANN401
    ) -> httpcore.AsyncNetworkStream:
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
            if _is_forbidden_ip(ip):
                raise httpcore.ConnectError(f"Connection to forbidden address {ip} blocked")
        except ValueError:
            pass

        stream = await super().connect_tcp(  # pylint: disable=no-member
            host=host,
            port=port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )
        server_addr = stream.get_extra_info("server_addr")
        if server_addr:
            ip = ipaddress.ip_address(server_addr[0])
            if _is_forbidden_ip(ip):
                await stream.aclose()
                raise httpcore.ConnectError(f"Connection to forbidden address {ip} blocked")
        return stream


class SafeAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """HTTP transport with post-connect TCP socket verification against SSRF and DNS rebinding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)
        self._pool._network_backend = SafeNetworkBackend()


async def _fetch_http_page_snapshot(  # noqa: PLR0911, PLR0912
    url: str, token: CancellationToken
) -> Result[dict[str, Any], str]:
    """Bounded plain-text HTTP fallback for browser.read_page when no CDP endpoint is configured."""
    if token.cancelled:
        return Failure("CANCELLED")
    if not web_url(url):
        return Failure("STALE_RESOURCE")

    current_url = url
    max_redirects = 5

    try:
        transport = SafeAsyncHTTPTransport(verify=True)
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAI-Reader/1.0)"},
        ) as client:
            raw_html = ""
            for _ in range(max_redirects + 1):
                if token.cancelled:
                    return Failure("CANCELLED")
                if not web_url(current_url):
                    return Failure("STALE_RESOURCE")
                parsed = urlsplit(current_url)
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                if not await _resolve_and_validate_public_host(parsed.hostname or "", port):
                    return Failure("STALE_RESOURCE")

                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            return Failure("BROWSER_READ_FAILED")
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    raw_bytes = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        if token.cancelled:
                            return Failure("CANCELLED")
                        raw_bytes.extend(chunk)
                        if len(raw_bytes) >= 2 * 1024 * 1024:
                            break
                    encoding = response.encoding or "utf-8"
                    raw_html = raw_bytes.decode(encoding, errors="replace")
                    break
            else:
                return Failure("BROWSER_READ_FAILED")

        title, text = _clean_html_to_text(raw_html, fallback_title=url)
        if not text:
            return Failure("BROWSER_READ_FAILED")
        is_complete = len(text) < MAX_PAGE_CHARACTERS and len(raw_bytes) < 2 * 1024 * 1024
        return Success({
            "target_id": f"http:{hashlib.sha256(url.encode()).hexdigest()[:12]}",
            "observed_url": url,
            "title": title,
            "text": text[:MAX_PAGE_CHARACTERS],
            "untrusted_content": True,
            "complete": is_complete,
            "reason": None if is_complete else f"Page truncated at {MAX_PAGE_CHARACTERS} characters",
        })
    except (httpx.ConnectError, httpcore.ConnectError):
        return Failure("STALE_RESOURCE")
    except asyncio.CancelledError:
        raise
    except Exception:  # pylint: disable=broad-exception-caught
        return Failure("BROWSER_READ_FAILED")


class BrowserCapability:
    def __init__(self, name: str, handles: SQLiteHandleStore, backend: BrowserBackend) -> None:
        self.name, self.handles, self.backend = name, handles, backend

    async def invoke(self, request: CapabilityRequest, cancellation: CancellationToken) -> Result[ActionResult, ActionFailure]:  # noqa: PLR0911, PLR0912
        if request.data_class != DataClass.PUBLIC and not (
            request.data_class == DataClass.PRIVATE and has_browser_consent(request)
        ):
            return Failure(failure(request, "PUBLIC_CONTEXT_REQUIRED"))
        if self.name == "browser.search":
            query = request.arguments["query"]
            if not query.strip() or len(query) > 256:  # noqa: PLR2004
                return Failure(failure(request, "INVALID_ARGUMENT"))
            limit_arg = request.arguments.get("limit") or request.arguments.get("max_results")
            try:
                requested_limit = int(limit_arg) if limit_arg is not None else DEFAULT_SEARCH_RESULTS
            except (ValueError, TypeError):
                requested_limit = DEFAULT_SEARCH_RESULTS
            search_limit = max(1, min(requested_limit, MAX_SEARCH_RESULTS))

            found = await self.backend.search(query, cancellation)
            if isinstance(found, Failure):
                return Failure(failure(request, found.failure()))
            raw_items = found.unwrap()
            entries = []
            for item in raw_items[:search_limit]:
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
                entry: dict[str, Any] = {"name": item["name"], "url": item["url"], "handle": issued.unwrap()}
                if "snippet" in item and item["snippet"]:
                    entry["snippet"] = str(item["snippet"])[:250]
                entries.append(entry)
            is_complete = len(raw_items) <= search_limit
            return Success(
                result(
                    request,
                    {
                        "results": entries,
                        "untrusted_content": True,
                        "complete": is_complete,
                        "reason": None if is_complete else f"Search results capped at limit of {search_limit}",
                    },
                    {"search_response_received": True},
                )
            )
        resolved = self.handles.resolve(request.arguments["handle"], actor_id=request.actor.producer_id,
                                        task_id=request.arguments["task_id"], operation=self.name)
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        if (handle.kind != "url" or not web_url(handle.target)
                or handle.data_class != request.data_class
                or handle.fingerprint != hashlib.sha256(handle.target.encode()).hexdigest()):
            return Failure(failure(request, "STALE_RESOURCE"))
        parsed_target = urlsplit(handle.target)
        target_port = parsed_target.port or (443 if parsed_target.scheme == "https" else 80)
        if not await _resolve_and_validate_public_host(parsed_target.hostname or "", target_port):
            return Failure(failure(request, "STALE_RESOURCE"))
        if self.name == "browser.open_result":
            invoked = await self.backend.open(handle.target, cancellation)
        else:
            invoked = await self.backend.read(handle.target, cancellation)
            if isinstance(invoked, Failure) and invoked.failure() == "AMBIGUOUS_OR_MISSING_TAB":
                open_res = await self.backend.open(handle.target, cancellation)
                if isinstance(open_res, Failure):
                    return Failure(failure(request, open_res.failure()))
                invoked = await self.backend.read(handle.target, cancellation)
            elif isinstance(invoked, Failure) and invoked.failure() == "BROWSER_NOT_CONFIGURED":
                invoked = await _fetch_http_page_snapshot(handle.target, cancellation)
        if isinstance(invoked, Failure):
            return Failure(failure(request, invoked.failure()))
        evidence = invoked.unwrap()
        if evidence.get("observed_url") != handle.target:
            return Failure(failure(request, "POSTCONDITION_FAILED"))
        return Success(result(request, evidence, {"target_id": evidence["target_id"], "observed_url": handle.target}))


def register_browser_capabilities(registry: CapabilityRegistry, handles: SQLiteHandleStore, backend: BrowserBackend) -> None:
    for name in ("browser.search", "browser.open_result", "browser.read_page"):
        if name == "browser.search":
            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (1-50)",
                        "default": DEFAULT_SEARCH_RESULTS,
                    },
                },
                "required": ["task_id", "query"],
                "additionalProperties": False,
            }
        else:
            fields = ("task_id", "handle")
            input_schema = {
                "type": "object",
                "properties": {key: {"type": "string"} for key in fields},
                "required": list(fields),
                "additionalProperties": False,
            }
        registry.register(RegisteredCapability(CapabilityDescriptor(
            name=name, description=f"Bounded {name} in an explicitly configured browser (v1).",
            input_schema=input_schema,
            risk_class=RiskClass.MODERATE, side_effects=("network", "browser") if name == "browser.open_result" else ("network",),
            isolation="host-api", verification_plan=("browser-url-identity",),
            timeout_seconds=60.0,
        ), implementation=BrowserCapability(name, handles, backend)))
