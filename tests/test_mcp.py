"""MCP conformance tests for the shared Stage 1 capability service."""

from __future__ import annotations

import asyncio

import pytest
from mcp import types

from conftest import ASGITestClient
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.routers.mcp import call_tool, list_tools

METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
PARSE_ERROR = -32700


def _service() -> CapabilityService:
    return CapabilityService(
        create_default_capability_registry(),
        PolicyEngine(isolation_available=lambda _isolation: False),
        InMemoryAuditLedger(),
    )


@pytest.mark.asyncio
async def test_mcp_list_tools_comes_from_capability_registry() -> None:
    tools = await list_tools(_service())
    names = {tool.name for tool in tools}
    assert names == {
        "calculate",
        "finance.quote",
        "get_desktop_weather",
        "run_python_code",
        "run_shell_command",
        "search.arxiv",
        "search.web",
        "search.wikipedia",
        "send_desktop_notification",
        "take_desktop_screenshot",
        "test.echo",
    }
    shell = next(tool for tool in tools if tool.name == "run_shell_command")
    assert tuple(shell.inputSchema["required"]) == ("command",)
    assert shell.outputSchema is not None


@pytest.mark.asyncio
async def test_mcp_call_returns_typed_shared_envelope() -> None:
    result = await call_tool(_service(), "test.echo", {"text": "hello"})
    assert isinstance(result, types.CallToolResult)
    assert not result.isError
    assert result.content[0].text == "hello"
    assert result.structuredContent is not None
    assert result.structuredContent["decision"]["outcome"] == "ALLOW"
    assert result.structuredContent["result"]["record_type"] == "action_result"


@pytest.mark.asyncio
async def test_mcp_rejects_missing_argument_and_unknown_capability() -> None:
    missing = await call_tool(_service(), "test.echo", {})
    unknown = await call_tool(_service(), "backend.private_tool", {})
    assert missing.isError
    assert missing.structuredContent["result"]["code"] == "INVALID_ARGUMENT"
    assert unknown.isError
    assert unknown.structuredContent["decision"] is None
    assert unknown.structuredContent["result"]["code"] == "CAPABILITY_NOT_FOUND"


def test_stateless_mcp_initialize(client: ASGITestClient) -> None:
    response = client.post(
        "/api/v1/mcp/sse",
        json={"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}},
    )
    assert response.status_code == 200  # noqa: PLR2004
    assert response.json()["result"]["serverInfo"]["name"] == "rai-secure-gateway"


def test_stateless_mcp_tools_list(client: ASGITestClient) -> None:
    response = client.post(
        "/api/v1/mcp/sse",
        json={"jsonrpc": "2.0", "id": 43, "method": "tools/list", "params": {}},
    )
    tools = response.json()["result"]["tools"]
    assert {tool["name"] for tool in tools} >= {"test.echo", "run_shell_command"}
    assert all("outputSchema" in tool for tool in tools)


def test_stateless_mcp_tools_call(client: ASGITestClient) -> None:
    response = client.post(
        "/api/v1/mcp/sse",
        json={
            "jsonrpc": "2.0",
            "id": 44,
            "method": "tools/call",
            "params": {"name": "test.echo", "arguments": {"text": "stateless"}},
        },
    )
    result = response.json()["result"]
    assert result["content"][0]["text"] == "stateless"
    assert result["structuredContent"]["ok"] is True


def test_stateless_mcp_ping_and_notifications(client: ASGITestClient) -> None:
    ping = client.post(
        "/api/v1/mcp/sse",
        json={"jsonrpc": "2.0", "id": 45, "method": "ping", "params": {}},
    )
    initialized = client.post(
        "/api/v1/mcp/sse",
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    )
    assert ping.json()["result"] == {}
    assert initialized.json() == {}


def test_stateless_mcp_errors_are_json_rpc(client: ASGITestClient) -> None:
    unsupported = client.post(
        "/api/v1/mcp/sse",
        json={"jsonrpc": "2.0", "id": 46, "method": "unsupported", "params": {}},
    )
    invalid_json = client.post("/api/v1/mcp/sse", content="invalid json")
    missing_name = client.post(
        "/api/v1/mcp/sse",
        json={
            "jsonrpc": "2.0",
            "id": 47,
            "method": "tools/call",
            "params": {"arguments": {}},
        },
    )
    assert unsupported.json()["error"]["code"] == METHOD_NOT_FOUND
    assert invalid_json.status_code == 400  # noqa: PLR2004
    assert invalid_json.json()["error"]["code"] == PARSE_ERROR
    assert missing_name.status_code == 400  # noqa: PLR2004
    assert missing_name.json()["error"]["code"] == INVALID_PARAMS


def test_mcp_runtime_is_owned_by_application_container() -> None:
    first = ApplicationContainer(config={}, testing=True)
    second = ApplicationContainer(config={}, testing=True)
    assert first.capability_service is not second.capability_service
    assert asyncio.run(call_tool(first.capability_service, "test.echo", {"text": "one"}))
