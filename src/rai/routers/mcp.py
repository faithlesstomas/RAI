"""MCP transport backed exclusively by the typed capability service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.types import Receive, Scope, Send

from rai.kernel.defaults import DEFAULT_ACTOR
from rai.kernel.capabilities import CapabilityDescriptor
from rai.kernel.records import ActionFailure, CapabilityRequest, ProducerIdentity
from rai.kernel.service import CapabilityService
from rai.kernel.transport import InvocationEnvelope, invoke_envelope, normalize_request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/mcp", tags=["MCP"])
MCP_ACTOR = ProducerIdentity(
    producer_id="rai.mcp-client", kind="transport-client", version="1.0.0"
)


class EmptyResponse(Response):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del scope, receive, send


def _service(request: Request) -> CapabilityService:
    return request.app.state.container.capability_service


def _mcp_tool(descriptor: CapabilityDescriptor) -> types.Tool:
    return types.Tool(
        name=descriptor.name,
        description=descriptor.description,
        inputSchema=descriptor.input_schema,
        outputSchema=InvocationEnvelope.model_json_schema(),
    )


async def list_tools(service: CapabilityService) -> list[types.Tool]:
    """Render the canonical registry as MCP descriptors."""
    return [_mcp_tool(descriptor) for descriptor in service.registry.descriptors()]


def _transport_failure(name: str, message: str) -> InvocationEnvelope:
    failure = ActionFailure(
        producer=DEFAULT_ACTOR,
        request_id="mcp-unregistered",
        capability=name,
        code="CAPABILITY_NOT_FOUND",
        message=message,
    )
    return InvocationEnvelope(ok=False, decision=None, result=failure)


async def call_tool(
    service: CapabilityService,
    name: str,
    arguments: dict[str, Any],
    request_record: CapabilityRequest | None = None,
) -> types.CallToolResult:
    """Invoke one MCP tool through the same service as CLI and REST."""
    descriptor = service.registry.descriptor(name)
    if descriptor is None:
        envelope = _transport_failure(name, "capability is not registered")
    else:
        request = request_record or normalize_request(
            descriptor, arguments, actor=MCP_ACTOR
        )
        envelope = await invoke_envelope(service, request)
    serialized = envelope.model_dump(mode="json")
    result = envelope.result
    if envelope.ok:
        text = str(result.output.get("text", json.dumps(result.output, sort_keys=True)))
    else:
        text = f"Error: {result.code}: {result.message}"
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structuredContent=serialized,
        isError=not envelope.ok,
    )


@dataclass(frozen=True)
class McpRuntime:
    server: Server
    transport: SseServerTransport


def create_mcp_runtime(service: CapabilityService) -> McpRuntime:
    """Create an app-scoped MCP server whose handlers close over the container."""
    server = Server("rai-secure-gateway")
    transport = SseServerTransport("/api/v1/mcp/messages")

    @server.list_tools()  # pylint: disable=no-member
    async def handle_list_tools() -> list[types.Tool]:
        return await list_tools(service)

    @server.call_tool()  # pylint: disable=no-member
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await call_tool(service, name, arguments)

    return McpRuntime(server, transport)


async def handle_stateless_mcp(  # noqa: PLR0911
    request: Request, service: CapabilityService
) -> Response:
    """Handle the JSON-RPC subset used by clients without MCP SSE sessions."""
    try:
        body = await request.json()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _rpc_error(None, -32700, f"Parse error: {exc}", status_code=400)
    if not isinstance(body, dict):
        return _rpc_error(None, -32600, "Invalid Request: expected an object", status_code=400)
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})
    if not method:
        return _rpc_error(request_id, -32600, "Invalid Request: missing method", status_code=400)
    if method == "initialize":
        return _rpc_result(
            request_id,
            {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "rai-secure-gateway", "version": "1.0.0"},
            },
        )
    if method == "notifications/initialized":
        return JSONResponse(content={})
    if method == "ping":
        return _rpc_result(request_id, {})
    if method == "tools/list":
        tools = await list_tools(service)
        return _rpc_result(
            request_id, {"tools": [tool.model_dump(exclude_none=True) for tool in tools]}
        )
    if method == "tools/call":
        name = params.get("name")
        if not name:
            return _rpc_error(
                request_id, -32602, "Invalid params: 'name' is required", status_code=400
            )
        result = await call_tool(service, name, params.get("arguments", {}))
        return _rpc_result(request_id, result.model_dump(exclude_none=True, by_alias=True))
    return _rpc_error(request_id, -32601, f"Method '{method}' not found")


def _rpc_result(request_id: object, result: object) -> JSONResponse:
    return JSONResponse(content={"jsonrpc": "2.0", "id": request_id, "result": result})


def _rpc_error(
    request_id: object, code: int, message: str, status_code: int = 200
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
    )


@router.get("/sse")
async def mcp_sse(request: Request) -> Response:
    runtime: McpRuntime = request.app.state.mcp_runtime
    async with runtime.transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await runtime.server.run(
            read_stream,
            write_stream,
            runtime.server.create_initialization_options(),
        )
    return EmptyResponse()


@router.post("/messages")
async def mcp_messages(request: Request) -> Response:
    if "session_id" not in request.query_params:
        return await handle_stateless_mcp(request, _service(request))
    runtime: McpRuntime = request.app.state.mcp_runtime
    await runtime.transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return EmptyResponse()


@router.post("/sse")
async def mcp_post_sse(request: Request) -> Response:
    if "session_id" not in request.query_params:
        return await handle_stateless_mcp(request, _service(request))
    runtime: McpRuntime = request.app.state.mcp_runtime
    await runtime.transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return EmptyResponse()
