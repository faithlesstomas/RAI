"""MCP transport backed exclusively by the typed capability service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
from rai.tools.security.auth import verify_session_token

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
    schema = json.loads(json.dumps(descriptor.input_schema))
    if isinstance(schema, dict) and "required" in schema and isinstance(schema["required"], list):
        schema["required"] = [r for r in schema["required"] if r != "session_id"]
    return types.Tool(
        name=descriptor.name,
        description=descriptor.description,
        inputSchema=schema,
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
    session_id: str | None = None,
) -> types.CallToolResult:
    """Invoke one MCP tool through the same service as CLI and REST."""
    descriptor = service.registry.descriptor(name)
    if descriptor is None:
        envelope = _transport_failure(name, "capability is not registered")
    else:
        bound_arguments = dict(arguments)
        expects_session = "session_id" in descriptor.input_schema.get("properties", {})
        if expects_session or "session_id" in bound_arguments:
            bound_arguments["session_id"] = session_id or ""

        if request_record is not None:
            request = request_record
            if expects_session or "session_id" in request.arguments:
                bound_args = dict(request.arguments)
                bound_args["session_id"] = session_id or ""
                request = request.model_copy(update={"arguments": bound_args})
        else:
            request = normalize_request(
                descriptor, bound_arguments, actor=MCP_ACTOR
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


@dataclass
class McpRuntime:
    server: Server
    transport: SseServerTransport
    transport_sessions: dict[str, str] = field(default_factory=dict)


def create_mcp_runtime(service: CapabilityService) -> McpRuntime:
    """Create an app-scoped MCP server whose handlers close over the container."""
    server = Server("rai-secure-gateway")
    transport = SseServerTransport("/api/v1/mcp/messages")
    runtime = McpRuntime(server, transport)

    @server.list_tools()  # pylint: disable=no-member
    async def handle_list_tools() -> list[types.Tool]:
        return await list_tools(service)

    @server.call_tool()  # pylint: disable=no-member
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        trusted_session_id = None
        try:
            ctx = server.request_context
            if ctx and ctx.request:
                transport_id = ctx.request.query_params.get("session_id") or ""
                session_token = (
                    ctx.request.headers.get("x-session-token")
                    or ctx.request.query_params.get("session_token")
                )
                verified_session = verify_session_token(session_token) if session_token else None
                if verified_session:
                    runtime.transport_sessions[transport_id] = verified_session
                    trusted_session_id = verified_session
                elif transport_id in runtime.transport_sessions:
                    trusted_session_id = runtime.transport_sessions[transport_id]
                elif transport_id:
                    assigned = f"mcp-session-{transport_id}"
                    runtime.transport_sessions[transport_id] = assigned
                    trusted_session_id = assigned
        except LookupError:
            pass
        return await call_tool(service, name, arguments, session_id=trusted_session_id)

    return runtime


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
        session_token = (
            request.headers.get("x-session-token")
            or request.query_params.get("session_token")
        )
        trusted_session_id = verify_session_token(session_token) or ""
        result = await call_tool(
            service, name, params.get("arguments", {}), session_id=trusted_session_id
        )
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
    session_token = (
        request.headers.get("x-session-token")
        or request.query_params.get("session_token")
    )
    verified_session = verify_session_token(session_token) if session_token else None
    writers_before = set(runtime.transport._read_stream_writers.keys())
    transport_uuid: str | None = None
    writer_key: Any = None
    try:
        async with runtime.transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            new_writers = set(runtime.transport._read_stream_writers.keys()) - writers_before
            if new_writers:
                writer_key = next(iter(new_writers))
                transport_uuid = (
                    writer_key.hex if hasattr(writer_key, "hex") else str(writer_key)
                )
                assigned_session = verified_session or f"mcp-session-{transport_uuid}"
                runtime.transport_sessions[transport_uuid] = assigned_session
            await runtime.server.run(
                read_stream,
                write_stream,
                runtime.server.create_initialization_options(),
            )
    finally:
        if transport_uuid is not None:
            runtime.transport_sessions.pop(transport_uuid, None)
        if writer_key is not None:
            runtime.transport._read_stream_writers.pop(writer_key, None)
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
