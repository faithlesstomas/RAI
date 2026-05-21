"""
FastAPI router that implements the Model Context Protocol (MCP) server over SSE.
Provides secure system tools (shell, python code execution) under sandboxing and HITL.
"""
import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.types import Receive, Scope, Send

from rai.tools.shell import run_secure_shell_command
from rai.tools.python import run_secure_python_code


class EmptyResponse(Response):
    """
    A response class that does nothing on call, preventing FastAPI from trying
    to write headers or body when they have already been sent by the MCP transport.
    """
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        pass


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/mcp",
    tags=["MCP"],
)

# 1. Initialize MCP Server and SSE Transport
mcp_server = Server("rai-secure-gateway")
sse_transport = SseServerTransport("/api/v1/mcp/messages")


# 2. Register MCP Tools
@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Exposes secure local execution tools to external MCP clients.
    """
    return [
        types.Tool(
            name="run_shell_command",
            description="Execute shell commands securely in a sandboxed Linux workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash shell command to execute."
                    },
                    "allow_network": {
                        "type": "boolean",
                        "description": "Allow network access inside the sandboxed environment (requires HITL approval).",
                        "default": False
                    }
                },
                "required": ["command"]
            }
        ),
        types.Tool(
            name="run_python_code",
            description="Execute Python code securely in an isolated, sandboxed space.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python script content to run."
                    },
                    "allow_network": {
                        "type": "boolean",
                        "description": "Allow network access inside the sandboxed environment (requires HITL approval).",
                        "default": False
                    }
                },
                "required": ["code"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
    """
    Handles MCP tool calls by routing them to the secure sandboxed runners.
    """
    logger.info("MCP Tool called: '%s' with arguments: %s", name, arguments)
    if name == "run_shell_command":
        cmd = arguments.get("command")
        allow_network = arguments.get("allow_network", False)
        if not cmd:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Error: Missing 'command' argument.")],
                isError=True
            )
        result = await run_secure_shell_command(cmd, allow_network=allow_network)
        is_error = result.startswith("Execution Error") or result.startswith("Execution Failure")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result)],
            isError=is_error
        )
    elif name == "run_python_code":
        code = arguments.get("code")
        allow_network = arguments.get("allow_network", False)
        if not code:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Error: Missing 'code' argument.")],
                isError=True
            )
        result = await run_secure_python_code(code, allow_network=allow_network)
        is_error = result.startswith("Execution Error") or result.startswith("Execution Failure")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result)],
            isError=is_error
        )
    else:
        logger.warning("MCP requested unknown tool: '%s'", name)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: Unknown tool name '{name}'.")],
            isError=True
        )


async def handle_stateless_mcp(request: Request) -> Response:
    """
    Handles stateless JSON-RPC POST messages directly without establishing an SSE stream.
    Used by MCP clients (like Antigravity CLI) that do not support standard SSE sessions.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.exception("Failed to parse JSON body for stateless MCP request")
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: expected a JSON-RPC request object"
                }
            }
        )

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    logger.info("Stateless MCP request received: method='%s', id=%s", method, req_id)

    if not method:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: missing method"
                }
            }
        )

    response_obj: Response

    # Handlers for specific methods
    if method == "initialize":
        response_obj = JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": "rai-secure-gateway",
                        "version": "1.19.0"
                    }
                }
            }
        )

    elif method == "notifications/initialized":
        response_obj = JSONResponse(status_code=200, content={})

    elif method == "ping":
        response_obj = JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }
        )

    elif method == "tools/list":
        try:
            tools = await list_tools()
            serialized_tools = [t.model_dump(exclude_none=True) for t in tools]
            response_obj = JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": serialized_tools
                    }
                }
            )
        except Exception as e:
            logger.exception("Failed to list tools in stateless mode")
            response_obj = JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error during tools/list: {str(e)}"
                    }
                }
            )

    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not name:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' is required for tools/call"
                    }
                }
            )
        try:
            res = await call_tool(name, arguments)
            response_obj = JSONResponse(
                status_code=200,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": res.model_dump(exclude_none=True)
                }
            )
        except Exception as e:
            logger.exception("Failed to call tool '%s' in stateless mode", name)
            response_obj = JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error calling tool: {str(e)}"
                    }
                }
            )

    else:
        logger.warning("Method '%s' is not supported in stateless mode", method)
        response_obj = JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }
        )

    return response_obj


# 3. Mount SSE transport endpoints onto the router
@router.get("/sse")
async def mcp_sse(request: Request) -> Response:
    """
    Establishes Server-Sent Events (SSE) stream transport for Model Context Protocol.
    """
    logger.info("Incoming SSE connection from MCP client")
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        logger.info("SSE transport connected. Running MCP server session...")
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )
        logger.info("MCP server session completed")
    return EmptyResponse()


@router.post("/messages")
async def mcp_messages(request: Request) -> Response:
    """
    Post endpoint to handle client messages over HTTP.
    """
    if "session_id" not in request.query_params:
        return await handle_stateless_mcp(request)

    logger.debug("Received POST message from MCP client")
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return EmptyResponse()


@router.post("/sse")
async def mcp_post_sse(request: Request) -> Response:
    """
    Fallback POST endpoint if client sends message to the main SSE endpoint instead of messages endpoint.
    """
    if "session_id" not in request.query_params:
        return await handle_stateless_mcp(request)

    logger.info("Received POST message on fallback /sse endpoint")
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
    return EmptyResponse()

