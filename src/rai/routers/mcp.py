"""
FastAPI router that implements the Model Context Protocol (MCP) server over SSE.
Provides secure system tools (shell, python code execution) under sandboxing and HITL.
"""
import logging
from fastapi import APIRouter, Request
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from rai.tools.shell import run_secure_shell_command
from rai.tools.python import run_secure_python_code

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


# 3. Mount SSE transport endpoints onto the router
@router.get("/sse")
async def mcp_sse(request: Request) -> None:
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


@router.post("/messages")
async def mcp_messages(request: Request) -> None:
    """
    Post endpoint to handle client messages over HTTP.
    """
    logger.debug("Received POST message from MCP client")
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )


@router.post("/sse")
async def mcp_post_sse(request: Request) -> None:
    """
    Fallback POST endpoint if client sends message to the main SSE endpoint instead of messages endpoint.
    """
    logger.info("Received POST message on fallback /sse endpoint")
    await sse_transport.handle_post_message(
        request.scope, request.receive, request._send
    )
