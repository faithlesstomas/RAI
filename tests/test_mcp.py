"""
Unit tests for the Model Context Protocol (MCP) router and server.
"""
import pytest
from unittest.mock import AsyncMock, patch
from mcp import types

from rai.routers.mcp import mcp_server, list_tools, call_tool


@pytest.mark.asyncio
async def test_mcp_list_tools() -> None:
    """
    Tests that the MCP server exposes the secure execution tools with correct schemas.
    """
    tools = await list_tools()
    assert len(tools) == 2

    tool_names = [tool.name for tool in tools]
    assert "run_shell_command" in tool_names
    assert "run_python_code" in tool_names

    # Verify run_shell_command schema
    shell_tool = next(t for t in tools if t.name == "run_shell_command")
    assert shell_tool.description is not None
    assert "command" in shell_tool.inputSchema["required"]
    assert "allow_network" in shell_tool.inputSchema["properties"]

    # Verify run_python_code schema
    python_tool = next(t for t in tools if t.name == "run_python_code")
    assert python_tool.description is not None
    assert "code" in python_tool.inputSchema["required"]


@pytest.mark.asyncio
@patch("rai.routers.mcp.run_secure_shell_command")
async def test_mcp_call_shell_command(mock_run_shell: AsyncMock) -> None:
    """
    Tests that run_shell_command invokes run_secure_shell_command.
    """
    mock_run_shell.return_value = "hello from sandbox"
    
    # 1. Success execution
    result = await call_tool("run_shell_command", {"command": "echo hello", "allow_network": True})
    assert isinstance(result, types.CallToolResult)
    assert not result.isError
    assert len(result.content) == 1
    assert result.content[0].text == "hello from sandbox"
    mock_run_shell.assert_awaited_once_with("echo hello", allow_network=True)

    # 2. Error handling (e.g. execution error)
    mock_run_shell.reset_mock()
    mock_run_shell.return_value = "Execution Error: Command was blocked"
    result = await call_tool("run_shell_command", {"command": "sudo rm -rf /"})
    assert result.isError
    assert "Execution Error" in result.content[0].text


@pytest.mark.asyncio
@patch("rai.routers.mcp.run_secure_python_code")
async def test_mcp_call_python_code(mock_run_python: AsyncMock) -> None:
    """
    Tests that run_python_code invokes run_secure_python_code.
    """
    mock_run_python.return_value = "python output"
    
    # 1. Success execution
    result = await call_tool("run_python_code", {"code": "print('hello')", "allow_network": False})
    assert isinstance(result, types.CallToolResult)
    assert not result.isError
    assert len(result.content) == 1
    assert result.content[0].text == "python output"
    mock_run_python.assert_awaited_once_with("print('hello')", allow_network=False)

    # 2. Invalid missing parameters
    result = await call_tool("run_python_code", {})
    assert result.isError
    assert "Error: Missing" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_call_unknown_tool() -> None:
    """
    Tests calling a tool that is not registered.
    """
    result = await call_tool("non_existent_tool", {})
    assert result.isError
    assert "Error: Unknown tool" in result.content[0].text
