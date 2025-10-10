"""
IPC Server for the Rich AI Assistant.

This module provides a server that listens on a Unix socket for commands,
processes them using the Agno agent, and returns the results.
"""
import asyncio
import json
import os
from functools import partial
from typing import Optional, Any, Dict
from unittest.mock import AsyncMock

from rich.console import Console

from .core import RAI_CONFIG, load_config, setup_agent, console

SOCKET_FILE = "/tmp/rai-ipc.sock"
console = Console()


class CommandHandler:
    """Handles processing of commands received by the IPC server."""

    def __init__(self, agent: Any, config_path: Optional[str] = None):
        self.agent = agent
        self.config_path = config_path

    async def handle_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'chat' command."""
        prompt = payload.get("prompt")
        session_id = payload.get("session_id", "default-ipc-session")

        if not self.agent or self.agent.session_id != session_id:
            console.log(f"IPC: Setting up new agent for session: {session_id}")
            self.agent, _ = setup_agent(session_id=session_id)

        if not prompt:
            return _build_error_response("Missing prompt in payload.")

        ai_response = await self.agent.arun(prompt)
        content = ai_response.content if ai_response else ""
        tool_calls = getattr(ai_response, "tool_calls", None)

        return {
            "status": "success",
            "payload": {"content": content, "tool_calls": tool_calls},
        }

    def handle_get_info(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'get_info' command."""
        return {
            "status": "success",
            "payload": {
                "backend": RAI_CONFIG.get("backend"),
                "model": RAI_CONFIG.get("model"),
                "system_prompt": RAI_CONFIG.get("system"),
                "server_version": "0.1.0",  # Hardcoded for now
            },
        }


def _build_error_response(message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Builds a standard error response dictionary."""
    return {"request_id": request_id, "status": "error", "error_message": message}


async def _initialize_agent_and_config(test_mode: bool, config_path: Optional[str]):
    """Initializes the agent and configuration based on the mode."""
    if test_mode:
        agent = AsyncMock()

        class MockResponse:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        agent.arun.return_value = MockResponse("This is a mocked AI response.")
        RAI_CONFIG.update({
            "model": "test-model",
            "backend": "test-backend",
            "system": "test-system-prompt",
        })
        return agent

    app_config = load_config(path=config_path)
    session_to_use = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(session_to_use, {})
    RAI_CONFIG.update({
        "model": session_config.get("model", "gemma3:1b"),
        "backend": session_config.get("backend", "ollama"),
        "system": session_config.get("system", "You are a helpful AI assistant."),
    })
    # The agent is initialized dynamically in the chat handler based on session_id
    return None


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    test_mode: bool = False,
    config_path: Optional[str] = None,
):
    """Coroutine to handle a single client connection."""
    peername = writer.get_extra_info("peername")
    console.log(f"IPC: Client connected: {peername}")

    agent = await _initialize_agent_and_config(test_mode, config_path)
    command_handler = CommandHandler(agent, config_path)

    command_map = {
        "chat": command_handler.handle_chat,
        "get_info": command_handler.handle_get_info,
    }

    try:
        while True:
            try:
                data = await reader.readuntil(b"\n")
                if not data:
                    break
            except asyncio.IncompleteReadError:
                break  # Client disconnected gracefully

            request_str = data.decode().strip()
            request_id = None
            try:
                request = json.loads(request_str)
                request_id = request.get("request_id")
                console.log(f"IPC: Received request: {request}")
            except json.JSONDecodeError:
                response = _build_error_response("Invalid JSON format.")
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
                continue

            command_name = request.get("command")
            handler_method = command_map.get(command_name)

            if handler_method:
                payload = request.get("payload", {})
                response_data = await handler_method(payload) if asyncio.iscoroutinefunction(handler_method) else handler_method(payload)
                response = {"request_id": request_id, **response_data}
            else:
                response = _build_error_response(f"Unknown command: {command_name}", request_id)

            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()

    except ConnectionResetError:
        console.log(f"IPC: Client {peername} reset the connection.")
    except Exception as e:
        console.log(f"[bold red]IPC: An error occurred with client {peername}: {e}[/bold red]")
    finally:
        console.log(f"IPC: Client disconnected: {peername}")
        writer.close()
        await writer.wait_closed()


async def start_server(test_mode: bool = False, config_path: Optional[str] = None):
    """Starts the IPC server on the Unix socket."""
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)

    handler = partial(handle_client, test_mode=test_mode, config_path=config_path)
    server = await asyncio.start_unix_server(handler, path=SOCKET_FILE)

    os.chmod(SOCKET_FILE, 0o600)

    addr = server.sockets[0].getsockname()
    console.log(f"IPC Server listening on {addr}")

    async with server:
        await server.serve_forever()


def run_ipc_server(test_mode: bool = False, config_path: Optional[str] = None):
    """Entry point to run the asyncio server."""
    try:
        asyncio.run(start_server(test_mode=test_mode, config_path=config_path))
    except KeyboardInterrupt:
        console.log("IPC Server shutting down.")
    finally:
        if os.path.exists(SOCKET_FILE):
            os.remove(SOCKET_FILE)