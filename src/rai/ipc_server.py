
"""
IPC Server for the Rich AI Assistant.

This module provides a server that listens on a Unix socket for commands,
processes them using the Agno agent, and returns the results.
"""
import asyncio
import json
import os
from typing import Optional
from rich.console import Console

# We need to import the agent setup logic from our existing CLI module.
# This is a pragmatic way to reuse code without a major refactor.
from unittest.mock import AsyncMock

# We need to import the agent setup logic from our existing CLI module.
# This is a pragmatic way to reuse code without a major refactor.
from .cli import setup_agent, RAI_CONFIG, load_config

SOCKET_FILE = "/tmp/rai-ipc.sock"
console = Console()


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, test_mode: bool = False, config_path: Optional[str] = None):
    """
    Coroutine to handle a single client connection.
    Each client gets its own instance of this coroutine.
    """
    peername = writer.get_extra_info('peername')
    console.log(f"IPC: Client connected: {peername}")

    agent = None
    if test_mode:
        agent = AsyncMock()
        class MockResponse:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
        agent.arun.return_value = MockResponse("This is a mocked AI response.")
        RAI_CONFIG["model"] = "test-model"
        RAI_CONFIG["backend"] = "test-backend"
        RAI_CONFIG["system"] = "test-system-prompt"
    else:
        app_config = load_config(path=config_path)
        session_to_use = app_config.get("active_session", "default")
        session_config = app_config.get("sessions", {}).get(session_to_use, {})
        RAI_CONFIG["model"] = session_config.get("model", "gemma3:1b")
        RAI_CONFIG["backend"] = session_config.get("backend", "ollama")
        RAI_CONFIG["system"] = session_config.get("system", "You are a helpful AI assistant.")

    try:
        while True:
            # ... (rest of the function is the same)
            # We use readuntil to handle our newline-delimited protocol.
            try:
                data = await reader.readuntil(b'\n')
                if not data:
                    break
            except asyncio.IncompleteReadError:
                # Client disconnected gracefully
                break

            request_str = data.decode().strip()
            
            # 2. Parse the JSON request.
            try:
                request = json.loads(request_str)
                console.log(f"IPC: Received request: {request}")
            except json.JSONDecodeError:
                error_response = {
                    "request_id": None,
                    "status": "error",
                    "error_message": "Invalid JSON format."
                }
                writer.write(json.dumps(error_response).encode() + b'\n')
                await writer.drain()
                continue

            # 3. Process the command.
            command = request.get("command")
            response = {}

            if command == "chat":
                prompt = request.get("payload", {}).get("prompt")
                session_id = request.get("payload", {}).get("session_id", "default-ipc-session")

                if not test_mode and (not agent or agent.session_id != session_id):
                    console.log(f"IPC: Setting up new agent for session: {session_id}")
                    agent, _ = setup_agent(session_id=session_id)

                if prompt and agent:
                    # Call the agent to get the AI response
                    ai_response = await agent.arun(prompt)
                    
                    content = ai_response.content if ai_response else ""
                    tool_calls = getattr(ai_response, 'tool_calls', None) if ai_response else None

                    response = {
                        "request_id": request.get("request_id"),
                        "status": "success",
                        "payload": {
                            "content": content,
                            "tool_calls": tool_calls
                        }
                    }
                else:
                    response = {
                        "request_id": request.get("request_id"),
                        "status": "error",
                        "error_message": "Missing prompt in payload."
                    }
            
            elif command == "get_info":
                response = {
                    "request_id": request.get("request_id"),
                    "status": "success",
                    "payload": {
                        "backend": RAI_CONFIG.get("backend"),
                        "model": RAI_CONFIG.get("model"),
                        "system_prompt": RAI_CONFIG.get("system"),
                        "server_version": "0.1.0" # Hardcoded for now
                    }
                }

            else:
                response = {
                    "request_id": request.get("request_id"),
                    "status": "error",
                    "error_message": f"Unknown command: {command}"
                }

            # 4. Send the response back to the client.
            writer.write(json.dumps(response).encode() + b'\n')
            await writer.drain()

    except Exception as e:
        console.log(f"[bold red]IPC: An error occurred with client {peername}: {e}[/bold red]")
    finally:
        console.log(f"IPC: Client disconnected: {peername}")
        writer.close()
        await writer.wait_closed()


async def start_server(test_mode: bool = False, config_path: Optional[str] = None):
    """
    Starts the IPC server on the Unix socket.
    """
    # Clean up old socket file if it exists
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)

    # Pass test_mode and config_path to the client handler
    handler = lambda r, w: handle_client(r, w, test_mode=test_mode, config_path=config_path)
    server = await asyncio.start_unix_server(handler, path=SOCKET_FILE)

    # Set socket permissions to be user-only
    os.chmod(SOCKET_FILE, 0o600)

    addr = server.sockets[0].getsockname()
    console.log(f"IPC Server listening on {addr}")

    async with server:
        await server.serve_forever()

def run_ipc_server(test_mode: bool = False, config_path: Optional[str] = None):
    """
    Entry point to run the asyncio server.
    """
    try:
        asyncio.run(start_server(test_mode=test_mode, config_path=config_path))
    except KeyboardInterrupt:
        console.log("IPC Server shutting down.")
    finally:
        if os.path.exists(SOCKET_FILE):
            os.remove(SOCKET_FILE)


