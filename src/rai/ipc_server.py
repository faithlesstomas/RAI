"""
IPC Server for the Rich AI Assistant.

This module provides a server that listens on a Unix socket for commands,
processes them using the core engine, and returns the results.
"""
import asyncio
import json
import os
from functools import partial
from typing import Optional, Any, Dict

from rich.console import Console

from .core import RAI_CONFIG, load_config
from .engine import run_chat, run_chain

SOCKET_FILE = "/tmp/rai-ipc.sock"
console = Console()


class CommandHandler:
    """Handles processing of commands received by the IPC server."""

    def __init__(self, config_path: Optional[str] = None, test_mode: bool = False) -> None:
        self.config_path = config_path
        if test_mode:
            self.session_config = {
                "model": "test-model",
                "backend": "test-backend",
                "system": "test-system-prompt",
            }
        else:
            app_config = load_config(path=self.config_path)
            session_to_use = app_config.get("active_session", "default")
            self.session_config = app_config.get("sessions", {}).get(session_to_use, {})

    async def handle_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'run' command by delegating to the engine's run_chain function."""
        chain_input = payload.get("input")
        chain_configs = payload.get("chain")

        if not chain_input or not chain_configs:
            return _build_error_response("Missing 'input' or 'chain' in payload.")

        return await run_chain(chain_input=chain_input, chain_configs=chain_configs)

    async def handle_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'chat' command by delegating to the engine."""
        prompt = payload.get("prompt")
        session_id = payload.get("session_id", "default-ipc-session")

        if not prompt:
            return _build_error_response("Missing prompt in payload.")

        return await run_chat(prompt=prompt, session_id=session_id)

    def handle_get_info(self, _payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handles the 'get_info' command."""
        return {
            "status": "success",
            "payload": {
                "backend": self.session_config.get("backend"),
                "model": self.session_config.get("model"),
                "system_prompt": self.session_config.get("system"),
                "server_version": "0.1.0",  # Hardcoded for now
            },
        }


def _build_error_response(message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Builds a standard error response dictionary."""
    return {"request_id": request_id, "status": "error", "error_message": message}


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    test_mode: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Coroutine to handle a single client connection."""
    peername = writer.get_extra_info("peername")
    console.log(f"IPC: Client connected: {peername}")

    command_handler = CommandHandler(config_path, test_mode)

    command_map = {
        "run": command_handler.handle_run,
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
                if asyncio.iscoroutinefunction(handler_method):
                    response_data = await handler_method(payload)
                else:
                    response_data = handler_method(payload)
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


async def start_server(test_mode: bool = False, config_path: Optional[str] = None) -> None:
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


def run_ipc_server(test_mode: bool = False, config_path: Optional[str] = None) -> None:
    """Entry point to run the asyncio server."""
    try:
        asyncio.run(start_server(test_mode=test_mode, config_path=config_path))
    except KeyboardInterrupt:
        console.log("IPC Server shutting down.")
    finally:
        if os.path.exists(SOCKET_FILE):
            os.remove(SOCKET_FILE)
