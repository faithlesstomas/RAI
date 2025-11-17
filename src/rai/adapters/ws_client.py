"""Adapter for the rai WebSocket client."""
import asyncio
import json
import socket
from typing import Any, Dict, List

import websockets
from returns.result import Failure, Result, Success
from websockets.exceptions import WebSocketException

from ..core import console, error_console
from ..exceptions import ChainExecutionError


class WebSocketAdapter:
    """
    An adapter that implements the Processor protocol by communicating
    with a remote rai server over WebSockets.
    """

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self.run_config = agent_config
        self.server_uri = agent_config.get("server_uri", "ws://127.0.0.1:8000/ws/v1/chat")
        self._websocket: websockets.WebSocketClientProtocol | None = None  # pylint: disable=no-member

    async def connect(self) -> Result[None, Exception]:
        """Establishes a WebSocket connection if one is not already open."""
        is_connected = False
        if self._websocket:
            try:
                await self._websocket.ping()
                is_connected = True
            except websockets.exceptions.ConnectionClosed:
                is_connected = False

        if not is_connected:
            try:
                if self.server_uri.startswith("unix://"):
                    uds_path = self.server_uri[len("unix://") :]
                    error_console.print(f"[dim]Connecting to Unix Domain Socket at {uds_path}...[/dim]")

                    # Manually create and connect the socket to avoid transport conflicts.
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    loop = asyncio.get_running_loop()
                    await loop.sock_connect(sock, uds_path)

                    uri_to_connect = "ws://localhost/ws/v1/chat"
                    self._websocket = await websockets.connect(uri_to_connect, sock=sock)

                else:
                    error_console.print(f"[dim]Connecting to WebSocket server at {self.server_uri}...[/dim]")
                    self._websocket = await websockets.connect(self.server_uri)

                error_console.print("[green]WebSocket connection established.[/green]")
                return Success(None)
            except Exception as e:
                error_console.print(
                    "[bold red]Connection Error:[/bold red]"
                    f" Could not connect to the rai server at {self.server_uri}."
                )
                error_console.print(f"[dim]Is the server running? ('rai serve'). Error: {e}[/dim]")
                return Failure(e)
        return Success(None)


    async def arun(self, prompt: str) -> Result[Dict[str, Any], Exception]:
        """Sends a prompt to the server and returns the response as a Result."""
        connect_result = await self.connect()
        if isinstance(connect_result, Failure):
            return connect_result

        if not self._websocket:
            return Failure(ChainExecutionError("WebSocket is not connected."))

        try:
            payload = {
                "chain_input": prompt,
                "chain_configs": [self.run_config],
            }
            await self._websocket.send(json.dumps(payload))

            response_str = await self._websocket.recv()
            response = json.loads(response_str)

            if response.get("status") == "success":
                return Success(response.get("payload", {}))

            return Failure(ChainExecutionError(response.get("detail", "Unknown server error")))
        except (WebSocketException, ConnectionRefusedError) as e:
            error_console.print(
                "[bold red]Connection Error:[/bold red] The connection to the server was lost."
            )
            return Failure(e)

    def get_history(self) -> List[Dict[str, str]]:
        """
        (Not Implemented) Returns the chat history.
        History is managed server-side in this adapter.
        """
        # TODO: Implement a server endpoint to fetch history for a session.
        console.print("[dim](History is managed by the server)[/dim]")
        return []

    def clear_history(self) -> None:
        """
        (Not Implemented) Clears the chat history.
        History is managed server-side in this adapter.
        """
        # TODO: Implement a server endpoint to clear history for a session.
        console.print("[dim](History is managed by the server)[/dim]")

    async def close(self) -> None:
        """Closes the WebSocket connection."""
        if self._websocket:
            try:
                await self._websocket.close()
            except websockets.exceptions.ConnectionClosed:
                # Connection is already closed, which is fine.
                pass
