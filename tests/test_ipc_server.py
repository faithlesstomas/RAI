
import asyncio
import json
import os
import socket
from contextlib import suppress
from pathlib import Path
from typing import Dict, Generator, Any
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from rai.ipc_server import start_server, SOCKET_FILE

IPC_SERVER_START_TIMEOUT = 5  # seconds


async def run_client_request(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to connect, send a request, and return the response asynchronously."""
    reader, writer = await asyncio.open_unix_connection(SOCKET_FILE)
    request_str = json.dumps(request_obj) + '\n'
    writer.write(request_str.encode('utf-8'))
    await writer.drain()

    response_data = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(response_data.decode())


@pytest.fixture
async def ipc_server() -> Generator[AsyncMock, Any, None]:
    """A pytest fixture to run the IPC server as an asyncio task."""
    # Patch the engine function directly at the point of use in the server module
    with patch('rai.ipc_server.run_chat', new_callable=AsyncMock) as mock_run_chat:
        with patch('rai.ipc_server.run_chain', new_callable=AsyncMock) as mock_run_chain:
            # Configure mocks to return successful payloads
            mock_run_chat.return_value = {
                "status": "success",
                "payload": {"content": "This is a mocked AI response."}
            }
            mock_run_chain.return_value = {"status": "success", "payload": {"content": "Chain processed."}}

            server_task = asyncio.create_task(start_server(test_mode=True))

            # Robustly wait for the server to be ready
            start_time = asyncio.get_event_loop().time()
            while True:
                try:
                    _, writer = await asyncio.open_unix_connection(SOCKET_FILE)
                    writer.close()
                    await writer.wait_closed()
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if asyncio.get_event_loop().time() - start_time > IPC_SERVER_START_TIMEOUT:
                        raise TimeoutError("IPC server failed to start in time.")
                    await asyncio.sleep(0.05)

            yield mock_run_chat, mock_run_chain

            # Teardown
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task
            if os.path.exists(SOCKET_FILE):
                os.remove(SOCKET_FILE)

@pytest.mark.asyncio
async def test_chat_command_success(ipc_server: tuple[AsyncMock, AsyncMock]) -> None:
    """Test a successful 'chat' command interaction."""
    mock_run_chat, _ = ipc_server
    prompt = "Hello, server!"
    request = {
        "request_id": "test-1",
        "command": "chat",
        "payload": {"prompt": prompt, "session_id": "test-session"}
    }

    response = await run_client_request(request)

    # Assert server response
    assert response["status"] == "success"
    assert response["request_id"] == "test-1"
    assert response["payload"]["content"] == "This is a mocked AI response."
    # Assert that the mock was called correctly
    mock_run_chat.assert_awaited_once_with(prompt=prompt, session_id="test-session")


@pytest.mark.asyncio
async def test_unknown_command(ipc_server: tuple[AsyncMock, AsyncMock]) -> None:
    """Test the server's response to an unknown command."""
    request = {
        "request_id": "test-2",
        "command": "nonexistent_command",
        "payload": {}
    }

    response = await run_client_request(request)

    assert response["status"] == "error"
    assert response["request_id"] == "test-2"
    assert "Unknown command" in response["error_message"]


@pytest.mark.asyncio
async def test_invalid_json(ipc_server: tuple[AsyncMock, AsyncMock]) -> None:
    """Test the server's response to a malformed JSON string."""
    reader, writer = await asyncio.open_unix_connection(SOCKET_FILE)
    malformed_request = "this is not json\n"
    writer.write(malformed_request.encode('utf-8'))
    await writer.drain()
    response_data = await reader.readline()
    writer.close()
    await writer.wait_closed()
    response = json.loads(response_data.decode())

    assert response["status"] == "error"
    assert response["request_id"] is None
    assert "Invalid JSON format" in response["error_message"]


@pytest.mark.asyncio
async def test_get_info_command(ipc_server: tuple[AsyncMock, AsyncMock]) -> None:
    """Test the 'get_info' command."""
    request = {
        "request_id": "test-info-1",
        "command": "get_info",
        "payload": {}
    }

    response = await run_client_request(request)

    assert response["status"] == "success"
    assert response["request_id"] == "test-info-1"
    payload = response["payload"]
    assert payload["backend"] == "test-backend"
    assert payload["model"] == "test-model"


@pytest.mark.asyncio
async def test_server_with_custom_config(tmp_path: Path) -> None:
    """Test running the server with a custom config file path."""
    config_content = {
        "active_session": "custom",
        "sessions": {
            "custom": {
                "model": "custom-model",
                "backend": "custom-backend",
                "system": "custom-system"
            }
        }
    }
    custom_config_file = tmp_path / "custom_config.json"
    custom_config_file.write_text(json.dumps(config_content))

    server_task = asyncio.create_task(start_server(test_mode=False, config_path=str(custom_config_file)))
    await asyncio.sleep(0.1)

    try:
        request = {"request_id": "test-custom-config", "command": "get_info"}
        response = await run_client_request(request)

        assert response["status"] == "success"
        payload = response["payload"]
        assert payload["model"] == "custom-model"
        assert payload["backend"] == "custom-backend"

    finally:
        server_task.cancel()
        with suppress(asyncio.CancelledError):
            await server_task
        if os.path.exists(SOCKET_FILE):
            os.remove(SOCKET_FILE)


