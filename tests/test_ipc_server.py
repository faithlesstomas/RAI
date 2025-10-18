
import asyncio
import json
import os
import socket
import time
from multiprocessing import Process
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

# Since tests are run from the root, we can import the server module
from rai.ipc_server import run_ipc_server, SOCKET_FILE

IPC_SERVER_START_TIMEOUT = 5  # seconds


@pytest.fixture
def ipc_server():
    """A pytest fixture to run the IPC server in a background process."""
    # Run the server in test mode
    with patch('rai.adapters.agno.setup_agent') as mock_setup_agent:
        mock_agent = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "This is a mocked AI response."
        mock_response.tool_calls = None
        mock_agent.arun.return_value = mock_response
        mock_setup_agent.return_value = (mock_agent, [])
        with patch('rai.core.sys.exit') as mock_exit:
            server_process = Process(target=run_ipc_server, args=(True,))
            server_process.start()

            # Wait for the server to start and the socket file to be created
            start_time = time.time()
            while not os.path.exists(SOCKET_FILE):
                if time.time() - start_time > IPC_SERVER_START_TIMEOUT:
                    server_process.terminate()
                    raise TimeoutError("IPC server failed to start in time.")
                time.sleep(0.1)

            yield mock_exit, mock_setup_agent # This is where the test runs

            # Teardown: stop the server process
            server_process.terminate()
            server_process.join()
            if os.path.exists(SOCKET_FILE):
                os.remove(SOCKET_FILE)


def run_client_request(request_obj):
    """Helper function to connect, send a request, and return the response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(SOCKET_FILE)
        request_str = json.dumps(request_obj) + '\n'
        sock.sendall(request_str.encode('utf-8'))
        with sock.makefile('r') as f:
            response_str = f.readline()
            return json.loads(response_str)


@pytest.mark.usefixtures("ipc_server")
def test_chat_command_success() -> None:
    """Test a successful 'chat' command interaction."""
    # Define the request
    prompt = "Hello, server!"
    request = {
        "request_id": "test-1",
        "command": "chat",
        "payload": {"prompt": prompt, "session_id": "test-session"}
    }

    # Send request and get response
    response = run_client_request(request)

    # Assertions
    assert response["status"] == "success"
    assert response["request_id"] == "test-1"
    assert response["payload"]["content"] == "This is a mocked AI response."


@pytest.mark.usefixtures("ipc_server")
def test_unknown_command() -> None:
    """Test the server's response to an unknown command."""
    request = {
        "request_id": "test-2",
        "command": "nonexistent_command",
        "payload": {}
    }

    response = run_client_request(request)

    assert response["status"] == "error"
    assert response["request_id"] == "test-2"
    assert "Unknown command" in response["error_message"]


@pytest.mark.usefixtures("ipc_server")
def test_invalid_json() -> None:
    """Test the server's response to a malformed JSON string."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(SOCKET_FILE)
        malformed_request = "this is not json\n"
        sock.sendall(malformed_request.encode('utf-8'))
        with sock.makefile('r') as f:
            response_str = f.readline()
            response = json.loads(response_str)

    assert response["status"] == "error"
    assert response["request_id"] is None
    assert "Invalid JSON format" in response["error_message"]


@pytest.mark.usefixtures("ipc_server")
def test_get_info_command() -> None:
    """Test the 'get_info' command."""
    request = {
        "request_id": "test-info-1",
        "command": "get_info",
        "payload": {}
    }

    response = run_client_request(request)

    assert response["status"] == "success"
    assert response["request_id"] == "test-info-1"
    payload = response["payload"]
    assert "backend" in payload
    assert "model" in payload
    assert "system_prompt" in payload
    assert payload["server_version"] == "0.1.0"


def test_server_with_custom_config(tmp_path) -> None:
    """Test running the server with a custom config file path."""
    # Create a temporary config file
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

    # Run the server in a separate process with the custom config path
    server_process = Process(target=run_ipc_server, args=(False, str(custom_config_file)))
    server_process.start()

    try:
        # Wait for the server to start
        start_time = time.time()
        while not os.path.exists(SOCKET_FILE):
            if time.time() - start_time > IPC_SERVER_START_TIMEOUT:
                raise TimeoutError("IPC server (custom config) failed to start in time.")
            time.sleep(0.1)

        # Send a get_info request
        request = {"request_id": "test-custom-config", "command": "get_info"}
        response = run_client_request(request)

        # Assert that the server is using the custom config
        assert response["status"] == "success"
        payload = response["payload"]
        assert payload["model"] == "custom-model"
        assert payload["backend"] == "custom-backend"

    finally:
        # Clean up the server process
        server_process.terminate()
        server_process.join()
        if os.path.exists(SOCKET_FILE):
            os.remove(SOCKET_FILE)

