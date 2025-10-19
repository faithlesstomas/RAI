
"""
Example Python client for the proposed IPC API v2.

This client demonstrates how to use the new `run` command to execute
single agents and chains of agents.
"""
import json
import socket
import uuid
from typing import Any, Dict, List

SOCKET_FILE = "/tmp/rai-ipc.sock"

def run_ipc_command(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Connects to the IPC server, sends a single command, and returns the response.
    """
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(SOCKET_FILE)
            request_obj = {
                "request_id": str(uuid.uuid4()),
                "command": "run",
                "payload": payload,
            }
            request_str = json.dumps(request_obj) + '\n'
            sock.sendall(request_str.encode('utf-8'))

            with sock.makefile('r') as f:
                response_str = f.readline()
                if response_str:
                    return json.loads(response_str)
                return {"status": "error", "error_message": "No response from server."}
    except FileNotFoundError:
        return {"status": "error", "error_message": f"Socket file not found at {SOCKET_FILE}. Is the server running?"}
    except ConnectionRefusedError:
        return {"status": "error", "error_message": "Connection refused. Is the server running?"}

# --- Agent Configuration Definitions ---

summarizer_config = {
    "agent_class": "AgentAgno",
    "backend": "ollama",
    "model": "gemma3:4b",  # You can change this to any model you have installed
    "system_prompt": "You are an expert summarizer. Condense the given text into a few key points.",
}

translator_config = {
    "agent_class": "AgentAgno",
    "backend": "ollama",
    "model": "gemma3:4b",  # You can change this to any model you have installed
    "system_prompt": "You are a translator. Translate the given text into Polish.",
}

# --- Client-Side Function Definitions ---

def agent_summarizer(text: str) -> str:
    """
    A client-side function that executes the summarizer agent on the server.
    This demonstrates the (f(x)) pattern.
    """
    print("--- Calling Summarizer Agent ---")
    payload = {
        "input": text,
        "chain": [summarizer_config]
    }
    response = run_ipc_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("error_message", "Unknown error")
        raise RuntimeError(f"Summarizer agent failed: {error_msg}")

def agent_translator(text: str) -> str:
    """
    A client-side function that executes the translator agent on the server.
    """
    print("--- Calling Translator Agent ---")
    payload = {
        "input": text,
        "chain": [translator_config]
    }
    response = run_ipc_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("error_message", "Unknown error")
        raise RuntimeError(f"Translator agent failed: {error_msg}")

def run_server_side_chain(text: str) -> str:
    """
    Executes a full chain of agents on the server in a single API call.
    This is the more efficient approach.
    """
    print("\n--- Running Full Chain on Server-Side ---")
    payload = {
        "input": text,
        "chain": [summarizer_config, translator_config]
    }
    response = run_ipc_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("error_message", "Unknown error")
        raise RuntimeError(f"Server-side chain failed: {error_msg}")


if __name__ == "__main__":
    long_text = (
        "The new IPC architecture is based on a stateless `run` command. "
        "This command accepts a chain of agent configurations, allowing for "
        "flexible and dynamic execution of AI tasks. The server processes the "
        "chain sequentially, passing the output of one agent as the input to the next."
    )
    print(f"Initial text:\n{long_text}\n")

    # --- Example 1: Client-side composition ---
    # This will make two separate API calls to the server.
    print("--- Example 1: Client-Side Composition (Two API calls) ---")
    try:
        summary = agent_summarizer(long_text)
        print(f"Intermediate Summary:\n{summary}\n")
        final_translation = agent_translator(summary)
        print(f"Final Result (from client-side chain):\n{final_translation}\n")
    except RuntimeError as e:
        print(f"Error: {e}")


    # --- Example 2: Server-side composition ---
    # This makes only one, more efficient API call.
    print("\n--- Example 2: Server-Side Composition (One API call) ---")
    try:
        final_result_server = run_server_side_chain(long_text)
        print(f"Final Result (from server-side chain):\n{final_result_server}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
