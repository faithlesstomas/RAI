"""
Example Python client for the API, using HTTP over a Unix socket.

This client demonstrates how to use the `run` endpoint to execute
chains of agents.
"""
import os
from typing import Any, Dict

import httpx

# The path to the Unix socket the server is listening on.
# This must match the --uds parameter of the `rai serve` command.
SOCKET_FILE = os.environ.get("RAI_SOCKET_FILE", "/tmp/rai.sock")
BASE_URL = "http://localhost" # Hostname is ignored when using UDS transport

def run_api_command(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Connects to the server via a Unix socket, sends a single command,
    and returns the response.
    """
    try:
        transport = httpx.HTTPTransport(uds=SOCKET_FILE)
        with httpx.Client(transport=transport, base_url=BASE_URL) as client:
            response = client.post("/api/v1/run", json=payload, timeout=120)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
            return response.json()
    except FileNotFoundError:
        return {"status": "error", "detail": f"Socket file not found at {SOCKET_FILE}. Is the server running with --uds?"}
    except ConnectionRefusedError:
        return {"status": "error", "detail": "Connection refused. Is the server running?"}
    except httpx.HTTPStatusError as e:
        # Try to return the JSON error detail from the server if possible
        try:
            return e.response.json()
        except Exception:
            return {"status": "error", "detail": f"HTTP error: {e}"}


# --- Agent Configuration Definitions ---

summarizer_config = {
    "agent_class": "agno",
    "backend": "ollama",
    "model": "gemma2:9b",
    "system_prompt": "You are an expert summarizer. Condense the given text into a few key points.",
}

translator_config = {
    "agent_class": "agno",
    "backend": "ollama",
    "model": "gemma2:9b",
    "system_prompt": "You are a translator. Translate the given text into Polish.",
}

# --- Client-Side Function Definitions ---

def agent_summarizer(text: str) -> str:
    """
    A client-side function that executes the summarizer agent on the server.
    """
    print("--- Calling Summarizer Agent ---")
    payload = {
        "chain_input": text,
        "chain_configs": [summarizer_config]
    }
    response = run_api_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("detail", "Unknown error")
        raise RuntimeError(f"Summarizer agent failed: {error_msg}")

def agent_translator(text: str) -> str:
    """
    A client-side function that executes the translator agent on the server.
    """
    print("--- Calling Translator Agent ---")
    payload = {
        "chain_input": text,
        "chain_configs": [translator_config]
    }
    response = run_api_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("detail", "Unknown error")
        raise RuntimeError(f"Translator agent failed: {error_msg}")

def run_server_side_chain(text: str) -> str:
    """
    Executes a full chain of agents on the server in a single API call.
    """
    print("\n--- Running Full Chain on Server-Side ---")
    payload = {
        "chain_input": text,
        "chain_configs": [summarizer_config, translator_config]
    }
    response = run_api_command(payload)
    if response.get("status") == "success":
        return response.get("payload", {}).get("content", "")
    else:
        error_msg = response.get("detail", "Unknown error")
        raise RuntimeError(f"Server-side chain failed: {error_msg}")


if __name__ == "__main__":
    long_text = (
        "The new API architecture is based on a stateless `run` endpoint. "
        "This endpoint accepts a chain of agent configurations, allowing for "
        "flexible and dynamic execution of AI tasks. The server processes the "
        "chain sequentially, passing the output of one agent as the input to the next."
    )
    print(f"Initial text:\n{long_text}\n")

    # --- Example 1: Client-side composition ---
    print("--- Example 1: Client-Side Composition (Two API calls) ---")
    try:
        summary = agent_summarizer(long_text)
        print(f"Intermediate Summary:\n{summary}\n")
        final_translation = agent_translator(summary)
        print(f"Final Result (from client-side chain):\n{final_translation}\n")
    except RuntimeError as e:
        print(f"Error: {e}")


    # --- Example 2: Server-side composition ---
    print("\n--- Example 2: Server-Side Composition (One API call) ---")
    try:
        final_result_server = run_server_side_chain(long_text)
        print(f"Final Result (from server-side chain):\n{final_result_server}\n")
    except RuntimeError as e:
        print(f"Error: {e}")
