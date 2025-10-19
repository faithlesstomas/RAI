
"""
Simple synchronous client to test the RAI IPC server.
"""
import json
import socket
import sys

SOCKET_FILE = "/tmp/rai-ipc.sock"

def get_server_info(sock: socket.socket) -> None:
    """Sends a get_info request and prints the response."""
    print("\n--- Getting Server Info ---")
    request_obj = {
        "request_id": "client-info-req-1",
        "command": "get_info",
        "payload": {}
    }
    request_str = json.dumps(request_obj) + '\n'
    sock.sendall(request_str.encode('utf-8'))

    with sock.makefile('r') as f:
        response_str = f.readline()
        if response_str:
            print("Received raw info response:", response_str.strip())
            response_obj = json.loads(response_str)
            print("\nServer Configuration:")
            print(json.dumps(response_obj.get("payload"), indent=2))
        else:
            print("Did not receive server info.")

def send_chat_request(sock: socket.socket, prompt: str) -> None:
    """Sends a chat request and prints the response."""
    print("\n--- Sending Chat Prompt ---")
    request_obj = {
        "request_id": "client-chat-req-1",
        "command": "chat",
        "payload": {
            "prompt": prompt,
            "session_id": "example-session"
        }
    }
    request_str = json.dumps(request_obj) + '\n'

    print(f"Sending: {request_obj}")
    sock.sendall(request_str.encode('utf-8'))

    with sock.makefile('r') as f:
        response_str = f.readline()
        if response_str:
            print(f"\nReceived raw chat response: {response_str.strip()}")
            response_obj = json.loads(response_str)
            print("\nAI Response Content:")
            print(response_obj.get("payload", {}).get("content"))
        else:
            print("\nReceived no response from server.")


def main() -> None:
    """Main function to connect and run requests."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        print(f"Connecting to {SOCKET_FILE}...")
        sock.connect(SOCKET_FILE)
        print("Connected.")
    except FileNotFoundError:
        print(f"Error: Socket file not found at {SOCKET_FILE}. Is the server running?")
        return
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the server running and accepting connections?")
        return

    try:
        # First, get server info
        get_server_info(sock)

        # Then, send a chat prompt
        if len(sys.argv) > 1:
            user_prompt = " ".join(sys.argv[1:])
        else:
            print("\nNo prompt provided. Sending a default prompt...")
            user_prompt = "Tell me a joke about Python."
        
        # Re-opening the file-like object for the second request
        # A more robust client would handle the socket differently, but this is a simple example
        send_chat_request(sock, user_prompt)

    finally:
        print("\nClosing socket.")
        sock.close()


if __name__ == "__main__":
    main()

