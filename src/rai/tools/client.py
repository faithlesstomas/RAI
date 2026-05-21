"""
Client-side tools for the RAI agent.
"""
import json

def eval_scheme(code: str) -> str:
    """
    Executes Scheme code on the client.

    Args:
        code: The Scheme code to execute.
    """
    # This function is a placeholder.
    # The adapter/daemon should intercept the call to this tool and return it to the client.
    # If it is executed, it returns a special string indicating it's a client tool.
    # We use a JSON structure with a prefix to make it easy to parse.
    return f"__CLIENT_TOOL_CALL__:{json.dumps({'tool': 'eval_scheme', 'code': code})}"
