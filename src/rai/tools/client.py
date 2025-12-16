"""
Client-side tools for the RAI agent.
"""
import json
from agno.tools import Toolkit

class ClientTools(Toolkit):
    """Tools for executing code on the client side."""
    def __init__(self) -> None:
        super().__init__(name="client_tools")
        self.register(self.eval_scheme)

    def eval_scheme(self, code: str) -> str:
        """
        Executes Scheme code on the client.

        Args:
            code: The Scheme code to execute.
        """
        # This function is a placeholder.
        # The adapter should intercept the call to this tool and return it to the client.
        # If it is executed, it returns a special string indicating it's a client tool.
        # We use a JSON structure with a prefix to make it easy to parse.
        return f"__CLIENT_TOOL_CALL__:{json.dumps({'tool': 'eval_scheme', 'code': code})}"
