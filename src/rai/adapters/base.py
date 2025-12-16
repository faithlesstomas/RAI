""" Base module for LLM/AI Procesor adapters
"""
from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable

from returns.result import Result


@runtime_checkable
class Processor(Protocol):
    """
    A protocol defining a standard interface for different AI agent
    processors or adapters. This allows the CLI to interact with various
    agent implementations (local, remote via WebSocket, etc.) in a uniform way.
    """

    async def arun(self, prompt: str) -> Result[Dict[str, Any], Exception]:
        """
        Asynchronously runs the agent with a given prompt and returns the result.

        Args:
            prompt: The user's input prompt.

        Returns:
            A Result monad containing either a dictionary with the agent's
            response on success, or an Exception on failure.
        """

    async def astream(self, prompt: str) -> AsyncIterator[Any]:
        """
        Asynchronously streams the agent's response.

        Args:
            prompt: The user's input prompt.

        Returns:
            An async iterator yielding chunks of the response.
        """

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the current chat history.

        Returns:
            A list of dictionaries, where each dictionary represents a
            message with "role" and "content" keys.
        """

    def reload(self) -> None:
        """
        Reloads the processor's configuration and re-initializes internal agents/clients.
        """

    def clear_history(self) -> None:
        """Clears the current chat history."""

    async def close(self) -> None:
        """
        Performs any necessary cleanup, such as closing network connections.
        """
