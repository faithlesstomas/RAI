"""Base class for all framework adapters."""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAdapter(ABC): # pylint: disable=too-few-public-methods
    """
    Abstract Base Class for AI framework adapters.
    """

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self.config = agent_config

    @abstractmethod
    async def arun(
        self,
        prompt: str,
        session_id: str,
        # We can add more common parameters here later, e.g., tool definitions
    ) -> Dict[str, Any]:
        """
        Asynchronously run a chat interaction.

        Args:
            prompt: The user's prompt.
            session_id: The ID of the current session.

        Returns:
            A dictionary containing the response from the AI, typically
            with keys like 'content' and 'tool_calls'.
        """
