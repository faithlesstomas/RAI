"""Adapter for the Agno framework."""
from typing import Any, Dict

from .base import BaseAdapter
from ..core import create_agent_from_config


class AgnoAdapter(BaseAdapter):  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Agno framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        super().__init__(agent_config)
        # Each adapter instance gets its own agent, configured dynamically.
        # A session_id is still needed for history management within the agent.
        session_id = self.config.get("session_id", "default-session")
        self.agent, _ = create_agent_from_config(self.config, session_id)

    async def arun(
        self,
        prompt: str,
        session_id: str,  # session_id is now mainly for engine-level context
    ) -> Dict[str, Any]:
        """
        Runs a chat interaction using the pre-configured Agno agent.
        """
        if not prompt:
            return {"status": "error", "error_message": "Missing prompt."}

        ai_response = await self.agent.arun(prompt)
        content = ai_response.content if ai_response else ""
        tool_calls = getattr(ai_response, "tool_calls", None)

        return {"content": content, "tool_calls": tool_calls}
