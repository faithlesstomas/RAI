"""Adapter for the Agno framework."""
from typing import Any, Dict

from .base import BaseAdapter
from ..core import setup_agent


class AgnoAdapter(BaseAdapter):
    """Adapter for running agents using the Agno framework."""

    async def arun(
        self,
        prompt: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Runs a chat interaction using the Agno agent.

        This logic was previously in the `engine.py` module.
        """
        # The setup_agent function handles agent caching and setup internally
        agent, _ = setup_agent(session_id=session_id)

        if not prompt:
            # This check should ideally be done before calling the adapter
            return {"status": "error", "error_message": "Missing prompt."}

        ai_response = await agent.arun(prompt)
        content = ai_response.content if ai_response else ""
        tool_calls = getattr(ai_response, "tool_calls", None)

        # The adapter's responsibility is to return the raw payload,
        # the engine will wrap it with status.
        return {"content": content, "tool_calls": tool_calls}
