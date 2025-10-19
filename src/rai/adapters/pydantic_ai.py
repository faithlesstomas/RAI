"""Adapter for the Pydantic AI framework."""
from typing import Any, Dict

from pydantic_ai import Agent

from .base import BaseAdapter
from ..core import get_session_config


class PydanticAIAdapter(BaseAdapter): #pylint: disable=too-few-public-methods
    """Adapter for running agents using the Pydantic AI framework."""

    async def arun(
        self,
        prompt: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Runs a chat interaction using the Pydantic AI agent.
        """
        # Get the configuration for the current session
        session_config = get_session_config(session_id)
        backend = session_config.get("backend", "ollama")
        model = session_config.get("model", "gemma3:4b")

        # Pydantic AI uses a single string for the model, e.g., "ollama/gemma2:9b"
        # or for other backends: "openai:gpt-4o"
        if backend in ["ollama", "groq"]:
            model_string = f"{backend}/{model}"
        else:
            model_string = f"{backend}:{model}"

        # For now, we instantiate the agent on each call.
        # We can optimize this later.
        agent = Agent(model_string)

        ai_response = await agent.run(prompt)

        # The response object in Pydantic AI has an `output` attribute.
        # We will need to investigate how to extract tool call information later.
        content = ai_response.output

        return {"content": content, "tool_calls": None}
