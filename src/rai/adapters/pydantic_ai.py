"""Adapter for the Pydantic AI framework."""
from typing import Any, Dict

from pydantic_ai import Agent

from .base import BaseAdapter


class PydanticAIAdapter(BaseAdapter):  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Pydantic AI framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        super().__init__(agent_config)

        backend = self.config.get("backend", "ollama")
        model = self.config.get("model", "gemma2:9b")

        # Pydantic AI uses a single string for the model, e.g., "ollama/gemma2:9b"
        if backend in ["ollama", "groq"]:
            model_string = f"{backend}/{model}"
        else:
            model_string = f"{backend}:{model}"

        # The agent is now instantiated once per adapter instance.
        self.agent = Agent(model_string)

    async def arun(
        self,
        prompt: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Runs a chat interaction using the pre-configured Pydantic AI agent.
        """
        # The session_id could be used in the future for history management
        # with Pydantic AI, if that feature is added/used.
        _ = session_id

        ai_response = await self.agent.run(prompt)

        # The response object in Pydantic AI has an `output` attribute.
        content = ai_response.output

        return {"content": content, "tool_calls": None}
