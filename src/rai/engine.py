"""
The core engine for the RAI platform.

This module provides the abstraction layer for interacting with different
AI agent frameworks.
"""
from typing import Any, Dict

from .core import setup_agent


async def run_chat(
    prompt: str,
    session_id: str,
    # In the future, we'll add a parameter here to select the framework,
    # e.g., framework: str = "agno"
) -> Dict[str, Any]:
    """
    Runs a chat interaction with the selected AI agent framework.

    This function acts as the abstraction layer. For now, it directly
    implements the logic for the 'agno' framework.
    """
    # This logic is moved from ipc_server.CommandHandler.handle_chat
    # Note: We might need to manage the agent instance more globally later.
    agent, _ = setup_agent(session_id=session_id)

    if not prompt:
        return {"status": "error", "error_message": "Missing prompt."}

    ai_response = await agent.arun(prompt)
    content = ai_response.content if ai_response else ""
    tool_calls = getattr(ai_response, "tool_calls", None)

    return {
        "status": "success",
        "payload": {"content": content, "tool_calls": tool_calls},
    }
