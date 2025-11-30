"""Adapter for the Agno framework."""
import sys
from typing import Any, Dict, List

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv
from returns.result import Failure, Result, Success

from ..core import error_console, setup_model, setup_tools


class AgnoAdapter:  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Agno framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self.config = agent_config
        self.agent = self._create_agent_from_config()

    def _create_agent_from_config(self) -> Agent:
        """Initializes an Agno agent from the adapter's configuration."""
        load_dotenv()

        run_config = self.config
        session_id = run_config.get("session_id", "default-session")
        backend = run_config.get("backend", "ollama")
        model_id = run_config.get("model", "gemma3:4b")
        ollama_host = run_config.get("ollama_host")
        system_prompt = run_config.get("system", "You are a helpful AI assistant.")
        enabled_tool_names = run_config.get("tools")
        use_markdown = run_config.get("markdown", not run_config.get("stream", False))

        # For now, assume tools are enabled if a list is provided or it's not ollama
        enable_tools = enabled_tool_names is not None or backend != "ollama"

        model_instance, _ = setup_model(
            backend, model_id, quiet=False, ollama_host=ollama_host
        )
        agent_tools, _ = setup_tools(
            enable_tools=enable_tools,
            quiet=False,
            enabled_tool_names=enabled_tool_names,
        )

        try:
            print(f"DEBUG: Initializing Agno Agent with session_id={session_id}")
            agent = Agent(
                model=model_instance,
                tools=agent_tools,
                markdown=use_markdown,
                add_history_to_context=True,
                db=SqliteDb(
                    session_table="agent_sessions",
                    db_file="tmp/data.db",
                ),
                session_id=session_id,
                instructions=system_prompt,
            )
            return agent
        except ImportError as e:
            error_console.print(
                f"[bold red]ERROR: Failed to import agent dependencies: {e}[/bold red]"
            )
            sys.exit(1)
        except Exception as e:  # pylint: disable=broad-except
            error_console.print(f"[bold red]ERROR: Failed to initialize agent: {e}[/bold red]")
            if backend == "ollama":
                error_console.print(
                    "[yellow]Is the Ollama server running and is the model pulled?[/yellow]"
                )
            sys.exit(1)

    async def arun(self, prompt: str) -> Result[Dict[str, Any], Exception]:
        """
        Runs a chat interaction using the pre-configured Agno agent.
        """
        try:
            if not prompt:
                return Failure(ValueError("Missing prompt."))

            ai_response = await self.agent.arun(prompt)

            # Handle potential list content from multimodal responses
            content = ai_response.content if ai_response else ""
            if isinstance(content, list):
                content = "\n".join(map(str, content))
            else:
                content = str(content)

            tool_calls = getattr(ai_response, "tool_calls", None)

            return Success({"content": content, "tool_calls": tool_calls})
        except Exception as e: # pylint: disable=broad-exception-caught
            return Failure(e)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves and formats chat history from the agent's storage.
        """
        try:
            if not (self.agent and self.agent.storage and self.agent.session_id):
                return []

            session = self.agent.storage.read(session_id=self.agent.session_id)
            if not (session and hasattr(session, "memory") and isinstance(session.memory, dict)):
                return []

            runs = session.memory.get("runs")
            if not isinstance(runs, list):
                return []

            all_messages = []
            for run in runs:
                if isinstance(run, dict) and "messages" in run and isinstance(run["messages"], list):
                    for msg in run["messages"]:
                        if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
                            all_messages.append({
                                "role": msg["role"],
                                "content": msg.get("content", "")
                            })

            # The logic to get unique messages seems to be causing issues.
            # For now, let's return all messages and see if that fixes the immediate problem.
            # If duplicates appear, this can be revisited.
            return all_messages
        except Exception:  # pylint: disable=broad-except
            return []

    def clear_history(self) -> None:
        """Clear the chat history in the agent's storage."""
        if self.agent.storage and hasattr(self.agent.storage, "delete"):
            self.agent.storage.delete(self.agent.session_id)

    async def close(self) -> None:
        """Clean up resources, like closing the model client."""
        if hasattr(self.agent, "model"):
            client = self.agent.model.get_client()
            if client and hasattr(client, "aio") and hasattr(client.aio, "aclose"):
                await client.aio.aclose()
