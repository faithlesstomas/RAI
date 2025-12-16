"""Adapter for the Agno framework."""
import json
import sys
from typing import Any, AsyncIterator, Dict, List

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv
from returns.result import Failure, Result, Success

from ..core import error_console, setup_model, setup_tools

import logging
logger = logging.getLogger(__name__)


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
        context = run_config.get("context")

        if context:
            # Append context to system prompt or handle it otherwise
            # For now, let's append it to system prompt for simplicity
            system_prompt += f"\n\nContext:\n{json.dumps(context, indent=2)}"

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
            logger.debug(f"Initializing Agno Agent with session_id={session_id}")
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

            # Ensure we are using the latest model/tools if reload happened
            ai_response = await self.agent.arun(prompt)

            # Handle potential list content from multimodal responses
            content = ai_response.content if ai_response else ""
            if isinstance(content, list):
                content = "\n".join(map(str, content))
            else:
                content = str(content)

            tool_calls = getattr(ai_response, "tool_calls", None)

            # Check for client tool calls in content (if executed and returned special string)
            # Or if tool_calls are present and we want to pass them

            # If the tool was executed and returned our special string, we can parse it back?
            # Actually, if we want the client to execute it, we should probably return the tool call
            # BEFORE it is executed, or if it is executed, we catch the "request" to execute.

            # For now, let's assume Agno executes it and returns the string "__CLIENT_TOOL_CALL__:..."
            # We can parse this and format the response accordingly.

            if "__CLIENT_TOOL_CALL__" in content:
                # Parse the content to extract tool calls
                try:
                    # Find the start of the marker
                    marker = "__CLIENT_TOOL_CALL__:"
                    start_idx = content.find(marker)
                    if start_idx != -1:
                        json_str = content[start_idx + len(marker):]
                        # It might be followed by other text, so we might need to be careful
                        # For now, assume it's the main part or at least parseable
                        # If there are multiple, we might need a regex
                        # Let's try to parse the first one
                        client_tool_data = json.loads(json_str.strip().split("\n")[0])

                        # Construct a tool call object
                        tool_call = {
                            "type": "function",
                            "function": {
                                "name": client_tool_data.get("tool"),
                                "arguments": json.dumps({"code": client_tool_data.get("code")})
                            }
                        }

                        # If tool_calls is None, initialize it
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(tool_call)

                        # Clean up content to remove the internal marker if desired
                        # content = content.replace(f"{marker}{json_str}", "[Client Tool Request]")
                except Exception as e:
                    pass # Failed to parse, ignore

            return Success({"content": content, "tool_calls": tool_calls})
        except Exception as e: # pylint: disable=broad-exception-caught
            return Failure(e)

    def reload(self) -> None:
        """Reloads the agent configuration."""
        logger.debug("Reloading AgnoAdapter configuration...")
        self.agent = self._create_agent_from_config()

    async def astream(self, prompt: str) -> AsyncIterator[Any]:
        """
        Streams the chat interaction using the pre-configured Agno agent.
        """
        try:
            if not prompt:
                yield Failure(ValueError("Missing prompt."))
                return

            # Agno agents usually have an astream method that yields chunks
            # We need to verify the exact return type of agent.astream
            async for chunk in self.agent.arun(prompt, stream=True):
                # Check if the chunk is a tool call or just text
                # For now, we assume it yields objects that have content or delta
                yield chunk

        except Exception as e: # pylint: disable=broad-exception-caught
            yield Failure(e)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieves and formats chat history from the agent's storage.
        """
        try:
            if not (self.agent and self.agent.session_id):
                return []

            # Try to find storage or db
            storage = getattr(self.agent, "storage", None)
            if not storage:
                 storage = getattr(self.agent, "db", None)
            
            if not storage:
                logger.error("Agent has no 'storage' or 'db' attribute.")
                return []

            # Assuming "agent" is a valid SessionType or string equivalent.
            session = storage.get_session(session_id=self.agent.session_id, session_type="agent")
            if not session:
                logger.warning(f"No session found for {self.agent.session_id}")
                return []

            # Try to get runs from session object
            runs = getattr(session, "runs", None)
            # logger.debug(f"DEBUG: session type: {type(session)}")
            # logger.debug(f"DEBUG: runs type: {type(runs)}")
            # logger.debug(f"DEBUG: runs content len: {len(runs) if runs else 0}")
            
            # Fallback to memory dict for older schemas
            if runs is None and hasattr(session, "memory") and isinstance(session.memory, dict):
                runs = session.memory.get("runs")

            if not runs:
                logger.warning("No runs found in session.")
                return []

            # Decode JSON if runs is a string (common in sqlite storage)
            # Handle potential double/triple encoding which seems to happen in current environment
            while isinstance(runs, str):
                try:
                    runs = json.loads(runs)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode runs JSON from session storage.")
                    return []

            if not isinstance(runs, list):
                # logger.warning(f"Runs is not a list: {type(runs)}")
                return []

            all_messages = []
            for run in runs:
                messages = []
                if isinstance(run, dict):
                    messages = run.get("messages", [])
                elif hasattr(run, "messages"):
                    messages = run.messages
                
                if not isinstance(messages, list):
                    continue

                for msg in messages:
                    role = None
                    content = ""
                    
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        content = msg.get("content", "")
                    elif hasattr(msg, "role") and hasattr(msg, "content"):
                         role = msg.role
                         content = msg.content
                    
                    if role in ["user", "assistant"]:
                        all_messages.append({
                            "role": role,
                            "content": str(content) if content is not None else ""
                        })
            
            return all_messages
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error retrieving history: {e}")
            return []

    def clear_history(self) -> None:
        """Clear the chat history in the agent's storage."""
        storage = getattr(self.agent, "storage", None)
        if not storage:
            storage = getattr(self.agent, "db", None)

        if storage and hasattr(storage, "delete"):
            storage.delete(self.agent.session_id)

    async def close(self) -> None:
        """Clean up resources, like closing the model client."""
        if hasattr(self.agent, "model"):
            client = self.agent.model.get_client()
            if client and hasattr(client, "aio") and hasattr(client.aio, "aclose"):
                await client.aio.aclose()
