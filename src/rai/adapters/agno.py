"""Adapter for the Agno framework."""
import json
import logging
import sys
from typing import Any, AsyncIterator, Dict, List, Optional

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from dotenv import load_dotenv
from returns.result import Failure, Result, Success

from ..core import error_console, validate_model_env, setup_tools

logger = logging.getLogger(__name__)


class AgnoAdapter:  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Agno framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self.config = agent_config
        self.agents: Dict[str, Agent] = {}  # Store agents per session
        # self.agent = self._create_agent_from_config() # Lazy creation

    def _get_or_create_agent(self, session_id: str) -> Agent:
        """Creates or retrieves an Agno agent for a given session."""
        if session_id in self.agents:
            return self.agents[session_id]

        logger.debug("Creating new Agno Agent for session_id=%s", session_id)

        load_dotenv()
        run_config = self.config
        # session_id passed as arg takes precedence, but config might have default?
        # Actually session_id should be passed from arun.
        # If not, use config default.

        backend = run_config.get("backend", "ollama")
        model_id = run_config.get("model", "gemma2:9b")
        ollama_host = run_config.get("ollama_host")
        system_prompt = run_config.get("system", "You are a helpful AI assistant.")
        enabled_tool_names = run_config.get("tools")
        context = run_config.get("context")

        if context:
            system_prompt += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        use_markdown = run_config.get("markdown", not run_config.get("stream", False))
        
        # Use config-provided flag or fall back to default logic
        enable_tools = run_config.get("enable_tools")
        if enable_tools is None:
             enable_tools = enabled_tool_names is not None or backend != "ollama"

        # Validate environment using Core
        model_config, _ = validate_model_env(
            backend, model_id, quiet=False, ollama_host=ollama_host
        )

        # Instantiate Model (Logic moved from core.py)
        model_instance = self._instantiate_model(backend, model_config)

        agent_tools, _ = setup_tools(
            enable_tools=enable_tools,
            quiet=False,
            enabled_tool_names=enabled_tool_names,
        )

        try:
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
            self.agents[session_id] = agent
            return agent
        except Exception as e:  # pylint: disable=broad-except
            error_console.print(f"[bold red]ERROR: Failed to initialize agent: {e}[/bold red]")
            if backend == "ollama":
                error_console.print(
                    "[yellow]Is the Ollama server running and is the model pulled?[/yellow]"
                )
            sys.exit(1)

    def _instantiate_model(self, backend: str, config: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Instantiates the Agno model object based on validated config."""

        # 1. Handle Local Inference (IREE / Llama.cpp)
        if backend == "local" or config.get("model_id", "").endswith((".vmfb", ".gguf", ".onnx")):
             # Lazy import to avoid circular dependencies if any
            try:
                from ..inference import load_local_model  # noqa: PLC0415
                from ..inference.bridges import LocalAgnoModel  # noqa: PLC0415
                from returns.result import Failure  # noqa: PLC0415

                model_path = config["model_id"]
                # We can pass explicit backend if needed, e.g. config.get("engine")
                engine_result = load_local_model(model_path)

                if isinstance(engine_result, Failure):
                    raise engine_result.failure()

                engine = engine_result.unwrap()
                return LocalAgnoModel(
                    id=model_path,
                    engine=engine,
                    name=config.get("model", "local-model")
                )
            except Exception as e: # pylint: disable=broad-except
                error_console.print(f"[bold red]ERROR: Failed to load local model: {e}[/bold red]")
                sys.exit(1)

        model_map = {
            "ollama": "agno.models.ollama.Ollama",
            "gemini": "agno.models.google.Gemini",
            "anthropic": "agno.models.anthropic.Claude",
            "openai": "agno.models.openai.chat.OpenAIChat",
            "groq": "agno.models.groq.Groq",
        }

        dependency_map = {
            "gemini": "gemini",
            "anthropic": "anthropic",
            "openai": "openai",
            "groq": "groq",
        }

        try:
            module_path, class_name = model_map[backend].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)

            if backend == "ollama":
                model_kwargs = {"id": config["model_id"]}
                if config.get("ollama_host"):
                    model_kwargs["host"] = config["ollama_host"]
                return model_class(**model_kwargs)

            return model_class(id=config["model_id"])
        except ImportError:
             # Should be caught by imports, but double check
            error_console.print(
                f"[bold red]ERROR: Backend '{backend}' requires dependency.[/bold red]"
            )
            if backend in dependency_map:
                error_console.print(
                    "[yellow]Please install it using: "
                    f"[bold]pip install .[{dependency_map[backend]}[/bold][/yellow]"
                )
            sys.exit(1)

    async def arun(
        self, 
        prompt: str, 
        session_id: Optional[str] = None, 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Result[Dict[str, Any], Exception]:
        """
        Runs a chat interaction using a session-specific Agno agent.
        """
        try:
            if not prompt:
                return Failure(ValueError("Missing prompt."))

            # Determine session_id
            target_session_id = session_id or self.config.get("session_id", "default-session")
            agent = self._get_or_create_agent(target_session_id)

            # Inject history if provided (Unified History Support)
            final_prompt = prompt
            if history:
                history_text = "\n".join(
                     f"{msg['role'].capitalize()}: {msg['content']}" 
                     for msg in history 
                     if msg.get('role') in ['user', 'assistant']
                )
                if history_text:
                    final_prompt = (
                        f"Below is the conversation history so far. Use it as context.\n\n"
                        f"{history_text}\n\n"
                        f"User: {prompt}"
                    )
                # Note: Agno might already wrap the prompt as "User: ...", so we just provide context.
                # Actually, if we change the prompt completely, Agno treats it as the new user message.
                # This works for statelessness since Agno's internal memory for this specific 'message' is fresh.

            # Ensure we are using the latest model/tools if reload happened (re-creation handles this)
            ai_response = await agent.arun(final_prompt)

            # Handle potential list content from multimodal responses
            content = ai_response.content if ai_response else ""
            if isinstance(content, list):
                content = "\n".join(map(str, content))
            else:
                content = str(content)

            tool_calls = getattr(ai_response, "tool_calls", None)

            # Check for client tool calls (legacy logic preserved)
            if "__CLIENT_TOOL_CALL__" in content:
                # Same logic as before...
                # For brevity I'm keeping the core logic but this block is unchanged from read
                try:
                    marker = "__CLIENT_TOOL_CALL__:"
                    start_idx = content.find(marker)
                    if start_idx != -1:
                        json_str = content[start_idx + len(marker):]
                        client_tool_data = json.loads(json_str.strip().split("\n")[0])
                        tool_call = {
                            "type": "function",
                            "function": {
                                "name": client_tool_data.get("tool"),
                                "arguments": json.dumps({"code": client_tool_data.get("code")})
                            }
                        }
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(tool_call)
                except Exception:
                    pass

            return Success({"content": content, "tool_calls": tool_calls})
        except Exception as e: # pylint: disable=broad-exception-caught
            return Failure(e)

    def reload(self) -> None:
        """Reloads the agent configuration."""
        logger.debug("Reloading AgnoAdapter configuration (clearing agent cache)...")
        self.agents.clear()

    async def astream(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Any]:
        """
        Streams the chat interaction using the pre-configured Agno agent.
        """
        try:
            if not prompt:
                yield Failure(ValueError("Missing prompt."))
                return

            # Use default session for streaming if not specified (TODO: support session in astream)
            session_id = self.config.get("session_id", "default-session")
            agent = self._get_or_create_agent(session_id)

            async for chunk in agent.arun(prompt, stream=True):
                yield chunk

        except Exception as e:  # pylint: disable=broad-exception-caught
            yield Failure(e)

    def get_history(self) -> List[Dict[str, str]]:  # noqa: PLR0911, PLR0912
        # pylint: disable=too-many-branches, too-many-return-statements
        """
        Retrieves and formats chat history from the agent's storage.
        """
        try:
            # We need to pick an agent to access storage.
            # If no agents created yet, we can't get history unless we access DB directly.
            # For now, let's try to get the 'default' or current session agent.
            session_id = self.config.get("session_id", "default-session")

            # Use get_or_create to ensure we have an agent with DB access
            # But wait, creating an agent might be heavy? No, it's just init.
            agent = self._get_or_create_agent(session_id)

            if not (agent and agent.session_id):
                return []

            # Try to find storage or db
            storage = getattr(agent, "storage", None)
            if not storage:
                storage = getattr(agent, "db", None)

            if not storage:
                logger.error("Agent has no 'storage' or 'db' attribute.")
                return []

            # Assuming "agent" is a valid SessionType or string equivalent.
            session = storage.get_session(session_id=agent.session_id, session_type="agent")
            if not session:
                logger.warning("No session found for %s", agent.session_id)
                return []

            # Try to get runs from session object
            runs = getattr(session, "runs", None)

            # Fallback to memory dict for older schemas
            if runs is None and hasattr(session, "memory") and isinstance(session.memory, dict):
                runs = session.memory.get("runs")

            if not runs:
                logger.warning("No runs found in session.")
                return []

            # Decode JSON if runs is a string
            while isinstance(runs, str):
                try:
                    runs = json.loads(runs)
                except json.JSONDecodeError:
                    return []

            if not isinstance(runs, list):
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
            logger.error("Error retrieving history: %s", e)
            return []

    def clear_history(self) -> None:
        """Clear the chat history in the agent's storage."""
        # Clear specific session
        session_id = self.config.get("session_id", "default-session")
        if session_id in self.agents:
            agent = self.agents[session_id]
            storage = getattr(agent, "storage", getattr(agent, "db", None))
            if storage and hasattr(storage, "delete"):
                storage.delete(agent.session_id)

        self.reload()  # Reset cache


    async def close(self) -> None:
        """Clean up resources, like closing the model client."""
        # Close all agents
        for agent in self.agents.values():
            if hasattr(agent, "model") and hasattr(agent.model, "get_client"):
                client = agent.model.get_client()
                if client and hasattr(client, "aio") and hasattr(client.aio, "aclose"):
                    await client.aio.aclose()
