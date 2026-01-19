"""Adapter for the Pydantic AI framework."""
import os
import json
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from returns.result import Failure, Result, Success

from agno.tools.calculator import CalculatorTools
from agno.tools.shell import ShellTools
from agno.utils.log import logger

# pylint: disable=logging-fstring-interpolation

class PydanticAIAdapter:  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Pydantic AI framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        self.config = agent_config
        self.agents: Dict[str, Agent] = {}  # Store agents per session

    def _get_or_create_agent(self, session_id: str) -> Agent:
        """Creates or retrieves a Pydantic AI agent for a given session."""
        if session_id in self.agents:
            return self.agents[session_id]

        backend = self.config.get("backend", "ollama")
        model_name = self.config.get("model", "gemma2:9b")
        # Check enabled_tools config flag passed from CLI
        enable_tools = self.config.get("enable_tools", True)
        enabled_tool_names = self.config.get("tools") if enable_tools else []
        
        # Use Core's setup_tools to get instances
        agno_tool_instances, _ = setup_tools(
            enable_tools=enable_tools,
            quiet=False,  # Core handles printing warnings
            enabled_tool_names=enabled_tool_names
        )
        pydantic_tools = _get_pydantic_compatible_tools(agno_tool_instances)

        if backend == "local" or model_name.endswith((".vmfb", ".gguf", ".onnx")):
             # Local Inference
            try:
                from ..inference import load_local_model  # noqa: PLC0415
                from ..inference.bridges import LocalPydanticModel  # noqa: PLC0415
                from returns.result import Failure  # noqa: PLC0415

                logger.debug(f"Creating PydanticAI agent with local model: {model_name}")
                engine_result = load_local_model(model_name)

                if isinstance(engine_result, Failure):
                     # Log error and raise
                     logger.error(f"Failed to load local model: {engine_result.failure()}")
                     raise engine_result.failure()

                engine = engine_result.unwrap()
                llm = LocalPydanticModel(engine=engine, _model_name=model_name)

            except Exception as e:
                logger.error(f"Error initializing local agent: {e}")
                raise e

        elif backend == "ollama":
            # Consistent with the rest of the app, use 'ollama_host'
            ollama_host = self.config.get(
                "ollama_host"
            ) or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_api_url = f"{ollama_host.rstrip('/')}/v1"
            logger.debug(
                f"Creating new PydanticAI agent for session '{session_id}' with Ollama model: {model_name}"
            )
            ollama_provider = OllamaProvider(base_url=ollama_api_url)
            llm = OpenAIChatModel(model_name=model_name, provider=ollama_provider)
        else:
            msg = "Creating new PydanticAI agent for session "
            msg += f"'{session_id}' with {backend} model: {model_name}"
            logger.debug(msg)
            llm = model_name

        # Extract system prompt
        system_prompt = self.config.get("system", "You are a helpful AI assistant.")

        agent = Agent(llm, tools=pydantic_tools, system_prompt=system_prompt)
        self.agents[session_id] = agent
        return agent

    async def arun(
        self, 
        prompt: str, 
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Result[Dict[str, Any], Exception]:
        """
        Runs a chat interaction using a session-specific Pydantic AI agent.
        """
        try:
            # Unified History Support
            final_prompt = prompt
            if history:
                 # Reconstruct conversation history including tool interactions
                 history_lines = []
                 for msg in history:
                     role = msg.get('role', 'unknown').capitalize()
                     content = msg.get('content', '')
                     tool_calls = msg.get('tool_calls')
                     
                     if tool_calls:
                         history_lines.append(f"{role}: {content} [Tool Calls: {tool_calls}]")
                     else:
                         history_lines.append(f"{role}: {content}")
                 
                 history_text = "\n".join(history_lines)

                 if history_text:
                     final_prompt = (
                         f"Below is the conversation history so far. Use it as context.\n\n"
                         f"{history_text}\n\n"
                         f"User: {prompt}"
                     )

            target_session_id = session_id or self.config.get("session_id", "default")
            agent = self._get_or_create_agent(target_session_id)
            ai_response = await agent.run(final_prompt)

            # Log tool usage from response messages
            if hasattr(ai_response, "new_messages"):
                for msg in ai_response.new_messages():
                    if hasattr(msg, "parts"):
                        for part in msg.parts:
                            if part.part_kind == 'tool-call':
                                logger.info(f"Tool Call: {part.tool_name}({part.args})")
                            elif part.part_kind == 'tool-return':
                                logger.info(f"Tool Result ({part.tool_name}): {part.content}")

            # The response object in Pydantic AI has an `output` attribute.
            content = ai_response.output

            return Success({"content": content, "tool_calls": None})
        except Exception as e:
            return Failure(e)

    def reload(self) -> None:
        """Reloads the agent configuration."""
        # For Pydantic AI, agents are cached by session_id in self.agents
        # Clearing this cache forces re-creation on next access.
        logger.debug("Reloading PydanticAIAdapter configuration (clearing agent cache)...")
        self.agents.clear()

    async def astream(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Any]:
        """Asynchronously streams the agent's response (simulated)."""
        # TODO: Implement true streaming for PydanticAI
        result = await self.arun(prompt, self.config.get("session_id", "default"))
        if isinstance(result, Success):
            yield result.unwrap().get("content", "")
        else:
            yield f"Error: {result.failure()}"

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieves the current chat history."""
        # TODO: Implement history retrieval for PydanticAI
        return []

    def clear_history(self) -> None:
        """Clears the current chat history."""
        # Clearing the cache effectively resets the agent for now
        self.reload()

    async def close(self) -> None:
        """Performs any necessary cleanup."""
        pass


import inspect
from ..core import error_console, validate_model_env, setup_tools

# ... (rest of imports)

# Remove manual imports of agno.tools.*
# from agno.tools.calculator import CalculatorTools
# from agno.tools.shell import ShellTools

def _get_pydantic_compatible_tools(agno_tools: List[Any]) -> List[Any]:
    """
    Dynamically extracts public methods from Agno tool instances to be used by PydanticAI.
    """
    from pydantic_ai import Tool

    pydantic_tools = []
    seen_names = set()

    for tool_instance in agno_tools:
        # Inspect for public methods
        for name, method in inspect.getmembers(tool_instance, predicate=inspect.ismethod):
            if name.startswith("_") or name == "register":
                continue
            
            tool_name = name
            if tool_name in seen_names:
                # Conflict detected! Namespace it.
                # E.g. FileTools_list_files
                new_name = f"{tool_instance.__class__.__name__}_{name}"
                logger.warning(f"Tool conflict for '{name}'. Renaming to '{new_name}'.")
                
                # Wrap in Tool object to rename
                # Note: PydanticAI Tool(fn, name=...) handles this
                pydantic_tools.append(Tool(method, name=new_name))
                seen_names.add(new_name)
            else:
                pydantic_tools.append(method)
                seen_names.add(tool_name)
            
    return pydantic_tools
