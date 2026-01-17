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
        enabled_tools = self.config.get("tools")
        pydantic_tools = setup_pydantic_tools(enabled_tools)

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

        agent = Agent(llm, tools=pydantic_tools)
        self.agents[session_id] = agent
        return agent

    async def arun(self, prompt: str, session_id: Optional[str] = None) -> Result[Dict[str, Any], Exception]:
        """
        Runs a chat interaction using a session-specific Pydantic AI agent.
        """
        try:
            target_session_id = session_id or self.config.get("session_id", "default")
            agent = self._get_or_create_agent(target_session_id)
            ai_response = await agent.run(prompt)

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

    async def astream(self, prompt: str) -> AsyncIterator[Any]:
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


def setup_pydantic_tools(
    enabled_tool_names: Optional[List[str]] = None,
) -> List[Any]:
    """
    Creates wrapper functions for Agno tools to make them compatible with Pydantic-AI.
    """
    if not enabled_tool_names:
        return []

    pydantic_tools = []
    if "CalculatorTools" in enabled_tool_names:
        _calculator = CalculatorTools()

        def run_calculator_operation(
            operation: str,
            a: Optional[float] = None,
            b: Optional[float] = None,
            n: Optional[int] = None,
        ) -> str:
            """
            Performs a mathematical operation.
            Supported operations: add, subtract, multiply, divide, exponentiate,
            factorial, is_prime, square_root.
            Args:
                operation (str): The name of the operation to perform.
                a (Optional[float]): The first number for binary operations.
                b (Optional[float]): The second number for binary operations.
                n (Optional[int]): The number for unary operations.
            """
            binary_ops = {
                "add": _calculator.add,
                "subtract": _calculator.subtract,
                "multiply": _calculator.multiply,
                "divide": _calculator.divide,
                "exponentiate": _calculator.exponentiate,
            }
            unary_ops = {
                "factorial": _calculator.factorial,
                "is_prime": _calculator.is_prime,
                "square_root": _calculator.square_root,
            }

            if operation in binary_ops and a is not None and b is not None:
                return binary_ops[operation](a, b)
            if operation in unary_ops and n is not None:
                return unary_ops[operation](n)

            return json.dumps({"error": f"Unsupported operation or missing arguments for {operation}"})


        pydantic_tools.append(run_calculator_operation)

    if "ShellTools" in enabled_tool_names:
        _shell = ShellTools()

        def run_shell(command: str) -> str:
            """
            Executes a shell command and returns its standard output.
            Note: This is not a full interactive shell.
            """
            return _shell.run_shell_command(args=[command])

        pydantic_tools.append(run_shell)

    # TODO: Add wrappers for other Agno tools here.

    return pydantic_tools
