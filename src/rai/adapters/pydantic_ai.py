"""Adapter for the Pydantic AI framework."""
import os
import json
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

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

        if backend == "ollama":
            ollama_base_url = self.config.get(
                "ollama_base_url"
            ) or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_api_url = f"{ollama_base_url.rstrip('/')}/v1"
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

    async def arun(self, prompt: str, session_id: str) -> Dict[str, Any]:
        """
        Runs a chat interaction using a session-specific Pydantic AI agent.
        """
        agent = self._get_or_create_agent(session_id)
        ai_response = await agent.run(prompt)

        # The response object in Pydantic AI has an `output` attribute.
        content = ai_response.output

        return {"content": content, "tool_calls": None}


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
