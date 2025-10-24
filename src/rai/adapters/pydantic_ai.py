"""Adapter for the Pydantic AI framework."""
import json
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent

from agno.tools.calculator import CalculatorTools
from agno.tools.shell import ShellTools
from agno.utils.log import logger

from .base import BaseAdapter


class PydanticAIAdapter(BaseAdapter):  # pylint: disable=too-few-public-methods
    """Adapter for running agents using the Pydantic AI framework."""

    def __init__(self, agent_config: Dict[str, Any]) -> None:
        super().__init__(agent_config)

        backend = self.config.get("backend", "ollama")
        model = self.config.get("model", "gemma2:9b")
        enabled_tools = self.config.get("tools")

        # Pydantic AI uses a single string for the model, e.g., "ollama/gemma2:9b"
        if backend in ["ollama", "groq", "gemini"]:
            model_string = f"{backend}/{model}"
        else:
            model_string = f"{backend}:{model}"

        # Get compatible tool wrappers
        pydantic_tools = setup_pydantic_tools(enabled_tools)

        logger.debug("Pydantic AI model string: %s", model_string)

        # The agent is now instantiated once per adapter instance.
        self.agent = Agent(model_string, tools=pydantic_tools)

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
            Performs a mathematical operation. Supported operations: add, subtract, multiply, divide, exponentiate, factorial, is_prime, square_root.
            Args:
                operation (str): The name of the operation to perform.
                a (Optional[float]): The first number for binary operations (add, subtract, multiply, divide, exponentiate).
                b (Optional[float]): The second number for binary operations.
                n (Optional[int]): The number for unary operations (factorial, is_prime, square_root).
            """
            if operation == "add" and a is not None and b is not None:
                return _calculator.add(a, b)
            if operation == "subtract" and a is not None and b is not None:
                return _calculator.subtract(a, b)
            if operation == "multiply" and a is not None and b is not None:
                return _calculator.multiply(a, b)
            if operation == "divide" and a is not None and b is not None:
                return _calculator.divide(a, b)
            if operation == "exponentiate" and a is not None and b is not None:
                return _calculator.exponentiate(a, b)
            if operation == "factorial" and n is not None:
                return _calculator.factorial(n)
            if operation == "is_prime" and n is not None:
                return _calculator.is_prime(n)
            if operation == "square_root" and n is not None:
                return _calculator.square_root(n)
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
