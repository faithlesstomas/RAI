"""
Core components for agent setup and configuration management.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable, Union, TypedDict



from rich.console import Console


# from agno.agent import Agent
# from agno.storage.sqlite import SqliteStorage

# Tools are now lazy-loaded in setup_tools
# from agno.tools.arxiv import ArxivTools
# from agno.tools.calculator import CalculatorTools
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.file import FileTools
# from agno.tools.python import PythonTools
# from agno.tools.shell import ShellTools
# from agno.tools.tavily import TavilyTools
# from agno.tools.webbrowser import WebBrowserTools
# from agno.tools.wikipedia import WikipediaTools
# from agno.tools.yfinance import YFinanceTools

from agno.utils.log import logger

# from pydantic_ai.exceptions import UserError
# from returns.iterables import Fold
# from returns.maybe import Maybe

# from rai.adapters.pydantic_ai import PydanticAIAdapter

# from rai.tools.gitlab import GitlabTools


@runtime_checkable
class AgentResponse(Protocol): # pylint: disable=too-few-public-methods
    """A protocol for the response object from an agent's arun method."""
    content: Union[str, List[object]]
    tool_calls: Optional[List[object]]

class ResponseDict(TypedDict, total=False):
    """A TypedDict for the dictionary response from PydanticAIAdapter."""
    content: str
    tool_calls: List[object]


_HAS_GNOME_TOOLS = False # pylint: disable=invalid-name
try:
    from rai.tools.gnome import send_notification, take_screenshot, weather

    _HAS_GNOME_TOOLS = True # pylint: disable=invalid-name
except ImportError:
    pass


# --- Globals ---
console = Console(record=True)
error_console = Console(stderr=True)


# --- Tool Registry ---
TOOL_REGISTRY = {
    "CalculatorTools": ("agno.tools.calculator", "CalculatorTools"),
    "ArxivTools": ("agno.tools.arxiv", "ArxivTools"),
    "WikipediaTools": ("agno.tools.wikipedia", "WikipediaTools"),
    "DuckDuckGoTools": ("agno.tools.duckduckgo", "DuckDuckGoTools"),
    "WebBrowserTools": ("agno.tools.webbrowser", "WebBrowserTools"),
    "FileTools": ("agno.tools.file", "FileTools"),
    "PythonTools": ("agno.tools.python", "PythonTools"),
    "ShellTools": ("agno.tools.shell", "ShellTools"),
    "GitlabTools": ("rai.tools.gitlab", "GitlabTools"),
    "YFinanceTools": ("agno.tools.yfinance", "YFinanceTools"),
    "ClientTools": ("rai.tools.client", "ClientTools"),
    # "TavilyTools": ("agno.tools.tavily", "TavilyTools"),
    # Requires API key, handled separately but class load is same
}


# --- Tool and Model Setup ---
def setup_tools( # noqa: PLR0912 # pylint: disable=too-many-branches, too-many-locals
    enable_tools: bool,
    quiet: bool,
    enabled_tool_names: Optional[List[str]] = None,
    has_prompt: bool = False,
) -> Tuple[List[Any], List[str]]:
    """Sets up the tools for the agent."""
    messages = []

    if _HAS_GNOME_TOOLS:
        # these are functions, not classes in modules,
        # so we handle them differently or add to a separate registry
        pass

    agent_tools: List[Any] = []
    if enable_tools:
        tools_to_enable = (
            enabled_tool_names
            if enabled_tool_names is not None
            else list(TOOL_REGISTRY.keys())
        )

        for tool_name in tools_to_enable:
            # Handle GNOME tools
            if _HAS_GNOME_TOOLS and tool_name in [
                "GnomeNotificationTool", "GnomeScreenshotTool", "GnomeWeatherTool"
            ]:
                if tool_name == "GnomeNotificationTool":
                    agent_tools.append(send_notification)
                elif tool_name == "GnomeScreenshotTool":
                    agent_tools.append(take_screenshot)
                elif tool_name == "GnomeWeatherTool":
                    agent_tools.append(weather)
                continue

            if tool_name not in TOOL_REGISTRY:
                if not quiet:
                    messages.append(
                        f"[bold yellow]WARNING: Unknown tool '{tool_name}' "
                        "specified in configuration. Skipping.[/bold yellow]"
                    )
                continue

            module_path, class_name = TOOL_REGISTRY[tool_name]
            logger.debug("Processing tool: %s", tool_name)

            # Special handling for tools requiring API keys
            tools_with_api_keys = {
                "TavilyTools": ["TAVILY_API_KEY"],
                "GitlabTools": ["GITLAB_ACCESS_TOKEN"],
            }

            if tool_name in tools_with_api_keys:
                required_vars = tools_with_api_keys[tool_name]
                if not all(os.getenv(var) for var in required_vars):
                    if not quiet and not has_prompt:
                        messages.append(
                            f"[bold yellow]WARNING: Missing {' or '.join(required_vars)} "
                            "env variable(s). "
                            f"{tool_name} will be disabled![/bold yellow]"
                        )
                    logger.debug(
                        "%s not found. Disabling %s.",
                        " or ".join(required_vars),
                        tool_name,
                    )
                    continue

            # Try to import and instantiate the tool
            try:
                module = __import__(module_path, fromlist=[class_name])
                tool_class = getattr(module, class_name)
                agent_tools.append(tool_class())
                logger.debug("%s successfully enabled.", tool_name)
            except ImportError as e:
                if not quiet:
                    messages.append(
                        f"[bold yellow]WARNING: Could not import {tool_name} ({e}). "
                        f"Install optional dependencies to use it.[/bold yellow]"
                    )
                logger.debug(f"Could not import {tool_name}: {str(e)}")
            except ValueError as e:
                if not quiet:
                    messages.append(
                        f"[bold yellow]WARNING: {tool_name} disabled due to "
                        f"configuration error: {e}[/bold yellow]"
                    )
                logger.debug(f"{tool_name} disabled due to configuration error: {e}")

    elif _HAS_GNOME_TOOLS and not quiet:
        messages.append(
            "[bold yellow]WARNING: GNOME tools are not installed. "
            "To enable them, run: pip install .[gnome-tools][/bold yellow]"
        )

    return agent_tools, messages



def validate_model_env(
    backend: str, model_id: str, quiet: bool, ollama_host: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates the environment for the specified backend and model.
    Returns a configuration dictionary for the adapter to use for instantiation,
    and a list of messages to display.
    """
    messages = []
    if backend == "gemini" and "GEMINI_API_KEY" in os.environ:
        if "GOOGLE_API_KEY" in os.environ and not quiet:
            messages.append(
                "[bold yellow]INFO: GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "Using GEMINI_API_KEY.[/bold yellow]"
            )
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    api_keys = {
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    # Supported backends list
    supported_backends = ["ollama", "gemini", "anthropic", "openai", "groq", "local"]

    if backend not in supported_backends:
        error_console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

    if backend in api_keys and not os.getenv(api_keys[backend]):
        error_console.print(
            f"[bold red]ERROR: {api_keys[backend]} environment variable not set.[/bold red]"
        )
        sys.exit(1)

    # Check for optional dependencies (rudimentary check).
    # We allow adapters to handle specific imports and fail gracefully if needed.

    config = {
        "backend": backend,
        "model_id": model_id,
        "api_key_env_var": api_keys.get(backend),
    }

    if backend == "ollama":
        config["ollama_host"] = ollama_host

    return config, messages
