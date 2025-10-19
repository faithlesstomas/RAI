"""
Core components for agent setup and configuration management.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from agno.agent import Agent
from agno.storage.sqlite import SqliteStorage
from agno.tools.arxiv import ArxivTools
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.tools.tavily import TavilyTools
from agno.tools.webbrowser import WebBrowserTools
from agno.tools.wikipedia import WikipediaTools
from agno.utils.log import logger
from dotenv import load_dotenv
from rich.console import Console

from rai.tools.gitlab import GitlabTools

_HAS_GNOME_TOOLS = False
try:
    from rai.tools.gnome import send_notification, take_screenshot, weather

    _HAS_GNOME_TOOLS = True
except ImportError:
    pass


# --- Constants ---
CONFIG_DIR = os.path.expanduser("~/.config/rai")
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

# --- Globals ---
RAI_CONFIG: Dict[str, Any] = {}
console = Console(record=True)
error_console = Console(stderr=True)


# --- Configuration ---
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the configuration from the given path or the default config file."""
    config_file = path or DEFAULT_CONFIG_FILE
    if not os.path.exists(config_file):
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Saves the provided configuration data to the given path or the default file."""
    config_file = path or DEFAULT_CONFIG_FILE
    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def get_session_config(session_id: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads the application config and returns the config for a specific session."""
    app_config = load_config(path=config_path)
    return app_config.get("sessions", {}).get(session_id, {})


# --- Tool and Model Setup ---
def _setup_tools(# noqa: PLR0912 # pylint: disable=too-many-branches
    enable_tools: bool,
    quiet: bool,
    enabled_tool_names: Optional[List[str]] = None,
) -> Tuple[List[Any], List[str]]:
    """Sets up the tools for the agent."""
    messages = []
    all_available_tools: Dict[str, Any] = {
        "CalculatorTools": CalculatorTools,
        "ArxivTools": ArxivTools,
        "WikipediaTools": WikipediaTools,
        "DuckDuckGoTools": DuckDuckGoTools,
        "WebBrowserTools": WebBrowserTools,
        "FileTools": FileTools,
        "PythonTools": PythonTools,
        "ShellTools": ShellTools,
        "GitlabTools": GitlabTools,
        "TavilyTools": TavilyTools,
    }

    if _HAS_GNOME_TOOLS:
        all_available_tools["GnomeNotificationTool"] = send_notification
        all_available_tools["GnomeScreenshotTool"] = take_screenshot
        all_available_tools["GnomeWeatherTool"] = weather

    agent_tools: List[Any] = []
    if enable_tools:
        tools_to_enable = (
            enabled_tool_names
            if enabled_tool_names is not None
            else all_available_tools.keys()
        )

        for tool_name in tools_to_enable:
            if tool_name not in all_available_tools:
                if not quiet:
                    messages.append(
                        f"[bold yellow]WARNING: Unknown tool '{tool_name}' "
                        "specified in configuration. Skipping.[/bold yellow]"
                    )
                continue

            tool_class = all_available_tools[tool_name]
            logger.debug(
                f"Processing tool: {tool_name}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
            )

            # Special handling for tools requiring API keys
            tools_with_api_keys = {
                "TavilyTools": ["TAVILY_API_KEY"],
                "GitlabTools": ["GITLAB_ACCESS_TOKEN"],
            }

            if tool_name in tools_with_api_keys:
                required_vars = tools_with_api_keys[tool_name]
                if all(os.getenv(var) for var in required_vars):
                    logger.debug(
                        f"{' or '.join(required_vars)} found. Attempting to enable {tool_name}."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                    )
                    try:
                        agent_tools.append(tool_class())
                        logger.debug(
                            f" {tool_name} successfully enabled."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                        )
                    except ValueError as e:
                        if not quiet:
                            messages.append(
                                f"[bold yellow]WARNING: {tool_name} disabled due to "
                                f"configuration error: {e}[/bold yellow]"
                            )
                        logger.debug(
                            f"{tool_name} disabled due to configuration error: {e}"  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                        )
                else:
                    if not quiet and not RAI_CONFIG.get("prompt"):
                        messages.append(
                            f"[bold yellow]WARNING: Missing {' or '.join(required_vars)} "
                            "env variable(s). "
                            f"{tool_name} will be disabled![/bold yellow]"
                        )
                    logger.debug(
                        f"{' or '.join(required_vars)} not found. Disabling {tool_name}."  # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                    )
            # Handle GNOME tools which are functions, not classes
            elif tool_name in [
                "GnomeNotificationTool",
                "GnomeScreenshotTool",
                "GnomeWeatherTool",
            ]:
                # These are already functions, just append them
                agent_tools.append(tool_class)
            else:
                # For other tools, just instantiate them
                agent_tools.append(tool_class())
    elif _HAS_GNOME_TOOLS and not quiet:
        messages.append(
            "[bold yellow]WARNING: GNOME tools are not installed. "
            "To enable them, run: pip install .[gnome-tools][/bold yellow]"
        )

    return agent_tools, messages


def _setup_model(backend: str, model_id: str, quiet: bool) -> Tuple[Any, List[str]]:
    """Sets up the model based on the backend and model ID."""
    messages = []
    if backend == "gemini" and "GEMINI_API_KEY" in os.environ:
        if "GOOGLE_API_KEY" in os.environ and not quiet:
            messages.append(
                "[bold yellow]INFO: GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "Using GEMINI_API_KEY.[/bold yellow]"
            )
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

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
    api_keys = {
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    if backend not in model_map:
        error_console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

    if backend in api_keys and not os.getenv(api_keys[backend]):
        error_console.print(
            f"[bold red]ERROR: {api_keys[backend]} environment variable not set.[/bold red]"
        )
        sys.exit(1)

    try:
        module_path, class_name = model_map[backend].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        return model_class(id=model_id), messages
    except ImportError:
        error_console.print(
            f"[bold red]ERROR: Backend '{backend}' requires an optional dependency.[/bold red]"
        )
        if backend in dependency_map:
            error_console.print(
                "[yellow]Please install it using: "
                f"[bold]pip install .[{dependency_map[backend]}[/bold][/yellow]"
            )
        sys.exit(1)


# --- Agent Setup ---
def setup_agent(
    enable_tools: bool = True,
    quiet: bool = False,
    use_markdown: bool = True,
    session_id: Optional[str] = "my_chat_session",
) -> Tuple[Agent, List[str]]:
    """Initializes the Agno agent, setting the model, prompt, and tools.
    """

    load_dotenv()

    backend = RAI_CONFIG.get("backend")
    model_id = RAI_CONFIG.get("model")
    system_prompt = RAI_CONFIG.get("system")
    enabled_tool_names = RAI_CONFIG.get("tools")

    model_instance, messages = _setup_model(backend, model_id, quiet)
    agent_tools, tool_messages = _setup_tools(enable_tools, quiet, enabled_tool_names)
    messages.extend(tool_messages)

    try:
        agent = Agent(
            model=model_instance,
            tools=agent_tools,
            show_tool_calls=False,
            markdown=use_markdown,
            add_history_to_messages=True,
            storage=SqliteStorage(
                table_name="agent_sessions",
                db_file="tmp/data.db",
                auto_upgrade_schema=True,
            ),
            session_id=session_id,
            instructions=system_prompt,
            telemetry=False,
        )
        return agent, messages
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