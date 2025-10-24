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
from rai.adapters.pydantic_ai import PydanticAIAdapter
from pydantic_ai.exceptions import UserError

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
def setup_tools( # noqa: PLR0912 # pylint: disable=too-many-branches
    enable_tools: bool,
    quiet: bool,
    enabled_tool_names: Optional[List[str]] = None,
    has_prompt: bool = False,
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
            logger.debug( # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                f"Processing tool: {tool_name}"
            )

            # Special handling for tools requiring API keys
            tools_with_api_keys = {
                "TavilyTools": ["TAVILY_API_KEY"],
                "GitlabTools": ["GITLAB_ACCESS_TOKEN"],
            }

            if tool_name in tools_with_api_keys:
                required_vars = tools_with_api_keys[tool_name]
                if all(os.getenv(var) for var in required_vars):
                    logger.debug( # pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                        f"{' or '.join(required_vars)} found. Attempting to enable {tool_name}."
                    )
                    try:
                        agent_tools.append(tool_class())
                        logger.debug(# pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                            f" {tool_name} successfully enabled."
                        )
                    except ValueError as e:
                        if not quiet:
                            messages.append(
                                f"[bold yellow]WARNING: {tool_name} disabled due to "
                                f"configuration error: {e}[/bold yellow]"
                            )
                        logger.debug(# pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                            f"{tool_name} disabled due to configuration error: {e}"
                        )
                else:
                    if not quiet and not has_prompt:
                        messages.append(
                            f"[bold yellow]WARNING: Missing {' or '.join(required_vars)} "
                            "env variable(s). "
                            f"{tool_name} will be disabled![/bold yellow]"
                        )
                    logger.debug(# pylint: disable=logging-fstring-interpolation # ruff: noqa: G004
                        f"{' or '.join(required_vars)} not found. Disabling {tool_name}."
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


def setup_model(backend: str, model_id: str, quiet: bool) -> Tuple[Any, List[str]]:
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
class AgnoResponseProxy:
    """A proxy to handle responses from the Agno agent, ensuring content is a string."""

    def __init__(self, original_response: Any) -> None:
        self._original_response = original_response

    @property
    def content(self) -> str:
        """
        Ensures that the content is a string. If the original content is a list,
        it joins the elements into a single string.
        # TODO: Add support for multimodal responses (e.g., images)
        """
        original_content = getattr(self._original_response, 'content', '')
        if isinstance(original_content, list):
            try:
                # For now, assume all parts are strings and join them.
                return "\n".join(map(str, original_content))
            except Exception:
                error_console.print(
                    "[bold yellow]Warning: Model returned a multi-part message"
                    "with non-textual content that is not yet supported.[/bold yellow]"
                )
                # Fallback to a simple string representation for now.
                return str(original_content)
        return str(original_content)

    def __getattr__(self, name: str) -> Any:
        """Delegates other attribute access to the original response."""
        return getattr(self._original_response, name)


class AgnoAgentProxy:
    """A proxy for the Agno agent to intercept and handle responses."""

    def __init__(self, real_agent: Agent) -> None:
        self._real_agent = real_agent

    async def arun(self, *args: Any, **kwargs: Any) -> AgnoResponseProxy:
        """
        Runs the agent and wraps the response in a proxy to handle content formatting.
        """
        response = await self._real_agent.arun(*args, **kwargs)
        return AgnoResponseProxy(response)

    def __getattr__(self, name: str) -> Any:
        """Delegates other attribute access to the real agent."""
        return getattr(self._real_agent, name)


class PydanticAIResponseProxy:
    """A proxy to handle responses from the PydanticAI adapter."""

    def __init__(self, response_dict: Dict[str, Any]) -> None:
        self._response_dict = response_dict

    @property
    def content(self) -> str:
        """Returns the content from the response dictionary."""
        return self._response_dict.get("content", "")

    # Add a placeholder for tool_calls if needed by the CLI response handling
    @property
    def tool_calls(self) -> Optional[Any]:
        """Returns the tool_calls from the response dictionary."""
        return self._response_dict.get("tool_calls")


class PydanticAIAgentProxy:
    """A proxy for the PydanticAI adapter to make it compatible with the CLI."""

    def __init__(self, adapter: PydanticAIAdapter, session_id: str) -> None:
        self._adapter = adapter
        self._session_id = session_id

    async def arun(self, prompt: str, stream: bool = False) -> PydanticAIResponseProxy:
        """
        Runs the adapter and wraps the dictionary response in a proxy object.
        NOTE: PydanticAIAdapter does not currently support streaming.
        """
        # The stream argument is ignored for now.
        _ = stream
        response_dict = await self._adapter.arun(prompt=prompt, session_id=self._session_id)
        return PydanticAIResponseProxy(response_dict)

    def __getattr__(self, name: str) -> Any:
        """Delegates other attribute access to the real adapter."""
        return getattr(self._adapter, name)


def create_agent_from_config(
    config: Dict[str, Any], session_id: str, use_markdown: bool = True
) -> Tuple[AgnoAgentProxy, List[str]]:
    """Initializes an Agno agent from a dynamic configuration dictionary."""
    load_dotenv()

    backend = config.get("backend", "ollama")
    model_id = config.get("model", "gemma3n:e4b")
    system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
    enabled_tool_names = config.get("tools")
    enable_tools = bool(enabled_tool_names)

    model_instance, messages = setup_model(backend, model_id, quiet=True)
    agent_tools, tool_messages = setup_tools(
        enable_tools=enable_tools,
        quiet=True,
        enabled_tool_names=enabled_tool_names,
        has_prompt=bool(system_prompt),
    )
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
        return AgnoAgentProxy(agent), messages
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


def setup_agent(
    framework: str = "agno",
    enable_tools: bool = True,
    quiet: bool = False,
    use_markdown: bool = True,
    session_id: Optional[str] = "my_chat_session",
) -> Tuple[Any, List[str]]:
    """Initializes the agent using the global RAI_CONFIG and chosen framework."""
    config = {
        "backend": RAI_CONFIG.get("backend"),
        "model": RAI_CONFIG.get("model"),
        "system_prompt": RAI_CONFIG.get("system"),
        "tools": RAI_CONFIG.get("tools"),
    }

    if framework == "pydantic_ai":
        if config.get("backend") == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            error_console.print(
                "[bold red]ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set for Gemini backend with Pydantic AI.[/bold red]"
            )
            sys.exit(1)
        try:
            # PydanticAIAdapter handles its own setup from the config dict.
            adapter = PydanticAIAdapter(agent_config=config)
            proxy = PydanticAIAgentProxy(adapter, session_id=session_id)
            return proxy, []  # No startup messages for now
        except UserError as e:
            error_message = f"[bold red]ERROR: Pydantic AI initialization failed: {e}[/bold red]"
            if config.get("backend") == "ollama":
                error_message += "\n[yellow]Is the Ollama server running and is the model pulled?[/yellow]"
            error_console.print(error_message)
            sys.exit(1)

    # Default to agno
    return create_agent_from_config(config, session_id, use_markdown)

