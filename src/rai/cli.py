"""
rai - Rich AI CLI assistant - ASYNC version
"""

import asyncio
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

from contextlib import redirect_stdout

import click
import ollama
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
from dotenv import load_dotenv
from ollama import ResponseError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from .tools import send_notification, take_screenshot



@dataclass
class CliOptions:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold CLI options."""

    prompt: Optional[str] = None
    system: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    no_markdown: bool = False
    json_output: bool = False
    quiet: bool = False
    list_config: bool = False
    get_config_key: Optional[str] = None
    set_config_pair: Optional[str] = None
    no_stream: bool = False


console = Console(force_terminal=True, record=True)
error_console = Console(stderr=True, force_terminal=True)

CONFIG_DIR = os.path.expanduser("~/.config/rai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
RAI_CONFIG = {}

SLASH_COMMANDS = [
    "/help",
    "/config",
    "/exit",
    "/quit",
    "/q",
]


# --- Configuration ---
def load_config():
    """Loads the configuration from the config file."""
    if not os.path.exists(CONFIG_FILE):
        return {}  # pylint: disable=unhashable-member
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config():
    """Saves the configuration to the config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(RAI_CONFIG, f, indent=2)


def _setup_tools(enable_tools, quiet):
    """Sets up the tools for the agent."""
    messages = []
    base_tools = [
        send_notification,
        take_screenshot,
        CalculatorTools(enable_all=True),
        ArxivTools(),
        WikipediaTools(),
        DuckDuckGoTools(),
        WebBrowserTools(),
        FileTools(),
        PythonTools(),
        ShellTools(),
    ]

    agent_tools = []
    if enable_tools:
        if os.getenv("TAVILY_API_KEY"):
            agent_tools = base_tools + [TavilyTools()]
        else:
            agent_tools = base_tools
            if not quiet and not RAI_CONFIG.get("prompt"):
                messages.append(
                    "[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. "
                    "Tavily will be disabled![/bold yellow]"
                )
    return agent_tools, messages


def _setup_model(backend, model_id, quiet):
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
def setup_agent(enable_tools: bool = True, quiet: bool = False):
    """Initializes the Agno agent, setting the model, prompt, and tools."""
    load_dotenv()
    backend = RAI_CONFIG.get("backend")
    model_id = RAI_CONFIG.get("model")
    system_prompt = RAI_CONFIG.get("system")

    model_instance, messages = _setup_model(backend, model_id, quiet)

    agent_tools, tool_messages = _setup_tools(enable_tools, quiet)
    messages.extend(tool_messages)

    try:
        agent = Agent(
            model=model_instance,
            tools=agent_tools,
            show_tool_calls=False,
            markdown=True,
            add_history_to_messages=True,
            storage=SqliteStorage(
                table_name="agent_sessions",
                db_file="tmp/data.db",
                auto_upgrade_schema=True,
            ),
            session_id="my_chat_session",
            instructions=system_prompt,
        )
        return agent, messages
    except ImportError as e:
        error_console.print(f"[bold red]ERROR: Failed to import agent dependencies: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        error_console.print(f"[bold red]ERROR: Failed to initialize agent: {e}[/bold red]")
        if backend == "ollama":
            error_console.print(
                "[yellow]Is the Ollama server running and is the model pulled?[/yellow]"
            )
        sys.exit(1)


def check_model_tool_support(model_id: str) -> bool:
    """Checks if the specified Ollama model supports tool use."""
    try:
        details = ollama.show(model_id)
        modelfile = details.get("modelfile", "")
        return "tool_use" in str(modelfile)
    except ResponseError:
        return False


# --- Non-Interactive Mode ---
def run_single_query(agent, prompt, no_markdown, json_output):
    """Executes a single query for non-interactive mode."""
    with redirect_stdout(io.StringIO()):
        initial_response = agent.run(prompt, stream=False)

    if initial_response and initial_response.content:
        if json_output:
            print(json.dumps({"response": initial_response.content}))
        else:
            temp_console = Console()
            temp_console.print(
                Markdown(initial_response.content)
                if not no_markdown
                else initial_response.content
            )


def _handle_slash_command(user_input):
    """Handles slash commands and returns True if the app should exit."""
    if user_input in ["/exit", "/quit", "/q"]:
        return True
    if user_input == "/help":
        console.print("Available commands: /help, /config, /q")
    elif user_input == "/config":
        console.print(
            f"Model: {RAI_CONFIG.get('model')}, Backend: {RAI_CONFIG.get('backend')}"
        )
    else:
        console.print(f"[red]Unknown command: {user_input}[/red]")
    return False


async def _handle_stream_response(agent, user_input, no_markdown):
    """Handles the streaming response from the agent."""
    response_content = ""  # Accumulate content here
    response_iterator = await agent.arun(user_input, stream=True)

    # Show spinner while waiting for the FIRST chunk
    with console.status("[bold green]Assistant is thinking..."):
        try:
            # Get the first event from the async iterator
            first_event = await anext(response_iterator)
        except StopAsyncIteration:
            # Handle case where there's no response at all (e.g., empty stream)
            first_event = None

    # Now, process and print events
    if first_event:
        # Debug print removed
        if first_event.content:
            response_content += first_event.content
            console.print(first_event.content, end="")  # Print raw content

        if hasattr(first_event, "tool_calls") and first_event.tool_calls:
            console.print()  # Add newline before tool calls panel
            tool_calls_json = json.dumps(first_event.tool_calls, indent=2)
            console.print(
                Panel(tool_calls_json, title="Tool Calls", border_style="yellow")
            )

    # Loop through the rest of the events
    async for event in response_iterator:
        # Debug print removed
        if event.content:
            response_content += event.content
            console.print(event.content, end="")  # Print raw content

        if hasattr(event, "tool_calls") and event.tool_calls:
            console.print()  # Add newline before tool calls panel
            tool_calls_json = json.dumps(event.tool_calls, indent=2)
            console.print(
                Panel(tool_calls_json, title="Tool Calls", border_style="yellow")
            )
    # After streaming, print the accumulated content as formatted Markdown
    if response_content:  # Check if there's any content to print
        console.print()  # Add newline before LLM response
        if not no_markdown:
            console.print(Markdown(response_content))
        else:
            console.print(response_content)


async def _handle_non_stream_response(agent, user_input, no_markdown):
    """Handles the non-streaming response from the agent."""
    response = None

    # Setup surgical logging capture for agno's logs
    log_capture_string = io.StringIO()
    log_handler = logging.StreamHandler(log_capture_string)
    agno_logger = logging.getLogger("agno")

    # Store original propagation and handlers
    original_propagate = agno_logger.propagate
    original_handlers = agno_logger.handlers[:]  # Make a copy

    # Clear existing handlers and prevent propagation to root
    agno_logger.propagate = False
    agno_logger.handlers.clear()
    agno_logger.addHandler(log_handler)

    original_level = agno_logger.level
    agno_logger.setLevel(logging.INFO)  # Capture INFO and above

    with console.status("[bold green]Assistant is thinking..."):
        response = await agent.arun(user_input, stream=False)  # Non-streaming call

    # Restore agno's logger settings
    agno_logger.removeHandler(log_handler)
    agno_logger.setLevel(original_level)
    agno_logger.propagate = original_propagate
    agno_logger.handlers = original_handlers  # Restore original handlers

    # Get captured logs and print them in a panel
    captured_agno_logs = log_capture_string.getvalue().strip()
    if captured_agno_logs:
        console.print(
            Panel(captured_agno_logs, title="[dim]Agno Logs[/dim]", border_style="dim")
        )

    if response and response.content:
        if not no_markdown:
            console.print(Markdown(response.content))
        else:
            console.print(response.content)


# --- Async Interactive Application ---
# --- Interactive Mode ---
async def run_interactive_chat(agent, quiet, no_markdown, no_stream):
    """Starts an interactive chat loop using PromptSession, rich.status, and direct console printing for streaming."""
    if not quiet:
        console.print(
            "**Welcome to Rich AI CLI Assistant!** Type your prompt and press Enter."
        )
        console.print(
            "[dim]Type /help for a list of commands, Ctrl+C or /q to exit.[/dim]"
        )

    history_file = os.path.join(CONFIG_DIR, "history.txt")
    session = PromptSession(
        history=FileHistory(history_file),
        completer=WordCompleter(SLASH_COMMANDS, ignore_case=True),
    )

    while True:
        try:
            user_input = await session.prompt_async("> ")
            if not user_input.strip():
                continue

            # Handle Slash Commands
            if user_input.startswith("/"):
                if _handle_slash_command(user_input):
                    break
                continue

            # --- Process AI Query ---
            if no_stream:
                await _handle_non_stream_response(agent, user_input, no_markdown)

            else: # --- OPTION A: Spinner then Stream (current implementation) ---
                await _handle_stream_response(agent, user_input, no_markdown)

            console.print(Rule(style="dim")) # Print a final rule

        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[yellow]Goodbye![/yellow]")


# --- Main CLI ---
@click.command()
@click.version_option(version="0.1.0")
@click.argument("prompt", required=False)
@click.option("-s", "--system", default=None,
    help="Defines the system prompt for the AI."
)
@click.option("-m", "--model", default=None,
    help="ID of the model to use."
)
@click.option("-b", "--backend", default=None,
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"])
)
@click.option(
    "--no-markdown", is_flag=True, help="Disable Markdown rendering for LLM responses."
)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--quiet", is_flag=True, help="Suppress informational messages.")
@click.option(
    "--list-config", is_flag=True, help="List all configuration parameters."
)
@click.option(
    "--get-config", "get_config_key", help="Get a configuration parameter."
)
@click.option(
    "--set-config",
    "set_config_pair",
    help="Set a configuration parameter (KEY=VALUE).",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable streaming of LLM responses.",
)
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    prompt: Optional[str],
    system: Optional[str],
    model: Optional[str],
    backend: Optional[str],
    no_markdown: bool,
    json_output: bool,
    quiet: bool,
    list_config: bool,
    get_config_key: Optional[str],
    set_config_pair: Optional[str],
    no_stream: bool,
):
    """AI assistant in the command line with tool support."""
    options = CliOptions(
        prompt=prompt,
        system=system,
        model=model,
        backend=backend,
        no_markdown=no_markdown,
        json_output=json_output,
        quiet=quiet,
        list_config=list_config,
        get_config_key=get_config_key,
        set_config_pair=set_config_pair,
        no_stream=no_stream,
    )
    asyncio.run(async_main(options))

def _handle_config_options(options: CliOptions):
    """Handles the configuration options."""
    config = load_config()
    if options.list_config:
        print(json.dumps(config, indent=2))
        return True
    if options.get_config_key:
        print(config.get(options.get_config_key, "not set"))
        return True
    if options.set_config_pair:
        if "=" in options.set_config_pair:
            key, val = options.set_config_pair.split("=", 1)
            RAI_CONFIG.update(load_config())
            RAI_CONFIG[key] = val
            save_config()
            print(f"{key} set to: {val}")
        else:
            error_console.print("Invalid format for --set-config. Use KEY=VALUE.")
        return True
    return False


async def async_main(options: CliOptions):
    """The actual async logic of the application."""
    if _handle_config_options(options):
        return

    config = load_config()
    RAI_CONFIG["prompt"] = options.prompt

    RAI_CONFIG["model"] = options.model or config.get("model") or "gemma3:1b"
    RAI_CONFIG["backend"] = options.backend or config.get("backend") or "ollama"
    RAI_CONFIG["system"] = (
        options.system
        or config.get("system")
        or "You are a versatile and helpful AI assistant."
    )

    if not options.quiet and not options.prompt:
        error_console.print(
            f"[dim]Using model: [bold]{RAI_CONFIG['model']}"
            f"[/bold] on backend: [bold]{RAI_CONFIG['backend']}[/bold][/dim]"
        )

    has_tools = True
    if RAI_CONFIG["backend"] == "ollama" and not options.prompt:
        try:
            ollama.show(RAI_CONFIG["model"])
        except ResponseError:
            error_console.print(
                f"\n[bold red]Error: Model '{RAI_CONFIG['model']}' not found in Ollama.[/bold red]"
            )
            sys.exit(1)
        except ConnectionError as e:
            error_console.print(f"\n[bold red]Error connecting to Ollama: {e}[/bold red]")
            sys.exit(1)

        has_tools = check_model_tool_support(RAI_CONFIG["model"])
        if not has_tools and not options.quiet:
            error_console.print(
                f"[yellow]Warning: Model '{RAI_CONFIG['model']}'"
                + "may not support tools. Text-only mode."
            )

    agent, startup_messages = setup_agent(
        enable_tools=has_tools,
        quiet=options.quiet,
    )

    if options.prompt:
        run_single_query(
            agent,
            options.prompt,
            no_markdown=options.no_markdown,
            json_output=options.json_output,
        )
    else:
        for msg in startup_messages:
            console.print(Panel(msg, border_style="yellow"))
        await run_interactive_chat(agent, options.quiet, options.no_markdown, options.no_stream)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
