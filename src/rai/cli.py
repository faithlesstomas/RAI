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
from prompt_toolkit.completion import NestedCompleter, WordCompleter
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

    stream: bool = False
    session_override: Optional[str] = None


console = Console(record=True)
error_console = Console(stderr=True)

CONFIG_DIR = os.path.expanduser("~/.config/rai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
RAI_CONFIG = {}

def _build_completer():
    """Builds a nested completer for interactive slash commands."""
    app_config = load_config()
    session_names = list(app_config.get("sessions", {}).keys())

    # Use WordCompleter for dynamic parts
    session_name_completer = WordCompleter(session_names, ignore_case=True)
    config_key_completer = WordCompleter(["model", "backend", "system"], ignore_case=True)

    return NestedCompleter.from_nested_dict({
        "/help": None,
        "/exit": None,
        "/quit": None,
        "/q": None,
        "/session": {
            "list": None,
            "switch": session_name_completer,
            "show": session_name_completer,
            "delete": session_name_completer,
            "rename": session_name_completer,
        },
        "/config": {
            "show": None,
            "get": config_key_completer,
            "set": config_key_completer,
        },
    })


# --- Configuration ---
def load_config():
    """Loads the configuration from the config file."""
    if not os.path.exists(CONFIG_FILE):
        return {}  # pylint: disable=unhashable-member
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config_data: dict):
    """Saves the provided configuration data to the config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


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
def setup_agent(
    enable_tools: bool = True,
    quiet: bool = False,
    use_markdown: bool = True,
    session_id: Optional[str] = "my_chat_session",
):
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
            markdown=use_markdown,
            add_history_to_messages=True,
            storage=SqliteStorage(
                table_name="agent_sessions",
                db_file="tmp/data.db",
                auto_upgrade_schema=True,
            ),
            session_id=session_id,
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





def _handle_help_command(args):
    """Handles the /help command."""
    # For now, a simple list. Can be expanded later.
    console.print("Available commands:")
    for cmd in _SLASH_COMMAND_HANDLERS.keys():
        console.print(f"  /{cmd}")
    console.print("  /exit, /quit, /q")


def _handle_session_command(args):
    """Handles /session slash commands."""
    if not args:
        console.print("Usage: /session [list|switch|show|delete|rename] [args...]")
        return

    subcommand = args[0].lower()
    if subcommand == "list":
        _list_sessions_logic()
    elif subcommand == "switch":
        if len(args) > 1:
            session_name = args[1]
            _switch_session_logic(session_name)
        else:
            error_console.print("[red]Usage: /session switch <session_name>[/red]")
    elif subcommand == "show":
        session_name = args[1] if len(args) > 1 else None
        _show_session_logic(session_name)
    elif subcommand == "delete":
        if len(args) > 1:
            session_name = args[1]
            _delete_session_logic(session_name)
        else:
            error_console.print("[red]Usage: /session delete <session_name>[/red]")
    elif subcommand == "rename":
        if len(args) > 2:
            old_name = args[1]
            new_name = args[2]
            _rename_session_logic(old_name, new_name)
        else:
            error_console.print("[red]Usage: /session rename <old_name> <new_name>[/red]")
    else:
        error_console.print(f"[red]Unknown session command: {subcommand}[/red]")


def _handle_config_command(args):
    """Handles /config slash commands."""
    subcommand = args[0].lower() if args else "show"

    if subcommand == "show":
        _show_config_logic()
    elif subcommand == "set":
        if len(args) > 2:
            key = args[1]
            value = " ".join(args[2:])
            _set_config_logic(key, value)
        else:
            error_console.print("[red]Usage: /config set <key> <value>[/red]")
    elif subcommand == "get":
        if len(args) > 1:
            key = args[1]
            _get_config_logic(key)
        else:
            error_console.print("[red]Usage: /config get <key>[/red]")
    else:
        error_console.print(f"[red]Unknown config command: {subcommand}[/red]")


_SLASH_COMMAND_HANDLERS = {
    "help": _handle_help_command,
    "session": _handle_session_command,
    "config": _handle_config_command,
}

def _handle_slash_command(user_input):
    """Handles slash commands and returns True if the app should exit."""
    parts = user_input.strip()[1:].split()
    if not parts:
        return False

    command = parts[0].lower()
    args = parts[1:]

    if command in ["exit", "quit", "q"]:
        return True

    handler = _SLASH_COMMAND_HANDLERS.get(command)
    if handler:
        handler(args)
    else:
        error_console.print(f"[red]Unknown command: /{command}[/red]")

    return False


async def _handle_stream_response(agent, user_input):
    """Handles the streaming response from the agent, printing raw content."""
    try:
        response_content_streamed = False
        response_iterator = await agent.arun(user_input, stream=True)

        # Show spinner while waiting for the FIRST chunk
        first_event = None
        if sys.stdout.isatty():
            with console.status("[bold green]Assistant is thinking..."):
                try:
                    # Get the first event from the async iterator
                    first_event = await anext(response_iterator)
                except StopAsyncIteration:
                    # Handle case where there's no response at all (e.g., empty stream)
                    first_event = None
        else:
            try:
                first_event = await anext(response_iterator)
            except StopAsyncIteration:
                first_event = None

        # Now, process and print events
        if first_event:
            if hasattr(first_event, "tool_calls") and first_event.tool_calls:
                console.print()  # Add newline before tool calls panel
                tool_calls_json = json.dumps(first_event.tool_calls, indent=2)
                console.print(
                    Panel(tool_calls_json, title="Tool Calls", border_style="yellow")
                )
            if first_event.content:
                response_content_streamed = True
                console.print(first_event.content, end="")  # Print raw content

        # Loop through the rest of the events
        async for event in response_iterator:
            if hasattr(event, "tool_calls") and event.tool_calls:
                console.print()  # Add newline before tool calls panel
                tool_calls_json = json.dumps(event.tool_calls, indent=2)
                console.print(
                    Panel(tool_calls_json, title="Tool Calls", border_style="yellow")
                )
            if event.content:
                response_content_streamed = True
                console.print(event.content, end="")  # Print raw content

        # If we streamed any content, print a final newline to finish the line.
        if response_content_streamed:
            console.print()
    except Exception as e:  # pylint: disable=broad-except
        error_console.print(f"[bold red]An error occurred during agent execution:[/bold red]\n{e}")
        return


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

    try:
        if sys.stdout.isatty():
            with console.status("[bold green]Assistant is thinking..."):
                response = await agent.arun(user_input, stream=False)
        else:
            response = await agent.arun(user_input, stream=False)
    except Exception as e:  # pylint: disable=broad-except
        error_console.print(f"[bold red]An error occurred during agent execution:[/bold red]\n{e}")
        return

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
async def run_interactive_chat(agent, quiet: bool, stream: bool, no_markdown_flag: bool):
    """Runs the main interactive chat loop."""
    if not quiet:
        console.print(
            "**Welcome to Rich AI CLI Assistant!** Type your prompt and press Enter."
        )
        console.print(
            "[dim]Type /help for a list of commands, Ctrl+C or /q to exit.[/dim]"
        )

    history_file = os.path.join(CONFIG_DIR, "history.txt")
    prompt_session = PromptSession(
        history=FileHistory(history_file), completer=_build_completer()
    )

    while True:
        try:
            user_input = await prompt_session.prompt_async("> ")
            if not user_input.strip():
                continue

            # Handle Slash Commands
            if user_input.startswith("/"):
                if _handle_slash_command(user_input):
                    break
                continue

            # --- Process AI Query ---
            if stream:
                # Streaming mode forces no markdown rendering on the client
                await _handle_stream_response(agent, user_input)
            else:
                # Non-streaming mode respects the original flag
                await _handle_non_stream_response(agent, user_input, no_markdown=no_markdown_flag)

            console.print(Rule(style="dim"))  # Print a final rule

        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[yellow]Goodbye![/yellow]")


# --- Main CLI ---
@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.option("-p", "--prompt", "prompt", default=None, help="The prompt to send to the AI.")
@click.option("-s", "--system", default=None, help="Defines the system prompt for the AI.")
@click.option("-m", "--model", default=None, help="ID of the model to use.")
@click.option(
    "-b",
    "--backend",
    default=None,
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"]),
)
@click.option(
    "--no-markdown", is_flag=True, help="Disable Markdown rendering for LLM responses."
)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--quiet", is_flag=True, help="Suppress informational messages.")
@click.option(
    "--stream",
    is_flag=True,
    help="Enable streaming of LLM responses (disables Markdown).",
)
@click.option(
    "--session",
    "session_override",
    default=None,
    help="Run in a specific session for this command only.",
)
@click.pass_context
def cli(ctx, **kwargs):
    """AI assistant in the command line with tool support."""
    # If a subcommand is invoked (like 'session', 'config'), let it handle execution.
    if ctx.invoked_subcommand is not None:
        return

    # Handle piped input if no prompt is given via -p
    if not kwargs.get("prompt") and not sys.stdin.isatty():
        kwargs["prompt"] = sys.stdin.read().strip()

    # Default action: run the main logic
    options = CliOptions(**kwargs)
    asyncio.run(async_main(options))




def _initialize_ollama_check(model_id: str, quiet: bool) -> bool:
    """Checks if the Ollama model is available and supports tools."""
    try:
        ollama.show(model_id)
    except ResponseError:
        error_console.print(
            f"\n[bold red]Error: Model '{model_id}' not found in Ollama.[/bold red]"
        )
        sys.exit(1)
    except ConnectionError as e:
        error_console.print(f"\n[bold red]Error connecting to Ollama: {e}[/bold red]")
        sys.exit(1)

    has_tools = check_model_tool_support(model_id)
    if not has_tools and not quiet:
        error_console.print(
            f"[yellow]Warning: Model '{model_id}' may not support tools. Text-only mode."
        )
    return has_tools


async def async_main(options: CliOptions):
    """The actual async logic of the application."""
    config = load_config()

    # --- New Session and Configuration Logic ---
    session_to_use = "default"
    if options.session_override:
        # A session was specified on the command line for a one-time run
        session_to_use = options.session_override
    else:
        # Default behavior: use the globally active session
        session_to_use = config.get("active_session", "default")

    # Ensure the session definition exists in the config, create it if not.
    if session_to_use not in config.get("sessions", {}):
        config.setdefault("sessions", {})[session_to_use] = {
            "model": "gemma3:1b",
            "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
        }
        save_config(config)  # Save the config since we added a new session

    # Get the configuration for the session we're using for this run
    session_config = config["sessions"][session_to_use]

    # Prioritize CLI options > session config
    RAI_CONFIG["model"] = options.model or session_config.get("model")
    RAI_CONFIG["backend"] = options.backend or session_config.get("backend")
    RAI_CONFIG["system"] = options.system or session_config.get("system")
    RAI_CONFIG["prompt"] = options.prompt

    # --- Core logic for streaming and markdown ---
    is_streaming = options.stream
    no_markdown_flag = options.no_markdown
    use_agent_markdown = not (no_markdown_flag or is_streaming)

    # --- Core logic for session management ---
    session_id = None
    if not options.prompt:  # Interactive mode uses a persistent session
        session_id = session_to_use  # Use the determined session name
    # For single-query mode (options.prompt is not None), session_id remains None for a temporary session.

    if not options.quiet and not options.prompt:
        active_session_name = config.get("active_session", "default")
        session_display = f"[bold]{session_to_use}[/bold]"
        if session_to_use == active_session_name:
            session_display += " (active)"
        else:
            session_display += " (override)"

        error_console.print(
            f"[dim]Session: {session_display} | "
            f"Model: [bold]{RAI_CONFIG['model']}[/bold] on "
            f"backend: [bold]{RAI_CONFIG['backend']}[/bold][/dim]"
        )

    has_tools = True
    if RAI_CONFIG["backend"] == "ollama":
        has_tools = _initialize_ollama_check(RAI_CONFIG["model"], options.quiet)

    agent, startup_messages = setup_agent(
        enable_tools=has_tools,
        quiet=options.quiet,
        use_markdown=use_agent_markdown,
        session_id=session_id,
    )

    if options.prompt:
        # Single query mode
        if is_streaming:
            await _handle_stream_response(agent, options.prompt)
        else:
            await _handle_non_stream_response(
                agent, options.prompt, no_markdown=no_markdown_flag
            )
    else:
        # Interactive mode
        for msg in startup_messages:
            console.print(Panel(msg, border_style="yellow"))
        await run_interactive_chat(
            agent,
            quiet=options.quiet,
            stream=is_streaming,
            no_markdown_flag=no_markdown_flag,
        )


@cli.group()
def session():
    """Manage chat sessions."""


def _switch_session_logic(session_name: str):
    """The actual logic for creating/switching sessions."""
    app_config = load_config()

    # Create session with defaults if it doesn't exist
    if session_name not in app_config.get("sessions", {}):
        console.print(f"Creating new session: [bold]{session_name}[/bold]")
        default_session = {
            "model": "gemma3:1b",
            "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
        }
        app_config.setdefault("sessions", {})[session_name] = default_session

    # Set the new active session
    app_config["active_session"] = session_name
    save_config(app_config)
    console.print(f"Switched to session: [bold green]{session_name}[/bold green]")
    console.print(
        "[yellow]Note: The new session will be used the next time you start rai.[/yellow]"
    )


@session.command(name="switch")
@click.argument("session_name")
def switch_session(session_name):
    """Creates a new session or switches to an existing one."""
    _switch_session_logic(session_name)


def _list_sessions_logic():
    """The actual logic for listing sessions."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})
    active_session = app_config.get("active_session", "default")

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    console.print("[bold]Available Sessions:[/bold]")
    for session_name in sessions:
        if session_name == active_session:
            console.print(f"- [bold green]{session_name} (active)[/bold green]")
        else:
            console.print(f"- {session_name}")


@session.command(name="list")
def list_sessions():
    """Lists all available sessions."""
    _list_sessions_logic()


def _show_session_logic(session_name: Optional[str] = None):
    """The actual logic for showing a session's configuration."""
    app_config = load_config()

    target_session = session_name or app_config.get("active_session", "default")

    session_config = app_config.get("sessions", {}).get(target_session)

    if not session_config:
        error_console.print(f"[bold red]Error: Session '{target_session}' not found.[/bold red]")
        return

    console.print(f"[bold]Configuration for session: [cyan]{target_session}[/cyan][/bold]")
    console.print(json.dumps(session_config, indent=2))


@session.command(name="show")
@click.argument("session_name", required=False)
def show_session(session_name):
    """Shows configuration for a specific or active session."""
    _show_session_logic(session_name)


def _delete_session_logic(session_name: str):
    """The actual logic for deleting a session."""
    app_config = load_config()

    active_session = app_config.get("active_session", "default")

    if session_name == active_session:
        error_console.print("[bold red]Error: Cannot delete the active session.[/bold red]")
        error_console.print(
            f"Switch to a different session before deleting '{session_name}'."
        )
        return

    if session_name not in app_config.get("sessions", {}):
        error_console.print(f"[bold red]Error: Session '{session_name}' not found.[/bold red]")
        return

    del app_config["sessions"][session_name]
    save_config(app_config)

    console.print(f"Session '[bold red]{session_name}[/bold red]' has been deleted.")


@session.command(name="delete")
@click.argument("session_name")
def delete_session(session_name):
    """Deletes a specified session."""
    _delete_session_logic(session_name)


def _rename_session_logic(old_name: str, new_name: str):
    """The actual logic for renaming a session."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})

    if old_name not in sessions:
        error_console.print(f"[bold red]Error: Session '{old_name}' not found.[/bold red]")
        return

    if new_name in sessions:
        error_console.print(
            f"[bold red]Error: Session name '{new_name}' already exists.[/bold red]"
        )
        return

    # Perform the rename
    sessions[new_name] = sessions.pop(old_name)
    console.print(
        f"Session '{old_name}' has been renamed to '[bold green]{new_name}[/bold green]'."
    )

    # If the renamed session was the active one, update the active_session key
    if app_config.get("active_session") == old_name:
        app_config["active_session"] = new_name
        console.print(
            f"Active session has been updated to "
            f"'[bold green]{new_name}[/bold green]'."
        )

    save_config(app_config)


@session.command(name="rename")
@click.argument("old_name")
@click.argument("new_name")
def rename_session(old_name, new_name):
    """Renames a session."""
    _rename_session_logic(old_name, new_name)


def _show_config_logic():
    """The actual logic for showing the active session's configuration."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(active_session)

    if not session_config:
        error_console.print(
            f"[bold red]Error: Active session '{active_session}' not found "
            f"in configuration.[/bold red]"
        )
        return

    console.print(
        f"[bold]Configuration for active session: [cyan]{active_session}[/cyan][/bold]"
    )
    console.print(json.dumps(session_config, indent=2))


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """View or manage the configuration of the active session."""
    if ctx.invoked_subcommand is None:
        _show_config_logic()


@config.command(name="show")
def show_config():
    """Shows configuration for the active session."""
    _show_config_logic()


def _set_config_logic(key: str, value: str):
    """The actual logic for setting a config value in the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")

    if active_session not in app_config.get("sessions", {}):
        error_console.print(
            f"[bold red]Error: Active session '{active_session}' not found.[/bold red]"
        )
        return

    # A list of keys that are allowed to be set
    allowed_keys = ["model", "backend", "system"]
    if key not in allowed_keys:
        error_console.print(
            f"[bold red]Error: Invalid configuration key '{key}'.[/bold red]"
        )
        error_console.print(f"Allowed keys are: {', '.join(allowed_keys)}")
        return

    app_config["sessions"][active_session][key] = value
    save_config(app_config)

    console.print(
        f"In session '[cyan]{active_session}[/cyan]', set '[bold]{key}[/bold]' to '[green]{value}[/green]'."
    )


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key, value):
    """Sets a configuration value for the active session."""
    _set_config_logic(key, value)


def _get_config_logic(key: str):
    """The actual logic for getting a config value from the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")

    session_config = app_config.get("sessions", {}).get(active_session)

    if not session_config:
        error_console.print(
            f"[bold red]Error: Active session '{active_session}' not found.[/bold red]"
        )
        return

    value = session_config.get(key)

    if value is None:
        error_console.print(
            f"[bold red]Error: Key '{key}' not found in session '{active_session}'.[/bold red]"
        )
    else:
        console.print(value)


@config.command(name="get")
@click.argument("key")
def get_config(key):
    """Gets a configuration value from the active session."""
    _get_config_logic(key)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
