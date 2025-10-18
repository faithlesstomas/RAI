"""
rai - Rich AI CLI assistant
"""

import asyncio
import io
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import ollama
from agno.agent import Agent
from ollama import ResponseError
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import NestedCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from agno.utils.log import logger # Import logger

from .core import (
    RAI_CONFIG, console, error_console, load_config, save_config, setup_agent
)
from .ipc_server import run_ipc_server
from .tts import TTS, resolve_voice_path


# --- Constants ---
MIN_RENAME_ARGS = 2
MIN_SET_CONFIG_ARGS = 2
DEFAULT_TTS_DATA_DIR = os.path.expanduser("~/.local/share/rai/piper_voices")


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
    config_path: Optional[str] = None
    debug: bool = False
    tts_voice_id: Optional[str] = None


async def _cancel_active_tts_task() -> None:
    """Safely cancels the currently active TTS task."""
    active_task = RAI_CONFIG.get('active_tts_task')
    if active_task and not active_task.done():
        console.print("[dim]Cancelling previous speech...[/dim]")
        active_task.cancel()
        try:
            await active_task
        except asyncio.CancelledError:
            pass  # Cancellation is expected


def _build_completer(config_path: Optional[str] = None) -> NestedCompleter:
    """Builds a nested completer for interactive slash commands."""
    app_config = load_config(path=config_path)
    session_names = list(app_config.get("sessions", {}).keys())
    session_name_completer = WordCompleter(session_names, ignore_case=True, match_middle=True)
    config_key_completer = WordCompleter(
        ["model", "backend", "system", "tools", "tts.data_dir", "tts.default_voice"],
        ignore_case=True,
        match_middle=True
    )
    return NestedCompleter.from_nested_dict({
        "/help": None, "/exit": None, "/quit": None, "/q": None,
        "/session": {
            "list": None, "switch": session_name_completer, "show": session_name_completer,
            "delete": session_name_completer, "rename": session_name_completer,
        },
        "/config": {"show": None, "get": config_key_completer, "set": config_key_completer},
    })


def check_model_tool_support(model_id: str) -> bool:
    """Checks if the specified Ollama model supports tool use."""
    try:
        details = ollama.show(model_id)
        # pylint: disable=unsupported-membership-test
        # pylint: disable=unsupported-membership-test
        # pylint: disable=unsupported-membership-test
        return "tool_use" in details.get("modelfile", "")
    except ResponseError:
        return False


# --- Slash Command Handlers ---
def _handle_help_command(_: List[str]) -> None:
    """Handles the /help command."""
    console.print("Available commands:")
    for cmd in _SLASH_COMMAND_HANDLERS:
        console.print(f"  /{cmd}")
    console.print("  /exit, /quit, /q")


def _handle_session_command(args: List[str]) -> None:
    """Handles /session slash commands."""
    if not args:
        console.print("Usage: /session [list|switch|show|delete|rename] [args...]")
        return

    subcommand, *command_args = args

    def _handle_rename() -> None:
        if len(command_args) >= MIN_RENAME_ARGS:
            _rename_session_logic(command_args[0], command_args[1])
        else:
            error_console.print("[red]Usage: /session rename <old_name> <new_name>[/red]")

    def _handle_delete() -> None:
        if command_args:
            _delete_session_logic(command_args[0])
        else:
            error_console.print("[red]Usage: /session delete <session_name>[/red]")

    def _handle_show() -> None:
        _show_session_logic(command_args[0] if command_args else None)

    def _handle_switch() -> None:
        if command_args:
            _switch_session_logic(command_args[0])
        else:
            error_console.print("[red]Usage: /session switch <session_name>[/red]")

    session_commands: Dict[str, Callable] = {
        "list": _list_sessions_logic, "switch": _handle_switch, "show": _handle_show,
        "delete": _handle_delete, "rename": _handle_rename,
    }

    handler = session_commands.get(subcommand.lower())
    if handler:
        handler()
    else:
        error_console.print(f"[red]Unknown session command: {subcommand}[/red]")


def _handle_config_command(args: List[str]) -> None:
    """Handles /config slash commands."""
    subcommand, *command_args = (args[0].lower(), args[1:]) if args else ("show", [])

    def _handle_set() -> None:
        if len(command_args) >= MIN_SET_CONFIG_ARGS:
            _set_config_logic(command_args[0], " ".join(command_args[1:]))
        else:
            error_console.print("[red]Usage: /config set <key> <value>[/red]")

    def _handle_get() -> None:
        if command_args:
            _get_config_logic(command_args[0])
        else:
            error_console.print("[red]Usage: /config get <key>[/red]")

    config_commands: Dict[str, Callable] = {
        "show": _show_config_logic, "set": _handle_set, "get": _handle_get,
    }

    handler = config_commands.get(subcommand)
    if handler:
        handler()
    else:
        error_console.print(f"[red]Unknown config command: {subcommand}[/red]")


_SLASH_COMMAND_HANDLERS: Dict[str, Callable[[List[str]], None]] = {
    "help": _handle_help_command,
    "session": _handle_session_command,
    "config": _handle_config_command,
}


def _handle_slash_command(user_input: str) -> bool:
    """Handles slash commands and returns True if the app should exit."""
    command, *args = user_input.strip()[1:].split()
    if not command:
        return False

    if command.lower() in ["exit", "quit", "q"]:
        return True

    handler = _SLASH_COMMAND_HANDLERS.get(command.lower())
    if handler:
        handler(args)
    else:
        error_console.print(f"[red]Unknown command: /{command}[/red]")

    return False


# --- Response Handling ---
async def _handle_stream_response(agent: Agent, user_input: str, tts_instance: Optional[TTS] = None) -> None:
    """Handles the streaming response from the agent."""
    full_response = ""
    try:
        response_content_streamed = False
        response_iterator = await agent.arun(user_input, stream=True)
        first_event = None
        with console.status("[bold green]Assistant is thinking..."):
            try:
                first_event = await anext(response_iterator)
            except StopAsyncIteration:
                pass  # No response

        if first_event:
            if hasattr(first_event, "tool_calls") and first_event.tool_calls:
                console.print()
                console.print(Panel(json.dumps(first_event.tool_calls, indent=2), title="Tool Calls", border_style="yellow"))
            if first_event.content:
                response_content_streamed = True
                console.print(first_event.content, end="")
                full_response += first_event.content

        async for event in response_iterator:
            if hasattr(event, "tool_calls") and event.tool_calls:
                console.print()
                console.print(Panel(json.dumps(event.tool_calls, indent=2), title="Tool Calls", border_style="yellow"))
            if event.content:
                response_content_streamed = True
                console.print(event.content, end="")
                full_response += event.content

        if response_content_streamed:
            console.print()

        if tts_instance and full_response:
            await _cancel_active_tts_task()
            console.print("[dim]Synthesizing speech in the background...[/dim]")
            new_task = asyncio.create_task(tts_instance.synthesize(full_response))
            RAI_CONFIG["active_tts_task"] = new_task

    except Exception as e:
        error_console.print(f"[bold red]An error occurred during agent execution:[/bold red]\n{e}")


async def _handle_non_stream_response(agent: Agent, user_input: str, no_markdown: bool, tts_instance: Optional[TTS] = None) -> None:
    """Handles the non-streaming response from the agent."""
    log_capture_string = io.StringIO()
    log_handler = logging.StreamHandler(log_capture_string)
    agno_logger = logging.getLogger("agno")
    original_handlers, agno_logger.handlers = agno_logger.handlers, [log_handler]
    original_propagate, agno_logger.propagate = agno_logger.propagate, False
    original_level, agno_logger.level = agno_logger.level, logging.INFO

    try:
        with console.status("[bold green]Assistant is thinking..."):
            response = await agent.arun(user_input, stream=False)
    except Exception as e:
        error_console.print(f"[bold red]An error occurred during agent execution:[/bold red]\n{e}")
        return
    finally:
        agno_logger.handlers, agno_logger.propagate, agno_logger.level = original_handlers, original_propagate, original_level

    if captured_logs := log_capture_string.getvalue().strip():
        console.print(Panel(captured_logs, title="[dim]Agno Logs[/dim]", border_style="dim"))

    if response and response.content:
        console.print(Markdown(response.content) if not no_markdown else response.content)
        if tts_instance:
            await _cancel_active_tts_task()
            console.print("[dim]Synthesizing speech in the background...[/dim]")
            new_task = asyncio.create_task(tts_instance.synthesize(response.content))
            RAI_CONFIG["active_tts_task"] = new_task


# --- Interactive Mode ---
def create_key_bindings() -> KeyBindings:
    """Creates custom key bindings for the prompt."""
    kb = KeyBindings()

    @kb.add(Keys.Tab)
    def _(event) -> None:
        b = event.app.current_buffer
        if b.suggestion:
            b.insert_text(b.suggestion.text)
        elif b.complete_state:
            b.complete_next()
        else:
            b.start_completion(select_first=True)

    return kb


async def run_interactive_chat(agent: Agent, quiet: bool, stream: bool, no_markdown_flag: bool, tts_instance: Optional[TTS] = None) -> None:
    """Runs the main interactive chat loop."""
    if not quiet:
        console.print("**Welcome to Rich AI CLI Assistant!** Type your prompt and press Enter.")
        console.print("[dim]Type /help for a list of commands, Ctrl+C or /q to exit.[/dim]")

    history_file = os.path.join(os.path.expanduser("~/.config/rai"), "history.txt")
    prompt_session = PromptSession(
        history=FileHistory(history_file), completer=_build_completer(),
        complete_while_typing=True, auto_suggest=AutoSuggestFromHistory(),
        key_bindings=create_key_bindings(),
    )

    try:
        while True:
            try:
                user_input = await prompt_session.prompt_async("> ")
                if not user_input.strip():
                    continue
                if user_input.startswith("/"):
                    if _handle_slash_command(user_input):
                        break
                    continue

                if stream:
                    await _handle_stream_response(agent, user_input, tts_instance=tts_instance)
                else:
                    await _handle_non_stream_response(agent, user_input, no_markdown=no_markdown_flag, tts_instance=tts_instance)
                console.print(Rule(style="dim"))
            except (KeyboardInterrupt, EOFError):
                break
    finally:
        await _cancel_active_tts_task()
        
    console.print("\n[yellow]Goodbye![/yellow]")


# --- Main Application Logic ---
def _initialize_ollama_check(model_id: str, quiet: bool) -> bool:
    """Checks if the Ollama model is available and supports tools."""
    try:
        ollama.show(model_id)
    except ResponseError:
        error_console.print(f"\n[bold red]Error: Model '{model_id}' not found in Ollama.[/bold red]")
        sys.exit(1)
    except ConnectionError as e:
        error_console.print(f"\n[bold red]Error connecting to Ollama: {e}[/bold red]")
        sys.exit(1)

    has_tools = check_model_tool_support(model_id)
    if not has_tools and not quiet:
        error_console.print(f"[yellow]Warning: Model '{model_id}' may not support tools. Text-only mode.")
    return has_tools


def _setup_session(app_config: Dict[str, Any], options: CliOptions) -> Tuple[str, Dict[str, Any]]:
    """Determines the session to use and its configuration."""
    session_to_use = options.session_override or app_config.get("active_session", "default")
    sessions = app_config.setdefault("sessions", {})

    # Ensure the session exists
    if session_to_use not in sessions:
        sessions[session_to_use] = {
            "model": "gemma3:1b", "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
            "tools": ["CalculatorTools", "ArxivTools", "WikipediaTools", "DuckDuckGoTools", "WebBrowserTools", "FileTools", "PythonTools", "ShellTools"],
        }

    # Ensure the TTS config exists in the session
    if "tts" not in sessions[session_to_use]:
        sessions[session_to_use]["tts"] = {
            "data_dir": DEFAULT_TTS_DATA_DIR,
            "default_voice": "pl_PL-gosia-medium",
        }

    save_config(app_config, path=options.config_path)
    return session_to_use, sessions[session_to_use]


async def async_main(options: CliOptions) -> None:  # noqa: PLR0912, PLR0915
    """The actual async logic of the application."""
    if options.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("gitlab").setLevel(logging.DEBUG)
    app_config = load_config(path=options.config_path)
    session_to_use, session_config = _setup_session(app_config, options)

    RAI_CONFIG.update({
        "model": options.model or session_config.get("model"),
        "backend": options.backend or session_config.get("backend"),
        "system": options.system or session_config.get("system"),
        "prompt": options.prompt,
    })

    # --- TTS Setup ---
    tts_instance = None
    RAI_CONFIG["active_tts_task"] = None
    if options.tts_voice_id:
        console.print("[dim][TTS Debug] --tts flag detected.[/dim]")
        tts_config = session_config.get("tts", {})
        data_dir = tts_config.get("data_dir", DEFAULT_TTS_DATA_DIR)
        console.print(f"[dim][TTS Debug] Using data_dir: {data_dir}[/dim]")

        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)

        voice_id_to_use = options.tts_voice_id
        if voice_id_to_use == "_default_":
            voice_id_to_use = tts_config.get("default_voice")
            console.print(f"[dim][TTS Debug] Using default voice_id: {voice_id_to_use}[/dim]")

        if voice_id_to_use:
            console.print(f"[dim][TTS Debug] Resolving voice_id: {voice_id_to_use}[/dim]")
            model_path = resolve_voice_path(voice_id_to_use, data_dir)
            if model_path:
                console.print(f"[dim][TTS Debug] Model path resolved: {model_path}[/dim]")
                tts_instance = TTS(model_path)
            else:
                error_console.print(f"[red]TTS Error: Could not resolve voice '{voice_id_to_use}'.[/red]")
        else:
            error_console.print("[red]TTS Error: --tts flag used, but no default voice is configured.[/red]")


    is_streaming = options.stream
    no_markdown_flag = options.no_markdown
    use_agent_markdown = not (no_markdown_flag or is_streaming)
    session_id = session_to_use if not options.prompt else None

    if not options.quiet and not options.prompt:
        active_session_name = app_config.get("active_session", "default")
        session_display = f"[bold]{session_to_use}[/bold]"
        if session_to_use == active_session_name:
            session_display += " (active)"
        else:
            session_display += " (override)"
        error_console.print(
            f"[dim]Session: {session_display} | Model: [bold]{RAI_CONFIG['model']}[/bold] on "
            f"backend: [bold]{RAI_CONFIG['backend']}[/bold][/dim]"
        )

    has_tools = RAI_CONFIG["backend"] != "ollama" or _initialize_ollama_check(RAI_CONFIG["model"], options.quiet)
    agent, startup_messages = setup_agent(
        enable_tools=has_tools, quiet=options.quiet,
        use_markdown=use_agent_markdown, session_id=session_id,
    )

    logger.debug(f"Agent tools after setup: {agent.tools}")

    if options.prompt:
        if is_streaming:
            await _handle_stream_response(agent, options.prompt, tts_instance=tts_instance)
        else:
            await _handle_non_stream_response(agent, options.prompt, no_markdown=no_markdown_flag, tts_instance=tts_instance)

        # In non-interactive mode, wait for the TTS task to finish before exiting
        if RAI_CONFIG.get("active_tts_task") and not RAI_CONFIG["active_tts_task"].done():
            try:
                await RAI_CONFIG["active_tts_task"]
            except asyncio.CancelledError:
                pass # Task was cancelled, which is fine
    else:
        for msg in startup_messages:
            console.print(Panel(msg, border_style="yellow"))
        await run_interactive_chat(
            agent, quiet=options.quiet, stream=is_streaming, no_markdown_flag=no_markdown_flag, tts_instance=tts_instance
        )


# --- CLI Command Groups ---
@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0")
@click.option("-p", "--prompt", "prompt", default=None, help="The prompt to send to the AI.")
@click.option("-s", "--system", default=None, help="Defines the system prompt for the AI.")
@click.option("-m", "--model", default=None, help="ID of the model to use.")
@click.option(
    "-b", "--backend", default=None,
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"]),
)
@click.option(
    "--config", "config_path", default=None, help="Path to a custom configuration file.",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option("--no-markdown", is_flag=True, help="Disable Markdown rendering for LLM responses.")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--quiet", is_flag=True, help="Suppress informational messages.")
@click.option("--stream", is_flag=True, help="Enable streaming of LLM responses (disables Markdown).")
@click.option(
    "--session", "session_override", default=None,
    help="Run in a specific session for this command only.",
)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option(
    "--tts",
    "tts_voice_id",
    default=None,
    is_flag=False,
    flag_value="_default_",  # Special value when only --tts is used
    help="Enable Text-to-Speech output. Optionally provide a voice ID.",
)
@click.pass_context
def cli(ctx: click.Context, **kwargs: Any) -> None:
    """AI assistant in the command line with tool support."""
    ctx.obj = kwargs
    if ctx.invoked_subcommand is not None:
        return

    if not kwargs.get("prompt") and not sys.stdin.isatty():
        kwargs["prompt"] = sys.stdin.read().strip()

    options = CliOptions(**kwargs)
    asyncio.run(async_main(options))


@cli.group()
def session() -> None:
    """Manage chat sessions."""


def _switch_session_logic(session_name: str) -> None:
    """The actual logic for creating/switching sessions."""
    app_config = load_config()
    if session_name not in app_config.get("sessions", {}):
        console.print(f"Creating new session: [bold]{session_name}[/bold]")
        app_config.setdefault("sessions", {})[session_name] = {
            "model": "gemma3:1b", "backend": "ollama",
            "system": "You are a versatile and helpful AI assistant.",
        }
    app_config["active_session"] = session_name
    save_config(app_config)
    console.print(f"Switched to session: [bold green]{session_name}[/bold green]")
    console.print("[yellow]Note: The new session will be used the next time you start rai.[/yellow]")


@session.command(name="switch")
@click.argument("session_name")
def switch_session(session_name: str) -> None:
    """Creates a new session or switches to an existing one."""
    _switch_session_logic(session_name)



def _list_sessions_logic() -> None:
    """The actual logic for listing sessions."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})
    active_session = app_config.get("active_session", "default")
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return
    console.print("[bold]Available Sessions:[/bold]")
    for name in sessions:
        if name == active_session:
            console.print(f"- [bold green]{name} (active)[/bold green]")
        else:
            console.print(f"- {name}")


@session.command(name="list")
def list_sessions() -> None:
    """Lists all available sessions."""
    _list_sessions_logic()


def _show_session_logic(session_name: Optional[str] = None) -> None:
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
def show_session(session_name: Optional[str]) -> None:
    """Shows configuration for a specific or active session."""
    _show_session_logic(session_name)


def _delete_session_logic(session_name: str) -> None:
    """The actual logic for deleting a session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    if session_name == active_session:
        error_console.print("[bold red]Error: Cannot delete the active session.[/bold red]")
        error_console.print(f"Switch to a different session before deleting '{session_name}'.")
        return
    if session_name not in app_config.get("sessions", {}):
        error_console.print(f"[bold red]Error: Session '{session_name}' not found.[/bold red]")
        return
    del app_config["sessions"][session_name]
    save_config(app_config)
    console.print(f"Session '[bold red]{session_name}[/bold red]' has been deleted.")


@session.command(name="delete")
@click.argument("session_name")
def delete_session(session_name: str) -> None:
    """Deletes a specified session."""
    _delete_session_logic(session_name)


def _rename_session_logic(old_name: str, new_name: str) -> None:
    """The actual logic for renaming a session."""
    app_config = load_config()
    sessions = app_config.get("sessions", {})
    if old_name not in sessions:
        error_console.print(f"[bold red]Error: Session '{old_name}' not found.[/bold red]")
        return
    if new_name in sessions:
        error_console.print(f"[bold red]Error: Session name '{new_name}' already exists.[/bold red]")
        return
    sessions[new_name] = sessions.pop(old_name)
    console.print(f"Session '{old_name}' has been renamed to '[bold green]{new_name}[/bold green]'.")
    if app_config.get("active_session") == old_name:
        app_config["active_session"] = new_name
        console.print(f"Active session has been updated to '[bold green]{new_name}[/bold green]'.")
    save_config(app_config)


@session.command(name="rename")
@click.argument("old_name")
@click.argument("new_name")
def rename_session(old_name: str, new_name: str) -> None:
    """Renames a session."""
    _rename_session_logic(old_name, new_name)


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context) -> None:
    """View or manage the configuration of the active session."""
    if ctx.invoked_subcommand is None:
        _show_config_logic()


def _show_config_logic() -> None:
    """The actual logic for showing the active session's configuration."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(active_session)
    if not session_config:
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found in configuration.[/bold red]")
        return
    console.print(f"[bold]Configuration for active session: [cyan]{active_session}[/cyan][/bold]")
    console.print(json.dumps(session_config, indent=2))


@config.command(name="show")
def show_config() -> None:
    """Shows configuration for the active session."""
    _show_config_logic()

def _set_config_logic(key: str, value: str) -> None:
    """The actual logic for setting a config value in the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    if active_session not in app_config.get("sessions", {}):
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found.[/bold red]")
        return
    allowed_keys = ["model", "backend", "system", "tools"]
    if key not in allowed_keys:
        error_console.print(f"[bold red]Error: Invalid configuration key '{key}'.[/bold red]")
        error_console.print(f"Allowed keys are: {', '.join(allowed_keys)}")
        return
    if key == "tools":
        app_config["sessions"][active_session][key] = [t.strip() for t in value.split(",")]
    else:
        app_config["sessions"][active_session][key] = value
    save_config(app_config)
    console.print(
        f"In session '[cyan]{active_session}[/cyan]', set '[bold]{key}[/bold]' to "
        f"'[green]{value}[/green]'.")


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str) -> None:
    """Sets a configuration value for the active session."""
    _set_config_logic(key, value)

def _get_config_logic(key: str) -> None:
    """The actual logic for getting a config value from the active session."""
    app_config = load_config()
    active_session = app_config.get("active_session", "default")
    session_config = app_config.get("sessions", {}).get(active_session)
    if not session_config:
        error_console.print(f"[bold red]Error: Active session '{active_session}' not found.[/bold red]")
        return
    value = session_config.get(key)
    if value is None:
        error_console.print(f"[bold red]Error: Key '{key}' not found in session '{active_session}'.[/bold red]")
    else:
        console.print(value)


@config.command(name="get")
@click.argument("key")
def get_config(key: str) -> None:
    """Gets a configuration value from the active session."""
    _get_config_logic(key)


@cli.command(name="serve-ipc")
@click.pass_context
def serve_ipc(ctx: click.Context) -> None:
    """
    Runs the IPC server to allow other processes to interact with the AI agent.
    """
    # pylint: disable=import-outside-toplevel
    # noqa: PLC0415, C0415
    # noqa: PLC0415, C0415
    # noqa: PLC0415, C0415
    config_path = ctx.obj.get('config_path')
    run_ipc_server(config_path=config_path)


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=no-value-for-parameter