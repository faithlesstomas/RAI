"""
Rich AI CLI module
"""

import asyncio
import collections
# import functools
# import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple # pylint: disable=unused-import

import click
import httpx
import ollama
# import websockets
# from agno.utils.log import logger  # Import logger
from ollama import ResponseError
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import NestedCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
# from pydantic_ai.exceptions import UserError
# from rich.console import Console, Group
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
# from websockets.exceptions import ConnectionClosed, WebSocketException
from returns.result import Success, Failure, Result

from . import config_manager
from .core import console, error_console
from .services.chat import ChatService
from typing import Union

class LocalProcessor:
    """Processor running agent executions locally inside the CLI process via google-antigravity."""
    def __init__(self, run_config: Dict[str, Any], session_name: str, enable_tools: bool = True) -> None:
        self.run_config = run_config
        self.session_name = session_name
        self.chat_service = ChatService()
        self.enable_tools = enable_tools

    async def arun(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Result[Dict[str, Any], Exception]:
        run_cfg = self.run_config.copy()
        if not self.enable_tools:
            run_cfg["tools"] = []
        return await self.chat_service.run_chain(
            chain_input=prompt,
            chain_configs=[run_cfg],
            session_id=self.session_name,
        )

    async def get_history(self) -> List[Dict[str, Any]]:
        result = await self.chat_service.get_session_history(self.session_name)
        if isinstance(result, Success):
            return result.unwrap()
        return []

    async def clear_history(self) -> None:
        await self.chat_service.clear_session_history(self.session_name)

    def reload(self) -> None:
        self.enable_tools = True
        if self.run_config.get("backend") == "ollama":
            try:
                self.enable_tools = check_model_tool_support(self.run_config.get("model", ""))
            except Exception:
                self.enable_tools = False

    async def close(self) -> None:
        pass


class ClientProcessor:
    """Processor running agent executions remotely by talking to the 'rai serve' daemon."""
    def __init__(self, run_config: Dict[str, Any], session_name: str, server_uri: str) -> None:
        self.run_config = run_config
        self.session_name = session_name
        self.server_uri = server_uri
        self.base_url = server_uri.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
        self.transport = None
        if server_uri.startswith("unix://"):
            uds_path = server_uri[len("unix://") :]
            self.transport = httpx.AsyncHTTPTransport(uds=uds_path)
            self.base_url = "http://localhost"
        self.client = httpx.AsyncClient(transport=self.transport, base_url=self.base_url, timeout=60)
        self.stateful_session_id = None

    async def connect(self) -> Result[None, Exception]:
        try:
            response = await self.client.get("/api/v1/models")
            response.raise_for_status()
            return Success(None)
        except Exception as e:
            error_console.print(
                "[bold red]Connection Error:[/bold red]"
                f" Could not connect to the rai server at {self.server_uri}. Error: {e}"
            )
            return Failure(e)

    async def arun(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> Result[Dict[str, Any], Exception]:
        payload = {
            "prompt": prompt,
            "agent_id": self.session_name,
            "session_id": self.stateful_session_id,
            "chain_configs": [self.run_config],
        }
        try:
            response = await self.client.post("/api/v1/run", json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                payload_data = data.get("payload", {})
                self.stateful_session_id = payload_data.get("session_id")
                return Success(payload_data)
            else:
                return Failure(Exception(data.get("detail", "Server error")))
        except Exception as e:
            return Failure(e)

    async def get_history(self) -> List[Dict[str, Any]]:
        session_to_query = self.stateful_session_id or self.session_name
        try:
            response = await self.client.get(f"/api/v1/history/sessions/{session_to_query}")
            response.raise_for_status()
            return response.json().get("messages", [])
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to fetch history from server: {e}")
            return []

    async def clear_history(self) -> None:
        session_to_clear = self.stateful_session_id or self.session_name
        try:
            response = await self.client.delete(f"/api/v1/history/sessions/{session_to_clear}")
            response.raise_for_status()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to clear history on server: {e}")

    def reload(self) -> None:
        pass

    async def close(self) -> None:
        await self.client.aclose()


Processor = Union[LocalProcessor, ClientProcessor]

# from .exceptions import ChainExecutionError
# from .tts import TTS, resolve_voice_path

# --- Custom Click Classes for Help Formatting ---

class SectionedOption(click.Option):
    """A click.Option that allows grouping options into sections."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: # noqa: ANN401
        self.section = kwargs.pop("section", "Options")
        super().__init__(*args, **kwargs)


class SectionedGroup(click.Group):
    """A click.Group that formats options into sections."""

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Writes all the options into the formatter grouped by section."""
        options_by_section = collections.defaultdict(list)
        other_opts = []

        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is None:
                continue
            if isinstance(param, SectionedOption):
                options_by_section[param.section].append(rv)
            else:
                other_opts.append(rv)

        section_order = [
            "Primary",
            "AI Configuration",
            "Output Formatting",
            "Speech",
            "Session & Debugging",
        ]

        for section_name in section_order:
            if section_name in options_by_section:
                with formatter.section(section_name):
                    formatter.write_dl(options_by_section[section_name])

        if other_opts:
            with formatter.section("Other Options"):
                formatter.write_dl(other_opts)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Ensures that the subcommands are always displayed."""
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None or cmd.hidden:
                continue
            commands.append((subcommand, cmd))

        if not commands:
            return

        # allow for 3 times the default spacing
        limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
        rows = [(subcommand, cmd.get_short_help_str(limit)) for subcommand, cmd in commands]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Formats the help string."""
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)
        self.format_options(ctx, formatter)
        self.format_commands(ctx, formatter)
        self.format_epilog(ctx, formatter)


# --- Constants ---
MIN_RENAME_ARGS = 2
MIN_SET_CONFIG_ARGS = 2


@dataclass
class CliOptions:  # pylint: disable=too-many-instance-attributes
    """Dataclass to hold CLI options."""

    prompt: Optional[str] = None
    connect_uri: Optional[str] = None
    system: Optional[str] = None
    model: Optional[str] = None
    backend: Optional[str] = None
    no_markdown: bool = False
    json_output: bool = False
    quiet: bool = False
    stream: bool = False
    session_override: Optional[str] = None
    config_path: Optional[str] = None
    log_level_opt: Optional[str] = None
    tts_voice_id: Optional[str] = None
    # This field will hold the active tts task, not a CLI option
    active_tts_task: Optional[asyncio.Task] = field(default=None, repr=False)


async def _cancel_active_tts_task(run_config: Dict[str, Any]) -> None:
    """Safely cancels the currently active TTS task."""
    active_task = run_config.get("active_tts_task")
    if active_task and not active_task.done():
        console.print("[dim]Cancelling previous speech...[/dim]")
        active_task.cancel()
        try:
            await active_task
        except asyncio.CancelledError:
            pass  # Cancellation is expected


def _build_completer(config_path: Optional[str] = None) -> NestedCompleter:
    """Builds a nested completer for interactive slash commands."""
    app_config = config_manager.load_config(path=config_path)
    session_names = list(app_config.get("sessions", {}).keys())
    # session_name_completer was unused
    config_key_completer = WordCompleter(
        ["model", "backend", "system", "tools", "tts.data_dir", "tts.default_voice"],
        ignore_case=True,
        match_middle=True,
    )
    return NestedCompleter.from_nested_dict(
        {
            "/help": None,
            "/exit": None,
            "/quit": None,
            "/q": None,
            "/config": {
                "show": None,
                "get": config_key_completer,
                "set": config_key_completer,
            },
        }
    )

def check_model_tool_support(model_id: str) -> bool:
    """Checks if the specified Ollama model supports tool use."""
    try:
        details = ollama.show(model_id)
        modelfile = str(details.get("modelfile", "") or "")
        # Robust check for tool support indicators
        indicators = [
            "tool_use",             # Explicit parameter
            "{{ .Tools",            # Template variable (standard)
            "{{.Tools",             # Template variable (standard)
            "{{ $.Tools",           # Template variable (with global context)
            "{{$.Tools",            # Template variable (with global context)
            "{{- if .Tools",        # Conditional check (standard)
            "{{- if $.Tools",       # Conditional check (global)
            "PARSER functiongemma", # specialized parser
            "RENDERER functiongemma" # specialized renderer
        ]
        return any(ind in modelfile for ind in indicators)
    except ResponseError:
        return False


# --- Slash Command Handlers ---
async def _handle_help_command(_: List[str], __: Dict[str, Any], ___: Processor) -> None:
    """Handles the /help command."""
    console.print("Available commands:")
    for cmd in _SLASH_COMMAND_HANDLERS:
        console.print(f"  /{cmd}")
    console.print("  /exit, /quit, /q")


async def _handle_config_command(args: List[str], run_config: Dict[str, Any], processor: Processor) -> None:
    """Handles /config slash commands for the current session."""
    if not args:
        subcommand = "show"
        command_args = []
    else:
        subcommand = args[0].lower()
        command_args = args[1:]

    if subcommand == "show":
        config_copy = {
            k: v for k, v in run_config.items() 
            if k not in ["chat_service", "active_tts_task"]
        }
        console.print(
            Panel(
                json.dumps(config_copy, indent=2),
                title="Current Session Config",
                border_style="yellow",
            )
        )
    elif subcommand == "get":
        if not command_args:
            error_console.print("[red]Usage: /config get <key>[/red]")
            return
        key = command_args[0]
        value = run_config.get(key)
        if isinstance(value, (dict, list)):
            console.print(f"{key}:")
            console.print(value)
        else:
            console.print(f"{key}: {value if value is not None else '[not set]'}")

    elif subcommand == "set":
        if len(command_args) < MIN_SET_CONFIG_ARGS:
            error_console.print("[red]Usage: /config set <key> <value>[/red]")
            return
        key = command_args[0]
        value = " ".join(command_args[1:])

        # Also persist the change to the configuration file using config_manager
        config_manager.set_config_logic(key, value) # This persists to disk

        # Refresh run_config in-place from disk
        refresh_run_config(run_config, processor)

        # Reload the processor to apply changes immediately
        console.print("[dim]Reloading agent...[/dim]")
        processor.reload()

        console.print(f"Set '{key}' to '{value}' (persisted).")
        console.print("[dim]Note: New settings will be used on the next interaction.[/dim]")
    else:
        error_console.print(f"[red]Unknown config command: {subcommand}. Available: show, get, set[/red]")

async def _handle_history_command(_: List[str], run_config: Dict[str, Any], processor: Processor) -> None:
    """Handles the /history command."""
    history = await processor.get_history()

    if not history:
        console.print("[dim]No history available.[/dim]")
        return

    for message in history:
        role = message.get("role", "user")
        content = message.get("content", "")
        console.print(Panel(content, title=role.capitalize(),
                            border_style="cyan" if role == "user" else "magenta"))

async def _handle_clear_command(_: List[str], run_config: Dict[str, Any], processor: Processor) -> None:
    """Handles the /clear command."""
    await processor.clear_history()
    console.print("[green]Chat history cleared.[/green]")

async def _handle_save_command(args: List[str], run_config: Dict[str, Any], processor: Processor) -> None:
    """Handles the /save command."""
    history = await processor.get_history()

    if not history:
        console.print("[dim]No history available to save.[/dim]")
        return
    if not args:
        error_console.print("[red]Usage: /save <filename.md>[/red]")
        return
    filename = args[0]
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for message in history:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                f.write(f"**{role.capitalize()}**\n\n{content}\n\n---\n\n")
        console.print(f"[green]Conversation saved to {filename}[/green]")
    except IOError as e:
        error_console.print(f"[red]Error saving file: {e}[/red]")

async def _handle_model_command(args: List[str], run_config: Dict[str, Any], processor: Processor) -> None:
    """Handles the /model command."""
    if not args:
        console.print(f"Current model: {run_config.get('model')}")
        return
    model_name = args[0]

    # Persist changes
    config_manager.set_config_logic("model", model_name)

    # Refresh run_config in-place from disk
    refresh_run_config(run_config, processor)

    # Reload processor
    console.print("[dim]Reloading agent...[/dim]")
    processor.reload()

    console.print(f"Set model to '{model_name}' (persisted).")
    console.print("[dim]Note: New settings will be used on the next interaction.[/dim]")


_SLASH_COMMAND_HANDLERS: Dict[str, Callable[[List[str], Dict[str, Any], Processor], Any]] = {
    "help": _handle_help_command,
    "config": _handle_config_command,
    "history": _handle_history_command,
    "clear": _handle_clear_command,
    "save": _handle_save_command,
    "model": _handle_model_command,
}

async def _handle_slash_command(user_input: str, run_config: Dict[str, Any], processor: Processor) -> bool:
    """Handles slash commands and returns True if the app should exit."""
    command, *args = user_input.strip()[1:].split()
    if not command:
        return False

    if command.lower() in ["exit", "quit", "q"]:
        return True

    handler = _SLASH_COMMAND_HANDLERS.get(command.lower())
    if handler:
        await handler(args, run_config, processor)
    else:
        error_console.print(f"[red]Unknown command: /{command}[/red]")

    return False


# --- Unified Interactive Loop ---

def create_key_bindings() -> KeyBindings:
    """Creates custom key bindings for the prompt."""
    kb = KeyBindings()

    @kb.add(Keys.Tab)
    def _(event) -> None:  # noqa: ANN001
        b = event.app.current_buffer
        if b.suggestion:
            b.insert_text(b.suggestion.text)
        elif b.complete_state:
            b.complete_next()
        else:
            b.start_completion(select_first=True)

    return kb


async def run_interactive_chat(
    processor: Processor,
    run_config: Dict[str, Any],
    options: CliOptions,
) -> None:
    """Runs the main interactive chat loop for any processor."""
    if not options.quiet:
        console.print("**Welcome to Rich AI CLI Assistant!** Type your prompt and press Enter.")
        console.print("[dim]Type /help for a list of commands, Ctrl+C or /q to exit.[/dim]")

    history_file = os.path.join(os.path.expanduser("~/.config/rai"), "history.txt")
    prompt_session = PromptSession(
        history=FileHistory(history_file),
        completer=_build_completer(options.config_path),
        complete_while_typing=True,
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=create_key_bindings(),
    )

    try:
        while True:
            try:
                user_input = await prompt_session.prompt_async("> ")
                if not user_input.strip():
                    continue
                if user_input.startswith("/"):
                    if await _handle_slash_command(user_input, run_config, processor):
                        break
                    continue

                # Unified History: Get ChatService and Session
                chat_service = run_config.get("chat_service")
                session_id = run_config.get("session_id", "default")
                history = []

                if chat_service:
                    # Fetch history
                    history_result = await chat_service.get_session_history(session_id)
                    history = history_result.unwrap() if isinstance(history_result, Success) else []
                    
                    # Save user input
                    await chat_service.add_message_to_history(session_id, "user", user_input)

                with console.status("[bold green]Assistant is thinking..."):
                    response_result = await processor.arun(user_input, history=history)

                match response_result:
                    case Success(response):
                        if response:
                            content = response.get("content", "")
                            # TODO: Add back streaming support
                            # TODO: Add back TTS support
                            console.print(Markdown(content))
                            
                            # Save assistant response
                            if chat_service:
                                await chat_service.add_message_to_history(
                                    session_id, 
                                    "assistant", 
                                    content,
                                    tool_calls=response.get("tool_calls")
                                )

                            if response.get("tool_calls"):
                                console.print(
                                    Panel(
                                        json.dumps(response["tool_calls"], indent=2),
                                        title="Tool Calls",
                                        border_style="yellow",
                                    )
                                )
                    case Failure(error):
                        # The adapter should have already printed a detailed error.
                        error_console.print(f"[red]Exiting due to a critical error: {error}[/red]")
                        break  # Exit the loop gracefully

                console.print(Rule(style="dim"))
            except (KeyboardInterrupt, EOFError):
                break
    finally:
        await _cancel_active_tts_task(run_config)
        await processor.close()

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


def _build_run_config(options: CliOptions) -> Tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Any]]:
    """Builds the run configuration from CLI options and the config file."""
    app_config = config_manager.load_config(path=options.config_path)
    session_to_use, session_config = config_manager.initialize_session(
        app_config, options.session_override, options.config_path
    )

    run_config = {
        "session_id": session_to_use,
        "options": options,
        "model": options.model or session_config.get("model"),
        "backend": options.backend or session_config.get("backend"),
        "system": options.system or session_config.get("system"),
        "tools": session_config.get("tools"),
        "active_tts_task": None,  # This is a client-side concern
    }
    return run_config, session_config, session_to_use, app_config


def refresh_run_config(run_config: Dict[str, Any], processor: Processor) -> None:
    """Refreshes the run_config dictionary in-place from disk/options."""
    session_id = run_config.get("session_id", "default")
    options = run_config.get("options")
    config_path = options.config_path if options else None

    app_config = config_manager.load_config(path=config_path)
    # Reinitialize session to get the latest config from agents.yaml / state
    _, session_config = config_manager.initialize_session(
        app_config, session_id, config_path
    )

    # Update top level config keys in place
    run_config["model"] = (options.model if options and options.model else None) or session_config.get("model")
    run_config["backend"] = (options.backend if options and options.backend else None) or session_config.get("backend")
    run_config["system"] = (options.system if options and options.system else None) or session_config.get("system")
    run_config["tools"] = session_config.get("tools")


def _setup_standalone_processor(
    run_config: Dict[str, Any],
    session_to_use: str,
    quiet: bool
) -> Tuple[LocalProcessor, ChatService]:
    """Sets up the processor for standalone execution."""
    # Perform backend-specific checks
    enable_tools = True
    if run_config.get("backend") == "ollama":
        has_tools = _initialize_ollama_check(run_config["model"], quiet)
        if not has_tools:
            # Explicitly disable tools for this session to avoid API errors (400)
            enable_tools = False
            if not quiet:
                 console.print(f"[yellow]Warning: Tools disabled for model '{run_config['model']}'.[/yellow]")

    chat_service = ChatService()
    processor = LocalProcessor(run_config, session_to_use, enable_tools=enable_tools)
    return processor, chat_service


async def async_run_standalone(options: CliOptions) -> None:
    """The main entry point for the CLI in standalone mode."""
    run_config, _, session_to_use, app_config = _build_run_config(options)

    if not options.quiet and not options.prompt:
        active_session_name = app_config.get("active_session", "default")
        session_display = f"[bold]{session_to_use}[/bold]"
        if session_to_use == active_session_name:
            session_display += " (active)"
        else:
            session_display += " (override)"
        error_console.print(
            f"[dim]Session: {session_display} | Model: [bold]{run_config['model']}[/bold] on "
            f"backend: [bold]{run_config['backend']}[/bold][/dim]"
        )

    processor, chat_service = _setup_standalone_processor(run_config, session_to_use, options.quiet)
    run_config["chat_service"] = chat_service

    if options.prompt:
        # Single-shot mode
        # Fetch history first? arun now accepts history
        history_result = await chat_service.get_session_history(session_to_use)
        history = history_result.unwrap() if isinstance(history_result, Success) else []
        
        # Save user message
        await chat_service.add_message_to_history(session_to_use, "user", options.prompt)

        with console.status("[bold green]Assistant is thinking..."):
            result = await processor.arun(options.prompt, history=history)

        if isinstance(result, Success):
            response_payload = result.unwrap()
            if response_payload:
                content = response_payload.get("content", "")
                console.print(Markdown(content))
                
                # Save assistant response
                await chat_service.add_message_to_history(
                    session_to_use, 
                    "assistant", 
                    content,
                    tool_calls=response_payload.get("tool_calls")
                )
    
        elif isinstance(result, Failure):
            error_console.print(f"[red]Error: {result.failure()}[/red]")
        await processor.close()
    else:
        # Interactive mode
        await run_interactive_chat(processor, run_config, options)


async def async_main_client(options: CliOptions) -> None: # noqa: PLR0912
    # pylint: disable=too-many-branches
    """The main entry point for the CLI in client mode."""
    run_config, _, session_to_use, _ = _build_run_config(options)

    # Determine server URI
    uri = options.connect_uri
    if uri == "_auto_":
        # TODO: Implement full auto-discovery (UDS, etc.)
        uri = "ws://127.0.0.1:8000/ws/v1/chat"

    if options.prompt:
        # Single-shot mode using REST
        payload = {"chain_input": options.prompt, "chain_configs": [run_config]}
        try:
            transport = None
            base_url = uri.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]

            if uri.startswith("unix://"):
                uds_path = uri[len("unix://") :]
                transport = httpx.AsyncHTTPTransport(uds=uds_path)
                base_url = "http://localhost"  # Dummy base URL for UDS
                if not options.quiet:
                    console.print(f"[dim]Connecting to server via UDS at {uds_path}...[/dim]")
            elif not options.quiet:
                console.print(f"[dim]Connecting to server at {base_url}...[/dim]")

            async with httpx.AsyncClient(transport=transport, base_url=base_url) as client:
                response = await client.post("/api/v1/run", json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    content = data.get("payload", {}).get("content", "")
                    if options.json_output:
                        console.print(json.dumps(data.get("payload"), indent=2))
                    else:
                        console.print(Markdown(content) if not options.no_markdown else content)
                else:
                    error_console.print(f"[red]Server Error: {data.get('detail')}[/red]")
        except httpx.RequestError as e:
            error_console.print(
                "[bold red]Connection Error:[/bold red]"
                f" Could not connect to the rai server at {uri}."
            )
            error_console.print(f"[dim]Is the server running? ('rai serve'). Error: {e}[/dim]")
        except Exception as e: # pylint: disable=broad-exception-caught # pylint: disable=broad-exception-caught # pylint: disable=broad-exception-caught
            error_console.print(f"[bold red]An unexpected client error occurred:[/bold red]\n{e}")
    else:
        # Interactive mode using ClientProcessor
        processor = ClientProcessor(
            run_config=run_config,
            session_name=session_to_use,
            server_uri=uri
        )

        # Eagerly connect to the server before starting the interactive chat.
        connect_result = await processor.connect()
        if isinstance(connect_result, Failure):
            return  # Exit gracefully

        await run_interactive_chat(processor, run_config, options)


@click.group(cls=SectionedGroup, invoke_without_command=True)
@click.version_option(version="0.1.0")
# Section: Primary
@click.option("-p", "--prompt", "prompt", default=None, help="The prompt to send to the AI.",
              cls=SectionedOption, section="Primary")
# Section: Connection
@click.option("--connect", "connect_uri", default=None,
              help="Connect to a 'rai serve' instance. If no URI is given, auto-discovers the server.",
              is_flag=False, flag_value="_auto_", cls=SectionedOption, section="Primary")
# Section: AI Configuration
@click.option("-s", "--system", default=None, help="Defines the system prompt for the AI.",
              cls=SectionedOption, section="AI Configuration")
@click.option("-m", "--model", default=None, help="ID of the model to use.",
              cls=SectionedOption, section="AI Configuration")
@click.option("-b", "--backend", default=None,
              type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq", "local"]),
              help="The backend to use.",
              cls=SectionedOption, section="AI Configuration")
# Section: Output Formatting
@click.option("--no-markdown", is_flag=True, help="Disable Markdown rendering for LLM responses.",
              cls=SectionedOption, section="Output Formatting")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.",
              cls=SectionedOption, section="Output Formatting")
@click.option("--quiet", is_flag=True, help="Suppress informational messages.",
              cls=SectionedOption, section="Output Formatting")
@click.option("--stream", is_flag=True,
              help="Enable streaming of LLM responses (disables Markdown).",
              cls=SectionedOption, section="Output Formatting")
# Section: Speech
@click.option("--tts", "tts_voice_id", default=None, is_flag=False, flag_value="_default_",
              help="Enable Text-to-Speech output. Optionally provide a voice ID.",
              cls=SectionedOption, section="Speech")
# Section: Session & Debugging
@click.option("--session", "session_override", default=None,
              help="Run in a specific session for this command only.",
              cls=SectionedOption, section="Session & Debugging")
@click.option("--log-level", "log_level_opt", default=None,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
              help="Set the logging level.",
              cls=SectionedOption, section="Session & Debugging")
@click.option("--config", "config_path", default=None,
              help="Path to a custom configuration file.",
              type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              cls=SectionedOption, section="Session & Debugging")
@click.pass_context
def cli(ctx: click.Context, **kwargs: Any) -> None: # noqa: ANN401
    """
    AI assistant in the command line.

    This command runs in standalone mode by default.
    Use '--connect' to connect to a server.
    Use 'rai serve' to run a server.
    """
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    ctx.obj = kwargs

    # Configure logging globally for all commands (standalone, client, serve)
    # Precedence: CLI --log-level > RAI_LOG_LEVEL > Default (INFO)
    
    # 1. Determine level string
    level_str = kwargs.get("log_level_opt")
    if not level_str:
        level_str = os.getenv("RAI_LOG_LEVEL", "INFO")
    
    # 2. Convert to logging constant
    log_level = getattr(logging, level_str.upper(), logging.INFO)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

    # Agno loggers and handlers are no longer needed as the library is deprecated.
    if ctx.invoked_subcommand is None:
        if not kwargs.get("prompt") and not sys.stdin.isatty():
            kwargs["prompt"] = sys.stdin.read().strip()

        options = CliOptions(**kwargs)

        if options.connect_uri:
            # Client Mode
            asyncio.run(async_main_client(options))
        else:
            # Standalone Mode
            asyncio.run(async_run_standalone(options))





@cli.group()
def sessions() -> None:
    """Manage chat sessions."""


@sessions.command(name="switch")
@click.argument("session_name")
def switch_session(session_name: str) -> None:
    """Creates a new session or switches to an existing one."""
    config_manager.switch_session_logic(session_name)


@sessions.command(name="list")
def list_sessions() -> None:
    """Lists all available sessions."""
    config_manager.list_sessions_logic()


@sessions.command(name="show")
@click.argument("session_name", required=False)
def show_session(session_name: Optional[str]) -> None:
    """Shows configuration for a specific or active session."""
    config_manager.show_session_logic(session_name)


@sessions.command(name="delete")
@click.argument("session_name")
def delete_session(session_name: str) -> None:
    """Deletes a specified session."""
    config_manager.delete_session_logic(session_name)


@sessions.command(name="rename")
@click.argument("old_name")
@click.argument("new_name")
def rename_session(old_name: str, new_name: str) -> None:
    """Renames a session."""
    config_manager.rename_session_logic(old_name, new_name)


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context) -> None:
    """View or manage the configuration of the active session."""
    if ctx.invoked_subcommand is None:
        config_manager.show_config_logic()


@config.command(name="show")
def show_config() -> None:
    """Shows configuration for the active session."""
    config_manager.show_config_logic()


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str) -> None:
    """Sets a configuration value for the active session."""
    config_manager.set_config_logic(key, value)


@config.command(name="get")
@click.argument("key")
def get_config(key: str) -> None:
    """Gets a configuration value from the active session."""
    config_manager.get_config_logic(key)


@config.command(name="edit")
def edit_config() -> None:
    """Opens the configuration file in the default editor."""
    config_path = config_manager.get_config_path()
    if config_path:
        console.print(f"[dim]Opening configuration file: {config_path}[/dim]")
        click.launch(str(config_path))
    else:
        error_console.print("[red]Could not determine configuration file path.[/red]")


@cli.command(name="serve")
@click.option("--host", default="127.0.0.1", help="Host to bind the server to.")
@click.option("--port", default=8000, type=int, help="Port to bind the server to.")
@click.option("--uds", default=None, help="Path to a Unix Domain Socket to bind to.")
@click.option("--workers", default=1, type=int, help="Number of worker processes.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(
    host: str, port: int, uds: Optional[str], workers: int, reload: bool
) -> None:
    """
    Runs the FastAPI server for REST, WebSocket, and IPC communication.
    """
    # pylint: disable=import-outside-toplevel
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        error_console.print("[bold red]Error: 'uvicorn' is not installed.[/bold red]")
        error_console.print("Please install it with: uv pip install uvicorn[standard]")
        sys.exit(1)

    if uds:
        console.print(f"🚀 Starting FastAPI server on Unix socket: [bold green]{uds}[/bold green]")
    else:
        console.print(f"🚀 Starting FastAPI server on [bold green]http://{host}:{port}[/bold green]")
    console.print("See API docs at [bold blue]/docs[/bold blue] or [bold blue]/redoc[/bold blue]")

    uvicorn.run(
        "rai.server:app",
        host=host if uds is None else None,
        port=port if uds is None else None,
        uds=uds,
        workers=workers,
        reload=reload,
        log_config=None,
    )


@cli.group()
def agents() -> None:
    """Manage AI agents."""


@agents.command(name="list")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--table", "table_output", is_flag=True, help="Output as a formatted table.")
@click.pass_context
def list_agents(ctx: click.Context, json_output: bool, table_output: bool) -> None:
    # pylint: disable=too-many-locals
    """Lists all available agents."""
    options = ctx.obj
    uri = options.get("connect_uri")

    agents_data = {}

    # Try to fetch from server if URI is provided explicitly
    if uri and uri != "_auto_":
        base_url = uri.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0]
        try:
            response = httpx.get(f"{base_url}/api/v1/agents/")
            response.raise_for_status()
            agents_data = response.json()
        except httpx.RequestError as e:
            error_console.print(f"[red]Error connecting to server at {base_url}: {e}[/red]")
            return
        except Exception as e: # pylint: disable=broad-exception-caught
            error_console.print(f"[red]Error fetching agents: {e}[/red]")
            return
    else:
        # Standalone mode: Load directly from config
        try:
            app_config = config_manager.load_config()
            agents_data = app_config.get("sessions", {})
        except Exception as e: # pylint: disable=broad-exception-caught
            error_console.print(f"[red]Error loading configuration: {e}[/red]")
            return

    if not agents_data:
        console.print("[dim]No agents found.[/dim]")
        return

    if json_output:
        console.print(json.dumps(agents_data, indent=2))
    elif table_output:
        table = Table(title="Available Agents", border_style="blue")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Backend", style="green")
        table.add_column("Description", style="white")

        for agent_id, agt_config in agents_data.items():
            model = agt_config.get("model", "N/A")
            backend = agt_config.get("backend", "N/A")
            system = agt_config.get("system", "")
            # Truncate system prompt for display
            max_len = 50
            description = (system[:max_len] + "...") if len(system) > max_len else system
            table.add_row(agent_id, model, backend, description)

        console.print(table)
    else:
        # Default: Simple list
        console.print("[bold]Available Agents:[/bold]")
        for agent_id in agents_data:
            console.print(f"- {agent_id}")


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=no-value-for-parameter
