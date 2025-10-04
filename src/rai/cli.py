"""
rai - Rich AI CLI assistant """

import io
import json
import os
import signal
import sys
from contextlib import nullcontext, redirect_stdout

import click
import ollama
from agno.agent import Agent
from agno.media import Image
from agno.models.ollama import Ollama
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
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from .tools import send_notification, take_screenshot

console = Console(force_terminal=True)
error_console = Console(stderr=True, force_terminal=True)

CONFIG_DIR = os.path.expanduser("~/.config/rai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
RAI_CONFIG = {}

SLASH_COMMANDS = [
    "/help",
    "/config",
    "/model",
    "/backend",
    "/system",
    "/exit",
    "/quit",
    "/q",
]


def load_config():
    """Loads the configuration from the config file."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config():
    """Saves the configuration to the config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(RAI_CONFIG, f, indent=2)


def on_resize(signum, frame):
    """Handles terminal resize events."""
    try:
        width, _ = os.get_terminal_size()
        console.width = width
        error_console.width = width
    except OSError:
        pass


base_tools = [
    send_notification,
    take_screenshot,
    CalculatorTools(
        enable_all=True,
    ),
    ArxivTools(),
    WikipediaTools(),
    DuckDuckGoTools(),
    WebBrowserTools(),
    FileTools(),
    PythonTools(),
    ShellTools(),
]


def setup_agent(enable_tools: bool = True, quiet: bool = False, json_output: bool = False):
    """Initializes the Agno agent, setting the model, prompt, and tools."""
    load_dotenv()
    backend = RAI_CONFIG.get("backend")
    model_id = RAI_CONFIG.get("model")
    system_prompt = RAI_CONFIG.get("system")

    # FIX: Handle GEMINI_API_KEY compatibility BEFORE the main key check.
    if backend == "gemini" and "GEMINI_API_KEY" in os.environ:
        if "GOOGLE_API_KEY" in os.environ and not quiet:
            error_console.print(
                "[bold yellow]INFO: Both GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "Prioritizing GEMINI_API_KEY.[/bold yellow]"
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
        # Dynamic import
        module_path, class_name = model_map[backend].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
    except ImportError:
        error_console.print(
            f"[bold red]ERROR: Backend '{backend}' requires an optional dependency.[/bold red]"
        )
        if backend in dependency_map:
            error_console.print(
                f"[yellow]Please install it using: [bold]pip install .[{dependency_map[backend]}][/bold][/yellow]"
            )
        sys.exit(1)

    agent_tools = []
    if enable_tools:
        if os.getenv("TAVILY_API_KEY"):
            agent_tools = base_tools + [TavilyTools()]
        else:
            agent_tools = base_tools
            if not quiet:
                error_console.print(
                    "[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. "
                    "Tavily will be disabled![/bold yellow]"
                )

    model_instance = model_class(id=model_id)

    try:
        agent = Agent(
            model=model_instance,
            tools=agent_tools,
            show_tool_calls=not json_output,
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
        return agent
    except Exception as e:
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
    except Exception:
        return False


def display_response(response, console_instance):
    """Displays the AI's response to the console."""
    console_instance.print(Markdown(response.content))


def handle_tool_calls(response, _agent, console_instance):
    """Handles the tool calls from the AI's response."""
    console_instance.print(
        Panel(
            json.dumps(response.tool_calls, indent=2),
            title="Tool Calls",
            border_style="yellow",
        )
    )


def _handle_tool_output(tool_output_str, quiet, json_output, response_data):
    """Handles the output of tools, extracting image data if present."""
    image_data = None
    if not tool_output_str:
        return None

    try:
        tool_call_data = json.loads(tool_output_str)
        if tool_call_data.get("type") == "image_data":
            image_data = tool_call_data
    except (json.JSONDecodeError, AttributeError):
        tool_call_data = None

    if json_output:
        response_data["tool_call_output"] = (
            tool_call_data if tool_call_data else tool_output_str
        )
    elif not quiet:
        panel_title = "[bold yellow]Tool Call[/bold yellow]"
        panel_content = (
            "Screenshot captured successfully."
            if image_data
            else tool_output_str
        )
        error_console.print(
            Panel(panel_content, title=panel_title, border_style="yellow")
        )
    return image_data


def _handle_image_response(
    agent, prompt, image_data, no_markdown, json_output, quiet, status
):
    """Handles the response when an image is present."""
    if not quiet and not json_output:
        error_console.print("\n[bold]AI Assistant (analyzing image):[/]")
        if status:
            status.update("[bold green]AI is analyzing...[/bold green]")
    img = Image(
        source=image_data.get("base64"),
        mime_type=f"image/{image_data.get('format', 'png')}",
    )
    response_stream = agent.run(prompt, images=[img], stream=True)
    full_response_content = "".join(
        chunk.content for chunk in response_stream if chunk.content
    )
    if not json_output:
        console.print(
            Markdown(full_response_content)
            if not no_markdown
            else full_response_content,
            end="",
        )
    return full_response_content


def run_single_query(
    agent, prompt, no_markdown, json_output, quiet, non_interactive
):
    """Executes a single query, handling image data and tool outputs."""
    response_data = {"prompt": prompt}
    status_context = (
        error_console.status(" thinking... ")
        if not quiet and not json_output
        else nullcontext()
    )

    initial_response = None
    try:
        with status_context as status:
            string_io = io.StringIO()
            with redirect_stdout(string_io):
                initial_response = agent.run(prompt, stream=False)

            tool_output_str = string_io.getvalue().strip()
            image_data = _handle_tool_output(
                tool_output_str, quiet, json_output, response_data
            )

            if image_data:
                response_data["ai_response"] = _handle_image_response(
                    agent, prompt, image_data, no_markdown, json_output, quiet, status
                )

    except ResponseError as e:
        if json_output:
            response_data["error"] = error_message
            print(json.dumps(response_data))
        else:
            error_console.print(f"\n[bold red]{error_message}[/bold red]")
        if non_interactive:
            sys.exit(1)
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        if json_output:
            response_data["error"] = error_message
            print(json.dumps(response_data))
        else:
            error_console.print(f"\n[bold red]{error_message}[/bold red]")
        if non_interactive:
            sys.exit(1)

    if not non_interactive:
        if (
            initial_response
            and hasattr(initial_response, "tool_calls")
            and initial_response.tool_calls
        ):
            handle_tool_calls(initial_response, agent, console)
        elif initial_response and initial_response.content:
            display_response(initial_response, console)
    else:
        if initial_response and initial_response.content:
            console.print(initial_response.content)
        elif (
            hasattr(initial_response, "tool_calls")
            and initial_response.tool_calls
        ):
            console.print(json.dumps(initial_response.tool_calls, indent=2))
            response_data["ai_response"] = initial_response.content

        if json_output:
            print(json.dumps(response_data))
        else:
            error_console.print()


# --- Interactive Mode ---
def handle_interactive_command(command):
    """Handles interactive commands."""
    parts = command.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else None

    if cmd == "/help":
        error_console.print("[bold green]Available commands:[/bold green]")
        error_console.print("  /help - Display this help message.")
        error_console.print("  /config - Display current configuration.")
        error_console.print("  /model [<model_id>] - Get or set the model.")
        error_console.print("  /backend [<backend_id>] - Get or set the backend.")
        error_console.print("  /system [<prompt>] - Get or set the system prompt.")
        error_console.print("  /exit, /quit, /q - Exit the chat.")
    elif cmd == "/config":
        error_console.print("[bold green]Current Configuration:[/bold green]")
        error_console.print(f"  Model: {RAI_CONFIG.get('model')}")
        error_console.print(f"  Backend: {RAI_CONFIG.get('backend')}")
        error_console.print(f"  System: {RAI_CONFIG.get('system')}")
    elif cmd == "/model":
        if arg:
            RAI_CONFIG["model"] = arg
            save_config()
            error_console.print(f"Model set to: {arg}. Restart for changes to take effect.")
        else:
            error_console.print(f"Current model: {RAI_CONFIG.get('model')}")
    elif cmd == "/backend":
        if arg:
            RAI_CONFIG["backend"] = arg
            save_config()
            error_console.print(f"Backend set to: {arg}. Restart for changes to take effect.")
        else:
            error_console.print(f"Current backend: {RAI_CONFIG.get('backend')}")
    elif cmd == "/system":
        if arg:
            RAI_CONFIG["system"] = arg
            save_config()
            error_console.print("System prompt set. Restart for changes to take effect.")
        else:
            error_console.print(f"Current system prompt: {RAI_CONFIG.get('system')}")
    elif cmd in ["/exit", "/quit", "/q"]:
        return False
    else:
        error_console.print(f"[bold red]Unknown command: {command}[/bold red]")
    return True


class CommandLexer(Lexer):
    def lex_document(self, document):
        words = document.text.split()
        text = document.text

        def get_line(lineno):
            if lineno == 0 and words and words[0] in SLASH_COMMANDS:
                # Style the first word as a command
                return [
                    ("class:command", words[0]),
                    ("", text[len(words[0]) :]),  # The rest of the text with default style
                ]
            # Default style for the whole line
            return [("", text)]

        return get_line

def run_interactive_chat(agent, no_markdown, json_output, quiet, non_interactive):
    """Starts an interactive chat loop with advanced prompt_toolkit features."""
    if not quiet and not json_output:
        error_console.print("[yellow]Interactive chat does not support images.[/yellow]")

    history_file = os.path.join(CONFIG_DIR, "history.txt")
    command_completer = WordCompleter(SLASH_COMMANDS, ignore_case=True)
    cli_style = Style.from_dict(
        {
            "toolbar": "bg:#333333 #ffffff",
            "command": "#00aa00 bold",
        }
    )

    def get_bottom_toolbar():
        return [
            (
                "class:toolbar",
                f" Model: {RAI_CONFIG.get('model')} | Backend: {RAI_CONFIG.get('backend')} ",
            )
        ]

    session = PromptSession(
        history=FileHistory(history_file),
        bottom_toolbar=get_bottom_toolbar,
        style=cli_style,
        completer=command_completer,
        lexer=CommandLexer(),
        refresh_interval=0.5,
    )

    while True:
        try:
            user_input = session.prompt("> ")
            if user_input.startswith("/"):
                if not handle_interactive_command(user_input):
                    break
            elif user_input.strip():
                run_single_query(
                    agent,
                    user_input,
                    no_markdown=no_markdown,
                    json_output=json_output,
                    quiet=quiet,
                    non_interactive=non_interactive,
                )
                console.print(Rule(style="dim"))
        except (KeyboardInterrupt, EOFError):
            break
    if not quiet and not json_output:
        error_console.print("\n[yellow]Goodbye![/yellow]")


@click.command()
@click.version_option(version="0.1.0")
@click.argument("prompt", required=False)
@click.option(
    "-s",
    "--system",
    default=None,
    help="Defines the system prompt for the AI.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    help="ID of the Ollama model to be used (e.g., gemma2:9b, llama3.2).",
)
@click.option(
    "-b",
    "--backend",
    default=None,
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"]),
    help="The LLM backend to use.",
)
@click.option(
    "--no-markdown", is_flag=True, help="Disable Markdown rendering for LLM responses."
)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format.")
@click.option("--quiet", is_flag=True, help="Suppress informational messages.")
@click.option("--list-config", is_flag=True, help="List all configuration parameters.")
@click.option("--get-config", "get_config_key", help="Get a configuration parameter.")
@click.option("--set-config", "set_config_pair", help="Set a configuration parameter (KEY=VALUE).")
def main(
    prompt,
    system,
    model,
    backend,
    no_markdown,
    json_output,
    quiet,
    list_config,
    get_config_key,
    set_config_pair,
):
    """AI assistant in the command line with tool support."""
    config = load_config()

    if list_config:
        print(json.dumps(config, indent=2))
        return
    if get_config_key:
        print(config.get(get_config_key, "not set"))
        return
    if set_config_pair:
        if "=" in set_config_pair:
            key, val = set_config_pair.split("=", 1)
            # Load current config to update it
            RAI_CONFIG.update(load_config())
            RAI_CONFIG[key] = val
            save_config()
            print(f"{key} set to: {val}")
        else:
            error_console.print("Invalid format for --set-config. Use KEY=VALUE.")
        return

    # Populate the global config
    RAI_CONFIG["model"] = model or config.get("model") or "gemma3:1b"
    RAI_CONFIG["backend"] = backend or config.get("backend") or "ollama"
    RAI_CONFIG["system"] = system or config.get("system") or "You are a versatile and helpful AI assistant."


    if not quiet:
        error_console.print(
            f"[dim]Using model: [bold]{RAI_CONFIG['model']}[/bold] on backend: [bold]{RAI_CONFIG['backend']}[/bold][/dim]"
        )

    has_tools = True
    if RAI_CONFIG["backend"] == "ollama":
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]
            if not any(m.startswith(RAI_CONFIG["model"]) for m in available_models):
                error_console.print(
                    f"\n[bold red]Error: Model '{RAI_CONFIG['model']}' not in Ollama.[/bold red]"
                )
                if not quiet:
                    error_console.print("\n[bold green]Available models:[/bold green]")
                    for m_name in sorted(list(set(available_models))):
                        error_console.print(f"- {m_name}", highlight=False)
                    error_console.print(
                        f"[yellow]Pull with: [bold]ollama pull {RAI_CONFIG['model']}[/bold][/yellow]"
                    )
                sys.exit(1)
        except ResponseError as e:
            error_console.print(
                f"\n[bold red]Error connecting to Ollama (status: {e.status_code}).[/bold red]"
            )
            sys.exit(1)

        has_tools = check_model_tool_support(RAI_CONFIG["model"])
        if not has_tools and not quiet:
            error_console.print(
                f"[yellow]Warning: Model '{RAI_CONFIG['model']}' may not support tools. "
                "Text-only mode."
            )

    agent = setup_agent(
        enable_tools=has_tools,
        quiet=quiet,
        json_output=json_output,
    )

    is_pipe = not sys.stdin.isatty()

    if not prompt and is_pipe:
        prompt = sys.stdin.read()
        quiet = True

    if prompt:
        non_interactive = True
        run_single_query(
            agent,
            prompt,
            no_markdown=no_markdown,
            json_output=json_output,
            quiet=quiet,
            non_interactive=non_interactive,
        )
    else:
        non_interactive = False
        signal.signal(signal.SIGWINCH, on_resize)
        on_resize(None, None)  # Set initial size
        run_interactive_chat(
            agent,
            no_markdown=no_markdown,
            json_output=json_output,
            quiet=quiet,
            non_interactive=non_interactive,
        )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
