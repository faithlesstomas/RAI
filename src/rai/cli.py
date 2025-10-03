"""
rai - Rich AI CLI assistant """

import io
import json
import os
import sys
from contextlib import nullcontext, redirect_stdout

import click
import ollama
from agno.agent import Agent
from agno.media import Image
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.models.openai.chat import OpenAIChat
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
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .tools import send_notification, take_screenshot

console = Console(force_terminal=True)
error_console = Console(stderr=True, force_terminal=True)

CONFIG_DIR = os.path.expanduser("~/.config/rai")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_config():
    """Loads the configuration from the config file."""
    if not os.path.exists(CONFIG_FILE):
        return {{}}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    """Saves the configuration to the config file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


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


def setup_agent(
    system_prompt: str,
    model_id: str,
    backend: str,
    enable_tools: bool = True,
    quiet: bool = False,
    json_output: bool = False,
):
    """Initializes the Agno agent, setting the model, prompt, and tools."""
    load_dotenv()

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

    model_map = {
        "ollama": Ollama,
        "gemini": Gemini,
        "anthropic": Claude,
        "openai": OpenAIChat,
        "groq": Groq,
    }
    api_keys = {
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    if backend in api_keys and not os.getenv(api_keys[backend]):
        error_console.print(
            f"[bold red]ERROR: {api_keys[backend]} environment variable not set.[/bold red]"
        )
        sys.exit(1)

    if backend in model_map:
        model_instance = model_map[backend](id=model_id)
    else:
        error_console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

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
        error_console.print(
            "[yellow]Is the Ollama server running and has the model?[/yellow]"
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
        error_message = f"Ollama API error (status: {e.status_code}): {e.error}"
        if json_output:
            response_data["error"] = error_message
            print(json.dumps(response_data))
        else:
            error_console.print(f"\n[bold red]{error_message}[/bold red]")
        sys.exit(1)
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        if json_output:
            response_data["error"] = error_message
            print(json.dumps(response_data))
        else:
            error_console.print(f"\n[bold red]{error_message}[/bold red]")
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


def run_interactive_chat(agent, no_markdown, json_output, quiet, non_interactive):
    """Starts an interactive chat loop with streaming response."""
    if not quiet and not json_output:
        error_console.print("[yellow]Interactive chat does not support images.[/yellow]")
    while True:
        try:
            user_input = console.input("> ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if user_input.startswith("/"):
                command_parts = user_input[1:].split(maxsplit=1)
                command = command_parts[0].lower()

                if command == "help":
                    error_console.print("[bold green]Available commands:[/bold green]")
                    error_console.print("  /help - Display this help message.")
                    error_console.print("  /config - Display current configuration.")
                    error_console.print("  /exit, /quit, /q - Exit the chat.")
                elif command == "config":
                    config = load_config()
                    error_console.print("[bold green]Current Configuration:[/bold green]")
                    error_console.print(f"  Model: {config.get('model')}")
                    error_console.print(f"  Backend: {config.get('backend')}")
                    error_console.print(f"  System: {config.get('system')}")
                elif command in ["exit", "quit", "q"]:
                    break
                else:
                    error_console.print(f"[bold red]Unknown command: {user_input}[/bold red]")
            else:
                run_single_query(
                    agent,
                    user_input,
                    no_markdown=no_markdown,
                    json_output=json_output,
                    quiet=quiet,
                    non_interactive=non_interactive,
                )
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
def main(prompt, system, model, backend, no_markdown, json_output, quiet, list_config, get_config_key, set_config_pair):
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
            config[key] = val
            save_config(config)
            print(f"{key} set to: {val}")
        else:
            error_console.print("Invalid format for --set-config. Use KEY=VALUE.")
        return

    model = model or config.get("model") or "gemma3:1b"
    backend = backend or config.get("backend") or "ollama"
    system = system or config.get("system") or "You are a versatile and helpful AI assistant."

    if not quiet:
        error_console.print(
            f"[dim]Using model: [bold]{model}[/bold] on backend: [bold]{backend}[/bold][/dim]"
        )

    has_tools = True
    if backend == "ollama":
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]
            if not any(m.startswith(model) for m in available_models):
                error_console.print(
                    f"\n[bold red]Error: Model '{model}' not in Ollama.[/bold red]"
                )
                if not quiet:
                    error_console.print("\n[bold green]Available models:[/bold green]")
                    for m_name in sorted(list(set(available_models))):
                        error_console.print(f"- {m_name}", highlight=False)
                    error_console.print(
                        f"[yellow]Pull with: [bold]ollama pull {model}[/bold][/yellow]"
                    )
                sys.exit(1)
        except ResponseError as e:
            error_console.print(
                f"\n[bold red]Error connecting to Ollama (status: {e.status_code}).[/bold red]"
            )
            sys.exit(1)

        has_tools = check_model_tool_support(model)
        if not has_tools and not quiet:
            error_console.print(
                f"[yellow]Warning: Model '{model}' may not support tools. "
                "Text-only mode.[/yellow]"
            )

    agent = setup_agent(
        system_prompt=system,
        model_id=model,
        enable_tools=has_tools,
        backend=backend,
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
        run_interactive_chat(
            agent,
            no_markdown=no_markdown,
            json_output=json_output,
            quiet=quiet,
            non_interactive=non_interactive,
        )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter