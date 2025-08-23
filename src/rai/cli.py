 #!/usr/bin/env python
""" rai - Rich AI CLI assistant
"""
import io
import os
import sys
import json
from contextlib import redirect_stdout

import click
import ollama
from dotenv import load_dotenv
from ollama import ResponseError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agno.agent import Agent
from agno.media import Image
from agno.models.anthropic import Claude
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.models.openai.chat import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from agno.tools.webbrowser import WebBrowserTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.shell import ShellTools
from agno.storage.sqlite import SqliteStorage


# from .tools import send_notification, take_screenshot, weather
from .tools import send_notification, take_screenshot

# global console
console = Console(force_terminal=True)


base_tools = [
    send_notification,
    take_screenshot,
    CalculatorTools(
        enable_all=True,
        # exclude_tools=["exponentiate", "factorial", "is_prime", "square_root"],
    ),
    ArxivTools(),
    WikipediaTools(),
    DuckDuckGoTools(),
    WebBrowserTools(),
    FileTools(),
    PythonTools(),
    ShellTools(),
    # weather, <-- disabled until fixing issue with dbus for org.gnome.Weather
    #              is this only ubunut issue or sth. else?
]


def setup_agent(
    system_prompt: str,
    model_id: str,
    backend: str,
    enable_tools: bool = True,
):
    """Initializes the Agno agent, setting the model, prompt, and tools."""
    load_dotenv()

    agent_tools = []
    if enable_tools:
        if os.getenv("TAVILY_API_KEY"):
            agent_tools = base_tools + [TavilyTools()]
        else:
            agent_tools = base_tools
            console.print(
                "[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. "
                "Tavily will be disabled![/bold yellow]"
            )

    model_instance = None
    if backend == "ollama":
        model_instance = Ollama(id=model_id)
    elif backend == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            console.print(
                "[bold red]ERROR: GOOGLE_API_KEY environment variable not set.[/bold red]"
            )
            sys.exit(1)
        model_instance = Gemini(id=model_id)
    elif backend == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print(
                "[bold red]ERROR: ANTHROPIC_API_KEY environment variable not set.[/bold red]"
            )
            sys.exit(1)
        model_instance = Claude(id=model_id)
    elif backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            console.print(
                "[bold red]ERROR: OPENAI_API_KEY environment variable not set.[/bold red]"
            )
            sys.exit(1)
        model_instance = OpenAIChat(id=model_id)
    elif backend == "groq":
        if not os.getenv("GROQ_API_KEY"):
            console.print(
                "[bold red]ERROR: GROQ_API_KEY environment variable not set.[/bold red]"
            )
            sys.exit(1)
        model_instance = Groq(id=model_id)
    else:
        console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

    try:
        agent = Agent(
            model=model_instance,
            tools=agent_tools,
            show_tool_calls=True,
            markdown=True,
            add_history_to_messages=True,
            storage=SqliteStorage(
                table_name="agent_sessions",
                db_file="tmp/data.db",
                auto_upgrade_schema=True
            ),
            session_id="my_chat_session",
            instructions=system_prompt,
        )
        return agent
    except Exception as e:  # pylint: disable=broad-exception-caught
        console.print(f"[bold red]ERROR: Failed to initialize agent: {e}[/bold red]")
        console.print(
            "[yellow]Is the Ollama server running and does it have the specified model?[/yellow]"
        )
        sys.exit(1)


def check_model_tool_support(model_id: str) -> bool:
    """
    Checks if the specified Ollama model supports tool use by inspecting its Modelfile.
    """
    try:
        details = ollama.show(model_id)
        modelfile = details.get("modelfile", "")
        # console.log(f'Modelfile for {model_id}: {modelfile}')
        # A simple heuristic: check for keywords related to tool use grammar.
        return "tool_use" in str(modelfile)
    except ResponseError:
        # This can happen if the model is not found, though `main` checks for this.
        return False
    except Exception:  # pylint: disable=broad-exception-caught
        # For any other unexpected errors, assume no tool support to be safe.
        return False


def run_single_query(agent, prompt, no_markdown: bool):
    """Executes a single query, handling potential image data from tools."""
    console.print(f"\n[bold green]Prompt: [/] {prompt}")
    console.print("\n[bold]AI Assistant (<model_id>): [/]") # TODO: print model name too
    try:
        # --- First Pass: Run with tools to get potential image data ---
        string_io = io.StringIO()
        with redirect_stdout(string_io):
            initial_response = agent.run(prompt, stream=False)

        tool_output_str = string_io.getvalue().strip()
        image_data = None

        if tool_output_str:
            panel_title = "[bold yellow]Tool Call[/bold yellow]"
            panel_content = "Screenshot captured successfully."
            try:
                tool_json = json.loads(tool_output_str)
                if tool_json.get("type") == "image_data":
                    image_data = tool_json
                else:
                    panel_content = tool_output_str
            except (json.JSONDecodeError, AttributeError):
                panel_content = tool_output_str

            console.print(Panel(panel_content, title=panel_title, border_style="yellow"))

        if image_data:
            console.print("\n[bold]AI Assistant (analyzing image):[/]")
            img = Image(
                source=image_data.get("base64"),
                mime_type=f"image/{image_data.get('format', 'png')}"
            )
            response_stream = agent.run(prompt, images=[img], stream=True)
            for response_chunk in response_stream:
                if response_chunk.content:
                    if not no_markdown:
                        console.print(Markdown(response_chunk.content), end="")
                    else:
                        console.print(response_chunk.content, end="")
        elif initial_response and initial_response.content:
            if not no_markdown:
                console.print(Markdown(initial_response.content))
            else:
                console.print(initial_response.content)

        console.print()

    except ResponseError as e:
        console.print(
            f"\n[bold red]Ollama API error (status: {e.status_code}): {e.error}[/bold red]"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")


def run_interactive_chat(agent, no_markdown: bool):
    """Starts an interactive chat loop with streaming response."""
    console.print("[yellow]Interactive chat does not support image analysis yet.[/yellow]")
    while True:
        try:
            user_input = console.input("[bold green]You: [/]")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            run_single_query(agent, user_input, no_markdown=no_markdown)
        except (KeyboardInterrupt, EOFError):
            break
    console.print("\n[yellow]Goodbye![/yellow]")


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "-s",
    "--system",
    default="You are a versatile and helpful AI assistant.",
    help="Defines the system prompt for the AI.",
)
@click.option(
    "-m",
    "--model",
    default="gemma3:1b",
    help="ID of the Ollama model to be used (e.g., gemma2:9b, llama3.2).",
)
@click.option(
    "-b",
    "--backend",
    default="ollama",
    type=click.Choice(["ollama", "gemini", "anthropic", "openai", "groq"]),
    help="The LLM backend to use.",
)
@click.option(
    "--no-markdown",
    is_flag=True,
    help="Disable Markdown rendering for LLM responses.",
)
def main(prompt, system, model, backend, no_markdown):
    """
    AI assistant in the command line with tool support and ready-made toolkits.
    """
    console.print(f"[dim]Using model: [bold]{model}[/bold] on backend: [bold]{backend}[/bold][/dim]")

    has_tools = True  # Assume tools are supported by default for non-ollama backends
    if backend == "ollama":
        try:
            available_models = [m["model"] for m in ollama.list()["models"]]
            model_found = any(m.startswith(model) for m in available_models)

            if not model_found:
                console.print(
                    f"\n[bold red]Error: Model '{model}' is not available in Ollama.[/bold red]"
                )
                console.print("\n[bold green]Available models:[/bold green]")
                for m_name in sorted(list(set(available_models))):
                    console.print(f"- {m_name}", highlight=False)
                console.print(
                    f"[yellow]You can download the missing model with the command: "
                    f"[bold]ollama pull {model}[/bold][/yellow]"
                )
                sys.exit(1)
        except ResponseError as e:
            console.print(
                f"\n[bold red]Error: Failed to connect to Ollama server to verify model "
                f"(status: {e.status_code}).[/bold red]"
            )
            sys.exit(1)

        has_tools = check_model_tool_support(model)
        if not has_tools:
            console.print(
                f"[yellow]Warning: Model '{model}' may not support tools. "
                "Proceeding in text-only mode.[/yellow]"
            )

    agent = setup_agent(
        system_prompt=system, model_id=model, enable_tools=has_tools, backend=backend
    )

    if prompt:
        if not sys.stdin.isatty():
            piped_content = sys.stdin.read()
            full_prompt = f"{prompt}\n\n---BEGIN CONTENT---\n{piped_content}\n---END CONTENT---"
            run_single_query(agent, full_prompt, no_markdown=no_markdown)
        else:
            run_single_query(agent, prompt, no_markdown=no_markdown)
    else:
        run_interactive_chat(agent, no_markdown=no_markdown)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
