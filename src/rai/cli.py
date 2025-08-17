#!/usr/bin/env python
""" rai - Rich AI CLI assistant
"""
import io
import os
import sys
from contextlib import redirect_stdout

import click
import ollama
from dotenv import load_dotenv
from ollama import ResponseError
from rich.console import Console
from rich.panel import Panel

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.arxiv import ArxivTools
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from agno.tools.webbrowser import WebBrowserTools
from agno.tools.wikipedia import WikipediaTools

from .tools import send_notification, take_screenshot, weather

# global console
console = Console(force_terminal=True)




base_tools=[
    send_notification, # New tool for sending notifications
    take_screenshot, # New tool for taking screenshots
    # count_files_in_path,
    CalculatorTools(
        enable_all=True,
        exclude_tools=["exponentiate", "factorial", "is_prime", "square_root"],
    ),
    ArxivTools(),
    WikipediaTools(),
    DuckDuckGoTools(),
    WebBrowserTools(),
    weather, # Custom weather tool
]


def setup_agent(system_prompt, model_id):
    """Initializes the Agno agent, setting the model, prompt, and tools."""
    load_dotenv() # Ensure environment variables are loaded

    if os.getenv("TAVILY_API_KEY"):
        tools = base_tools + [TavilyTools()]
    else:
        tools = base_tools
        console.print(
            "[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. "
            "Tavily will be disabled![/bold yellow]"
        )

    try:
        # First attempt with tools
        agent = Agent(
            model=Ollama(id=model_id),
            tools=tools,
            show_tool_calls=True,
            markdown=True,
            add_history_to_messages=True,
            session_id="my_chat_session",
            instructions=system_prompt,
        )
        # Pre-flight check for tool support
        try:
            list(agent.run(".", stream=True))
        except ResponseError as e:
            if "does not support tools" in str(e.error):
                console.print(
                    f"[bold yellow]WARNING: Model '{model_id}' does not support tools. "
                    "Running in no-tools mode.[/bold yellow]"
                )
                # Initialize again without tools
                agent = Agent(
                    model=Ollama(id=model_id),
                    tools=[],
                    show_tool_calls=True,
                    markdown=True,
                    add_history_to_messages=True,
                    session_id="my_chat_session",
                )
                return agent
            raise
        return agent
    except Exception as e:
        console.print(f"[bold red]ERROR: Failed to initialize agent: {e}[/bold red]")
        console.print("[yellow]Is the Ollama server running and does it have the specified model?[/yellow]")
        sys.exit(1)

def run_single_query(agent, prompt):
    """Executes a single query and streams the response."""
    console.print("\n[bold blue]AI Assistant:[/]")
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            response_stream = agent.run(prompt, stream=True)

        tool_output = f.getvalue()
        if tool_output:
            tool_panel = Panel(
                tool_output.strip(),
                title="[bold yellow]Tool Call[/bold yellow]",
                border_style="yellow"
            )
            console.print(tool_panel)

        for response_chunk in response_stream:
            if response_chunk.content:
                console.out(response_chunk.content, end="", style="bright_blue")

        console.print()
    except ResponseError as e:
        console.print(
            f"\n[bold red]Ollama API error (status: {e.status_code}): {e.error}[/bold red]"
        )
    except Exception as e:
        console.print(
            f"\n[bold red]An unexpected error occurred during streaming: {e}[/bold red]"
        )


def run_interactive_chat(agent):
    """Starts an interactive chat loop with streaming response."""
    welcome_message = (
        "[bold]Welcome to Super-Assistant![/bold]\n\n" +        "I can count files AND search the internet!\n" +        "Ask me about the weather or the latest news."
    )
    console.print(Panel(
        welcome_message,
        title="Assistant with Toolkits",
        border_style="magenta"
    ))

    while True:
        try:
            user_input = console.input("[bold green]You:[/]")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue

            console.print("\n[bold blue]Asystent AI:[/]")
            f = io.StringIO()
            with redirect_stdout(f):
                response_stream = agent.run(user_input, stream=True)

            tool_output = f.getvalue()
            if tool_output:
                console.print(Panel(
                    tool_output.strip(),
                    title="[bold yellow]Tool Call[/bold yellow]",
                    border_style="yellow"
                ))

            for response_chunk in response_stream:
                if response_chunk.content:
                    console.out(response_chunk.content, end="", style="bright_blue")

            console.print("\n---")

        except (KeyboardInterrupt, EOFError):
            break
        except ResponseError as e:
            console.print(
                f"\n[bold red]Wystąpił błąd API Ollama (status: {e.status_code}): {e.error}[/bold red]"
            )
        except Exception as e:
            console.print(f"\n[bold red]Wystąpił nieoczekiwany błąd podczas strumieniowania: {e}[/bold red]")



    console.print("\n[yellow]Goodbye![/yellow]")


@click.command()
@click.argument('prompt', required=False)
@click.option(
    '-s', '--system',
    # Change #4: Improve the default system prompt to encourage AI to search the web
    default=(
        "You are a versatile and helpful AI assistant."
    ),
    help=(
        "Defines the system prompt "
        "for the AI."
    )
)
@click.option(
    '-m', '--model',
    default="gemma2:9b",
    help=(
        "ID of the Ollama model to be used "
        "(e.g., gemma2:9b, "
        "llama3.2)."
    )
)
def main(prompt, system, model):
    """
    AI assistant in the command line with tool support and ready-made toolkits.
    """
    console.print(f"[dim]Using model: [bold]{model}[/bold][/dim]")

    try:
        available_models = [m['model'] for m in ollama.list()['models']]
        if model not in available_models:
            console.print(
                f"\n[bold red]Error: Model '{model}' is not available in Ollama.[/bold red]"
            )
            console.print("\n[bold green]Available models:[/bold green]")
            # Use a set to avoid duplicates and sort for consistent output
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
        console.print(f"Details: {e.error}")
        console.print("[yellow]Ensure the Ollama server is running.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error: An unexpected problem occurred during model verification.[/bold red]")
        console.print(f"Details: {e}")
        console.print("[yellow]Ensure the Ollama server is running.[/yellow]")
        sys.exit(1)

    agent = setup_agent(system_prompt=system, model_id=model)

    if prompt:
        if not sys.stdin.isatty():
            piped_content = sys.stdin.read()
            full_prompt = (
                f"{prompt}\n\nHere is the content to process:\n"
                + "\n---\n" +
                f"{piped_content}"
            )
            run_single_query(agent, full_prompt)
        else:
            run_single_query(agent, prompt)
    else:
        run_interactive_chat(agent)


if __name__ == "__main__":
    main()
