#!/usr/bin/env python
""" rai - Rich AI CLI assistant
"""
import os
import sys
import json
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout
import datetime
import subprocess
import ollama

from agno.agent import Agent
from agno.models.ollama import Ollama
from ollama import ResponseError
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.webbrowser import WebBrowserTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
from agno.tools.tavily import TavilyTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.openweather import OpenWeatherTools
from agno.tools import tool

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

import click

from .tools import send_notification, take_screenshot, weather

# global console
console = Console(force_terminal=True)




base_tools=[
    send_notification, # Nowy tool do wysyłania powiadomień
    take_screenshot, # Nowy tool do robienia zrzutów ekranu
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
    """Inicjalizuje agenta Agno, ustawiając model, prompt i narzędzia."""
    load_dotenv() # Upewnijmy się, że zmienne środowiskowe są załadowane

    if os.getenv("TAVILY_API_KEY"):
        tools = base_tools.append(TavilyTools())
    else:
        tools = base_tools
        console.print("[bold yellow]WARNING: Missing TAVILY_API_KEY env variable. Tavily will be disabled![/bold yellow]")

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
                console.print(f"[bold yellow]WARNING: Model '{model_id}' does not support tools. Running in no-tools mode.[/bold yellow]")
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
            else:
                raise
        return agent
    except Exception as e:
        console.print(f"[bold red]BŁĄD: Nie udało się zainicjalizować agenta: {e}[/bold red]")
        console.print("[yellow]Czy na pewno serwer Ollama jest uruchomiony i posiada wskazany model?[/yellow]")
        sys.exit(1)

def run_single_query(agent, prompt, system):
    """Wykonuje pojedyncze zapytanie i wyświetla odpowiedź strumieniowo."""
    console.print("\n[bold blue]Asystent AI:[/]")
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            response_stream = agent.run(prompt, stream=True)

        tool_output = f.getvalue()
        if tool_output:
            console.print(Panel(tool_output.strip(), title="[bold yellow]Wywołanie Narzędzia[/bold yellow]", border_style="yellow"))

        for response_chunk in response_stream:
            if response_chunk.content:
                console.out(response_chunk.content, end="", style="bright_blue")

        console.print()
    except ResponseError as e:
        console.print(f"\n[bold red]Wystąpił błąd API Ollama (status: {e.status_code}): {e.error}[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Wystąpił nieoczekiwany błąd podczas strumieniowania: {e}[/bold red]")


def run_interactive_chat(agent, system):
    """Uruchamia pętlę interaktywnego czatu z odpowiedzią strumieniową."""
    welcome_message = "[bold]Witaj w Super-Asystencie![/bold]\n\nPotrafię liczyć pliki ORAZ szukać w internecie!\nZapytaj mnie o pogodę lub o najnowsze wiadomości."
    console.print(Panel(welcome_message, title="Asystent z Toolkitami", border_style="magenta"))

    while True:
        try:
            user_input = console.input("[bold green]Ty:[/]")
            if user_input.lower() in ["wyjdź", "exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue

            console.print("\n[bold blue]Asystent AI:[/]")
            f = io.StringIO()
            with redirect_stdout(f):
                response_stream = agent.run(user_input, stream=True)

            tool_output = f.getvalue()
            if tool_output:
                console.print(Panel(tool_output.strip(), title="[bold yellow]Wywołanie Narzędzia[/bold yellow]", border_style="yellow"))

            for response_chunk in response_stream:
                if response_chunk.content:
                    console.out(response_chunk.content, end="", style="bright_blue")

            console.print("\n---")

        except (KeyboardInterrupt, EOFError):
            break
        except ResponseError as e:
            console.print(f"\n[bold red]Wystąpił błąd API Ollama (status: {e.status_code}): {e.error}[/bold red]")
        except Exception as e:
            console.print(f"\n[bold red]Wystąpił nieoczekiwany błąd podczas strumieniowania: {e}[/bold red]")



    console.print("\n[yellow]Do widzenia![/yellow]")


@click.command()
@click.argument('prompt', required=False)
@click.option(
    '-s', '--system',
    # ZMIANA #4: Ulepszamy domyślny system prompt, aby zachęcić AI do szukania w sieci
    default="You are versatile and helpful AI assistant.",
    help="Definiuje prompt systemowy dla AI."
)
@click.option(
    '-m', '--model',
    default="gemma2:9b",
    help="ID modelu Ollama, który ma być użyty (np. gemma2:9b, llama3.2)."
)
def main(prompt, system, model):
    """
    Asystent AI w wierszu poleceń z obsługą narzędzi i gotowych toolkitów.
    """
    console.print(f"[dim]Używam modelu: [bold]{model}[/bold][/dim]")

    try:
        available_models = [m['model'] for m in ollama.list()['models']]
        if model not in available_models:
            console.print(f"\n[bold red]Błąd: Model '{model}' nie jest dostępny w Ollama.[/bold red]")
            console.print("\n[bold green]Dostępne modele:[/bold green]")
            # Use a set to avoid duplicates and sort for consistent output
            for m_name in sorted(list(set(available_models))):
                console.print(f"- {m_name}", highlight=False)
            console.print(f"\n[yellow]Możesz pobrać brakujący model poleceniem: [bold]ollama pull {model}[/bold][/yellow]")
            sys.exit(1)
    except ResponseError as e:
        console.print(f"\n[bold red]Błąd: Nie udało się połączyć z serwerem Ollama, aby zweryfikować model (status: {e.status_code}).[/bold red]")
        console.print(f"Szczegóły: {e.error}")
        console.print("[yellow]Upewnij się, że serwer Ollama jest uruchomiony.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Błąd: Wystąpił nieoczekiwany problem podczas weryfikacji modelu.[/bold red]")
        console.print(f"Szczegóły: {e}")
        console.print("[yellow]Upewnij się, że serwer Ollama jest uruchomiony.[/yellow]")
        sys.exit(1)

    agent = setup_agent(system_prompt=system, model_id=model)

    if prompt:
        if not sys.stdin.isatty():
            piped_content = sys.stdin.read()
            full_prompt = f"{prompt}\n\nOto treść do przetworzenia:\n\n---\n{piped_content}"
            run_single_query(agent, full_prompt, system)
        else:
            run_single_query(agent, prompt, system)
    else:
        run_interactive_chat(agent, system)


if __name__ == "__main__":
    main()
