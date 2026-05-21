"""
Core components for agent setup and configuration management.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable, Union, TypedDict

from rich.console import Console

logger = logging.getLogger(__name__)

@runtime_checkable
class AgentResponse(Protocol):  # pylint: disable=too-few-public-methods
    """A protocol for the response object from an agent's arun method."""
    content: Union[str, List[object]]
    tool_calls: Optional[List[object]]

class ResponseDict(TypedDict, total=False):
    """A TypedDict for the dictionary response."""
    content: str
    tool_calls: List[object]

from rai.tools.desktop import get_desktop_adapter

# --- Globals ---
console = Console(record=True)
error_console = Console(stderr=True)

# --- Standard Tool Implementations (Decoupled from Agno) ---

def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.

    Args:
        expression (str): The mathematical expression to evaluate (e.g. "2 * 3 + 4").
    """
    try:
        import math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        return str(eval(expression, {"__builtins__": None}, allowed))  # pylint: disable=eval-used
    except Exception as e:
        return f"Error: {e}"

def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia for the given query.

    Args:
        query (str): The search query.
    """
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {e}"

def web_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo for the given query.

    Args:
        query (str): The search query.
    """
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        return "\n\n".join([f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Error: {e}"

def arxiv_search(query: str) -> str:
    """
    Searches arXiv for scientific papers.

    Args:
        query (str): The scientific search query.
    """
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=3)
        results = []
        for r in client.results(search):
            results.append(f"Title: {r.title}\nAuthors: {', '.join(a.name for a in r.authors)}\nSummary: {r.summary}\nURL: {r.entry_id}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Error: {e}"

def get_stock_price(ticker: str) -> str:
    """
    Gets stock market information for a ticker using yfinance.

    Args:
        ticker (str): The stock ticker symbol (e.g. "GOOG", "AAPL").
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return f"Ticker: {ticker}\nPrice: {info.get('regularMarketPrice') or info.get('currentPrice')}\nCurrency: {info.get('currency')}\nSummary: {info.get('longBusinessSummary', 'N/A')}"
    except Exception as e:
        return f"Error: {e}"


# --- Tool Registry ---
TOOL_REGISTRY = {
    "CalculatorTools": [calculate],
    "ArxivTools": [arxiv_search],
    "WikipediaTools": [wikipedia_search],
    "DuckDuckGoTools": [web_search],
    "YFinanceTools": [get_stock_price],
}


# --- Tool and Model Setup ---
def setup_tools(  # noqa: PLR0912 # pylint: disable=too-many-branches, too-many-locals
    enable_tools: bool,
    quiet: bool,
    enabled_tool_names: Optional[List[str]] = None,
    has_prompt: bool = False,
) -> Tuple[List[Any], List[str]]:
    """Sets up the tools for the agent."""
    messages = []
    agent_tools: List[Any] = []

    if not enable_tools:
        return agent_tools, messages

    tools_to_enable = (
        enabled_tool_names
        if enabled_tool_names is not None
        else list(TOOL_REGISTRY.keys()) + ["ClientTools", "GitlabTools"]
    )

    for tool_name in tools_to_enable:
        # Handle Desktop tools
        if tool_name in [
            "DesktopNotificationTool", "DesktopScreenshotTool", "DesktopWeatherTool"
        ]:
            try:
                adapter = get_desktop_adapter()
                if tool_name == "DesktopNotificationTool":
                    agent_tools.append(adapter.send_notification)
                elif tool_name == "DesktopScreenshotTool":
                    agent_tools.append(adapter.take_screenshot)
                elif tool_name == "DesktopWeatherTool":
                    agent_tools.append(adapter.weather)
                logger.debug("%s successfully enabled.", tool_name)
            except Exception as e:
                logger.debug("Could not enable %s: %s", tool_name, e)
            continue

        # Handle ClientTools
        if tool_name == "ClientTools":
            try:
                from rai.tools.client import eval_scheme
                agent_tools.append(eval_scheme)
                logger.debug("ClientTools successfully enabled.")
            except Exception as e:
                logger.debug("Could not enable ClientTools: %s", e)
            continue

        # Handle GitlabTools
        if tool_name == "GitlabTools":
            if not os.getenv("GITLAB_ACCESS_TOKEN"):
                if not quiet and not has_prompt:
                    messages.append(
                        "[bold yellow]WARNING: Missing GITLAB_ACCESS_TOKEN env variable. "
                        "GitlabTools will be disabled![/bold yellow]"
                    )
                continue
            try:
                from rai.tools.gitlab import GitlabTools
                gitlab_inst = GitlabTools()
                # Get all public methods of GitlabTools as callables
                for attr_name in dir(gitlab_inst):
                    if attr_name.startswith("_"):
                        continue
                    attr = getattr(gitlab_inst, attr_name)
                    if callable(attr):
                        agent_tools.append(attr)
                logger.debug("GitlabTools successfully enabled.")
            except Exception as e:
                logger.debug("Could not enable GitlabTools: %s", e)
            continue

        if tool_name not in TOOL_REGISTRY:
            if not quiet:
                messages.append(
                    f"[bold yellow]WARNING: Unknown tool '{tool_name}' "
                    "specified in configuration. Skipping.[/bold yellow]"
                )
            continue

        # Enable mapped standalone functions
        for func in TOOL_REGISTRY[tool_name]:
            agent_tools.append(func)
            logger.debug("%s successfully enabled.", tool_name)

    return agent_tools, messages


def validate_model_env(
    backend: str, model_id: str, quiet: bool, ollama_host: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates the environment for the specified backend and model.
    Returns a configuration dictionary for the adapter to use for instantiation,
    and a list of messages to display.
    """
    messages = []
    if backend == "gemini" and "GEMINI_API_KEY" in os.environ:
        if "GOOGLE_API_KEY" in os.environ and not quiet:
            messages.append(
                "[bold yellow]INFO: GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "Using GEMINI_API_KEY.[/bold yellow]"
            )
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    api_keys = {
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    supported_backends = ["ollama", "gemini", "anthropic", "openai", "groq", "local"]

    if backend not in supported_backends:
        error_console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

    if backend in api_keys and not os.getenv(api_keys[backend]):
        error_console.print(
            f"[bold red]ERROR: {api_keys[backend]} environment variable not set.[/bold red]"
        )
        sys.exit(1)

    config = {
        "backend": backend,
        "model_id": model_id,
        "api_key_env_var": api_keys.get(backend),
    }

    if backend == "ollama":
        config["ollama_host"] = ollama_host

    return config, messages
