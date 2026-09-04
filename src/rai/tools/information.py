"""Compatibility information capabilities used by legacy agent configs."""

from __future__ import annotations


def calculate(expression: str) -> str:
    try:
        import math  # noqa: PLC0415

        allowed = {key: getattr(math, key) for key in dir(math) if not key.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        return str(eval(expression, {"__builtins__": None}, allowed))  # pylint: disable=eval-used
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"Error: {exc}"


def wikipedia_search(query: str) -> str:
    try:
        import wikipedia  # noqa: PLC0415

        return wikipedia.summary(query, sentences=3)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"Error: {exc}"


def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS  # noqa: PLC0415

        results = DDGS().text(query, max_results=5)
        return "\n\n".join(
            f"Title: {result['title']}\nLink: {result['href']}\nSnippet: {result['body']}"
            for result in results
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"Error: {exc}"


def arxiv_search(query: str) -> str:
    try:
        import arxiv  # noqa: PLC0415

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=3)
        return "\n\n".join(
            f"Title: {result.title}\n"
            f"Authors: {', '.join(author.name for author in result.authors)}\n"
            f"Summary: {result.summary}\nURL: {result.entry_id}"
            for result in client.results(search)
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"Error: {exc}"


def get_stock_price(ticker: str) -> str:
    try:
        import yfinance as yf  # noqa: PLC0415

        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        return (
            f"Ticker: {ticker}\nPrice: {price}\nCurrency: {info.get('currency')}\n"
            f"Summary: {info.get('longBusinessSummary', 'N/A')}"
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"Error: {exc}"
