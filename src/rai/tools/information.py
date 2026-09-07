"""Compatibility information capabilities used by legacy agent configs."""

from __future__ import annotations

import ast
from collections.abc import Callable
import math
import operator

Number = int | float
CalcValue = Number | tuple[Number, ...]

MAX_EXPRESSION_CHARS = 256
MAX_AST_NODES = 64
MAX_COLLECTION_ITEMS = 32
MAX_ABS_VALUE = 1e100
MAX_POWER_EXPONENT = 100

_BINARY_OPERATORS: dict[
    type[ast.operator], Callable[[Number, Number], Number]
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CONSTANTS: dict[str, Number] = {
    "e": math.e,
    "pi": math.pi,
    "tau": math.tau,
}
_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "abs": abs,
    "acos": math.acos,
    "asin": math.asin,
    "atan": math.atan,
    "atan2": math.atan2,
    "ceil": math.ceil,
    "cos": math.cos,
    "degrees": math.degrees,
    "exp": math.exp,
    "fabs": math.fabs,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "max": max,
    "min": min,
    "radians": math.radians,
    "round": round,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "sum": sum,
    "tan": math.tan,
}


def _bounded_number(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("calculator results must be real numbers")
    if abs(value) > MAX_ABS_VALUE:
        raise ValueError("calculator result exceeds the numeric limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    return value


def _evaluate_expression(node: ast.AST) -> CalcValue:  # noqa: PLR0911
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)
    if isinstance(node, ast.Constant):
        return _bounded_number(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _bounded_number(_evaluate_expression(node.operand))
        return _bounded_number(_UNARY_OPERATORS[type(node.op)](operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _bounded_number(_evaluate_expression(node.left))
        right = _bounded_number(_evaluate_expression(node.right))
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POWER_EXPONENT:
            raise ValueError("calculator exponent exceeds the limit")
        return _bounded_number(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, (ast.List, ast.Tuple)):
        if len(node.elts) > MAX_COLLECTION_ITEMS:
            raise ValueError("calculator collection exceeds the item limit")
        return tuple(
            _bounded_number(_evaluate_expression(item)) for item in node.elts
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords or node.func.id not in _FUNCTIONS:
            raise ValueError("calculator function is not allowed")
        arguments = [_evaluate_expression(argument) for argument in node.args]
        return _bounded_number(_FUNCTIONS[node.func.id](*arguments))
    raise ValueError("calculator expression contains unsupported syntax")


def calculate(expression: str) -> str:
    """Evaluate a small, bounded arithmetic language without Python ``eval``."""
    try:
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ValueError("calculator expression exceeds the length limit")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError("calculator expression exceeds the complexity limit")
        return str(_evaluate_expression(tree))
    except (ArithmeticError, SyntaxError, TypeError, ValueError) as exc:
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
