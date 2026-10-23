"""Extraction and conversion of in-turn tool calls emitted by LLM backends."""

from __future__ import annotations

import ast
import contextlib
import json
import re
from typing import Any

from rai.actions.intent import ActionIntent
from rai.assistant.records import ConversationTurn


def _normalize_jsonish(s: str) -> str:
    """Normalize JSON-like strings with unquoted keys, single quotes, or literal newlines."""
    pattern = re.compile(
        r'("(?:\\.|[^"\\])*")'
        r"|('(?:\\.|[^'\\])*')"
        r"|([{,]\s*)([a-zA-Z_]\w*)(\s*:)"
        r"|(:\s*)(True|False|None)\b",
        re.DOTALL,
    )

    def _repl(m: re.Match[str]) -> str:
        if m.group(1) is not None:
            inner = m.group(1)[1:-1]
            inner = (
                inner.replace("\r\n", "\\n")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            return f'"{inner}"'
        if m.group(2) is not None:
            inner = m.group(2)[1:-1].replace("\\'", "'")
            return json.dumps(inner)
        if m.group(3) is not None:
            return f'{m.group(3)}"{m.group(4)}"{m.group(5)}'
        if m.group(6) is not None:
            lit = {"True": "true", "False": "false", "None": "null"}[m.group(7)]
            return f"{m.group(6)}{lit}"
        return m.group(0)

    return pattern.sub(_repl, s)


def parse_tool_args(raw_args: str) -> dict[str, Any]:
    """Parse JSON or Python-like tool call argument strings safely."""
    s = raw_args.strip()
    if not s:
        return {}

    # Properly serialize Gemma delimiter blocks (<|"|>...<|"|> and <|'|>...<|'|>)
    # into valid JSON strings so internal quotes and newlines are preserved.
    if '<|\\"|>' in s or '<|"|>' in s or "<|'|>" in s:
        s = s.replace('<|\\"|>', '<|"|>')
        s = re.sub(
            r'<\|"\|>(.*?)<\|"\|>',
            lambda m: json.dumps(m.group(1)),
            s,
            flags=re.DOTALL,
        )
        s = re.sub(
            r"<\|'\|>(.*?)<\|'\|>",
            lambda m: json.dumps(m.group(1)),
            s,
            flags=re.DOTALL,
        )
        # Handle unclosed trailing delimiter before final brace
        s = re.sub(
            r'<\|"\|>(.*?)\s*\}$',
            lambda m: json.dumps(m.group(1)) + "}",
            s,
            flags=re.DOTALL,
        )
        s = s.replace('<|"|>', '"').replace("<|'|>", "'")

    with contextlib.suppress(json.JSONDecodeError, TypeError):
        data = json.loads(s)
        if isinstance(data, dict):
            return data

    s_norm = _normalize_jsonish(s)
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        data = json.loads(s_norm)
        if isinstance(data, dict):
            return data

    with contextlib.suppress(ValueError, SyntaxError):
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            return data

    # Fallback for single key-value pair with raw multiline or unescaped quotes
    m_kv = re.match(
        r'^\{\s*["\']?([a-zA-Z_]\w*)["\']?\s*:\s*(.*?)\s*\}$',
        s,
        flags=re.DOTALL,
    )
    if m_kv:
        key = m_kv.group(1)
        val = m_kv.group(2).strip()
        for quote in ('"""', "'''", '"', "'"):
            if val.startswith(quote) and val.endswith(quote) and len(val) >= 2 * len(quote):
                val = val[len(quote) : -len(quote)]
                val = val.replace('\\"', '"').replace("\\'", "'").replace("\\n", "\n")
                return {key: val}

    return {}


_TOOL_ALIASES: dict[str, str] = {
    "search.web": "browser.search",
    "web.search": "browser.search",
    "web_search": "browser.search",
    "search": "browser.search",
    "internet.search": "browser.search",
    "browser_search": "browser.search",
    "browser.read": "browser.read_page",
    "read_page": "browser.read_page",
    "browser_read_page": "browser.read_page",
    "web.read": "browser.read_page",
    "page.read": "browser.read_page",
    "browser.open": "browser.open_result",
    "open_result": "browser.open_result",
    "browser_open_result": "browser.open_result",
    "web.open": "browser.open_result",
    "app.list": "application.list",
    "apps.list": "application.list",
    "applications.list": "application.list",
    "list_applications": "application.list",
    "app.launch": "application.launch",
    "apps.launch": "application.launch",
    "launch_application": "application.launch",
    "launch_app": "application.launch",
    "application.open": "application.launch",
    "open_application": "application.launch",
    "files.list": "file.list",
    "list_files": "file.list",
    "dir.list": "file.list",
    "directory.list": "file.list",
    "files.search": "file.search",
    "find_files": "file.search",
    "file.open": "document.open",
    "doc.open": "document.open",
    "volume.get": "system.volume.get",
    "audio.volume.get": "system.volume.get",
    "volume.set": "system.volume.set",
    "audio.volume.set": "system.volume.set",
    "process.list": "process.inspect",
    "processes.inspect": "process.inspect",
    "history.query": "activity.query",
    "activity.list": "activity.query",
    "activity.search": "activity.query",
    "python": "run_python_code",
    "python.run": "run_python_code",
    "run_python": "run_python_code",
    "exec_python": "run_python_code",
    "execute_python": "run_python_code",
    "shell": "run_shell_command",
    "bash": "run_shell_command",
    "run_shell": "run_shell_command",
    "shell.run": "run_shell_command",
}


def normalize_tool_capability(capability: str) -> str:
    """Normalize common model-emitted tool name aliases to registered capability names."""
    cleaned = capability.strip()
    return _TOOL_ALIASES.get(cleaned.lower(), cleaned)


def _normalize_capability_args(capability: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize common parameter alias names for code/shell execution tools."""
    if capability == "run_python_code" and "code" not in args:
        for alt in ("script", "python", "python_code", "source", "expression", "cmd", "command", "query"):
            if alt in args and isinstance(args[alt], str):
                return {**args, "code": args[alt]}
    elif capability == "run_shell_command" and "command" not in args:
        for alt in ("cmd", "shell", "script", "bash", "code", "query"):
            if alt in args and isinstance(args[alt], str):
                return {**args, "command": args[alt]}
    return args


def strip_tool_call_tags(text: str) -> str:
    """Remove raw <tool_call>...</tool_call> blocks from historical assistant text."""
    if not text or ("<tool_call>" not in text and "<|tool_call>" not in text):
        return text
    cleaned = re.sub(
        r"(?:<\|tool_call>|<tool_call>).*?(?:<tool_call\|>|<\|tool_call\|>|<\|tool_call_end\|>|</tool_call>|$)",
        "",
        text,
        flags=re.DOTALL,
    )
    return cleaned.strip()


def extract_model_tool_call(text: str) -> tuple[str, str, dict[str, Any]] | None:
    """Extract a tool call from LLM output (e.g. Gemma, ChatML, or markdown block).

    Returns:
        (preamble, capability_name, arguments) or None if no valid tool call pattern is found.
    """
    if not text:
        return None

    # 1. Gemma / ChatML syntax: <|tool_call>[call:] cap[{args}]</tool_call>
    m = re.search(
        r"(.*?)(?:<\|tool_call>|<tool_call>)\s*(?:call:\s*)?([a-zA-Z_][\w\.-]*)\s*(\{.*?\})?\s*(?:<tool_call\|>|<\|tool_call\|>|<\|tool_call_end\|>|</tool_call>|$)",
        text,
        re.DOTALL,
    )
    if m:
        preamble = m.group(1).strip()
        capability = normalize_tool_capability(m.group(2))
        raw_args = m.group(3)
        args = parse_tool_args(raw_args) if raw_args else {}
        return (preamble, capability, _normalize_capability_args(capability, args))

    # 2. JSON tool call: <tool_call>{"name": ..., "arguments": ...}</tool_call> or ```tool_call
    m2 = re.search(
        r"(.*?)(?:<\|tool_call>|<tool_call>|```(?:json\s*)?tool_call)\s*(\{.*\})\s*(?:<tool_call\|>|<\|tool_call\|>|<\|tool_call_end\|>|</tool_call>|```|$)",
        text,
        re.DOTALL,
    )
    if m2:
        preamble = m2.group(1).strip()
        obj = parse_tool_args(m2.group(2))
        cap = (
            obj.get("name")
            or obj.get("capability")
            or obj.get("function")
            or obj.get("tool")
        )
        if cap and isinstance(cap, str):
            raw_inner_args = (
                obj.get("arguments") or obj.get("args") or obj.get("parameters")
            )
            if isinstance(raw_inner_args, dict):
                args = raw_inner_args
            else:
                args = {
                    k: v
                    for k, v in obj.items()
                    if k not in ("name", "capability", "function", "tool")
                }
            norm_cap = normalize_tool_capability(cap)
            return (preamble, norm_cap, _normalize_capability_args(norm_cap, args))

    return None


_TOOL_PARAM_HINTS: dict[str, str] = {
    "application.list": '{"query": "<optional literal app name filter, or empty string>"}',
    "application.launch": '{"query": "<app name or desktop_id>", "selection": "<optional 1-based index>"}',
    "file.access": "{}",
    "file.list": '{"query": "<absolute directory path>"}',
    "file.search": '{"query": "<filename, pattern e.g. *.pdf, or substring>", "directory": "<optional allowed directory path>"}',
    "document.open": '{"query": "<file/dir path or name>", "selection": "<optional 1-based index>", "application": "<optional app>"}',
    "system.volume.get": "{}",
    "system.volume.set": '{"percent": <integer 0-100>}',
    "process.inspect": '{"query": "<process name or PID>"}',
    "browser.search": '{"query": "<search query>"}',
    "browser.open_result": '{"url": "<web URL to open>", "selection": "<optional 1-based result index from previous search>"}',
    "browser.read_page": '{"url": "<web URL to read>", "selection": "<optional 1-based result index from previous search>"}',
    "activity.query": '{"query": "<optional app/project filter>", "lookback_minutes": <optional int>}',
    "capabilities.list": "{}",
    "run_python_code": '{"code": "<python code snippet to execute in persistent REPL session>"}',
    "run_shell_command": '{"command": "<shell command string>"}',
    "calculate": '{"expression": "<mathematical expression>"}',
    "search.wikipedia": '{"query": "<search query>"}',
    "search.arxiv": '{"query": "<search query>"}',
    "finance.quote": '{"symbol": "<ticker symbol>"}',
    "send_desktop_notification": '{"title": "<title>", "message": "<body text>"}',
    "take_desktop_screenshot": "{}",
    "get_desktop_weather": '{"location": "<city or empty string>"}',
}


def format_tool_signature(name: str, description: str) -> str:
    """Format a capability descriptor with its expected tool-call argument schema."""
    params = _TOOL_PARAM_HINTS.get(name, "{}")
    return f"- {name} (args: {params}): {description}"


def _parse_int_arg(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def intent_from_tool_call(  # noqa: PLR0911, PLR0912
    turn: ConversationTurn,
    capability: str,
    args: dict[str, Any],
) -> ActionIntent | None:
    """Convert an extracted tool call into a strongly typed, validated ActionIntent."""
    lang = turn.metadata.get("language") if turn.metadata.get("language") in ("pl", "en") else "pl"
    source_turn_id = turn.record_id
    capability = normalize_tool_capability(capability)

    try:
        if capability == "browser.search":
            query = args.get("query") or args.get("q")
            if not query and isinstance(args.get("queries"), (list, tuple)) and args["queries"]:
                query = args["queries"][0]
            query_str = str(query or "").strip()
            if not query_str:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="browser.search",
                language=lang,
                query=query_str[:256],
            )

        if capability in {"browser.read_page", "browser.open_result"}:
            selection_int = _parse_int_arg(
                args.get("selection") or args.get("index") or args.get("result")
            )
            url_query = str(
                args.get("url") or args.get("query") or args.get("target") or ""
            ).strip()
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome=capability,
                language=lang,
                query=url_query[:256],
                selection=selection_int,
            )

        if capability == "process.inspect":
            query_str = str(
                args.get("query") or args.get("name") or args.get("pid") or ""
            ).strip()
            if not query_str:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="process.inspect",
                language=lang,
                query=query_str[:256],
            )

        if capability == "activity.query":
            lookback_int = _parse_int_arg(
                args.get("lookback_minutes") or args.get("minutes")
            )
            query_str = str(args.get("query") or args.get("filter") or "").strip()
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="activity.query",
                language=lang,
                query=query_str[:256],
                lookback_minutes=lookback_int,
            )

        if capability == "system.volume.get":
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="system.volume.get",
                language=lang,
            )

        if capability == "system.volume.set":
            pct = _parse_int_arg(
                args.get("percent")
                if "percent" in args
                else (args.get("volume") if "volume" in args else args.get("level"))
            )
            if pct is None:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="system.volume.set",
                language=lang,
                percent=pct,
            )

        if capability == "application.list":
            query_str = str(args.get("query") or args.get("name") or "").strip()
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="application.list",
                language=lang,
                query=query_str[:256],
            )

        if capability == "application.launch":
            query_str = str(
                args.get("query")
                or args.get("name")
                or args.get("application")
                or args.get("desktop_id")
                or args.get("app")
                or ""
            ).strip()
            selection_int = _parse_int_arg(args.get("selection") or args.get("index"))
            if not query_str and selection_int is None:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="application.launch",
                language=lang,
                query=query_str[:256],
                selection=selection_int,
            )

        if capability in {"file.search", "file.list"}:
            dir_str = str(args.get("directory") or args.get("dir") or "").strip()
            query_str = str(
                args.get("query")
                or args.get("path")
                or args.get("name")
                or args.get("filename")
                or args.get("pattern")
                or ""
            ).strip()
            if not query_str and dir_str:
                query_str = dir_str
                dir_str = ""
            if not query_str:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome=capability,
                language=lang,
                query=query_str[:256],
                directory=dir_str[:256],
            )

        if capability == "document.open":
            query_str = str(
                args.get("query")
                or args.get("path")
                or args.get("file")
                or args.get("filename")
                or args.get("name")
                or ""
            ).strip()
            selection_int = _parse_int_arg(args.get("selection") or args.get("index"))
            app_str = str(
                args.get("application") or args.get("app") or args.get("desktop_id") or ""
            ).strip()
            if not query_str and selection_int is None:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="document.open",
                language=lang,
                query=query_str[:256],
                selection=selection_int,
                application=app_str[:256],
            )

        if capability == "file.access":
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="file.access",
                language=lang,
            )

        if capability == "capabilities.list":
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="capabilities.list",
                language=lang,
            )

    except Exception:  # noqa: BLE001
        return None

    return None
