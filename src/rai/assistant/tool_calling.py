"""Extraction and conversion of in-turn tool calls emitted by LLM backends."""

from __future__ import annotations

import ast
import contextlib
import json
import re
from typing import Any

from rai.actions.intent import ActionIntent
from rai.assistant.records import ConversationTurn


def parse_tool_args(raw_args: str) -> dict[str, Any]:
    """Parse JSON or Python-like tool call argument strings safely."""
    s = raw_args.strip()
    s = s.replace('<|\\\"|>', '"').replace('<|"|>', '"').replace("<|'|>", "'")
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        data = json.loads(s)
        if isinstance(data, dict):
            return data

    # Normalize unquoted keys: {key: ...} -> {"key": ...}
    s_norm = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', s)
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        data = json.loads(s_norm)
        if isinstance(data, dict):
            return data

    with contextlib.suppress(ValueError, SyntaxError):
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            return data

    return {}


def extract_model_tool_call(text: str) -> tuple[str, str, dict[str, Any]] | None:
    """Extract a tool call from LLM output (e.g. Gemma, ChatML, or markdown block).

    Returns:
        (preamble, capability_name, arguments) or None if no valid tool call pattern is found.
    """
    if not text:
        return None

    # 1. Gemma / syntax: <|tool_call>call: cap{args}...
    m = re.search(
        r"(.*?)(?:<\|tool_call>|<tool_call>)\s*call:\s*([\w\.]+)\s*(\{.*?\})(?:<tool_call\|>|<\|tool_call\|>|<\|tool_call_end\|>|</tool_call>|$)",
        text,
        re.DOTALL,
    )
    if m:
        preamble = m.group(1).strip()
        capability = m.group(2).strip()
        args = parse_tool_args(m.group(3))
        return (preamble, capability, args)

    # 2. JSON tool call: <tool_call>{"name": ..., "arguments": ...}</tool_call> or ```tool_call
    m2 = re.search(
        r"(.*?)(?:<tool_call>|```(?:json\s*)?tool_call)\s*(\{.*?\})\s*(?:</tool_call>|```|$)",
        text,
        re.DOTALL,
    )
    if m2:
        preamble = m2.group(1).strip()
        obj = parse_tool_args(m2.group(2))
        cap = obj.get("name") or obj.get("capability") or obj.get("function")
        if cap and isinstance(cap, str):
            args = obj.get("arguments") or obj.get("args") or {}
            if isinstance(args, dict):
                return (preamble, cap.strip(), args)

    return None


def intent_from_tool_call(  # noqa: PLR0911, PLR0912
    turn: ConversationTurn,
    capability: str,
    args: dict[str, Any],
) -> ActionIntent | None:
    """Convert an extracted tool call into a strongly typed, validated ActionIntent."""
    lang = turn.metadata.get("language") if turn.metadata.get("language") in ("pl", "en") else "pl"
    source_turn_id = turn.record_id

    try:
        if capability == "browser.search":
            query = args.get("query")
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
            sel = args.get("selection") or args.get("index")
            selection_int = None
            if sel is not None:
                try:
                    selection_int = int(sel)
                except (ValueError, TypeError):
                    pass
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome=capability,
                language=lang,
                selection=selection_int,
            )

        if capability == "process.inspect":
            query_str = str(args.get("query") or "").strip()
            if not query_str:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="process.inspect",
                language=lang,
                query=query_str[:256],
            )

        if capability == "activity.query":
            lookback = args.get("lookback_minutes")
            lookback_int = None
            if lookback is not None:
                try:
                    lookback_int = int(lookback)
                except (ValueError, TypeError):
                    pass
            query_str = str(args.get("query") or "").strip()
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
            pct = args.get("percent")
            if pct is None:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="system.volume.set",
                language=lang,
                percent=int(pct),
            )

        if capability == "application.list":
            query_str = str(args.get("query") or "").strip()
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome="application.list",
                language=lang,
                query=query_str[:256],
            )

        if capability in {"file.search", "file.list"}:
            query_str = str(args.get("query") or "").strip()
            if not query_str:
                return None
            return ActionIntent(
                source_turn_id=source_turn_id,
                outcome=capability,
                language=lang,
                query=query_str[:256],
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
