"""Unit tests for rai.inference.prompting."""

from typing import Any, Dict, List

from rai.inference.prompting import (
    clean_function_gemma_response_text,
    clean_response_text,
    format_tools_for_function_gemma,
    format_tools_to_system_prompt,
    parse_function_gemma_tool_calls,
    parse_tool_calls,
)


def test_format_tools_to_system_prompt_empty() -> None:
    assert format_tools_to_system_prompt([]) == ""


def test_format_tools_to_system_prompt_success() -> None:
    tools: List[Dict[str, Any]] = [
        {
            "name": "get_weather",
            "description": "Fetch weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    prompt = format_tools_to_system_prompt(tools)
    assert "[AVAILABLE TOOLS]" in prompt
    assert "get_weather" in prompt
    assert "<tool_code>" in prompt


def test_parse_tool_calls_no_tags() -> None:
    text = "Hello, I can help you without tools."
    assert parse_tool_calls(text) == []


def test_parse_tool_calls_invalid_json() -> None:
    text = "Here is a call: <tool_code>{not valid json</tool_code>"
    assert parse_tool_calls(text) == []


def test_parse_tool_calls_no_tool_calls_key() -> None:
    text = '<tool_code>{"other_key": 123}</tool_code>'
    assert parse_tool_calls(text) == []


def test_parse_tool_calls_valid_multiple() -> None:
    payload = """Thinking...
<tool_code>
{
    "tool_calls": [
        {"name": "calc", "arguments": {"expr": "2+2"}},
        {"name": "lookup", "arguments": "raw_query_string"},
        {"arguments": {"missing": "name"}}
    ]
}
</tool_code>
Done!"""
    calls = parse_tool_calls(payload)
    assert len(calls) == 2
    assert calls[0]["id"] == "call_0"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "calc"
    assert '"expr": "2+2"' in calls[0]["function"]["arguments"]

    assert calls[1]["id"] == "call_1"
    assert calls[1]["function"]["name"] == "lookup"
    assert calls[1]["function"]["arguments"] == "raw_query_string"


def test_clean_response_text() -> None:
    text = 'Prefix message.<tool_code>{"tool_calls": []}</tool_code>Suffix response.'
    cleaned = clean_response_text(text)
    assert "<tool_code>" not in cleaned
    assert "Prefix message." in cleaned
    assert "Suffix response." in cleaned


def test_format_tools_for_function_gemma() -> None:
    assert format_tools_for_function_gemma([]) == ""
    tools: List[Dict[str, Any]] = [
        {"name": "search", "description": "Search web"}
    ]
    formatted = format_tools_for_function_gemma(tools)
    assert "<start_function_declaration>" in formatted
    assert "<end_function_declaration>" in formatted
    assert '"name": "search"' in formatted


def test_parse_function_gemma_json_payload() -> None:
    text = '<start_function_call>{"name": "check_status", "arguments": {"host": "localhost"}}<end_function_call>'
    calls = parse_function_gemma_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "check_status"
    assert '"host": "localhost"' in calls[0]["function"]["arguments"]


def test_parse_function_gemma_call_syntax() -> None:
    text = "<start_function_call>call:get_weather{city:<escape>Warsaw<escape>}<end_function_call>"
    calls = parse_function_gemma_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert '"city": "Warsaw"' in calls[0]["function"]["arguments"]


def test_parse_function_gemma_python_syntax() -> None:
    text = "<start_function_call>call:add(a=1, b=2)<end_function_call>"
    calls = parse_function_gemma_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "add"
    assert '"a": 1' in calls[0]["function"]["arguments"]
    assert '"b": 2' in calls[0]["function"]["arguments"]


def test_parse_function_gemma_python_positional_syntax() -> None:
    text = "<start_function_call>calculate('2 * 3')<end_function_call>"
    calls = parse_function_gemma_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "calculate"
    assert "2 * 3" in calls[0]["function"]["arguments"]


def test_clean_function_gemma_response_text() -> None:
    text = "Output text.<start_function_call>call:test()<end_function_call>After call."
    cleaned = clean_function_gemma_response_text(text)
    assert "<start_function_call>" not in cleaned
    assert "Output text." in cleaned
    assert "After call." in cleaned
