"""
Prompt engineering and parsing for local tools.
"""
import json
import re
from typing import Any, Dict, List, Optional
from returns.result import Result, Success, Failure

def format_tools_to_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Formats a list of tool definitions (JSON schema) into a system prompt addition.
    """
    if not tools:
        return ""
        
    tool_desc = json.dumps(tools, indent=2)
    
    prompt = f"""
\n[AVAILABLE TOOLS]
You have access to the following tools. 
To use a tool, you MUST respond with a JSON object wrapped in <tool_code> tags.
The JSON object must follow this schema:
{{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{ "arg1": "value1" }}
        }}
    ]
}}

Tool Definitions:
{tool_desc}

If you do not need to use a tool, just respond with natural text.
[/AVAILABLE TOOLS]
"""
    return prompt

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extracts tool calls from the generated text.
    Returns a list of dicts compatible with OpenAI tool calls structure:
    [
        {
            "id": "call_id",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "json_string" 
            }
        }
    ]
    """
    # Regex to find <tool_code>...</tool_code> (dotall)
    pattern = re.compile(r"<tool_code>(.*?)</tool_code>", re.DOTALL)
    match = pattern.search(text)
    
    if not match:
        return []
    
    json_str = match.group(1).strip()
    try:
        data = json.loads(json_str)
        if "tool_calls" not in data:
            return []
            
        parsed_calls = []
        for i, call in enumerate(data["tool_calls"]):
            tool_name = call.get("name")
            tool_args = call.get("arguments")
            
            if not tool_name:
                continue
                
            # Convert args back to string if they are dict (OpenAI format expects stringified JSON)
            if isinstance(tool_args, dict):
                tool_args_str = json.dumps(tool_args)
            else:
                tool_args_str = str(tool_args)
                
            parsed_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": tool_args_str
                }
            })
            
        return parsed_calls
        
    except json.JSONDecodeError:
        return []
    except Exception:
        return []

def clean_response_text(text: str) -> str:
    """Removes the tool code block from the text to be shown to the user."""
    pattern = re.compile(r"<tool_code>.*?</tool_code>", re.DOTALL)
    return pattern.sub("", text).strip()

def format_tools_for_function_gemma(tools: List[Dict[str, Any]]) -> str:
    """
    Formats tools for FunctionGemma using specialized tokens.
    """
    if not tools:
        return ""
    
    # FunctionGemma expects:
    # <start_of_turn>developer
    # You are a model that can do function calling with the following functions
    # <start_function_declaration>
    # ...
    # <end_function_declaration>
    # <end_of_turn>
    
    # We strip the outer <start_of_turn> as _messages_to_prompt handles turns?
    # No, we return the inner content or the full block?
    # Bridges inject this into SYSTEM message or prepend to prompt.
    # FunctionGemma usually treats this as 'developer' turn or 'system' role.
    
    # Let's return just the declaration block, and the bridge will wrap it in <start_of_turn>developer
    # if it implements the chat template. 
    # OR, we return the plain text that goes INSIDE the developer message.
    
    formatted_tools = []
    for tool in tools:
        schema = _fg_format_function_declaration(tool)
        formatted_tools.append(f"<start_function_declaration>{schema}<end_function_declaration>")
    
    # Template intro: <start_of_turn>developer\n...
    # Template tool loop: ...declarations... (no separator?)
    # Template end: <end_of_turn>\n
    
    # Template loop does not add separator explicitly.
    # But usually models prefer strict adjacency?
    # Let's try joining with nothing or newline. 
    # The template has:
    # {%- for tool in tools %}
    #    ...
    # {%- endfor %}
    # This implies concatenation without separator unless inside format_function_declaration.
    # I'll try NO separator.
    return "".join(formatted_tools)

def parse_function_gemma_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Parses FunctionGemma output.
    Format: <start_function_call>func_name(arg1=val1, ...)<end_function_call>
    OR: <start_function_call>{"name": "func", "arguments": {...}}<end_function_call> (if JSON mode)
    
    For now, assume Python-like call or JSON.
    Documentation implies: tool_code(params)
    We will regex for <start_function_call>(.*?)<end_function_call>
    """
    pattern = re.compile(r"<start_function_call>(.*?)<end_function_call>", re.DOTALL)
    matches = pattern.findall(text)
    
    parsed_calls = []
    for i, content in enumerate(matches):
        content = content.strip()
        
        # 1. Try standard JSON payload
        if content.startswith("{"):
            try:
                data = json.loads(content)
                if "name" in data:
                     parsed_calls.append({
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {}))
                        }
                     })
                continue
            except:
                pass
                
        # 2. Try 'call:name{args}' format (FunctionGemma specific)
        # e.g. call:get_current_temperature{location:<escape>London<escape>}
        match = re.match(r"^call:([a-zA-Z0-9_]+)\{(.*)\}$", content, re.DOTALL)
        if match:
            name = match.group(1)
            raw_args = match.group(2)
            
            # Helper to convert FunctionGemma args to JSON
            # Replace <escape> with "
            # Try to fix unquoted keys?
            # Example: location:<escape>London<escape> -> location:"London"
            # We need "location":"London"
            
            # Crude approach: 
            # 1. Replace <escape> with "
            args_str = raw_args.replace("<escape>", '"')
            
            # 2. Quote keys? regex for (\w+):
            # Be careful not to quote valid JSON structure if mixed.
            # Assuming simple keys:
            args_str = re.sub(r'([a-zA-Z0-9_]+):', r'"\1":', args_str)
            
            try:
                # Wrap in {} since we matched content inside {}
                data = json.loads(f"{{{args_str}}}")
                parsed_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(data)
                    }
                })
                continue
            except:
                # Fallback: empty args or partial parsing
                pass

        # 3. Python style fallback: name(k=v)
        match = re.match(r"^([a-zA-Z0-9_]+)\((.*)\)$", content, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            
            import ast
            try:
                # Construct call: dict(args) -> gives dict.
                val = ast.literal_eval(f"dict({args_str})")
                parsed_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(val)
                    }
                })
            except:
                pass
    
    return parsed_calls

def clean_function_gemma_response_text(text: str) -> str:
    """Removes the tool call block from FunctionGemma text."""
    pattern = re.compile(r"<start_function_call>.*?<end_function_call>", re.DOTALL)
    return pattern.sub("", text).strip()
