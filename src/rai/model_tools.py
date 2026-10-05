"""Legacy TUI model metadata helper; no CLI dependency."""
import ollama
from ollama import ResponseError

def check_model_tool_support(model_id: str) -> bool:
    """Checks if the specified Ollama model supports tool use."""
    try:
        details = ollama.show(model_id)
        modelfile = str(details.get("modelfile", "") or "")
        # Robust check for tool support indicators
        indicators = [
            "tool_use",             # Explicit parameter
            "{{ .Tools",            # Template variable (standard)
            "{{.Tools",             # Template variable (standard)
            "{{ $.Tools",           # Template variable (with global context)
            "{{$.Tools",            # Template variable (with global context)
            "{{- if .Tools",        # Conditional check (standard)
            "{{- if $.Tools",       # Conditional check (global)
            "PARSER functiongemma", # specialized parser
            "RENDERER functiongemma" # specialized renderer
        ]
        return any(ind in modelfile for ind in indicators)
    except ResponseError:
        return False


