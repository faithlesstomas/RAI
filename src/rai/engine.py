"""
The core engine for the RAI platform.

This module discovers and manages framework adapters, and provides the
main entry point for running chat interactions.
"""
import pkgutil
import inspect
import importlib
from typing import Any, Dict, List, Type, Optional
import uuid

from . import adapters
from .adapters.base import BaseAdapter
from .core import get_session_config # Keep for backward compatibility of run_chat

# A dictionary to hold discovered adapter classes
_ADAPTERS: Dict[str, Type[BaseAdapter]] = {}

def _discover_adapters() -> None:
    """Dynamically discovers and loads adapter classes."""
    if _ADAPTERS:
        return

    for module_info in pkgutil.iter_modules(adapters.__path__, adapters.__name__ + "."):
        if module_info.name.endswith('.base') or module_info.name.endswith('.__init__'):
            continue

        module = importlib.import_module(module_info.name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                adapter_name = module_info.name.split('.')[-1]
                _ADAPTERS[adapter_name] = obj

_discover_adapters()


async def run_chain(chain_input: str, chain_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ 
    Runs a chat interaction by dispatching to the appropriate framework adapter.
    """
    if not chain_input:
        return {"status": "error", "error_message": "Missing input."}
    if not chain_configs:
        return {"status": "error", "error_message": "Missing chain configuration."}

    current_input = chain_input
    final_payload = {}
    session_id = str(uuid.uuid4()) # A unique session for the entire chain execution

    for i, agent_config in enumerate(chain_configs):
        framework = agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
        adapter_class = _ADAPTERS.get(framework)

        if not adapter_class:
            return {"status": "error", "error_message": f"Framework '{framework}' not supported."}

        try:
            # Add session_id to the config for the adapter to use
            agent_config["session_id"] = session_id
            adapter_instance = adapter_class(agent_config=agent_config)
            
            # The prompt for the arun method is the output of the previous step
            payload = await adapter_instance.arun(prompt=current_input, session_id=session_id)
            
            # The input for the next agent is the content from the current one
            current_input = payload.get("content", "")
            final_payload = payload # Store the last payload

        except Exception as e: # pylint: disable=broad-exception-caught
            return {
                "status": "error",
                "error_message": f"Error in chain step {i+1} with framework '{framework}': {e}",
            }

    return {"status": "success", "payload": final_payload}


async def run_chat(prompt: str, session_id: str, framework: str = "agno") -> Dict[str, Any]:
    """
    Runs a single chat interaction using the old session-based config.
    This is kept for backward compatibility.
    """
    if not prompt:
        return {"status": "error", "error_message": "Missing prompt."}

    adapter_class = _ADAPTERS.get(framework)
    if not adapter_class:
        return {"status": "error", "error_message": f"Framework '{framework}' not supported."}

    try:
        # Get static config from file, as this is the old flow
        agent_config = get_session_config(session_id)
        if not agent_config:
            # Provide a default config if none is found
            agent_config = {"backend": "ollama", "model": "gemma2:9b"}

        adapter_instance = adapter_class(agent_config=agent_config)
        payload = await adapter_instance.arun(prompt=prompt, session_id=session_id)
        return {"status": "success", "payload": payload}
    except Exception as e: # pylint: disable=broad-exception-caught
        return {"status": "error", "error_message": f"Error running framework '{framework}': {e}"}
