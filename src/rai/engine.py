"""
The core engine for the RAI platform.

This module discovers and manages framework adapters, and provides the
main entry point for running chat interactions.
"""
import pkgutil
import inspect
import importlib
from typing import Any, Dict, Type

from . import adapters
from .adapters.base import BaseAdapter

# A dictionary to hold discovered adapter classes
_ADAPTERS: Dict[str, Type[BaseAdapter]] = {}

def _discover_adapters() -> None:
    """Dynamically discovers and loads adapter classes."""
    if _ADAPTERS:
        return

    # Iterate through the modules in the adapters package
    for module_info in pkgutil.iter_modules(adapters.__path__, adapters.__name__ + "."):
        if module_info.name.endswith('.base') or module_info.name.endswith('.__init__'):
            continue

        # Dynamically import the module
        module = importlib.import_module(module_info.name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                # Use the last part of the module name as the adapter key
                adapter_name = module_info.name.split('.')[-1]
                _ADAPTERS[adapter_name] = obj

_discover_adapters()


async def run_chat(prompt: str, session_id: str, framework: str = "agno") -> Dict[str, Any]:
    """
    Runs a chat interaction by dispatching to the appropriate framework adapter.
    """
    if not prompt:
        return {"status": "error", "error_message": "Missing prompt."}

    adapter_class = _ADAPTERS.get(framework)
    if not adapter_class:
        return {"status": "error", "error_message": f"Framework '{framework}' not supported."}

    # TODO: fix dangerous broad-exception-caught warning i this block
    try:
        adapter_instance = adapter_class()
        payload = await adapter_instance.arun(prompt=prompt, session_id=session_id)
        return {"status": "success", "payload": payload}
    except Exception as e: # pylint: disable=broad-exception-caught
        # Basic error handling for now
        return {"status": "error", "error_message": f"Error running framework '{framework}': {e}"}
