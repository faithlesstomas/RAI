"""
The core engine for the RAI platform.

This module discovers and manages framework adapters, and provides the
main entry point for running chat interactions.
"""

import functools
import inspect

import importlib
import pkgutil
import uuid
from typing import Any, Dict, List, Optional, Type

from returns.result import Failure, Result, Success

from . import adapters
from .adapters.base import Processor
from .exceptions import AdapterNotFoundError, ChainExecutionError


@functools.lru_cache(maxsize=None)
def _discover_adapters() -> Dict[str, Type[Processor]]:
    """Dynamically discovers and loads adapter classes."""
    discovered_adapters: Dict[str, Type[Processor]] = {}
    for module_info in pkgutil.iter_modules(adapters.__path__, adapters.__name__ + "."):
        if module_info.name.endswith(".base") or module_info.name.endswith(".__init__"):
            continue

        module = importlib.import_module(module_info.name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Check if the class is a concrete implementation of the Processor protocol
            if issubclass(obj, Processor) and not inspect.isabstract(obj):
                adapter_name = module_info.name.split(".")[-1]
                discovered_adapters[adapter_name] = obj
    return discovered_adapters


async def run_chain(
    chain_input: str,
    chain_configs: List[Dict[str, Any]],
    app_config: Optional[Dict[str, Any]] = None, # Added for consistency, not used yet
) -> Result[Dict[str, Any], Exception]:
    """
    Runs a chat interaction by dispatching to the appropriate framework adapter.
    """
    _ = app_config # Unused for now, but here for future-proofing the API
    try:
        if not chain_input:
            return Failure(ValueError("Missing input."))
        if not chain_configs:
            return Failure(ValueError("Missing chain configuration."))

        rai_adapters = _discover_adapters()
        current_input = chain_input
        final_payload: Dict[str, Any] = {}
        session_id = str(uuid.uuid4())  # A unique session for the entire chain execution

        for _, agent_config in enumerate(chain_configs):
            framework = (
                agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
            )
            adapter_class = rai_adapters.get(framework)

            if not adapter_class:
                return Failure(
                    AdapterNotFoundError(f"Framework '{framework}' not supported.")
                )

            # Add session_id to the config for the adapter to use
            agent_config["session_id"] = session_id
            adapter_instance = adapter_class(agent_config=agent_config)

            # The prompt for the arun method is the output of the previous step
            result = await adapter_instance.arun(prompt=current_input)

            if isinstance(result, Failure):
                return result # Propagate the failure

            payload = result.unwrap()

            # The input for the next agent is the content from the current one
            current_input = payload.get("content", "")
            final_payload = payload  # Store the last payload

        return Success(final_payload)

    except Exception as e: # pylint: disable=broad-exception-caught
        return Failure(ChainExecutionError(f"An error occurred during chain execution: {e}"))


async def run_chat(
    prompt: str,
    session_id: str,
    app_config: Dict[str, Any],
    framework: str = "agno",
) -> Result[Dict[str, Any], Exception]:
    """
    Runs a single chat interaction using the old session-based config.
    This is kept for backward compatibility.
    """
    try:
        if not prompt:
            return Failure(ValueError("Missing prompt."))

        rai_adapters = _discover_adapters()
        adapter_class = rai_adapters.get(framework)
        if not adapter_class:
            return Failure(AdapterNotFoundError(f"Framework '{framework}' not supported."))

        # Get static config from the passed app_config
        agent_config = app_config.get("sessions", {}).get(session_id)
        if not agent_config:
            # Provide a default config if none is found
            agent_config = {"backend": "ollama", "model": "gemma2:9b"}

        adapter_instance = adapter_class(agent_config=agent_config)
        payload = await adapter_instance.arun(prompt=prompt, session_id=session_id)
        return Success(payload)

    except Exception as e: # pylint: disable=broad-exception-caught
        return Failure(ChainExecutionError(f"An error occurred during chat execution: {e}"))
