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
from typing import Any, AsyncIterator, Dict, List, Optional, Type

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


async def run_chain( # noqa: PLR0912
    chain_input: str,
    chain_configs: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    app_config: Optional[Dict[str, Any]] = None, # Added for consistency, not used yet
    context: Optional[Dict[str, Any]] = None,
) -> Result[Dict[str, Any], Exception]: # noqa: PLR0912
    # pylint: disable=too-many-branches, too-many-locals
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

        # Use provided session_id or generate a new one
        if not session_id:
            session_id = str(uuid.uuid4())

        for _, agent_config in enumerate(chain_configs):
            framework = (
                agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
            )
            adapter_class = rai_adapters.get(framework)

            # Infer backend if not provided
            if "backend" not in agent_config:
                if "model" in agent_config:
                    model = agent_config["model"]
                    if model.startswith("gemini"):
                        agent_config["backend"] = "gemini"
                    if model.startswith("claude"):
                        agent_config["backend"] = "anthropic"
                    if model.startswith("gpt"):
                        agent_config["backend"] = "openai"

                    # Ensure default backend is set if still missing
                    if "backend" not in agent_config:
                        agent_config["backend"] = "ollama"

            if not adapter_class:
                return Failure(
                    AdapterNotFoundError(f"Framework '{framework}' not supported.")
                )

            # Add session_id to the config for the adapter to use
            agent_config["session_id"] = session_id
            if context:
                agent_config["context"] = context
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


async def stream_chain( # noqa: PLR0912, PLR0915, PLR0914
    chain_input: str,
    chain_configs: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    app_config: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Any]: # noqa: PLR0912, PLR0915, PLR0914
    # pylint: disable=too-many-branches, too-many-statements, too-many-locals
    """
    Streams the execution of a chain of agents.
    Only the last agent's response is streamed.
    """
    _ = app_config
    try:
        if not chain_input:
            yield Failure(ValueError("Missing input."))
            return
        if not chain_configs:
            yield Failure(ValueError("Missing chain configuration."))
            return

        rai_adapters = _discover_adapters()
        current_input = chain_input

        if not session_id:
            session_id = str(uuid.uuid4())

        # Execute all agents except the last one normally
        for _, agent_config in enumerate(chain_configs[:-1]):
            framework = (
                agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
            )
            adapter_class = rai_adapters.get(framework)

            # Infer backend if not provided
            if "backend" not in agent_config:
                if "model" in agent_config:
                    model = agent_config["model"]
                    if model.startswith("gemini"):
                        agent_config["backend"] = "gemini"
                    if model.startswith("claude"):
                        agent_config["backend"] = "anthropic"
                    if model.startswith("gpt"):
                        agent_config["backend"] = "openai"

                    # Ensure default backend is set if still missing
                    if "backend" not in agent_config:
                        agent_config["backend"] = "ollama"

            if not adapter_class:
                yield Failure(
                    AdapterNotFoundError(f"Framework '{framework}' not supported.")
                )
                return

            agent_config["session_id"] = session_id
            if context:
                agent_config["context"] = context
            adapter_instance = adapter_class(agent_config=agent_config)

            result = await adapter_instance.arun(prompt=current_input)

            if isinstance(result, Failure):
                yield result
                return

            payload = result.unwrap()
            current_input = payload.get("content", "")

        # Stream the last agent
        last_config = chain_configs[-1]
        framework = (
            last_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()
        )
        adapter_class = rai_adapters.get(framework)

        # Infer backend if not provided
        if "backend" not in chain_configs[-1]:
            if "model" in chain_configs[-1]:
                model = chain_configs[-1]["model"]
                if model.startswith("gemini"):
                    chain_configs[-1]["backend"] = "gemini"
                if model.startswith("claude"):
                    chain_configs[-1]["backend"] = "anthropic"
                if model.startswith("gpt"):
                    chain_configs[-1]["backend"] = "openai"

                    if "backend" not in chain_configs[-1]:
                        chain_configs[-1]["backend"] = "ollama"

        if not adapter_class:
            yield Failure(
                AdapterNotFoundError(f"Framework '{framework}' not supported.")
            )
            return

        last_config["session_id"] = session_id
        if context:
            last_config["context"] = context
        # Ensure stream is enabled in config for the adapter
        last_config["stream"] = True
        adapter_instance = adapter_class(agent_config=last_config)

        async for chunk in adapter_instance.astream(prompt=current_input):
            yield chunk

    except Exception as e: # pylint: disable=broad-exception-caught
        yield Failure(ChainExecutionError(f"An error occurred during chain streaming: {e}"))


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
