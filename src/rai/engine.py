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

from returns.result import Failure, Result, Success, safe
from returns.pipeline import flow
from returns.pointfree import bind

from . import adapters
from .adapters.base import Processor
from .exceptions import AdapterNotFoundError, ChainExecutionError, AgentConfigError


@functools.lru_cache(maxsize=None)
def _discover_adapters() -> Dict[str, Type[Processor]]:
    """Dynamically discovers and loads adapter classes."""
    discovered_adapters: Dict[str, Type[Processor]] = {}
    for module_info in pkgutil.iter_modules(adapters.__path__, adapters.__name__ + "."):
        if module_info.name.endswith(".base") or module_info.name.endswith(".__init__"):
            continue

        module = importlib.import_module(module_info.name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Processor) and not inspect.isabstract(obj):
                adapter_name = module_info.name.split(".")[-1]
                discovered_adapters[adapter_name] = obj
    return discovered_adapters


def _validate_input(chain_input: str, chain_configs: List[Dict[str, Any]]) -> Result[str, Exception]:
    if not chain_input:
        return Failure(ValueError("Missing input."))
    if not chain_configs:
        return Failure(ValueError("Missing chain configuration."))
    return Success(chain_input)


def _infer_backend(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """Infers the backend based on the model name if not explicitly provided."""
    if "backend" not in agent_config:
        model = agent_config.get("model", "")
        if model.startswith("gemini"):
            agent_config["backend"] = "gemini"
        elif model.startswith("claude"):
            agent_config["backend"] = "anthropic"
        elif model.startswith("gpt"):
            agent_config["backend"] = "openai"
        else:
            agent_config["backend"] = "ollama"
    return agent_config


def _get_adapter_class(framework: str) -> Result[Type[Processor], Exception]:
    rai_adapters = _discover_adapters()
    adapter_class = rai_adapters.get(framework)
    if not adapter_class:
        return Failure(AdapterNotFoundError(f"Framework '{framework}' not supported."))
    return Success(adapter_class)


async def _execute_step(
    current_input: str,
    agent_config: Dict[str, Any],
    session_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> Result[Dict[str, Any], Exception]:
    """Executes a single step in the chain."""
    framework = agent_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()

    # 1. Get Adapter Class
    adapter_class_result = _get_adapter_class(framework)
    if isinstance(adapter_class_result, Failure):
        return adapter_class_result

    adapter_class = adapter_class_result.unwrap()

    # 2. Prepare Config
    config = _infer_backend(agent_config.copy())
    config["session_id"] = session_id
    if context:
        config["context"] = context

    # 3. Instantiate and Run
    try:
        adapter_instance = adapter_class(agent_config=config)
        return await adapter_instance.arun(prompt=current_input)
    except Exception as e:
        return Failure(ChainExecutionError(f"Error executing step with framework {framework}: {e}"))


async def run_chain(
    chain_input: str,
    chain_configs: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    app_config: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Result[Dict[str, Any], Exception]:
    """
    Runs a chat interaction by dispatching to the appropriate framework adapter.
    """
    _ = app_config

    # Validation
    validation = _validate_input(chain_input, chain_configs)
    if isinstance(validation, Failure):
        return validation

    final_session_id = session_id or str(uuid.uuid4())
    current_input = chain_input
    final_payload: Dict[str, Any] = {}

    for agent_config in chain_configs:
        result = await _execute_step(current_input, agent_config, final_session_id, context)

        if isinstance(result, Failure):
            return result

        final_payload = result.unwrap()
        current_input = final_payload.get("content", "")

    return Success(final_payload)


async def stream_chain(
    chain_input: str,
    chain_configs: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    app_config: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Any]:
    """
    Streams the execution of a chain of agents.
    Only the last agent's response is streamed.
    """
    _ = app_config

    validation = _validate_input(chain_input, chain_configs)
    if isinstance(validation, Failure):
        yield validation
        return

    final_session_id = session_id or str(uuid.uuid4())
    current_input = chain_input

    # Execute all previous steps
    for agent_config in chain_configs[:-1]:
        result = await _execute_step(current_input, agent_config, final_session_id, context)
        if isinstance(result, Failure):
            yield result
            return
        current_input = result.unwrap().get("content", "")

    # Stream the last step
    last_config = chain_configs[-1]
    framework = last_config.get("agent_class", "AgentAgno").replace("Agent", "").lower()

    adapter_class_result = _get_adapter_class(framework)
    if isinstance(adapter_class_result, Failure):
        yield adapter_class_result
        return

    adapter_class = adapter_class_result.unwrap()
    config = _infer_backend(last_config.copy())
    config["session_id"] = final_session_id
    if context:
        config["context"] = context
    config["stream"] = True

    try:
        adapter_instance = adapter_class(agent_config=config)
        async for chunk in adapter_instance.astream(prompt=current_input):
            yield chunk
    except Exception as e:
        yield Failure(ChainExecutionError(f"Error during streaming: {e}"))


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
    if not prompt:
        return Failure(ValueError("Missing prompt."))

    adapter_class_result = _get_adapter_class(framework)
    if isinstance(adapter_class_result, Failure):
        return adapter_class_result

    adapter_class = adapter_class_result.unwrap()

    # Get static config from the passed app_config
    # TODO: This logic will need to be updated when separating agents from sessions
    agent_config = app_config.get("sessions", {}).get(session_id)
    if not agent_config:
        # Provide a default config if none is found
        agent_config = {"backend": "ollama", "model": "gemma2:9b"}

    # Ensure backend is inferred
    agent_config = _infer_backend(agent_config)

    try:
        adapter_instance = adapter_class(agent_config=agent_config)
        # Note: run_chat implies single turn or session-based memory handled by adapter
        return await adapter_instance.arun(prompt=prompt, session_id=session_id)
    except Exception as e:
        return Failure(ChainExecutionError(f"An error occurred during chat execution: {e}"))
