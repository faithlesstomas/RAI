"""
Core components for agent setup and configuration management.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable, Union, TypedDict

from rich.console import Console

logger = logging.getLogger(__name__)

@runtime_checkable
class AgentResponse(Protocol):  # pylint: disable=too-few-public-methods
    """A protocol for the response object from an agent's arun method."""
    content: Union[str, List[object]]
    tool_calls: Optional[List[object]]

class ResponseDict(TypedDict, total=False):
    """A TypedDict for the dictionary response."""
    content: str
    tool_calls: List[object]

from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.compatibility import policy_wrapped_handlers
from rai.kernel.defaults import (
    HitlApprovalBroker,
    create_default_capability_registry,
    isolation_available,
)
from rai.kernel.audit import JsonlAuditLedger
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService

# --- Globals ---
console = Console(record=True)
error_console = Console(stderr=True)
active_status = None

# --- Tool and Model Setup ---
def setup_tools(
    enable_tools: bool,
    quiet: bool,
    enabled_tool_names: Optional[List[str]] = None,
    has_prompt: bool = False,
    capability_service: CapabilityService | None = None,
    capability_registry: CapabilityRegistry | None = None,
) -> Tuple[List[Any], List[str]]:
    """Sets up the tools for the agent with strict capability policy mediation."""
    messages = []
    agent_tools: List[Any] = []

    if not enable_tools:
        return agent_tools, messages

    active_service = capability_service or CapabilityService(
        capability_registry or create_default_capability_registry(),
        PolicyEngine(isolation_available=isolation_available),
        JsonlAuditLedger(),
        HitlApprovalBroker(),
    )
    active_registry = active_service.registry
    tools_to_enable = (
        enabled_tool_names
        if enabled_tool_names is not None
        else list(active_registry.compatibility_groups())
    )

    for tool_name in tools_to_enable:
        if tool_name == "GitlabTools" and not os.getenv("GITLAB_ACCESS_TOKEN"):
            if not quiet and not has_prompt:
                messages.append(
                    "[bold yellow]WARNING: Missing GITLAB_ACCESS_TOKEN env variable. "
                    "GitlabTools will be disabled![/bold yellow]"
                )
            continue

        secured_handlers = policy_wrapped_handlers(active_service, tool_name)
        if secured_handlers:
            agent_tools.extend(secured_handlers)
            logger.debug("%s successfully enabled.", tool_name)
            continue

        if not quiet:
            messages.append(
                f"[bold yellow]WARNING: Unknown tool '{tool_name}' "
                "specified in configuration. Skipping.[/bold yellow]"
            )

    return agent_tools, messages


def validate_model_env(
    backend: str, model_id: str, quiet: bool, ollama_host: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validates the environment for the specified backend and model.
    Returns a configuration dictionary for the adapter to use for instantiation,
    and a list of messages to display.
    """
    messages = []
    if backend == "gemini" and "GEMINI_API_KEY" in os.environ:
        if "GOOGLE_API_KEY" in os.environ and not quiet:
            messages.append(
                "[bold yellow]INFO: GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "Using GEMINI_API_KEY.[/bold yellow]"
            )
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    api_keys = {
        "gemini": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    supported_backends = ["ollama", "llama", "gemini", "anthropic", "openai", "groq"]

    if backend not in supported_backends:
        error_console.print(f"[bold red]ERROR: Unsupported backend '{backend}'.[/bold red]")
        sys.exit(1)

    if backend in api_keys and not os.getenv(api_keys[backend]):
        error_console.print(
            f"[bold red]ERROR: {api_keys[backend]} environment variable not set.[/bold red]"
        )
        sys.exit(1)

    config = {
        "backend": backend,
        "model_id": model_id,
        "api_key_env_var": api_keys.get(backend),
    }

    if backend == "ollama":
        config["ollama_host"] = ollama_host

    return config, messages
