"""Transport-neutral operations used by Stage 1 CLI commands."""

from __future__ import annotations

from . import config_manager
from .container import ApplicationContainer
from .kernel.records import CapabilityRequest
from .kernel.transport import InvocationEnvelope, invoke_envelope


async def invoke_local(
    request: CapabilityRequest, container: ApplicationContainer | None = None
) -> InvocationEnvelope:
    """Invoke through an explicit local container, sharing REST/MCP semantics."""
    owned = container is None
    active = container or ApplicationContainer(config_manager.load_config())
    try:
        return await invoke_envelope(active.capability_service, request)
    finally:
        if owned:
            await active.close()


def list_local_capabilities(
    container: ApplicationContainer | None = None,
) -> list[dict[str, object]]:
    active = container or ApplicationContainer(config_manager.load_config())
    return [
        descriptor.model_dump(mode="json")
        for descriptor in active.capability_registry.descriptors()
    ]
