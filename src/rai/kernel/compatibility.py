"""Policy-preserving wrappers for legacy backend tool-call signatures."""

from __future__ import annotations

import inspect
import json
from functools import wraps
from typing import Any

from .defaults import DEFAULT_ACTOR
from .records import ProducerIdentity
from .service import CapabilityService
from .transport import invoke_envelope, normalize_request


def policy_wrapped_handlers(
    service: CapabilityService,
    group: str,
    actor: ProducerIdentity = DEFAULT_ACTOR,
) -> tuple[Any, ...]:
    """Expose familiar call signatures while retaining registry and policy authority."""
    wrapped = []
    for capability in service.registry.compatibility_capabilities(group):
        raw = capability.compatibility_handler
        if raw is None:
            continue

        @wraps(raw)
        async def invoke(*args: Any, __capability: Any = capability, __raw: Any = raw, **kwargs: Any) -> str:  # noqa: ANN401
            bound = inspect.signature(__raw).bind(*args, **kwargs)
            allowed = set(__capability.descriptor.input_schema.get("properties", {}))
            unknown = set(bound.arguments) - allowed
            if unknown:
                return f"Execution Error: INVALID_ARGUMENT: unknown argument: {sorted(unknown)[0]}"
            bound.apply_defaults()
            arguments = {
                name: value for name, value in bound.arguments.items() if name in allowed
            }
            request = normalize_request(
                __capability.descriptor,
                arguments,
                actor=actor,
                target_resource=f"capability://{__capability.name}",
            )
            envelope = await invoke_envelope(service, request)
            if not envelope.ok:
                failure = envelope.result
                return f"Execution Error: {failure.code}: {failure.message}"
            output = envelope.result.output
            return str(output.get("text", json.dumps(output, sort_keys=True)))

        wrapped.append(invoke)
    return tuple(wrapped)
