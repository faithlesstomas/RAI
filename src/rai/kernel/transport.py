"""Shared request normalization and typed transport envelope."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from returns.result import Failure

from .capabilities import CapabilityDescriptor
from .defaults import DEFAULT_ACTOR
from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    PolicyDecision,
    ProducerIdentity,
)
from .service import CapabilityService


class InvocationEnvelope(BaseModel):
    """Identical result shape returned by CLI, REST and MCP adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    decision: PolicyDecision | None
    result: ActionResult | ActionFailure


def normalize_request(  # noqa: PLR0913
    descriptor: CapabilityDescriptor,
    arguments: dict[str, Any],
    *,
    actor: ProducerIdentity = DEFAULT_ACTOR,
    request_id: str | None = None,
    correlation_id: str | None = None,
    timestamp: datetime | None = None,
    data_class: DataClass = DataClass.LOCAL,
    target_resource: str | None = None,
) -> CapabilityRequest:
    values: dict[str, Any] = {
        "producer": actor,
        "actor": actor,
        "correlation_id": correlation_id,
        "capability": descriptor.name,
        "arguments": arguments,
        "data_class": data_class,
        "target_resource": target_resource or f"capability://{descriptor.name}",
        "requested_side_effects": descriptor.side_effects,
        "isolation": descriptor.isolation,
        "verification_plan": descriptor.verification_plan,
    }
    if request_id is not None:
        values["record_id"] = request_id
    if timestamp is not None:
        values["timestamp"] = timestamp
    return CapabilityRequest.model_validate(values)


async def invoke_envelope(
    service: CapabilityService, request: CapabilityRequest
) -> InvocationEnvelope:
    decision, result = await service.invoke(request)
    if isinstance(result, Failure):
        return InvocationEnvelope(ok=False, decision=decision, result=result.failure())
    return InvocationEnvelope(ok=True, decision=decision, result=result.unwrap())
