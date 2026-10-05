"""REST transport for bounded local inference tasks and contracts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..dependencies import get_capability_service
from ..inference.capabilities import BOUNDED_TASK_CAPABILITY
from ..inference.tasks import (
    BOUNDED_TASK_CONTRACTS,
    BoundedTaskKind,
)
from ..kernel.records import (
    CapabilityRequest,
    DataClass,
    ProducerIdentity,
    _new_id,
    _utc_now,
)
from ..kernel.service import CapabilityService
from ..kernel.transport import InvocationEnvelope, invoke_envelope

router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])


class BoundedTaskContractInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    version: str
    instruction: str
    minimum_confidence: float
    max_input_tokens: int
    max_output_tokens: int
    max_latency_seconds: float
    max_ram_bytes: int
    max_vram_bytes: int
    output_schema: dict[str, Any]


class BoundedTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: BoundedTaskKind
    objective: str = Field(min_length=1)
    content: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = None
    data_class: DataClass | None = None
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_latency_seconds: float | None = Field(default=None, ge=0.1)


@router.get("/contracts")
async def list_contracts() -> dict[str, list[BoundedTaskContractInfo]]:
    """List supported bounded task contracts and their strict schema definitions."""
    contracts = [
        BoundedTaskContractInfo(
            kind=kind.value,
            version=contract.version,
            instruction=contract.instruction,
            minimum_confidence=contract.failure_policy.minimum_confidence,
            max_input_tokens=contract.limits.max_input_tokens,
            max_output_tokens=contract.limits.max_output_tokens,
            max_latency_seconds=contract.limits.max_latency_seconds,
            max_ram_bytes=contract.limits.max_ram_bytes,
            max_vram_bytes=contract.limits.max_vram_bytes,
            output_schema=contract.output_model.model_json_schema(),
        )
        for kind, contract in BOUNDED_TASK_CONTRACTS.items()
    ]
    return {"contracts": contracts}


@router.post("/tasks", response_model=InvocationEnvelope)
async def dispatch_task(
    payload: BoundedTaskRequest,
    service: CapabilityService = Depends(get_capability_service),
) -> InvocationEnvelope:
    """Dispatch a schema-constrained bounded local inference task through capability governance."""
    descriptor = service.registry.descriptor(BOUNDED_TASK_CAPABILITY)
    if descriptor is None:
        raise HTTPException(
            status_code=503,
            detail="Bounded inference capability is not registered",
        )

    actor = ProducerIdentity(
        producer_id="rai.api.inference", kind="api", version="1.0.0"
    )
    correlation_id = f"corr:{_new_id()}"
    request_id = f"req:{_new_id()}"

    args: dict[str, Any] = {
        "kind": payload.kind.value,
        "objective": payload.objective,
        "content": payload.content,
    }
    if payload.task_id:
        args["task_id"] = payload.task_id
    if payload.data_class:
        args["data_class"] = payload.data_class.value
    if payload.max_input_tokens is not None:
        args["max_input_tokens"] = payload.max_input_tokens
    if payload.max_output_tokens is not None:
        args["max_output_tokens"] = payload.max_output_tokens
    if payload.max_latency_seconds is not None:
        args["max_latency_seconds"] = payload.max_latency_seconds

    request = CapabilityRequest(
        record_id=request_id,
        timestamp=_utc_now(),
        producer=actor,
        actor=actor,
        correlation_id=correlation_id,
        capability=BOUNDED_TASK_CAPABILITY,
        arguments=args,
        data_class=payload.data_class or DataClass.LOCAL,
        target_resource=f"capability://{BOUNDED_TASK_CAPABILITY}",
        requested_side_effects=descriptor.side_effects,
        isolation=descriptor.isolation,
        verification_plan=descriptor.verification_plan,
    )

    return await invoke_envelope(service, request)
