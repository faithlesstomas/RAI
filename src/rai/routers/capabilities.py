"""REST transport for the shared capability registry and invocation service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_capability_service
from ..kernel.records import CapabilityRequest
from ..kernel.service import CapabilityService
from ..kernel.transport import InvocationEnvelope, invoke_envelope

router = APIRouter(prefix="/api/v1/capabilities", tags=["Capabilities"])


@router.get("")
async def list_capabilities(
    service: CapabilityService = Depends(get_capability_service),
) -> dict[str, object]:
    return {
        "capabilities": [
            descriptor.model_dump(mode="json")
            for descriptor in service.registry.descriptors()
        ]
    }


@router.post("/invoke", response_model=InvocationEnvelope)
async def invoke_capability(
    request: CapabilityRequest,
    service: CapabilityService = Depends(get_capability_service),
) -> InvocationEnvelope:
    return await invoke_envelope(service, request)
