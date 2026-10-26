"""REST transport for the shared capability registry and invocation service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

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
    http_request: Request,
    request: CapabilityRequest,
    service: CapabilityService = Depends(get_capability_service),
) -> InvocationEnvelope:
    trusted_session_id = (
        http_request.headers.get("x-session-id")
        or http_request.query_params.get("session_id")
        or ""
    )
    descriptor = service.registry.descriptor(request.capability)
    expects_session = bool(
        descriptor and "session_id" in descriptor.input_schema.get("properties", {})
    )
    if expects_session or "session_id" in request.arguments:
        arguments = dict(request.arguments)
        arguments["session_id"] = trusted_session_id
        request = request.model_copy(update={"arguments": arguments})
    return await invoke_envelope(service, request)
