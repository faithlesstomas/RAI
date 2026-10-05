"""API endpoints for runtime execution, approvals, and models."""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..dependencies import get_model_registry
from ..services.model_registry import ModelRegistry
from ..tools.security.hitl import get_approval_manager

router = APIRouter(
    tags=["Execution and Models"],
    responses={404: {"description": "Not found"}},
)


class ApprovalResolutionRequest(BaseModel):
    """Request model to resolve a pending HITL authorization request."""

    approved: bool


@router.get("/api/v1/approvals")
async def list_approvals() -> JSONResponse:
    """Lists all pending HITL tool execution approval requests."""
    manager = get_approval_manager()
    return JSONResponse(content={"approvals": manager.list_pending()})


@router.post("/api/v1/approvals/{approval_id}/resolve")
async def resolve_approval(
    approval_id: str,
    request: ApprovalResolutionRequest,
) -> JSONResponse:
    """Resolves a pending HITL authorization request (approves or denies execution)."""
    manager = get_approval_manager()
    success = manager.resolve_request(approval_id, request.approved)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Pending approval request '{approval_id}' not found or already resolved.",
        )
    return JSONResponse(
        content={
            "status": "success",
            "message": f"Approval '{approval_id}' resolved to: {request.approved}",
        }
    )


@router.get("/api/v1/models")
async def get_all_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> JSONResponse:
    """Returns a list of available models for all backends."""
    models = await registry.get_all_models()
    return JSONResponse(content={"models": models})


@router.get("/api/v1/models/{backend}")
async def get_models_for_backend(
    backend: str,
    registry: ModelRegistry = Depends(get_model_registry),
) -> JSONResponse:
    """Returns a list of available models for a given backend."""
    logging.info("Fetching models for backend: %s", backend)
    models = await registry.get_models(backend)

    if not models and backend != "ollama":
        return JSONResponse(content={"models": ["default-model"]})

    return JSONResponse(content={"models": models})
