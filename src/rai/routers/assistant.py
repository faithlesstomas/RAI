"""HTTP transport for the RAI-owned graph-memory assistant."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from returns.result import Failure

from rai.assistant.records import AssistantSessionId, ConversationTurn
from rai.assistant.service import AssistantService
from rai.dependencies import get_assistant_service
from rai.kernel.records import DataClass, ProducerIdentity, _new_id

router = APIRouter(prefix="/api/v1/assistant", tags=["Rich Assistant"])


class AssistantTurnRequest(BaseModel):
    """One provider-neutral assistant turn."""

    prompt: str = Field(min_length=1, max_length=16 * 1024)
    session_id: AssistantSessionId | None = None
    request_id: str | None = Field(default=None, min_length=1)
    reply_to_turn_id: str | None = None
    data_class: DataClass = DataClass.LOCAL


class AssistantTurnResult(BaseModel):
    """Terminal assistant response with reproducibility references."""

    session_id: AssistantSessionId
    request_id: str
    turn_id: str
    content: str
    status: str
    manifest_id: str
    admitted_memory_ids: tuple[str, ...] = ()


def _turn_from_request(request: AssistantTurnRequest) -> ConversationTurn:
    return ConversationTurn(
        record_id=_new_id(),
        producer=ProducerIdentity(
            producer_id="assistant-http", kind="user", version="1.0.0"
        ),
        session_id=request.session_id or _new_id(),
        role="user",
        text=request.prompt,
        reply_to_turn_id=request.reply_to_turn_id,
        data_class=request.data_class,
    )


@router.post("/turn", response_model=AssistantTurnResult)
async def accept_assistant_turn(
    request: AssistantTurnRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> AssistantTurnResult:
    """Run one bounded turn reconstructed from RAI-owned context and memory."""
    turn = _turn_from_request(request)
    result = await service.accept_turn(turn, request_id=request.request_id)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=503 if failure.retryable else 422,
            detail={"code": failure.code, "message": failure.message},
        )
    response = result.unwrap()
    return AssistantTurnResult(
        session_id=response.session_id,
        request_id=response.request_id,
        turn_id=response.turn_id,
        content=response.text,
        status=response.status,
        manifest_id=response.manifest_id,
        admitted_memory_ids=response.admitted_memory_ids,
    )


@router.post("/stream")
async def stream_assistant_turn(
    request: AssistantTurnRequest,
    service: AssistantService = Depends(get_assistant_service),
) -> StreamingResponse:
    """Emit validated assistant output as SSE without persisting partial chunks."""
    turn = _turn_from_request(request)

    async def events() -> AsyncGenerator[str, None]:
        async for chunk in service.accept_turn_stream(
            turn, request_id=request.request_id
        ):
            if isinstance(chunk, Failure):
                failure = chunk.failure()
                payload = {"code": failure.code, "message": failure.message}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                return
            yield f"data: {json.dumps({'content': chunk.unwrap()})}\n\n"
        yield f"event: done\ndata: {json.dumps({'session_id': turn.session_id})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/manifests/{manifest_id}")
async def get_context_manifest(
    manifest_id: str,
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, object]:
    """Return the inspectable context selection record for one response."""
    result = await service.store.get_manifest(manifest_id)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    manifest = result.unwrap()
    if manifest is None:
        raise HTTPException(status_code=404, detail="Context manifest not found")
    return manifest.model_dump(mode="json")
