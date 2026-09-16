"""HTTP transport for the RAI-owned graph-memory assistant."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from returns.result import Failure, Success

from rai.assistant.diagnostics import diagnose_memory
from rai.assistant.records import (
    AssistantResponse,
    AssistantSessionId,
    ConversationTurn,
)
from rai.assistant.service import AssistantService
from rai.dependencies import get_assistant_service
from rai.kernel.records import DataClass, ProducerIdentity, _new_id
from rai.kernel.ports import LifecycleState
from rai.tools.security.auth import is_authorized

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
    memory_operation_ids: tuple[str, ...] = ()


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
    return _turn_result(response)


def _turn_result(response: AssistantResponse) -> AssistantTurnResult:
    return AssistantTurnResult(
        session_id=response.session_id,
        request_id=response.request_id,
        turn_id=response.turn_id,
        content=response.text,
        status=response.status,
        manifest_id=response.manifest_id,
        admitted_memory_ids=response.admitted_memory_ids,
        memory_operation_ids=response.memory_operation_ids,
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


@router.get("/contexts/{manifest_id}")
async def get_context_window(
    manifest_id: str,
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, object]:
    """Return the exact bounded context package used for a response."""
    result = await service.store.get_context_package(manifest_id)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    package = result.unwrap()
    if package is None:
        raise HTTPException(status_code=404, detail="Context package not found")
    return package.model_dump(mode="json")


@router.get("/sessions")
async def list_assistant_sessions(
    limit: int = Query(default=20, ge=1, le=200),
    service: AssistantService = Depends(get_assistant_service),
) -> list[dict[str, object]]:
    """List persisted sessions so clients can resume a conversation."""
    result = await service.store.list_sessions(limit=limit)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    return [
        {
            "session_id": item.session_id,
            "started_at": item.started_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "turn_count": item.turn_count,
            "last_role": item.last_role,
            "preview": item.preview,
        }
        for item in result.unwrap()
    ]


@router.get("/sessions/{session_id}/turns")
async def get_chat_history(
    session_id: AssistantSessionId,
    limit: int = Query(default=20, ge=1, le=200),
    service: AssistantService = Depends(get_assistant_service),
) -> list[dict[str, object]]:
    """Return the bounded persisted user/assistant history for one session."""
    result = await service.get_recent_turns(session_id, limit=limit)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    return [turn.model_dump(mode="json") for turn in result.unwrap()]


@router.get("/sessions/{session_id}/context")
async def get_latest_session_context(
    session_id: AssistantSessionId,
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, object]:
    """Return the latest exact context package for a session."""
    manifest_result = await service.store.get_latest_manifest_for_session(session_id)
    if isinstance(manifest_result, Failure):
        failure = manifest_result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    manifest = manifest_result.unwrap()
    if manifest is None:
        raise HTTPException(status_code=404, detail="Context manifest not found")
    return await get_context_window(manifest.record_id, service)


@router.get("/memories")
async def list_assistant_memories(
    limit: int = Query(default=20, ge=1, le=200),
    service: AssistantService = Depends(get_assistant_service),
) -> list[dict[str, object]]:
    """List active memories in the authenticated assistant profile."""
    result = await service.store.retrieve_relevant_memories(
        profile_scope=service.profile_scope,
        data_classes=(DataClass.PUBLIC, DataClass.LOCAL, DataClass.PRIVATE),
        limit=limit,
    )
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    return [
        {**memory.model_dump(mode="json"), "ranking_reason": reason}
        for memory, reason in result.unwrap()
    ]


@router.get("/memory-operations")
async def list_memory_operations(
    limit: int = Query(default=20, ge=1, le=200),
    service: AssistantService = Depends(get_assistant_service),
) -> list[dict[str, object]]:
    """Return the append-only memory mutation and admission trace."""
    result = await service.store.list_memory_operations(
        profile_scope=service.profile_scope, limit=limit
    )
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    return [operation.model_dump(mode="json") for operation in result.unwrap()]


@router.get("/diagnostics/memory")
async def get_memory_diagnostics(
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, object]:
    """Return stage-specific integrity evidence for the active memory profile."""
    result = await diagnose_memory(service.store, profile_scope=service.profile_scope)
    if isinstance(result, Failure):
        failure = result.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    return result.unwrap().model_dump(mode="json")


@router.websocket("/ws")
async def assistant_websocket(websocket: WebSocket) -> None:  # noqa: PLR0912
    """Serve stateful native assistant turns over one authenticated WebSocket."""
    if not is_authorized(websocket.headers, websocket.query_params.get("token")):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    await websocket.accept()
    default_session_id = websocket.query_params.get("session_id") or _new_id()
    try:
        service = websocket.app.state.container.assistant_service
        if service.state != LifecycleState.RUNNING:
            start_result = await service.start()
            if isinstance(start_result, Failure):
                failure = start_result.failure()
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {"code": failure.code, "message": failure.message},
                    }
                )
                await websocket.close(code=1011)
                return
        await websocket.send_json({"type": "session", "session_id": default_session_id})
        reply_by_session: dict[str, str] = {}
        while True:
            try:
                payload = await websocket.receive_json()
                request = AssistantTurnRequest.model_validate(payload)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {"code": "INVALID_REQUEST", "message": str(exc)},
                    }
                )
                continue

            session_id = request.session_id or default_session_id
            reply_to = request.reply_to_turn_id or reply_by_session.get(session_id)
            if reply_to is None:
                recent = await service.get_recent_turns(session_id, limit=1)
                if isinstance(recent, Success) and recent.unwrap():
                    reply_to = recent.unwrap()[-1].record_id
            turn_request = request.model_copy(
                update={"session_id": session_id, "reply_to_turn_id": reply_to}
            )
            turn = _turn_from_request(turn_request)
            result = await service.accept_turn(turn, request_id=request.request_id)
            if isinstance(result, Failure):
                failure = result.failure()
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {"code": failure.code, "message": failure.message},
                    }
                )
                continue
            response = result.unwrap()
            reply_by_session[session_id] = response.turn_id
            await websocket.send_json(
                {
                    "type": "response",
                    "payload": _turn_result(response).model_dump(mode="json"),
                }
            )
    except WebSocketDisconnect:
        return
