"""HTTP transport for the RAI-owned graph-memory assistant."""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from returns.result import Failure, Success

from rai.assistant.diagnostics import diagnose_memory
from rai.assistant.notebook import export_turns_to_notebook
from rai.assistant.records import (
    AssistantResponse,
    AssistantSessionId,
    ConversationTurn,
)
from rai.assistant.service import AssistantService
from rai.dependencies import get_assistant_service, get_monty_runner
from rai.kernel.records import DataClass, ProducerIdentity, _new_id
from rai.kernel.ports import LifecycleState
from rai.tools.security.auth import create_session_token, is_authorized
from rai.tools.security.monty import MontyRunner

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
    session_token: str | None = None
    request_id: str
    turn_id: str
    content: str
    status: str
    manifest_id: str
    admitted_memory_ids: tuple[str, ...] = ()
    memory_operation_ids: tuple[str, ...] = ()


class SessionTokenResult(BaseModel):
    """Authenticated session token descriptor."""

    session_id: str
    session_token: str


class REPLExecuteRequest(BaseModel):
    """Execution request inside a session's persistent Monty REPL."""

    code: str = Field(min_length=1, max_length=64 * 1024)


class REPLExecuteResult(BaseModel):
    """Result of evaluating a code cell in session REPL."""

    session_id: str
    output: str
    variables: dict[str, str] = Field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "ok"


class REPLStateResult(BaseModel):
    """Current variable and function inventory for a session's REPL."""

    session_id: str
    variables: dict[str, str] = Field(default_factory=dict)
    has_active_state: bool = False


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
        session_token=create_session_token(response.session_id),
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
        yield f"event: done\ndata: {json.dumps({'session_id': turn.session_id, 'session_token': create_session_token(turn.session_id)})}\n\n"

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


@router.get("/sessions/{session_id}/token", response_model=SessionTokenResult)
async def get_session_token(session_id: str) -> SessionTokenResult:
    """Return an authenticated session token for binding capabilities and MCP."""
    return SessionTokenResult(
        session_id=session_id,
        session_token=create_session_token(session_id),
    )


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


from rai.tools.python import get_session_repl_lock as _get_session_repl_lock


@router.post("/sessions/{session_id}/repl/execute", response_model=REPLExecuteResult)
async def execute_session_repl(
    session_id: str,
    request: REPLExecuteRequest,
    service: AssistantService = Depends(get_assistant_service),
    runner: MontyRunner = Depends(get_monty_runner),
) -> REPLExecuteResult:
    if not isinstance(runner, MontyRunner):
        from rai.tools.security.monty import get_monty_runner  # noqa: PLC0415

        runner = get_monty_runner()

    lock = await _get_session_repl_lock(service.profile_scope, session_id)
    async with lock:
        saved_blob_res = await service.store.get_repl_session(
            session_id, profile_scope=service.profile_scope
        )
        if isinstance(saved_blob_res, Failure):
            failure = saved_blob_res.failure()
            return REPLExecuteResult(
                session_id=session_id,
                output=f"State Load Error: {failure.message}",
                variables={},
                duration_ms=0.0,
                status="error",
            )
        session_blob = (
            saved_blob_res.unwrap()[0]
            if saved_blob_res.unwrap() is not None
            else None
        )

        start_t = time.perf_counter()
        eval_res = await asyncio.to_thread(
            runner.run, request.code, session_blob=session_blob
        )
        duration_ms = (time.perf_counter() - start_t) * 1000.0

        if isinstance(eval_res, Failure):
            return REPLExecuteResult(
                session_id=session_id,
                output=f"Execution error: {eval_res.failure()}",
                variables={},
                duration_ms=round(duration_ms, 2),
                status="error",
            )

        res = eval_res.unwrap()
        if res.session_blob:
            save_res = await service.store.save_repl_session(
                session_id,
                res.session_blob,
                res.variables,
                profile_scope=service.profile_scope,
            )
            if isinstance(save_res, Failure):
                failure = save_res.failure()
                return REPLExecuteResult(
                    session_id=session_id,
                    output=f"{res.output}\n\n[State Persistence Error: {failure.message}]"
                    if res.output
                    else f"[State Persistence Error: {failure.message}]",
                    variables=res.variables,
                    duration_ms=round(duration_ms, 2),
                    status="error",
                )

        return REPLExecuteResult(
            session_id=session_id,
            output=res.output,
            variables=res.variables,
            duration_ms=round(duration_ms, 2),
            status="ok",
        )


@router.get("/sessions/{session_id}/repl/state", response_model=REPLStateResult)
async def get_session_repl_state(
    session_id: str,
    service: AssistantService = Depends(get_assistant_service),
    runner: MontyRunner = Depends(get_monty_runner),
) -> REPLStateResult:
    """Inspect variables and functions present in session's persistent REPL."""
    if not isinstance(runner, MontyRunner):
        from rai.tools.security.monty import get_monty_runner  # noqa: PLC0415

        runner = get_monty_runner()

    saved_res = await service.store.get_repl_session(
        session_id, profile_scope=service.profile_scope
    )
    if isinstance(saved_res, Failure):
        failure = saved_res.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    session_data = saved_res.unwrap()
    if not session_data:
        return REPLStateResult(
            session_id=session_id,
            variables={},
            has_active_state=False,
        )

    session_blob, variables = session_data
    if not variables and session_blob:
        inspect_res = await asyncio.to_thread(runner.inspect_state, session_blob)
        if isinstance(inspect_res, Success):
            variables = inspect_res.unwrap()

    return REPLStateResult(
        session_id=session_id,
        variables=variables,
        has_active_state=True,
    )


@router.post("/sessions/{session_id}/repl/reset")
async def reset_session_repl(
    session_id: str,
    service: AssistantService = Depends(get_assistant_service),
) -> dict[str, str]:
    """Reset persistent Monty REPL state for the session."""
    lock = await _get_session_repl_lock(service.profile_scope, session_id)
    async with lock:
        clear_res = await service.store.clear_repl_session(
            session_id, profile_scope=service.profile_scope
        )
        if isinstance(clear_res, Failure):
            failure = clear_res.failure()
            raise HTTPException(
                status_code=500,
                detail={"code": failure.code, "message": failure.message},
            )
        return {"status": "ok", "session_id": session_id}


@router.get("/sessions/{session_id}/notebook.ipynb")
async def export_session_notebook(
    session_id: str,
    service: AssistantService = Depends(get_assistant_service),
) -> Response:
    """Export conversation turns and code cells of a session as a Jupyter notebook (.ipynb)."""
    turns_res = await service.get_recent_turns(session_id, limit=200)
    if isinstance(turns_res, Failure):
        failure = turns_res.failure()
        raise HTTPException(
            status_code=500,
            detail={"code": failure.code, "message": failure.message},
        )
    turns = list(turns_res.unwrap())
    notebook_dict = export_turns_to_notebook(
        session_id=session_id,
        turns=turns,
        profile_scope=service.context_builder.profile_scope,
    )
    content = json.dumps(notebook_dict, indent=2)
    return Response(
        content=content,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="session_{session_id}.ipynb"'
        },
    )


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
        await websocket.send_json(
            {
                "type": "session",
                "session_id": default_session_id,
                "session_token": create_session_token(default_session_id),
            }
        )
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
