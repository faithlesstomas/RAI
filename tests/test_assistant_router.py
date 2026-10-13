"""HTTP transport tests for the RAI-owned assistant path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.routers.assistant import (
    AssistantTurnRequest,
    REPLExecuteRequest,
    accept_assistant_turn,
    execute_session_repl,
    export_session_notebook,
    get_chat_history,
    get_context_manifest,
    get_context_window,
    get_latest_session_context,
    get_session_repl_state,
    list_assistant_memories,
    list_assistant_sessions,
    list_memory_operations,
    reset_session_repl,
    stream_assistant_turn,
)


@pytest.fixture
def assistant_service(tmp_path: Path) -> AssistantService:
    return AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "assistant-http.sqlite3"),
        backend=DeterministicAssistantBackend(),
    )


@pytest.mark.asyncio
async def test_assistant_turn_and_manifest_transport(
    assistant_service: AssistantService,
) -> None:
    start = await assistant_service.start()
    assert isinstance(start, Success)

    response = await accept_assistant_turn(
        AssistantTurnRequest(
            prompt="Zapamiętaj, że w przykładach kodu preferuję Guile.",
            session_id="http-session",
            request_id="http-request",
        ),
        assistant_service,
    )
    manifest = await get_context_manifest(response.manifest_id, assistant_service)

    assert response.session_id == "http-session"
    assert response.status == "COMPLETED"
    assert response.admitted_memory_ids
    assert manifest["backend_name"] == "deterministic"


@pytest.mark.asyncio
async def test_assistant_sse_uses_same_memory_committing_pipeline(
    assistant_service: AssistantService,
) -> None:
    response = await stream_assistant_turn(
        AssistantTurnRequest(
            prompt="Zapamiętaj, że w przykładach kodu preferuję Guile.",
            session_id="stream-session",
        ),
        assistant_service,
    )

    chunks = [chunk async for chunk in response.body_iterator]
    payload = "".join(
        chunk.decode() if isinstance(chunk, bytes) else chunk for chunk in chunks
    )
    memories = await assistant_service.store.retrieve_relevant_memories()

    assert "event: done" in payload
    assert isinstance(memories, Success)
    assert memories.unwrap()


@pytest.mark.asyncio
async def test_assistant_inspection_api_exposes_user_visible_state(
    assistant_service: AssistantService,
) -> None:
    response = await accept_assistant_turn(
        AssistantTurnRequest(
            prompt="Mój ulubiony kolor jest zielony.",
            session_id="inspection-session",
            request_id="inspection-request",
        ),
        assistant_service,
    )

    memories = await list_assistant_memories(20, assistant_service)
    operations = await list_memory_operations(20, assistant_service)
    sessions = await list_assistant_sessions(20, assistant_service)
    history = await get_chat_history("inspection-session", 20, assistant_service)
    context = await get_context_window(response.manifest_id, assistant_service)
    latest_context = await get_latest_session_context(
        "inspection-session", assistant_service
    )

    assert memories[0]["content"]["value"] == "zielony"
    assert operations[0]["operation"] == "REMEMBER"
    assert sessions[0]["session_id"] == "inspection-session"
    assert [turn["role"] for turn in history] == ["user", "assistant"]
    assert context["manifest"]["record_id"] == response.manifest_id
    assert latest_context == context


@pytest.mark.asyncio
async def test_assistant_repl_and_notebook_endpoints(
    assistant_service: AssistantService,
) -> None:
    start = await assistant_service.start()
    assert isinstance(start, Success)
    session_id = "repl-test-session"

    # 1. Initially state is inactive
    init_state = await get_session_repl_state(session_id, assistant_service)
    assert init_state.has_active_state is False
    assert init_state.variables == {}

    # 2. Execute cell 1
    exec1 = await execute_session_repl(
        session_id,
        REPLExecuteRequest(code="x = 10\ny = 25\nx + y"),
        assistant_service,
    )
    assert exec1.status == "ok"
    assert exec1.output == "35"
    assert exec1.variables.get("x") == "int"
    assert exec1.variables.get("y") == "int"

    # 3. Execute cell 2 using state from cell 1
    exec2 = await execute_session_repl(
        session_id,
        REPLExecuteRequest(code="z = x * 2\nz"),
        assistant_service,
    )
    assert exec2.status == "ok"
    assert exec2.output == "20"
    assert exec2.variables.get("z") == "int"

    # 4. Check REPL state endpoint
    state = await get_session_repl_state(session_id, assistant_service)
    assert state.has_active_state is True
    assert state.variables.get("x") == "int"
    assert state.variables.get("z") == "int"

    # 5. Reset REPL session
    reset_res = await reset_session_repl(session_id, assistant_service)
    assert reset_res == {"status": "ok", "session_id": session_id}

    state_after_reset = await get_session_repl_state(session_id, assistant_service)
    assert state_after_reset.has_active_state is False

    # 6. Test notebook export
    # Accept a turn to populate session history
    await accept_assistant_turn(
        AssistantTurnRequest(
            prompt="Napisz skrypt liczący sumę:\n```python\na = 1\nb = 2\na + b\n```",
            session_id=session_id,
        ),
        assistant_service,
    )

    nb_resp = await export_session_notebook(session_id, assistant_service)
    assert nb_resp.media_type == "application/x-ipynb+json"
    assert f'filename="session_{session_id}.ipynb"' in nb_resp.headers["Content-Disposition"]

    nb_dict = json.loads(nb_resp.body.decode("utf-8"))
    assert nb_dict["nbformat"] == 4
    assert len(nb_dict["cells"]) >= 2

