"""HTTP transport tests for the RAI-owned assistant path."""

from __future__ import annotations

from pathlib import Path

import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.routers.assistant import (
    AssistantTurnRequest,
    accept_assistant_turn,
    get_chat_history,
    get_context_manifest,
    get_context_window,
    get_latest_session_context,
    list_assistant_memories,
    list_assistant_sessions,
    list_memory_operations,
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
