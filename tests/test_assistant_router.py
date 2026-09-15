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
    get_context_manifest,
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
