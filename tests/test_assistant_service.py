"""Tests for AssistantService: pipeline, failure modes, context manifest, and streaming."""

from __future__ import annotations

from pathlib import Path
import pytest
from returns.result import Failure, Success

from rai.assistant.audit import InMemoryAssistantAuditLedger
from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.records import ConversationTurn
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity

PRODUCER = ProducerIdentity(producer_id="service-test", kind="test", version="1.0.0")


@pytest.fixture
def service(tmp_path: Path) -> AssistantService:
    store = SQLiteMemoryGraphStore(path=tmp_path / "memory_graph.sqlite3")
    backend = DeterministicAssistantBackend()
    audit = InMemoryAssistantAuditLedger()
    return AssistantService(store=store, backend=backend, audit_ledger=audit)


@pytest.mark.asyncio
async def test_service_successful_turn_with_preference(service: AssistantService) -> None:
    turn = ConversationTurn(
        record_id="turn-pref-1",
        producer=PRODUCER,
        session_id="session-A",
        role="user",
        text="Zapamiętaj, że w przykładach kodu preferuję Guile zamiast Pythona.",
    )
    resp_res = await service.accept_turn(turn)
    assert isinstance(resp_res, Success)
    response = resp_res.unwrap()
    assert response.status == "COMPLETED"
    assert "Guile" in response.text
    assert len(response.admitted_memory_ids) == 1

    # Verify manifest was persisted
    man_res = await service.store.get_manifest(response.manifest_id)
    assert isinstance(man_res, Success)
    manifest = man_res.unwrap()
    assert manifest is not None
    assert manifest.session_id == "session-A"

    # Verify audit entry was written
    entries = await service.audit_ledger.list_for_session("session-A")
    assert len(entries) == 1
    assert entries[0].status == "COMPLETED"
    assert entries[0].turn_id == "turn-pref-1"


@pytest.mark.asyncio
async def test_service_idempotent_retry(service: AssistantService) -> None:
    turn = ConversationTurn(
        record_id="turn-idem-1",
        producer=PRODUCER,
        session_id="session-A",
        role="user",
        text="Hello assistant",
    )
    first_res = await service.accept_turn(turn, request_id="req-idem-1")
    assert isinstance(first_res, Success)
    first_resp = first_res.unwrap()

    # Second invocation with same request_id -> returns cached response
    second_res = await service.accept_turn(turn, request_id="req-idem-1")
    assert isinstance(second_res, Success)
    second_resp = second_res.unwrap()
    assert second_resp.record_id == first_resp.record_id
    assert second_resp.text == first_resp.text

    # Audit ledger should only have 1 entry
    entries = await service.audit_ledger.list_for_session("session-A")
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_service_backend_timeout_records_failed_terminal(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "memory_graph.sqlite3")
    backend = DeterministicAssistantBackend(fail_mode="timeout")
    audit = InMemoryAssistantAuditLedger()
    service = AssistantService(store=store, backend=backend, audit_ledger=audit)

    turn = ConversationTurn(
        record_id="turn-timeout-1",
        producer=PRODUCER,
        session_id="session-fail",
        role="user",
        text="A query that will time out",
    )
    res = await service.accept_turn(turn, request_id="req-timeout-1")
    assert isinstance(res, Failure)
    assert res.failure().code == "TIMEOUT"

    # Verify terminal response was recorded as FAILED
    terminal_res = await store.get_response_by_request_id("req-timeout-1")
    assert isinstance(terminal_res, Success)
    stored = terminal_res.unwrap()
    assert stored is not None
    assert stored.status == "FAILED"
    assert "timed out" in (stored.error_message or "")

    # Assistant turn should NOT have been committed as successful turn
    chain = await store.get_recent_reply_chain("session-fail")
    assert isinstance(chain, Success)
    # Only the user turn was accepted
    assert len([t for t in chain.unwrap() if t.role == "assistant"]) == 0


@pytest.mark.asyncio
async def test_service_cancellation_records_cancelled_terminal(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "memory_graph.sqlite3")
    backend = DeterministicAssistantBackend(fail_mode="cancellation")
    audit = InMemoryAssistantAuditLedger()
    service = AssistantService(store=store, backend=backend, audit_ledger=audit)

    turn = ConversationTurn(
        record_id="turn-cancel-1",
        producer=PRODUCER,
        session_id="session-cancel",
        role="user",
        text="A query that is cancelled",
    )
    token = CancellationToken()
    token.cancel()

    res = await service.accept_turn(turn, cancellation=token, request_id="req-cancel-1")
    assert isinstance(res, Failure)
    assert res.failure().code == "CANCELLED"

    # Verify terminal response was recorded as CANCELLED
    terminal_res = await store.get_response_by_request_id("req-cancel-1")
    assert isinstance(terminal_res, Success)
    stored = terminal_res.unwrap()
    assert stored is not None
    assert stored.status == "CANCELLED"


@pytest.mark.asyncio
async def test_service_streaming(service: AssistantService) -> None:
    turn = ConversationTurn(
        record_id="turn-stream-1",
        producer=PRODUCER,
        session_id="session-stream",
        role="user",
        text="Zapamiętaj, że w przykładach kodu preferuję Guile zamiast Pythona.",
    )
    chunks: list[str] = []
    res = await service.stream_turn(turn, on_chunk=chunks.append, request_id="req-stream-1")
    assert isinstance(res, Success)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Guile" in full_text

    # Terminal state was committed
    terminal = await service.store.get_response_by_request_id("req-stream-1")
    assert isinstance(terminal, Success)
    assert terminal.unwrap() is not None
    assert terminal.unwrap().status == "COMPLETED"
