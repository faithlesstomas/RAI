"""Tests for AssistantService: pipeline, failure modes, context manifest, and streaming."""

from __future__ import annotations

from pathlib import Path
import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock
import pytest
from returns.result import Failure, Success

from rai.assistant.audit import InMemoryAssistantAuditLedger
from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.context import AssistantContextBuilder
from rai.assistant.records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    MemoryRecord,
)
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, ProducerIdentity, _utc_now

PRODUCER = ProducerIdentity(producer_id="service-test", kind="test", version="1.0.0")


@pytest.fixture
def service(tmp_path: Path) -> AssistantService:
    store = SQLiteMemoryGraphStore(path=tmp_path / "memory_graph.sqlite3")
    backend = DeterministicAssistantBackend()
    audit = InMemoryAssistantAuditLedger()
    return AssistantService(store=store, backend=backend, audit_ledger=audit)


@pytest.mark.asyncio
async def test_service_successful_turn_with_preference(
    service: AssistantService,
) -> None:
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

    # A turn has one terminal response even when a retry changes request_id.
    third_res = await service.accept_turn(turn, request_id="req-idem-2")
    assert isinstance(third_res, Success)
    assert third_res.unwrap().record_id == first_resp.record_id

    reused_request = await service.accept_turn(
        turn.model_copy(update={"record_id": "turn-idem-other"}),
        request_id="req-idem-1",
    )
    assert isinstance(reused_request, Failure)
    assert reused_request.failure().code == "ID_CONFLICT"

    # Audit ledger should only have 1 entry
    entries = await service.audit_ledger.list_for_session("session-A")
    assert len(entries) == 1


@pytest.mark.asyncio
async def test_service_reuses_partially_persisted_turn_on_retry(
    service: AssistantService,
) -> None:
    await service.start()
    parent = ConversationTurn(
        record_id="partial-parent",
        producer=PRODUCER,
        session_id="partial-session",
        role="assistant",
        text="Earlier response",
        status="COMPLETED",
    )
    assert isinstance(await service.store.accept_turn(parent), Success)
    original = ConversationTurn(
        record_id="partial-user",
        producer=PRODUCER,
        session_id="partial-session",
        role="user",
        text="Continue after a partial write",
    )
    persisted = original.model_copy(
        update={
            "reply_to_turn_id": parent.record_id,
            "metadata": {"profile_scope": "default"},
            "domain_scope": "global",
            "purpose": "assistant",
        }
    )
    assert isinstance(await service.store.accept_turn(persisted), Success)

    retried = await service.accept_turn(original, request_id="partial-retry")

    assert isinstance(retried, Success)
    stored = await service.store.get_turn(original.record_id)
    assert isinstance(stored, Success)
    assert stored.unwrap() is not None
    assert stored.unwrap().reply_to_turn_id == parent.record_id


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
    res = await service.stream_turn(
        turn, on_chunk=chunks.append, request_id="req-stream-1"
    )
    assert isinstance(res, Success)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert "Guile" in full_text

    # Terminal state was committed
    terminal = await service.store.get_response_by_request_id("req-stream-1")
    assert isinstance(terminal, Success)
    assert terminal.unwrap() is not None
    assert terminal.unwrap().status == "COMPLETED"


@pytest.mark.asyncio
async def test_concurrent_same_turn_invokes_backend_once(
    tmp_path: Path,
) -> None:
    backend = DeterministicAssistantBackend(delay_seconds=0.05)
    original_generate = backend.generate
    backend.generate = AsyncMock(wraps=original_generate)  # type: ignore[method-assign]
    service = AssistantService(
        store=SQLiteMemoryGraphStore(path=tmp_path / "concurrent.sqlite3"),
        backend=backend,
    )
    turn = ConversationTurn(
        record_id="turn-concurrent",
        producer=PRODUCER,
        session_id="session-concurrent",
        role="user",
        text="Hello",
    )

    first, second = await asyncio.gather(
        service.accept_turn(turn, request_id="request-concurrent-a"),
        service.accept_turn(turn, request_id="request-concurrent-b"),
    )

    assert isinstance(first, Success)
    assert isinstance(second, Success)
    assert first.unwrap().record_id == second.unwrap().record_id
    assert backend.generate.await_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_invalid_backend_candidate_is_committed_as_failure(
    tmp_path: Path,
) -> None:
    backend = DeterministicAssistantBackend()
    backend.generate = AsyncMock(return_value=Success(object()))  # type: ignore[method-assign]
    store = SQLiteMemoryGraphStore(path=tmp_path / "invalid.sqlite3")
    service = AssistantService(store=store, backend=backend)
    turn = ConversationTurn(
        record_id="turn-invalid",
        producer=PRODUCER,
        session_id="session-invalid",
        role="user",
        text="Hello",
    )

    result = await service.accept_turn(turn, request_id="request-invalid")

    assert isinstance(result, Failure)
    assert result.failure().code == "INVALID_OUTPUT"
    stored = await store.get_response_by_request_id("request-invalid")
    assert isinstance(stored, Success)
    assert stored.unwrap() is not None
    assert stored.unwrap().status == "FAILED"


@pytest.mark.asyncio
async def test_memory_privacy_class_cannot_be_downgraded(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "privacy.sqlite3")
    service = AssistantService(store=store, backend=DeterministicAssistantBackend())
    turn = ConversationTurn(
        record_id="turn-private",
        producer=PRODUCER,
        session_id="session-private",
        role="user",
        text="Preferuję Guile w przykładach kodu.",
        data_class=DataClass.PRIVATE,
    )

    result = await service.accept_turn(turn)
    assert isinstance(result, Success)
    public_retrieval = await store.retrieve_relevant_memories()
    private_retrieval = await store.retrieve_relevant_memories(
        data_classes=(DataClass.PRIVATE,)
    )

    assert isinstance(public_retrieval, Success)
    assert public_retrieval.unwrap() == ()
    assert isinstance(private_retrieval, Success)
    assert private_retrieval.unwrap()[0][0].data_class == DataClass.PRIVATE


@pytest.mark.asyncio
async def test_context_builder_excludes_poisoned_expired_retrieval(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "poisoned.sqlite3")
    source = ConversationTurn(
        record_id="turn-source",
        producer=PRODUCER,
        session_id="session-poisoned",
        role="user",
        text="old statement",
        status="COMPLETED",
    )
    await store.accept_turn(source)
    expired = MemoryRecord(
        record_id="memory-expired",
        producer=PRODUCER,
        topic="other",
        content={"statement": "stale"},
        source_turn_id=source.record_id,
        valid_from=_utc_now() - timedelta(days=2),
        valid_until=_utc_now() - timedelta(days=1),
    )
    store.retrieve_relevant_memories = AsyncMock(  # type: ignore[method-assign]
        return_value=Success(((expired, "poisoned adapter result"),))
    )
    builder = AssistantContextBuilder(store=store)
    current = ConversationTurn(
        record_id="turn-current",
        producer=PRODUCER,
        session_id="session-poisoned",
        role="user",
        text="what is current?",
    )

    result = await builder.build_context(current)

    assert isinstance(result, Success)
    manifest = result.unwrap().manifest
    assert manifest.durable_memory_ids == ()
    assert manifest.exclusions == ("memory-expired:temporal_validity",)


@pytest.mark.asyncio
async def test_context_builder_fails_closed_when_current_turn_exceeds_budget(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "budget.sqlite3")
    builder = AssistantContextBuilder(store=store, max_context_characters=256)
    current = ConversationTurn(
        record_id="turn-over-budget",
        producer=PRODUCER,
        session_id="session-budget",
        role="user",
        text="x" * 300,
    )

    result = await builder.build_context(current)

    assert isinstance(result, Failure)
    assert result.failure().code == "CONTEXT_BUDGET_EXCEEDED"


@pytest.mark.asyncio
async def test_recent_context_enforces_query_privacy_classes(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "recent-privacy.sqlite3")
    await store.start()
    private_turn = ConversationTurn(
        record_id="private-recent",
        producer=PRODUCER,
        session_id="privacy-session",
        role="user",
        text="Mam na imię Sekret.",
        data_class=DataClass.PRIVATE,
        domain_scope="personal",
        status="COMPLETED",
    )
    assert isinstance(await store.accept_turn(private_turn), Success)
    current = ConversationTurn(
        record_id="privacy-query",
        producer=PRODUCER,
        session_id="privacy-session",
        role="user",
        text="Jak mam na imię?",
        domain_scope="personal",
    )
    assert isinstance(await store.accept_turn(current), Success)

    result = await AssistantContextBuilder(store=store).build_context(current)

    assert isinstance(result, Success)
    assert "private-recent" not in result.unwrap().manifest.recent_turn_ids
    assert result.unwrap().manifest.routing_decision == "no_evidence"


@pytest.mark.asyncio
async def test_route_is_recomputed_after_budget_removes_compact_memory(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(path=tmp_path / "route-budget.sqlite3")
    await store.start()
    source = ConversationTurn(
        record_id="large-memory-source",
        producer=PRODUCER,
        session_id="source-session",
        role="user",
        text="Mam na imię Tomasz.",
        domain_scope="personal",
        status="COMPLETED",
    )
    assert isinstance(await store.accept_turn(source), Success)
    memory = MemoryRecord(
        record_id="large-memory",
        producer=PRODUCER,
        kind="claim",
        topic="user.identity.name",
        content={
            "predicate": "name",
            "value": "Tomasz",
            "modality": "direct",
            "confidence": 1.0,
            "padding": "x" * 2_000,
        },
        source_turn_id=source.record_id,
        domain_scope="personal",
    )
    manifest = AssistantContextManifest(
        record_id="large-memory-manifest",
        producer=PRODUCER,
        session_id=source.session_id,
        turn_id=source.record_id,
    )
    response = AssistantResponse(
        record_id="large-memory-response",
        producer=PRODUCER,
        session_id=source.session_id,
        turn_id="large-memory-assistant",
        user_turn_id=source.record_id,
        request_id="large-memory-request",
        manifest_id=manifest.record_id,
        text="stored",
    )
    assert isinstance(
        await store.commit_terminal(response, manifest, None, (memory,), ()), Success
    )
    current = ConversationTurn(
        record_id="large-memory-query",
        producer=PRODUCER,
        session_id="other-session",
        role="user",
        text="Jak mam na imię?",
        domain_scope="personal",
    )
    builder = AssistantContextBuilder(store=store, max_context_characters=700)

    result = await builder.build_context(current)

    assert isinstance(result, Success)
    assert result.unwrap().manifest.durable_memory_ids == ()
    assert result.unwrap().manifest.routing_decision == "no_evidence"
    assert "compact_memory:removed_by_character_budget" in (
        result.unwrap().manifest.rejected_routes
    )
