"""Tests for SQLiteMemoryGraphStore: CRUD, idempotency, supersession, deletion cascading."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pytest
from returns.result import Failure, Success

from rai.assistant.ports import MemoryQuery
from rai.assistant.records import (
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.ports import LifecycleState
from rai.kernel.records import DataClass, ProducerIdentity, _new_id, _utc_now

PRODUCER = ProducerIdentity(producer_id="store-test", kind="test", version="1.0.0")


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    db_path = tmp_path / "sub" / "memory_graph.sqlite3"
    return SQLiteMemoryGraphStore(path=db_path)


@pytest.mark.asyncio
async def test_store_lifecycle_and_permissions(store: SQLiteMemoryGraphStore) -> None:
    res = await store.start()
    assert isinstance(res, Success)
    assert store.state == LifecycleState.RUNNING

    # Verify file permissions
    stat = os.stat(store.path)
    assert oct(stat.st_mode)[-3:] == "600"
    dir_stat = os.stat(store.path.parent)
    assert oct(dir_stat.st_mode)[-3:] == "700"

    stop_res = await store.stop()
    assert isinstance(stop_res, Success)
    assert store.state == LifecycleState.STOPPED


@pytest.mark.asyncio
async def test_accept_turn_and_idempotency(store: SQLiteMemoryGraphStore) -> None:
    await store.start()
    turn = ConversationTurn(
        record_id="turn-1",
        producer=PRODUCER,
        session_id="session-1",
        role="user",
        text="Hello world",
    )
    res = await store.accept_turn(turn)
    assert isinstance(res, Success)

    # Identical retry -> Success
    retry = await store.accept_turn(turn)
    assert isinstance(retry, Success)

    # Immutable graph position is part of turn identity.
    different_parent = turn.model_copy(update={"reply_to_turn_id": "other-turn"})
    parent_conflict = await store.accept_turn(different_parent)
    assert isinstance(parent_conflict, Failure)
    assert parent_conflict.failure().code == "ID_CONFLICT"

    # Same id, differing content -> Failure
    conflicting = ConversationTurn(
        record_id="turn-1",
        producer=PRODUCER,
        session_id="session-1",
        role="user",
        text="Different text",
    )
    conflict_res = await store.accept_turn(conflicting)
    assert isinstance(conflict_res, Failure)
    assert conflict_res.failure().code == "ID_CONFLICT"

    # Verify get_turn
    get_res = await store.get_turn("turn-1")
    assert isinstance(get_res, Success)
    retrieved = get_res.unwrap()
    assert retrieved is not None
    assert retrieved.text == "Hello world"


@pytest.mark.asyncio
async def test_dependent_lookup_never_crosses_profile_scope(
    store: SQLiteMemoryGraphStore,
) -> None:
    await store.start()
    turn = ConversationTurn(
        record_id="profile-source-turn",
        producer=PRODUCER,
        session_id="profile-session",
        role="user",
        text="Profile-scoped source",
        status="COMPLETED",
    )
    assert isinstance(await store.accept_turn(turn), Success)
    foreign_turn = ConversationTurn(
        record_id="profile-foreign-turn",
        producer=PRODUCER,
        session_id="profile-session",
        role="user",
        text="Foreign profile source",
        status="COMPLETED",
    )
    assert isinstance(await store.accept_turn(foreign_turn), Success)
    source = MemoryRecord(
        record_id="profile-source-memory",
        producer=PRODUCER,
        kind="claim",
        topic="claim.profile.source",
        content={"fact": "source"},
        source_turn_id=turn.record_id,
        profile_scope="profile-a",
    )
    foreign = MemoryRecord(
        record_id="profile-foreign-memory",
        producer=PRODUCER,
        kind="derived_claim",
        topic="claim.profile.foreign",
        content={"fact": "foreign"},
        source_turn_id=foreign_turn.record_id,
        profile_scope="profile-b",
    )
    manifest = AssistantContextManifest(
        record_id="profile-manifest",
        producer=PRODUCER,
        session_id=turn.session_id,
        turn_id=turn.record_id,
    )
    response = AssistantResponse(
        record_id="profile-response",
        producer=PRODUCER,
        session_id=turn.session_id,
        turn_id="profile-assistant-turn",
        user_turn_id=turn.record_id,
        request_id="profile-request",
        manifest_id=manifest.record_id,
        text="stored",
    )
    relation = MemoryRelation(
        source_id=source.record_id,
        target_id=foreign.record_id,
        kind=MemoryRelationKind.SUPPORTS,
    )
    denied_relation = relation.model_copy(
        update={
            "relation_id": "profile-denied-relation",
            "target_id": foreign.record_id,
            "policy_outcome": "DENY",
            "eligible": False,
        }
    )
    assert isinstance(
        await store.commit_terminal(
            response, manifest, None, (source, foreign), (relation,)
        ),
        Success,
    )

    unscoped = await store.get_dependent_memory_ids((source.record_id,))
    scoped = await store.get_dependent_memory_ids(
        (source.record_id,), profile_scope="profile-a"
    )

    assert isinstance(unscoped, Success)
    assert unscoped.unwrap() == (foreign.record_id,)
    assert isinstance(scoped, Success)
    assert scoped.unwrap() == ()

    # A denied graph edge must never drive a destructive cascade, even unscoped.
    source_two = source.model_copy(
        update={
            "record_id": "profile-source-memory-two",
            "topic": "claim.profile.source-two",
        }
    )
    second_manifest = manifest.model_copy(update={"record_id": "profile-manifest-two"})
    second_response = response.model_copy(
        update={
            "record_id": "profile-response-two",
            "request_id": "profile-request-two",
            "manifest_id": second_manifest.record_id,
        }
    )
    assert isinstance(
        await store.commit_terminal(
            second_response,
            second_manifest,
            None,
            (source_two,),
            (denied_relation.model_copy(update={"source_id": source_two.record_id}),),
        ),
        Success,
    )
    denied = await store.get_dependent_memory_ids((source_two.record_id,))
    assert isinstance(denied, Success)
    assert denied.unwrap() == ()

    deleted = await store.delete_turn(turn.record_id)
    surviving_foreign = await store.get_memory(foreign.record_id)
    assert isinstance(deleted, Success)
    assert isinstance(surviving_foreign, Success)
    assert surviving_foreign.unwrap() is not None


@pytest.mark.asyncio
async def test_reply_chain_and_limits(store: SQLiteMemoryGraphStore) -> None:
    await store.start()
    for i in range(5):
        turn = ConversationTurn(
            record_id=f"turn-{i}",
            producer=PRODUCER,
            session_id="session-1",
            role="user" if i % 2 == 0 else "assistant",
            text=f"Message {i}",
            status="COMPLETED",
        )
        await store.accept_turn(turn)

    limit = 3
    chain_res = await store.get_recent_reply_chain(session_id="session-1", limit=limit)
    assert isinstance(chain_res, Success)
    turns = chain_res.unwrap()
    assert len(turns) == limit
    assert [t.record_id for t in turns] == ["turn-2", "turn-3", "turn-4"]


@pytest.mark.asyncio
async def test_reply_chain_follows_links_without_crossing_branches(
    store: SQLiteMemoryGraphStore,
) -> None:
    await store.start()
    turns = (
        ConversationTurn(
            record_id="root",
            producer=PRODUCER,
            session_id="branched",
            role="user",
            text="root",
            status="COMPLETED",
        ),
        ConversationTurn(
            record_id="branch-a",
            producer=PRODUCER,
            session_id="branched",
            role="assistant",
            text="a",
            reply_to_turn_id="root",
            status="COMPLETED",
        ),
        ConversationTurn(
            record_id="branch-b",
            producer=PRODUCER,
            session_id="branched",
            role="assistant",
            text="b",
            reply_to_turn_id="root",
            status="COMPLETED",
        ),
        ConversationTurn(
            record_id="leaf-a",
            producer=PRODUCER,
            session_id="branched",
            role="user",
            text="leaf",
            reply_to_turn_id="branch-a",
            status="COMPLETED",
        ),
    )
    for turn in turns:
        assert isinstance(await store.accept_turn(turn), Success)

    chain = await store.get_recent_reply_chain("branched", limit=10)

    assert isinstance(chain, Success)
    assert [turn.record_id for turn in chain.unwrap()] == ["root", "branch-a", "leaf-a"]


@pytest.mark.asyncio
async def test_supersession_and_neutral_retrieval(
    store: SQLiteMemoryGraphStore,
) -> None:
    await store.start()

    turn_1 = ConversationTurn(
        record_id="turn-source-1",
        producer=PRODUCER,
        session_id="session-1",
        role="user",
        text="Preferuję Guile",
        status="COMPLETED",
    )
    await store.accept_turn(turn_1)

    # Memory 1: Guile
    mem_1 = MemoryRecord(
        record_id="mem-1",
        producer=PRODUCER,
        kind="preference",
        topic="code_examples",
        content={"preference": "Guile"},
        source_turn_id="turn-source-1",
    )
    manifest_1 = AssistantContextManifest(
        record_id="man-1",
        producer=PRODUCER,
        session_id="session-1",
        turn_id="turn-source-1",
    )
    resp_1 = AssistantResponse(
        record_id="resp-1",
        producer=PRODUCER,
        session_id="session-1",
        turn_id="asst-1",
        user_turn_id="turn-source-1",
        request_id="req-1",
        manifest_id="man-1",
        text="Zapamiętano Guile",
        status="COMPLETED",
    )
    commit_1 = await store.commit_terminal(
        response=resp_1,
        manifest=manifest_1,
        assistant_turn=None,
        memories=(mem_1,),
        relations=(),
    )
    assert isinstance(commit_1, Success)
    historical_cutoff = _utc_now()

    # Query 1: Should retrieve Guile with topic exact match
    query = MemoryQuery(topic="code_examples")
    ret_1 = await store.retrieve_relevant_memories(query=query)
    assert isinstance(ret_1, Success)
    memories_1 = ret_1.unwrap()
    assert len(memories_1) == 1
    rec_1, reason_1 = memories_1[0]
    assert rec_1.content["preference"] == "Guile"
    assert "topic exact match: 'code_examples'" in reason_1

    # Now superseding with Memory 2: Python
    turn_2 = ConversationTurn(
        record_id="turn-source-2",
        producer=PRODUCER,
        session_id="session-2",
        role="user",
        text="Używaj Pythona",
        status="COMPLETED",
    )
    await store.accept_turn(turn_2)

    mem_2 = MemoryRecord(
        record_id="mem-2",
        producer=PRODUCER,
        kind="preference",
        topic="code_examples",
        content={"preference": "Python"},
        source_turn_id="turn-source-2",
    )
    manifest_2 = AssistantContextManifest(
        record_id="man-2",
        producer=PRODUCER,
        session_id="session-2",
        turn_id="turn-source-2",
    )
    resp_2 = AssistantResponse(
        record_id="resp-2",
        producer=PRODUCER,
        session_id="session-2",
        turn_id="asst-2",
        user_turn_id="turn-source-2",
        request_id="req-2",
        manifest_id="man-2",
        text="Zapamiętano Python",
        status="COMPLETED",
    )
    commit_2 = await store.commit_terminal(
        response=resp_2,
        manifest=manifest_2,
        assistant_turn=None,
        memories=(mem_2,),
        relations=(),
    )
    assert isinstance(commit_2, Success)

    # Verify SUPERSEDES relation was created
    rel_res = await store.get_relations(kind=MemoryRelationKind.SUPERSEDES)
    assert isinstance(rel_res, Success)
    relations = rel_res.unwrap()
    assert len(relations) == 1
    assert relations[0].source_id == "mem-2"
    assert relations[0].target_id == "mem-1"

    # Query 2: MUST only retrieve Python, Guile is superseded
    ret_2 = await store.retrieve_relevant_memories(query=query)
    assert isinstance(ret_2, Success)
    memories_2 = ret_2.unwrap()
    assert len(memories_2) == 1
    rec_2, reason_2 = memories_2[0]
    assert rec_2.content["preference"] == "Python"

    # Bitemporal query: reconstruct what the system knew before the correction.
    historical = await store.retrieve_relevant_memories(
        query=MemoryQuery(
            topic="code_examples",
            transaction_at=historical_cutoff,
            valid_at=historical_cutoff,
        )
    )
    assert isinstance(historical, Success)
    assert [memory.record_id for memory, _ in historical.unwrap()] == ["mem-1"]
    previous = await store.get_memory("mem-1")
    assert isinstance(previous, Success)
    previous_record = previous.unwrap()
    assert previous_record is not None
    assert previous_record[0].expired_at is not None
    assert previous_record[0].recorded_at <= historical_cutoff

    # Source turn deletion cascading:
    # Delete turn-source-2 -> mem-2 should be DELETED, but mem-1 MUST NOT be revived!
    del_res = await store.delete_turn("turn-source-2")
    assert isinstance(del_res, Success)
    assert del_res.unwrap() == 1

    # Query 3: MUST return empty tuple (neither Python nor Guile is active)
    ret_3 = await store.retrieve_relevant_memories(query=query)
    assert isinstance(ret_3, Success)
    memories_3 = ret_3.unwrap()
    assert len(memories_3) == 0
    deleted_memory = await store.get_memory("mem-2")
    assert isinstance(deleted_memory, Success)
    assert deleted_memory.unwrap() is None
    replayed = await store.replay_memory_projection()
    assert isinstance(replayed, Success)
    assert replayed.unwrap() == ()


@pytest.mark.asyncio
async def test_exactly_once_idempotency_for_responses(
    store: SQLiteMemoryGraphStore,
) -> None:
    await store.start()
    turn = ConversationTurn(
        record_id="turn-once-1",
        producer=PRODUCER,
        session_id="session-1",
        role="user",
        text="Question",
        status="COMPLETED",
    )
    await store.accept_turn(turn)

    manifest = AssistantContextManifest(
        record_id="man-once-1",
        producer=PRODUCER,
        session_id="session-1",
        turn_id="turn-once-1",
    )
    resp = AssistantResponse(
        record_id="resp-once-1",
        producer=PRODUCER,
        session_id="session-1",
        turn_id="asst-once-1",
        user_turn_id="turn-once-1",
        request_id="req-once-1",
        manifest_id="man-once-1",
        text="Answer 1",
        status="COMPLETED",
    )
    await store.commit_terminal(
        response=resp,
        manifest=manifest,
        assistant_turn=None,
        memories=(),
        relations=(),
    )

    # Second commit with same request_id returns early without error
    second_commit = await store.commit_terminal(
        response=resp,
        manifest=manifest,
        assistant_turn=None,
        memories=(),
        relations=(),
    )
    assert isinstance(second_commit, Success)
    assert second_commit.unwrap().record_id == resp.record_id

    different_request = resp.model_copy(
        update={
            "record_id": "resp-once-2",
            "request_id": "req-once-2",
            "text": "Conflicting second answer",
        }
    )
    turn_commit = await store.commit_terminal(
        response=different_request,
        manifest=manifest,
        assistant_turn=None,
        memories=(),
        relations=(),
    )
    assert isinstance(turn_commit, Success)
    assert turn_commit.unwrap().record_id == resp.record_id
    assert turn_commit.unwrap().text == "Answer 1"

    other_turn = turn.model_copy(
        update={"record_id": "turn-once-2", "text": "Other question"}
    )
    assert isinstance(await store.accept_turn(other_turn), Success)
    reused_request = resp.model_copy(
        update={
            "record_id": "resp-once-3",
            "user_turn_id": other_turn.record_id,
            "request_id": resp.request_id,
        }
    )
    request_conflict = await store.commit_terminal(
        response=reused_request,
        manifest=manifest,
        assistant_turn=None,
        memories=(),
        relations=(),
    )
    assert isinstance(request_conflict, Failure)
    assert request_conflict.failure().code == "ID_CONFLICT"

    # Fetch by request_id
    fetch = await store.get_response_by_request_id("req-once-1")
    assert isinstance(fetch, Success)
    assert fetch.unwrap() is not None
    assert fetch.unwrap().text == "Answer 1"
