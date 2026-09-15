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

    # Fetch by request_id
    fetch = await store.get_response_by_request_id("req-once-1")
    assert isinstance(fetch, Success)
    assert fetch.unwrap() is not None
    assert fetch.unwrap().text == "Answer 1"
