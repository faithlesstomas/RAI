"""Unit and integration tests for ObservationStore and SQLite observation artifacts."""
# ruff: noqa: PLR2004

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from rai.assistant.records import (
    ObservationArtifact,
    ObservationSchemaSummary,
    summarize_observation_schema,
)
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.records import DataClass, ProducerIdentity, ProvenanceReference, _utc_now

PRODUCER = ProducerIdentity(
    producer_id="test-producer", kind="test", version="1.0.0"
)


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryGraphStore:
    db_path = tmp_path / "memory_graph.sqlite3"
    return SQLiteMemoryGraphStore(path=db_path)


@pytest.mark.asyncio
async def test_observation_store_lifecycle_and_allocation(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-1"
    id1 = (await store.allocate_artifact_id(session_id)).unwrap()
    assert id1 == "obs_1"

    # Save an artifact with obs_1
    artifact1 = ObservationArtifact(
        artifact_id=id1,
        session_id=session_id,
        turn_id="turn-1",
        capability="application.list",
        request_id="req-1",
        data_class=DataClass.LOCAL,
        content={"applications": [{"name": "Firefox"}, {"name": "VLC"}]},
        schema_summary=summarize_observation_schema({"applications": [{"name": "Firefox"}, {"name": "VLC"}]}),
        size_chars=50,
        producer=PRODUCER,
    )
    saved = (await store.put(artifact1)).unwrap()
    assert saved.artifact_id == "obs_1"

    # Next allocation should be obs_2
    id2 = (await store.allocate_artifact_id(session_id)).unwrap()
    assert id2 == "obs_2"


@pytest.mark.asyncio
async def test_get_by_alias_and_stable_id(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-lookup"
    artifact = ObservationArtifact(
        artifact_id="obs_1",
        session_id=session_id,
        turn_id="turn-1",
        capability="browser.search",
        request_id="req-search",
        data_class=DataClass.PUBLIC,
        content={"results": [{"title": "Example", "url": "https://example.com"}]},
        schema_summary=summarize_observation_schema({"results": [{"title": "Example", "url": "https://example.com"}]}),
        size_chars=60,
        producer=PRODUCER,
    )
    await store.put(artifact)

    # Lookup by session alias
    by_alias = (await store.get(session_id, "obs_1")).unwrap()
    assert by_alias is not None
    assert by_alias.record_id == artifact.record_id
    assert by_alias.capability == "browser.search"
    assert by_alias.content["results"][0]["title"] == "Example"

    # Lookup by stable UUID
    by_uuid = (await store.get(session_id, artifact.record_id)).unwrap()
    assert by_uuid is not None
    assert by_uuid.artifact_id == "obs_1"

    # Lookup non-existent
    missing = (await store.get(session_id, "obs_999")).unwrap()
    assert missing is None


@pytest.mark.asyncio
async def test_privacy_rejection(store: SQLiteMemoryGraphStore) -> None:
    # Validation at record level:
    with pytest.raises(ValueError, match="SECRET"):
        ObservationArtifact(
            artifact_id="obs_sec",
            session_id="s1",
            turn_id="t1",
            capability="keys.get",
            request_id="req-sec",
            data_class=DataClass.SECRET,
            content={"secret": "super-password"},
            schema_summary=ObservationSchemaSummary(),
            size_chars=30,
            producer=PRODUCER,
        )


@pytest.mark.asyncio
async def test_list_and_delete_for_session(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-list"
    for i in range(1, 4):
        art = ObservationArtifact(
            artifact_id=f"obs_{i}",
            session_id=session_id,
            turn_id=f"turn-{i}",
            capability=f"cap_{i}",
            request_id=f"req-{i}",
            data_class=DataClass.LOCAL,
            content={"val": i},
            schema_summary=ObservationSchemaSummary(),
            size_chars=10,
            producer=PRODUCER,
        )
        await store.put(art)

    listing = (await store.list_for_session(session_id)).unwrap()
    assert len(listing) == 3
    assert [a.artifact_id for a in listing] == ["obs_1", "obs_2", "obs_3"]

    # Delete session
    deleted = (await store.delete_for_session(session_id)).unwrap()
    assert deleted == 3

    empty_listing = (await store.list_for_session(session_id)).unwrap()
    assert len(empty_listing) == 0


@pytest.mark.asyncio
async def test_prune_expired(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-expire"
    now = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)

    # Expired
    art_expired = ObservationArtifact(
        artifact_id="obs_old",
        session_id=session_id,
        turn_id="turn-old",
        capability="cache.get",
        request_id="req-old",
        data_class=DataClass.LOCAL,
        content={"data": "stale"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=20,
        expires_at=now - timedelta(minutes=10),
        producer=PRODUCER,
    )
    await store.put(art_expired)

    # Active
    art_active = ObservationArtifact(
        artifact_id="obs_fresh",
        session_id=session_id,
        turn_id="turn-fresh",
        capability="cache.get",
        request_id="req-fresh",
        data_class=DataClass.LOCAL,
        content={"data": "fresh"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=20,
        expires_at=now + timedelta(minutes=10),
        producer=PRODUCER,
    )
    await store.put(art_active)

    pruned = (await store.prune_expired(now=now)).unwrap()
    assert pruned == 1

    remaining = (await store.list_for_session(session_id)).unwrap()
    assert len(remaining) == 1
    assert remaining[0].artifact_id == "obs_fresh"
