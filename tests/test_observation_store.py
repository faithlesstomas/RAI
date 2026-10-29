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

    remaining = (await store.list_for_session(session_id, now=now)).unwrap()
    assert len(remaining) == 1
    assert remaining[0].artifact_id == "obs_fresh"


@pytest.mark.asyncio
async def test_get_rejects_expired_artifact(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-get-expired"
    past = _utc_now() - timedelta(minutes=5)
    artifact = ObservationArtifact(
        artifact_id="obs_exp",
        session_id=session_id,
        turn_id="turn-1",
        capability="browser.read_page",
        request_id="req-exp",
        data_class=DataClass.PUBLIC,
        content={"text": "should not be visible"},
        schema_summary=summarize_observation_schema({"text": "should not be visible"}),
        size_chars=30,
        expires_at=past,
        producer=PRODUCER,
    )
    await store.put(artifact)

    # get should fail closed and return None because the artifact is expired
    res = (await store.get(session_id, "obs_exp")).unwrap()
    assert res is None


@pytest.mark.asyncio
async def test_get_requires_session_id(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-scoped"
    artifact = ObservationArtifact(
        artifact_id="obs_leak",
        session_id=session_id,
        turn_id="turn-1",
        capability="test",
        request_id="req-leak",
        data_class=DataClass.PUBLIC,
        content={"data": "isolated"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=10,
        producer=PRODUCER,
    )
    await store.put(artifact)

    # Querying without a session_id must not return session-scoped artifact
    res = (await store.get("", "obs_leak")).unwrap()
    assert res is None


@pytest.mark.asyncio
async def test_unique_alias_collision_handling(store: SQLiteMemoryGraphStore) -> None:
    session_id = "test-session-unique-alias"
    art1 = ObservationArtifact(
        artifact_id="obs_1",
        session_id=session_id,
        turn_id="turn-1",
        capability="test",
        request_id="req-u1",
        data_class=DataClass.LOCAL,
        content={"val": 1},
        schema_summary=ObservationSchemaSummary(),
        size_chars=10,
        producer=PRODUCER,
    )
    saved1 = (await store.put(art1)).unwrap()
    assert saved1.artifact_id == "obs_1"

    # Insert a second artifact with different record_id but intentionally conflicting artifact_id
    art2 = ObservationArtifact(
        artifact_id="obs_1",
        session_id=session_id,
        turn_id="turn-2",
        capability="test",
        request_id="req-u2",
        data_class=DataClass.LOCAL,
        content={"val": 2},
        schema_summary=ObservationSchemaSummary(),
        size_chars=10,
        producer=PRODUCER,
    )
    saved2 = (await store.put(art2)).unwrap()
    # Collision is safely resolved to the next free alias (obs_2)
    assert saved2.artifact_id == "obs_2"

    # Both are accessible under their distinct aliases
    get1 = (await store.get(session_id, "obs_1")).unwrap()
    assert get1 is not None and get1.content["val"] == 1
    get2 = (await store.get(session_id, "obs_2")).unwrap()
    assert get2 is not None and get2.content["val"] == 2


@pytest.mark.asyncio
async def test_history_service_cascades_observation_cleanup(
    tmp_path: Path, store: SQLiteMemoryGraphStore
) -> None:
    from returns.result import Success  # noqa: PLC0415
    from rai.services.history import HistoryService  # noqa: PLC0415

    session_id = "session-cascade-test"
    art = ObservationArtifact(
        artifact_id="obs_cascade",
        session_id=session_id,
        turn_id="turn-cascade",
        capability="test",
        request_id="req-c1",
        data_class=DataClass.LOCAL,
        content={"data": "to-be-deleted"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=20,
        producer=PRODUCER,
    )
    await store.put(art)

    # Verify it exists in store
    pre_list = (await store.list_for_session(session_id)).unwrap()
    assert len(pre_list) == 1

    # Create history service with observation_store=store
    hist_db = str(tmp_path / "hist.db")
    service = HistoryService(db_path=hist_db, observation_store=store)
    clear_res = await service.clear_history(session_id)
    assert isinstance(clear_res, Success)

    # Verify observations for the session were purged
    post_list = (await store.list_for_session(session_id)).unwrap()
    assert len(post_list) == 0


@pytest.mark.asyncio
async def test_observation_content_aes_gcm_encrypted_on_disk(
    store: SQLiteMemoryGraphStore,
) -> None:
    """Artifact content_json must be encrypted with AES-256-GCM and not appear as plaintext on disk."""
    session_id = "session-encrypted-test"
    secret_text = "SUPER_SECRET_PAYLOAD_TOKEN_778899"
    art = ObservationArtifact(
        artifact_id="obs_secret",
        session_id=session_id,
        turn_id="turn-secret",
        capability="browser.read_page",
        request_id="req-secret",
        data_class=DataClass.PRIVATE,
        content={"secret": secret_text, "nested": {"key": "value"}},
        schema_summary=ObservationSchemaSummary(),
        size_chars=100,
        producer=PRODUCER,
    )
    saved = (await store.put(art)).unwrap()
    assert saved.artifact_id == "obs_secret"

    # Verify raw disk bytes do not contain plaintext secret
    disk_bytes = store.path.read_bytes()
    assert secret_text.encode("utf-8") not in disk_bytes

    # Verify retrieval decrypts transparently
    retrieved = (await store.get(session_id, "obs_secret")).unwrap()
    assert retrieved is not None
    assert retrieved.content["secret"] == secret_text
    assert retrieved.content["nested"]["key"] == "value"


@pytest.mark.asyncio
async def test_observation_legacy_plaintext_json_compatibility(
    store: SQLiteMemoryGraphStore,
) -> None:
    """Store reads legacy unencrypted JSON rows without error."""
    session_id = "session-legacy"
    conn = store._connect()
    with conn:
        store._init_db(conn)
        conn.execute(
            """
            INSERT INTO observation_artifacts (
                record_id, artifact_id, session_id, turn_id, capability,
                request_id, data_class, content_json, schema_summary_json,
                size_chars, created_at, expires_at, provenance_json,
                derived_from_json, producer_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "rec_legacy",
                "obs_legacy",
                session_id,
                "turn-1",
                "legacy.tool",
                "req-leg",
                "PUBLIC",
                '{"legacy_field": "legacy_value"}',
                ObservationSchemaSummary().model_dump_json(),
                30,
                _utc_now().isoformat(),
                None,
                "[]",
                "[]",
                PRODUCER.model_dump_json(),
            ),
        )
    conn.close()

    retrieved = (await store.get(session_id, "obs_legacy")).unwrap()
    assert retrieved is not None
    assert retrieved.content == {"legacy_field": "legacy_value"}


@pytest.mark.asyncio
async def test_history_service_clears_repl_session_and_artifacts(
    tmp_path: Path, store: SQLiteMemoryGraphStore
) -> None:
    """HistoryService.clear_history cleans both observation artifacts and REPL sessions."""
    from returns.result import Success  # noqa: PLC0415
    from rai.services.history import HistoryService  # noqa: PLC0415

    session_id = "session-repl-purge"
    art = ObservationArtifact(
        artifact_id="obs_purge",
        session_id=session_id,
        turn_id="turn-p",
        capability="test",
        request_id="req-p1",
        data_class=DataClass.LOCAL,
        content={"data": "to-be-deleted"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=20,
        producer=PRODUCER,
    )
    await store.put(art)
    await store.save_repl_session(session_id, b"snapshot_bytes", {"x": "int"})

    # Verify both exist
    assert len((await store.list_for_session(session_id)).unwrap()) == 1
    assert (await store.get_repl_session(session_id)).unwrap() is not None

    hist_db = str(tmp_path / "hist_repl.db")
    service = HistoryService(db_path=hist_db, observation_store=store)
    res = await service.clear_history(session_id)
    assert isinstance(res, Success)

    # Verify both are purged
    assert len((await store.list_for_session(session_id)).unwrap()) == 0
    assert (await store.get_repl_session(session_id)).unwrap() is None


@pytest.mark.asyncio
async def test_retention_ttl_and_auto_pruning(store: SQLiteMemoryGraphStore) -> None:
    """Retention TTL parsing and auto-pruning works correctly."""
    from rai.assistant.observation_budget import parse_retention_ttl, persist_observation_artifact  # noqa: PLC0415

    assert parse_retention_ttl(None) is None
    assert parse_retention_ttl("session") is None
    assert parse_retention_ttl("profile") is None
    assert parse_retention_ttl("1h") == timedelta(hours=1)
    assert parse_retention_ttl("7d") == timedelta(days=7)
    assert parse_retention_ttl(3600) == timedelta(seconds=3600)

    # Persist with retention
    session_id = "session-retention"
    art_id = await persist_observation_artifact(
        store,
        session_id=session_id,
        turn_id="turn-1",
        capability="test",
        request_id="req-ret",
        data_class=DataClass.LOCAL,
        output={"key": "value"},
        retention="24h",
    )
    assert art_id is not None
    art = (await store.get(session_id, art_id)).unwrap()
    assert art is not None
    assert art.expires_at is not None
    assert art.expires_at > _utc_now()

    # Expired artifact
    past_dt = _utc_now() - timedelta(minutes=10)
    exp_art = ObservationArtifact(
        artifact_id="obs_exp",
        session_id=session_id,
        turn_id="turn-2",
        capability="test",
        request_id="req-exp",
        data_class=DataClass.LOCAL,
        content={"key": "expired"},
        schema_summary=ObservationSchemaSummary(),
        size_chars=15,
        producer=PRODUCER,
        expires_at=past_dt,
    )
    await store.put(exp_art)

    # get should not return expired artifact
    assert (await store.get(session_id, "obs_exp")).unwrap() is None

    # prune_expired should delete it from disk
    pruned = (await store.prune_expired()).unwrap()
    assert pruned >= 1
