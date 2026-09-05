"""Acceptance and security failure tests for the Stage 2 local event plane."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from returns.result import Failure, Result, Success

from conftest import ASGITestClient
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    RegisteredCapability,
)
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.dispatch import DeterministicEventDispatcher
from rai.kernel.event_service import EventService
from rai.kernel.events import EventCursor, JournalFailure
from rai.kernel.journal import SQLiteEventJournal
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken, EventJournal
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    Observation,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
)
from rai.kernel.service import CapabilityService
from rai.kernel.schemas import event_json_schema
from rai.kernel.socket_transport import EventSocketServer
from rai.server import create_app
from rai.tools.security.auth import TOKEN_HEADER

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)
OK = 200
UNAUTHORIZED = 401
EXPECTED_UNCERTAINTY_MS = 7
PRIVATE_SOCKET_MODE = 0o600
PRODUCER = ProducerIdentity(
    producer_id="synthetic-stage2", kind="collector", version="1.0.0"
)


def observation(
    record_id: str = "person-1",
    *,
    timestamp: datetime = NOW,
    data_class: DataClass = DataClass.LOCAL,
) -> Observation:
    return Observation(
        record_id=record_id,
        timestamp=timestamp,
        producer=PRODUCER,
        kind="person_present",
        payload={"present": True},
        data_class=data_class,
    )


@pytest.mark.asyncio
async def test_ordering_idempotency_conflict_and_out_of_order_source_time(
    tmp_path: Path,
) -> None:
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    later = observation("later", timestamp=NOW + timedelta(hours=1))
    earlier = observation("earlier", timestamp=NOW - timedelta(hours=1))
    first = await journal.append(later, clock_source="device-monotonic", clock_uncertainty_ms=7)
    duplicate = await journal.append(later, clock_source="ignored-retry", clock_uncertainty_ms=99)
    second = await journal.append(earlier)
    conflict = await journal.append(later.model_copy(update={"payload": {"present": False}}))

    assert isinstance(first, Success) and isinstance(duplicate, Success)
    assert duplicate.unwrap() == first.unwrap()
    assert isinstance(second, Success)
    assert [first.unwrap().sequence, second.unwrap().sequence] == [1, 2]
    assert first.unwrap().clock_source == "device-monotonic"
    assert first.unwrap().clock_uncertainty_ms == EXPECTED_UNCERTAINTY_MS
    assert isinstance(conflict, Failure) and conflict.failure().code == "CONFLICT"

    replay = await journal.read(EventCursor.from_sequence(0), 10)
    assert isinstance(replay, Success)
    assert [item.record.record_id for item in replay.unwrap().events] == ["later", "earlier"]


@pytest.mark.asyncio
async def test_restart_replay_and_durable_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    first_process = SQLiteEventJournal(path)
    appended = await first_process.append(observation())
    assert isinstance(appended, Success)

    before_ack_restart = SQLiteEventJournal(path)
    pending = await before_ack_restart.read(EventCursor.from_sequence(0), 1)
    assert isinstance(pending, Success) and len(pending.unwrap().events) == 1
    acknowledged = await before_ack_restart.acknowledge(
        "consumer-a", pending.unwrap().next_cursor
    )
    assert isinstance(acknowledged, Success)

    after_ack_restart = SQLiteEventJournal(path)
    position = await after_ack_restart.position("consumer-a")
    assert isinstance(position, Success)
    assert position.unwrap() == pending.unwrap().next_cursor
    empty = await after_ack_restart.read(position.unwrap(), 1)
    assert isinstance(empty, Success) and not empty.unwrap().events
    regression = await after_ack_restart.acknowledge(
        "consumer-a", EventCursor.from_sequence(0)
    )
    assert isinstance(regression, Failure)
    assert regression.failure().code == "ACK_REGRESSION"
    retention = await after_ack_restart.request_retention(
        pending.unwrap().next_cursor, "policy-retention-window"
    )
    assert isinstance(retention, Success)
    retained = await after_ack_restart.read(EventCursor.from_sequence(0), 1)
    assert isinstance(retained, Success) and len(retained.unwrap().events) == 1


@pytest.mark.asyncio
async def test_malformed_oversized_and_sensitive_events_fail_closed(tmp_path: Path) -> None:
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3", max_event_bytes=4096)
    service = EventService(journal, max_event_bytes=4096)
    malformed = await service.ingest(b"not-json")
    invalid_producer_payload = observation("bad-producer").model_dump(mode="json")
    invalid_producer_payload["producer"]["producer_id"] = ""
    invalid_producer = await service.ingest(
        json.dumps(invalid_producer_payload).encode()
    )
    unsupported_version_payload = observation("bad-version").model_dump(mode="json")
    unsupported_version_payload["schema_version"] = "2.0.0"
    unsupported_version = await service.ingest(
        json.dumps(unsupported_version_payload).encode()
    )
    oversized = await service.ingest(b"{" + b"x" * 4097)
    blocked = await journal.append(observation("blocked", data_class=DataClass.BLOCKED))
    invalid_cursor = await journal.read(EventCursor(value="broken"), 1)
    excessive_batch = await journal.read(EventCursor.from_sequence(0), 101)
    mismatch = await journal.append(
        observation("mismatch"), data_class=DataClass.PRIVATE
    )

    assert isinstance(malformed, Failure) and malformed.failure().code == "INVALID_EVENT"
    assert (
        isinstance(invalid_producer, Failure)
        and invalid_producer.failure().code == "INVALID_EVENT"
    )
    assert (
        isinstance(unsupported_version, Failure)
        and unsupported_version.failure().code == "INVALID_EVENT"
    )
    assert isinstance(oversized, Failure) and oversized.failure().code == "OVERSIZED_EVENT"
    assert isinstance(blocked, Failure) and blocked.failure().code == "INVALID_EVENT"
    assert isinstance(invalid_cursor, Failure) and invalid_cursor.failure().code == "INVALID_CURSOR"
    assert isinstance(excessive_batch, Failure) and excessive_batch.failure().code == "OVERLOADED"
    assert isinstance(mismatch, Failure) and mismatch.failure().code == "INVALID_EVENT"


@pytest.mark.asyncio
async def test_terminal_requires_classification_and_retry_cannot_change_it(
    tmp_path: Path,
) -> None:
    journal = SQLiteEventJournal(tmp_path / "classified.sqlite3")
    terminal = ActionFailure(
        record_id="failure:classified:CANCELLED",
        timestamp=NOW,
        producer=PRODUCER,
        request_id="classified",
        capability="test.echo",
        code="CANCELLED",
        message="cancelled",
    )
    missing = await journal.append(terminal)
    accepted = await journal.append(terminal, data_class=DataClass.LOCAL)
    changed = await journal.append(terminal, data_class=DataClass.PRIVATE)
    contradictory = await journal.append(
        ActionResult(
            record_id="result:restart-contradiction",
            timestamp=NOW,
            producer=PRODUCER,
            request_id="classified",
            capability="test.echo",
            output={"text": "should-not-commit"},
            verification={"compare-output": True},
        ),
        data_class=DataClass.LOCAL,
    )

    assert isinstance(missing, Failure) and missing.failure().code == "INVALID_EVENT"
    assert isinstance(accepted, Success)
    assert accepted.unwrap().data_class == DataClass.LOCAL
    assert isinstance(changed, Failure) and changed.failure().code == "CONFLICT"
    assert (
        isinstance(contradictory, Failure)
        and contradictory.failure().code == "CONFLICT"
    )


@pytest.mark.asyncio
async def test_slow_subscription_is_bounded_without_blocking_append(tmp_path: Path) -> None:
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    service = EventService(journal)
    for index in range(4):
        assert isinstance(await journal.append(observation(f"event-{index}")), Success)

    subscription = service.subscribe(
        EventCursor.from_sequence(0), batch_size=1, queue_size=1, poll_interval=0
    )
    first = await anext(subscription)
    assert isinstance(first, Success)
    await asyncio.sleep(0.01)
    appended = await journal.append(observation("collector-not-blocked"))
    overload = await anext(subscription)
    assert isinstance(appended, Success)
    assert isinstance(overload, Failure) and overload.failure().code == "OVERLOADED"


@pytest.mark.asyncio
async def test_acceptance_slice_persists_one_terminal_before_ack(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    journal = SQLiteEventJournal(path)
    audit = InMemoryAuditLedger()
    service = CapabilityService(
        create_default_capability_registry(), PolicyEngine(), audit
    )
    source = await journal.append(observation())
    assert isinstance(source, Success)
    dispatcher = DeterministicEventDispatcher(journal, service)
    terminal = await dispatcher.process_next()

    assert isinstance(terminal, Success)
    assert terminal.unwrap() is not None
    assert terminal.unwrap().record_type == "action_result"
    assert terminal.unwrap().output == {"text": "person_present:true"}
    assert [entry.stage for entry in audit.entries] == ["DECISION", "TERMINAL"]
    position = await journal.position(dispatcher.consumer_id)
    assert isinstance(position, Success) and position.unwrap() == source.unwrap().cursor

    restarted = SQLiteEventJournal(path)
    all_events = await restarted.read(EventCursor.from_sequence(0), 10)
    assert isinstance(all_events, Success)
    record_types = [item.record.record_type for item in all_events.unwrap().events]
    assert record_types == ["observation", "policy_decision", "action_result"]
    assert record_types.count("action_result") == 1


@pytest.mark.asyncio
async def test_cancelled_terminal_survives_restart_before_ack_without_success_retry(
    tmp_path: Path,
) -> None:
    class FailFirstAck:
        def __init__(self, journal: SQLiteEventJournal) -> None:
            self.journal = journal

        def __getattr__(self, name: str) -> Any:  # noqa: ANN401
            return getattr(self.journal, name)

        async def acknowledge(
            self, consumer_id: str, cursor: EventCursor
        ) -> Result[EventCursor, JournalFailure]:
            del consumer_id, cursor
            return Failure(
                JournalFailure(
                    code="JOURNAL_UNAVAILABLE", message="simulated crash", retryable=True
                )
            )

    path = tmp_path / "restart.sqlite3"
    journal = SQLiteEventJournal(path)
    await journal.append(observation("restart-source"))
    service = CapabilityService(
        create_default_capability_registry(), PolicyEngine(), InMemoryAuditLedger()
    )
    cancellation = CancellationToken()
    cancellation.cancel()
    interrupted = await DeterministicEventDispatcher(
        FailFirstAck(journal),  # type: ignore[arg-type]
        service,
        consumer_id="restart-consumer",
    ).process_next(cancellation)
    assert isinstance(interrupted, Failure)

    restarted = SQLiteEventJournal(path)
    completed = await DeterministicEventDispatcher(
        restarted, service, consumer_id="restart-consumer"
    ).process_next()
    assert isinstance(completed, Success)
    replay = await restarted.read(EventCursor.from_sequence(0), 10)
    assert isinstance(replay, Success)
    terminals = [
        item
        for item in replay.unwrap().events
        if item.record.record_type in {"action_result", "action_failure"}
    ]
    assert len(terminals) == 1
    assert terminals[0].record.record_type == "action_failure"
    assert terminals[0].record.code == "CANCELLED"


@pytest.mark.asyncio
async def test_unavailable_actuator_policy_denial_and_cancellation_are_terminal(
    tmp_path: Path,
) -> None:
    class DenyPolicy(PolicyEngine):
        def evaluate(
            self, request: CapabilityRequest, descriptor: CapabilityDescriptor
        ) -> PolicyDecision:
            return super().evaluate(request, descriptor).model_copy(
                update={"outcome": PolicyOutcome.DENY, "reason_codes": ("TEST_DENIAL",)}
            )

    def unavailable(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("test actuator unavailable")

    unavailable_registry = CapabilityRegistry()
    echo_descriptor = create_default_capability_registry().descriptor("test.echo")
    assert echo_descriptor is not None
    unavailable_registry.register(
        RegisteredCapability(echo_descriptor, unavailable)
    )

    for name, registry, policy, cancellation, expected in (
        (
            "unavailable",
            unavailable_registry,
            PolicyEngine(),
            None,
            "CAPABILITY_FAILED",
        ),
        (
            "denied",
            create_default_capability_registry(),
            DenyPolicy(),
            None,
            "POLICY_DENIED",
        ),
        (
            "cancelled",
            create_default_capability_registry(),
            PolicyEngine(),
            CancellationToken(),
            "CANCELLED",
        ),
    ):
        journal = SQLiteEventJournal(tmp_path / f"{name}.sqlite3")
        item = observation(name)
        await journal.append(item)
        if cancellation is not None:
            cancellation.cancel()
        audit = InMemoryAuditLedger()
        dispatcher = DeterministicEventDispatcher(
            journal,
            CapabilityService(registry, policy, audit),
            consumer_id=f"consumer-{name}",
        )
        result = await dispatcher.process_next(cancellation)
        assert isinstance(result, Success)
        if expected is not None:
            assert result.unwrap() is not None and result.unwrap().code == expected
            persisted = await journal.terminal_for(f"request:{name}")
            assert isinstance(persisted, Success)
            assert persisted.unwrap() == result.unwrap()
            assert [entry.stage for entry in audit.entries] == [
                "DECISION",
                "TERMINAL",
            ]


def test_http_ingest_replay_ack_and_authentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "stage2-secret")
    container = ApplicationContainer(
        config={}, testing=True, event_journal_path=tmp_path / "events.sqlite3"
    )
    client = ASGITestClient(create_app(container))
    payload = observation().model_dump(mode="json")
    assert client.post("/api/v1/events", json=payload).status_code == UNAUTHORIZED

    headers = {TOKEN_HEADER: "stage2-secret"}
    accepted = client.post("/api/v1/events", json=payload, headers=headers)
    assert accepted.status_code == OK
    cursor = accepted.json()["cursor"]
    zero = EventCursor.from_sequence(0).value
    replay = client.get(f"/api/v1/events?cursor={zero}&limit=1", headers=headers)
    assert replay.status_code == OK
    assert replay.json()["events"][0]["record"]["record_id"] == "person-1"
    ack = client.post(
        "/api/v1/events/consumers/http-test/ack", json=cursor, headers=headers
    )
    assert ack.status_code == OK
    subscribed = client.get(
        f"/api/v1/events/subscriptions/http-test?cursor={zero}&limit=1", headers=headers
    )
    assert subscribed.status_code == OK
    assert subscribed.json()["events"] == []


@pytest.mark.asyncio
async def test_unix_socket_is_private_authenticated_and_uses_same_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAI_DISABLE_AUTH", raising=False)
    monkeypatch.setenv("RAI_API_TOKEN", "socket-secret")
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    socket_path = tmp_path / "events.sock"
    server = EventSocketServer(EventService(journal), journal, socket_path)
    await server.start()
    try:
        assert socket_path.stat().st_mode & 0o777 == PRIVATE_SOCKET_MODE
        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            json.dumps(
                {"operation": "ingest", "token": "wrong", "event": observation().model_dump(mode="json")}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        denied = json.loads(await reader.readline())
        assert denied["error"]["code"] == "UNAUTHORIZED"
        writer.close()
        await writer.wait_closed()

        reader, writer = await asyncio.open_unix_connection(socket_path)
        writer.write(
            json.dumps(
                {"operation": "ingest", "token": "socket-secret", "event": observation().model_dump(mode="json")}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        accepted = json.loads(await reader.readline())
        assert accepted["result"]["record"]["record_id"] == "person-1"
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_unix_socket_refuses_to_replace_non_socket_path(tmp_path: Path) -> None:
    journal = SQLiteEventJournal(tmp_path / "events.sqlite3")
    socket_path = tmp_path / "events.sock"
    socket_path.write_text("do-not-delete", encoding="utf-8")
    server = EventSocketServer(EventService(journal), journal, socket_path)

    with pytest.raises(RuntimeError, match="refusing to replace non-socket"):
        await server.start()
    assert socket_path.read_text(encoding="utf-8") == "do-not-delete"


def test_sqlite_journal_satisfies_versioned_port(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite3"
    assert isinstance(SQLiteEventJournal(path), EventJournal)
    assert not path.exists()


@pytest.mark.asyncio
async def test_runtime_lifecycle_starts_and_stops_local_event_plane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAI_RUNTIME_DIR", str(tmp_path / "run"))
    container = ApplicationContainer(
        config={},
        testing=False,
        audit_ledger=InMemoryAuditLedger(),
        event_journal_path=tmp_path / "events.sqlite3",
    )
    await container.start()
    socket_path = tmp_path / "run" / "events-v1.sock"
    try:
        assert socket_path.exists()
        assert container._dispatcher_task is not None  # pylint: disable=protected-access
        assert not container._dispatcher_task.done()  # pylint: disable=protected-access
    finally:
        await container.close()
    assert not socket_path.exists()


def test_published_event_schema_matches_runtime_contract() -> None:
    published = Path("schemas/rai.events.v1.schema.json")
    assert json.loads(published.read_text(encoding="utf-8")) == event_json_schema()
