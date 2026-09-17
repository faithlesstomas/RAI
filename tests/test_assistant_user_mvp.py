"""User-visible acceptance tests for the Rich Assistant MVP after Issue #34."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
from fastapi import WebSocketDisconnect
import pytest
from returns.result import Success

from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.derived import VerifiedDerivedClaim, write_verified_derived_claim
from rai.assistant.diagnostics import diagnose_memory
from rai.assistant.ports import MemoryQuery
from rai.assistant.records import ConversationTurn, MemoryOperationKind
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.cli import cli
from rai.container import ApplicationContainer
from rai.kernel.records import DataClass, ProducerIdentity
from rai.routers.assistant import assistant_websocket
from rai.server import create_app
from conftest import ASGITestClient

PRODUCER = ProducerIdentity(producer_id="user-mvp-test", kind="test", version="1.0.0")
HTTP_OK = 200


def _turn(record_id: str, session_id: str, text: str) -> ConversationTurn:
    return ConversationTurn(
        record_id=record_id,
        producer=PRODUCER,
        session_id=session_id,
        role="user",
        text=text,
    )


@pytest.mark.asyncio
async def test_natural_memory_recall_context_inspection_forget_and_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "assistant.sqlite3"
    store = SQLiteMemoryGraphStore(database)
    service = AssistantService(store=store, backend=DeterministicAssistantBackend())
    await service.start()

    remembered = await service.accept_turn(
        _turn("turn-colour", "session-a", "Mój ulubiony kolor jest zielony."),
        request_id="remember-colour",
    )
    assert isinstance(remembered, Success)
    assert remembered.unwrap().admitted_memory_ids
    assert remembered.unwrap().memory_operation_ids

    recalled = await service.accept_turn(
        _turn("turn-question", "session-b", "Jaki jest mój ulubiony kolor?"),
        request_id="recall-colour",
    )
    assert isinstance(recalled, Success)
    response = recalled.unwrap()
    assert "zielony" in response.text

    package = await store.get_context_package(response.manifest_id)
    assert isinstance(package, Success)
    assert package.unwrap() is not None
    assert package.unwrap().manifest.durable_memory_ids == (
        remembered.unwrap().admitted_memory_ids[0],
    )
    assert package.unwrap().content["recent_turns"] == ()

    diagnostic = await diagnose_memory(
        store,
        response=response,
        expected_answer_substring="zielony",
    )
    assert isinstance(diagnostic, Success)
    assert diagnostic.unwrap().healthy
    assert {stage.stage: stage.status for stage in diagnostic.unwrap().stages}[
        "ANSWER_USE"
    ] == "PASS"

    derived = await write_verified_derived_claim(
        store,
        VerifiedDerivedClaim(
            verification_id="verify-colour-derived",
            verifier_id="user-mvp-test-verifier",
            verifier_version="1.0.0",
            policy_version="derived-writeback-v1",
            topic="derived.personal.colour-summary",
            content={"finding": "preferred colour is green"},
            source_memory_ids=(remembered.unwrap().admitted_memory_ids[0],),
            domain_scope="personal",
        ),
    )
    assert isinstance(derived, Success)

    forgotten = await service.accept_turn(
        _turn("turn-forget", "session-c", "Zapomnij mój ulubiony kolor."),
        request_id="forget-colour",
    )
    assert isinstance(forgotten, Success)
    assert "Usunąłem 2" in forgotten.unwrap().text
    removed_derived = await store.get_memory(derived.unwrap().record_id)
    assert isinstance(removed_derived, Success)
    removed_derived_record = removed_derived.unwrap()
    assert removed_derived_record is not None
    assert removed_derived_record[1] == "DELETED"

    active = await store.retrieve_relevant_memories(
        query=MemoryQuery(keywords=("ulubiony", "kolor"))
    )
    replayed = await store.replay_memory_projection()
    operations = await store.list_memory_operations()
    assert isinstance(active, Success) and active.unwrap() == ()
    assert isinstance(replayed, Success) and replayed.unwrap() == ()
    raw = await store.retrieve_relevant_turns(
        "default",
        MemoryQuery(keywords=("ulubiony", "kolor")),
        (DataClass.PUBLIC, DataClass.LOCAL),
    )
    assert isinstance(raw, Success)
    assert {
        turn.record_id for turn, _reason in raw.unwrap()
    }.isdisjoint({"turn-colour", remembered.unwrap().turn_id})
    assert all("zielony" not in turn.text.casefold() for turn, _reason in raw.unwrap())
    suppressed_recent = await store.get_recent_reply_chain(
        "session-a", profile_scope="default"
    )
    assert isinstance(suppressed_recent, Success)
    assert all(
        turn.record_id
        not in {"turn-colour", remembered.unwrap().turn_id}
        for turn in suppressed_recent.unwrap()
    )
    assert isinstance(operations, Success)
    assert [operation.operation for operation in operations.unwrap()] == [
        MemoryOperationKind.REMEMBER.value,
        MemoryOperationKind.REFLECT.value,
        MemoryOperationKind.FORGET.value,
    ]

    await service.stop()
    store = SQLiteMemoryGraphStore(database)
    service = AssistantService(
        store=store, backend=DeterministicAssistantBackend()
    )
    await service.start()
    raw_after_restart = await store.retrieve_relevant_turns(
        "default",
        MemoryQuery(keywords=("ulubiony", "kolor")),
        (DataClass.PUBLIC, DataClass.LOCAL),
    )
    assert isinstance(raw_after_restart, Success)
    assert {
        turn.record_id for turn, _reason in raw_after_restart.unwrap()
    }.isdisjoint({"turn-colour", remembered.unwrap().turn_id})
    assert all(
        "zielony" not in turn.text.casefold()
        for turn, _reason in raw_after_restart.unwrap()
    )

    recalled_after_forget = await service.accept_turn(
        _turn("turn-after-forget", "session-d", "Jaki jest mój ulubiony kolor?"),
        request_id="recall-after-forget",
    )
    assert isinstance(recalled_after_forget, Success)
    assert "zielony" not in recalled_after_forget.unwrap().text.casefold()
    await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Chyba mój ulubiony napój to herbata.",
        'Ktoś powiedział: "Mój ulubiony napój to herbata."',
    ],
)
async def test_uncertain_or_hearsay_memory_is_visible_as_rejected_operation(
    tmp_path: Path, text: str
) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3")
    service = AssistantService(store=store, backend=DeterministicAssistantBackend())
    await service.start()

    result = await service.accept_turn(_turn("turn-unsafe", "session", text))
    memories = await store.retrieve_relevant_memories()
    operations = await store.list_memory_operations()

    assert isinstance(result, Success)
    assert result.unwrap().admitted_memory_ids == ()
    assert "Nie zapisałem" in result.unwrap().text
    assert isinstance(memories, Success) and memories.unwrap() == ()
    assert isinstance(operations, Success)
    assert operations.unwrap()[-1].status == "REJECTED"
    assert operations.unwrap()[-1].stage == "ADMISSION"


@pytest.mark.asyncio
async def test_natural_correction_supersedes_memory_and_replays_current_value(
    tmp_path: Path,
) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3")
    service = AssistantService(store=store, backend=DeterministicAssistantBackend())
    await service.start()

    first = await service.accept_turn(
        _turn("turn-colour-green", "session", "Mój ulubiony kolor jest zielony.")
    )
    corrected = await service.accept_turn(
        _turn("turn-colour-blue", "session", "Mój ulubiony kolor jest niebieski.")
    )
    recalled = await service.accept_turn(
        _turn("turn-colour-recall", "other-session", "Jaki jest mój ulubiony kolor?")
    )
    operations = await store.list_memory_operations()
    replayed = await store.replay_memory_projection()

    assert isinstance(first, Success)
    assert isinstance(corrected, Success)
    assert isinstance(recalled, Success) and "niebieski" in recalled.unwrap().text
    assert isinstance(operations, Success)
    assert [operation.operation for operation in operations.unwrap()] == [
        MemoryOperationKind.REMEMBER.value,
        MemoryOperationKind.SUPERSEDE.value,
    ]
    assert isinstance(replayed, Success)
    assert replayed.unwrap() == corrected.unwrap().admitted_memory_ids


def test_cli_exposes_sessions_history_memory_context_and_operations(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    environment = {"RAI_DATA_DIR": str(tmp_path / "data")}
    config = {"assistant": {"backend": "deterministic"}}

    with patch("rai.cli_commands._assistant_config", return_value=config):
        answer = runner.invoke(
            cli,
            [
                "assistant",
                "ask",
                "Mój ulubiony kolor jest zielony.",
                "--backend",
                "deterministic",
                "--session-id",
                "useful-cli",
            ],
            env=environment,
        )
        sessions = runner.invoke(cli, ["assistant", "sessions"], env=environment)
        history = runner.invoke(
            cli,
            ["assistant", "history", "--session-id", "useful-cli"],
            env=environment,
        )
        memories = runner.invoke(cli, ["assistant", "memories"], env=environment)
        context = runner.invoke(
            cli,
            ["assistant", "context", "--session-id", "useful-cli"],
            env=environment,
        )
        operations = runner.invoke(cli, ["assistant", "operations"], env=environment)
        diagnostics = runner.invoke(cli, ["assistant", "diagnostics"], env=environment)

    for result in (
        answer,
        sessions,
        history,
        memories,
        context,
        operations,
        diagnostics,
    ):
        assert result.exit_code == 0, result.output
    assert "useful-cli" in sessions.output
    assert "Mój ulubiony kolor" in history.output
    assert "zielony" in memories.output
    assert '"context_window"' in context.output
    assert "REMEMBER APPLIED" in operations.output
    assert '"healthy": true' in diagnostics.output


class _FakeWebSocket:
    def __init__(self, service: AssistantService) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {"session_id": "native-ws"}
        self.app = SimpleNamespace(
            state=SimpleNamespace(container=SimpleNamespace(assistant_service=service))
        )
        self.sent: list[dict[str, object]] = []
        self._received = False
        self.closed: tuple[int, str] | None = None

    async def accept(self) -> None:
        return None

    async def receive_json(self) -> dict[str, object]:
        if self._received:
            raise WebSocketDisconnect()
        self._received = True
        return {"prompt": "Mój ulubiony kolor jest zielony."}

    async def send_json(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


@pytest.mark.asyncio
async def test_native_websocket_uses_the_same_memory_committing_service(
    tmp_path: Path,
) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3"),
        backend=DeterministicAssistantBackend(),
    )
    websocket = _FakeWebSocket(service)

    await assistant_websocket(websocket)  # type: ignore[arg-type]

    assert websocket.sent[0] == {"type": "session", "session_id": "native-ws"}
    response = websocket.sent[1]
    assert response["type"] == "response"
    assert response["payload"]["session_id"] == "native-ws"  # type: ignore[index]
    assert response["payload"]["admitted_memory_ids"]  # type: ignore[index]
    assert response["payload"]["memory_operation_ids"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_native_websocket_rejects_unauthorized_clients(tmp_path: Path) -> None:
    service = AssistantService(
        store=SQLiteMemoryGraphStore(tmp_path / "assistant.sqlite3"),
        backend=DeterministicAssistantBackend(),
    )
    websocket = _FakeWebSocket(service)

    with patch("rai.routers.assistant.is_authorized", return_value=False):
        await assistant_websocket(websocket)  # type: ignore[arg-type]

    assert websocket.closed == (1008, "Unauthorized")
    assert websocket.sent == []


def test_authenticated_rest_surface_supports_external_clients(tmp_path: Path) -> None:
    container = ApplicationContainer(
        config={},
        testing=True,
        assistant_memory_path=tmp_path / "assistant.sqlite3",
    )
    client = ASGITestClient(create_app(container))

    turn = client.post(
        "/api/v1/assistant/turn",
        json={
            "prompt": "Mój ulubiony kolor jest zielony.",
            "session_id": "rest-client",
            "request_id": "rest-request",
        },
    )
    assert turn.status_code == HTTP_OK
    payload = turn.json()
    assert payload["admitted_memory_ids"]
    assert payload["memory_operation_ids"]

    memories = client.get("/api/v1/assistant/memories")
    history = client.get("/api/v1/assistant/sessions/rest-client/turns")
    context = client.get("/api/v1/assistant/sessions/rest-client/context")
    operations = client.get("/api/v1/assistant/memory-operations")
    diagnostics = client.get("/api/v1/assistant/diagnostics/memory")
    sessions = client.get("/api/v1/assistant/sessions")

    assert memories.status_code == HTTP_OK and memories.json()[0]["topic"]
    assert history.status_code == HTTP_OK
    assert [item["role"] for item in history.json()] == ["user", "assistant"]
    assert context.status_code == HTTP_OK
    assert context.json()["manifest"]["record_id"] == payload["manifest_id"]
    assert operations.status_code == HTTP_OK
    assert operations.json()[0]["operation"] == "REMEMBER"
    assert diagnostics.status_code == HTTP_OK
    assert diagnostics.json()["healthy"] is True
    assert sessions.status_code == HTTP_OK
    assert sessions.json()[0]["session_id"] == "rest-client"
