"""Additional edge-case tests for ActionIntent validation, LocalIntentRecognizer, and AssistantActions."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from returns.result import Failure, Success

from rai.actions.assistant import AssistantActions
from rai.actions.capabilities import register_application_capabilities
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.files import Document, register_file_capabilities
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent, LocalIntentRecognizer
from rai.actions.service import ActionCapabilityService
from rai.assistant.records import ConversationTurn
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.inference.protocols import GenerationStats, InferenceResult
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ProducerIdentity
from rai.kernel.synthetic import SyntheticApprovalBroker

ACTOR = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")


def user_turn(text: str = "test", session_id: str = "sess", role: str = "user") -> ConversationTurn:
    return ConversationTurn(producer=ACTOR, session_id=session_id, role=role, text=text)


# --- ActionIntent validation edges ---


def test_action_intent_validation_rules() -> None:
    base = {"source_turn_id": "turn-1", "language": "pl"}

    with pytest.raises(ValidationError, match="this action needs a target query"):
        ActionIntent(**base, outcome="file.search", query="")

    with pytest.raises(ValidationError, match="this action needs a target query"):
        ActionIntent(**base, outcome="browser.search", query="   ")

    with pytest.raises(ValidationError, match="this action needs a target query"):
        ActionIntent(**base, outcome="process.inspect", query="")

    with pytest.raises(ValidationError, match="launch needs a query or previous selection"):
        ActionIntent(**base, outcome="application.launch", query="", selection=None)

    with pytest.raises(ValidationError, match="volume change needs an absolute percentage"):
        ActionIntent(**base, outcome="system.volume.set", percent=None)

    with pytest.raises(ValidationError, match="clarification needs a question and at least two choices"):
        ActionIntent(**base, outcome="clarify", question="", options=("A", "B"))

    with pytest.raises(ValidationError, match="clarification needs a question and at least two choices"):
        ActionIntent(**base, outcome="clarify", question="Why?", options=("A",))

    with pytest.raises(ValidationError, match="invalid clarification option"):
        ActionIntent(**base, outcome="clarify", question="Why?", options=("A", "   "))

    with pytest.raises(ValidationError, match="invalid clarification option"):
        ActionIntent(**base, outcome="clarify", question="Why?", options=("A", "x" * 300))

    valid = ActionIntent(**base, outcome="system.volume.set", percent=50)
    assert valid.percent == 50


# --- LocalIntentRecognizer edges ---


async def test_local_intent_recognizer_backend_failures_and_timeouts() -> None:
    turn = user_turn("otwórz")

    class FailingEngine:
        async def generate(self, **_: Any) -> Any:
            return Failure("MODEL_ERROR")

    recognizer = LocalIntentRecognizer(FailingEngine())
    assert (await recognizer.recognize(turn, CancellationToken())) == Failure("INTENT_BACKEND_FAILED")

    class TimeoutEngine:
        async def generate(self, **_: Any) -> Any:
            await asyncio.Event().wait()

    recognizer = LocalIntentRecognizer(TimeoutEngine())
    token = CancellationToken()
    token.cancel()
    assert (await recognizer.recognize(turn, token)) == Failure("CANCELLED")

    class OversizedEngine:
        async def generate(self, **_: Any) -> Any:
            return Success(InferenceResult(text="x" * 10000, stats=GenerationStats(1, 1, 0.1, 10)))

    recognizer = LocalIntentRecognizer(OversizedEngine())
    assert (await recognizer.recognize(turn, CancellationToken())) == Failure("INTENT_OUTPUT_TOO_LARGE")

    class CrashingEngine:
        async def generate(self, **_: Any) -> Any:
            raise RuntimeError("crash")

    recognizer = LocalIntentRecognizer(CrashingEngine())
    assert (await recognizer.recognize(turn, CancellationToken())) == Failure("INTENT_BACKEND_FAILED")


async def test_local_intent_recognizer_catalog_errors() -> None:
    turn = user_turn("otwórz")

    class FailingCatalog:
        async def discover(self) -> Any:
            return Failure("IO_ERROR")

    class DummyEngine:
        async def generate(self, **_: Any) -> Any:
            return Success(InferenceResult(text="{}", stats=GenerationStats(1, 1, 0.1, 10)))

    recognizer = LocalIntentRecognizer(DummyEngine(), applications=FailingCatalog())
    assert (await recognizer.recognize(turn, CancellationToken())) == Failure("APPLICATION_CATALOG_UNAVAILABLE")

    class TimeoutCatalog:
        async def discover(self) -> Any:
            await asyncio.Event().wait()

    recognizer = LocalIntentRecognizer(DummyEngine(), applications=TimeoutCatalog())
    assert (await recognizer.recognize(turn, CancellationToken())) == Failure("APPLICATION_CATALOG_UNAVAILABLE")


# --- AssistantActions edges ---


class MockRecognizer:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Any:
        return self.result


async def test_assistant_actions_non_user_or_no_action(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    rec = MockRecognizer(Success(ActionIntent(source_turn_id="t1", outcome="no_action", language="pl")))
    actions = AssistantActions(service, rec)

    # non-user turn
    assert (await actions.handle(user_turn(role="assistant"), CancellationToken())) is None

    # no-action turn
    assert (await actions.handle(user_turn(role="user"), CancellationToken())) is None


async def test_assistant_actions_recognizer_failure(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    rec = MockRecognizer(Failure("UNRECOGNIZED"))
    actions = AssistantActions(service, rec)

    res = await actions.handle(user_turn(), CancellationToken())
    assert res is not None
    assert "Nie udało się" in res.text
    assert res.metadata["intent_failure"] == "UNRECOGNIZED"


async def test_assistant_actions_clarify(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    intent = ActionIntent(source_turn_id="t1", outcome="clarify", language="pl",
                          question="Co zrobić?", options=("Opcja A", "Opcja B"))
    rec = MockRecognizer(Success(intent))
    actions = AssistantActions(service, rec)

    res = await actions.handle(user_turn(), CancellationToken())
    assert res is not None
    assert "Co zrobić?" in res.text
    assert "1. Opcja A" in res.text
    assert "2. Opcja B" in res.text
    assert res.metadata["action_clarification"]["outcome"] == "clarify"


async def test_assistant_actions_application_missing_or_failed_discovery(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")

    intent = ActionIntent(source_turn_id="t1", outcome="application.launch", language="pl", query="app")
    rec = MockRecognizer(Success(intent))
    actions = AssistantActions(service, rec)

    # application.list is not registered in registry
    res = await actions.handle(user_turn(), CancellationToken())
    assert res is not None
    assert "niedostępne" in res.text

    # now register a failing application backend
    class FailingApps:
        async def discover(self) -> Any:
            return Failure("BACKEND_DOWN")
        async def launch(self, app: Any, token: Any) -> Any:
            return Failure("NO")

    register_application_capabilities(registry, FailingApps(), handles)
    res2 = await actions.handle(user_turn(), CancellationToken())
    assert res2 is not None
    assert "Nie mogę sprawdzić aplikacji" in res2.text


async def test_assistant_actions_document_file_search_and_open(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "my_file.txt").write_text("hello")
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()

    class DocBackend:
        async def open(self, document: Document, cancellation: CancellationToken) -> Any:
            return Success({"pid": 100, "open_file_descriptor": True})

    register_file_capabilities(registry, (root,), handles, DocBackend())
    service = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    service.handles = handles
    service.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")

    # Search files
    intent_search = ActionIntent(source_turn_id="t1", outcome="file.search", language="pl", query="my_file")
    actions = AssistantActions(service, MockRecognizer(Success(intent_search)), store=store)
    res_search = await actions.handle(user_turn(), CancellationToken())
    assert res_search is not None
    assert "1. my_file.txt" in res_search.text
    choices = res_search.metadata["action_choices"]
    assert choices["kind"] == "file"

    # Open without previous choices -> prompt user
    intent_open = ActionIntent(source_turn_id="t2", outcome="document.open", language="pl", selection=1)
    actions_no_store = AssistantActions(service, MockRecognizer(Success(intent_open)))
    res_no_store = await actions_no_store.handle(user_turn(), CancellationToken())
    assert res_no_store is not None
    assert "Który dokument mam otworzyć?" in res_no_store.text
