"""Additional edge-case tests for ActionIntent validation, LocalIntentRecognizer, and AssistantActions."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from returns.result import Failure, Result, Success

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
from rai.kernel.records import ActionFailure, ProducerIdentity
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


async def test_local_intent_recognizer_includes_recent_dialogue_from_store(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    await store.start()
    t1 = ConversationTurn(producer=ACTOR, session_id="s1", role="user", text="Poprzednie pytanie", metadata={"profile_scope": "default"}, status="COMPLETED")
    t2 = ConversationTurn(producer=ACTOR, session_id="s1", role="assistant", text="Poprzednia odpowiedź", metadata={"profile_scope": "default"}, status="COMPLETED", reply_to_turn_id=t1.record_id)
    await store.accept_turn(t1)
    await store.accept_turn(t2)

    current_turn = ConversationTurn(producer=ACTOR, session_id="s1", role="user", text="Bieżące pytanie")

    class CapturingEngine:
        captured_source = None
        async def generate(self, **kwargs: Any) -> Any:
            self.captured_source = json.loads(kwargs["messages"][1]["content"])
            return Success(InferenceResult(
                text=json.dumps({"source_turn_id": current_turn.record_id, "outcome": "no_action", "language": "pl"}),
                stats=GenerationStats(10, 10, 0.1, 100),
            ))

    engine = CapturingEngine()
    recognizer = LocalIntentRecognizer(engine, store=store)
    result = await recognizer.recognize(current_turn, CancellationToken())
    assert isinstance(result, Success)
    assert engine.captured_source is not None
    assert "recent_dialogue" in engine.captured_source
    dialogue = engine.captured_source["recent_dialogue"]
    assert len(dialogue) == 2
    assert dialogue[0]["text"] == "Poprzednie pytanie"
    assert dialogue[1]["text"] == "Poprzednia odpowiedź"
    await store.stop()


async def test_local_intent_recognizer_graceful_fallbacks() -> None:
    turn = user_turn("Zadna z podanych. Chodzi o cos innego")

    class EngineFallback:
        payload: dict[str, Any] = {}
        async def generate(self, **kwargs: Any) -> Any:
            return Success(InferenceResult(
                text=json.dumps(self.payload),
                stats=GenerationStats(10, 10, 0.1, 100),
            ))

    engine = EngineFallback()
    recognizer = LocalIntentRecognizer(engine)

    # Empty clarify choices gracefully falls back to no_action
    engine.payload = {"source_turn_id": turn.record_id, "outcome": "clarify", "language": "pl", "question": "Co?", "options": []}
    res_clarify = await recognizer.recognize(turn, CancellationToken())
    assert isinstance(res_clarify, Success)
    assert res_clarify.unwrap().outcome == "no_action"

    # Empty launch query gracefully falls back to no_action
    engine.payload = {"source_turn_id": turn.record_id, "outcome": "application.launch", "language": "pl", "query": ""}
    res_launch = await recognizer.recognize(turn, CancellationToken())
    assert isinstance(res_launch, Success)
    assert res_launch.unwrap().outcome == "no_action"


async def test_application_list_matches_executable_name(tmp_path: Path) -> None:
    from rai.actions.applications import Application
    from rai.actions.capabilities import register_application_capabilities
    from rai.kernel.service import CapabilityService
    from rai.kernel.transport import normalize_request

    class MockAppBackend:
        async def discover(self) -> Any:
            return Success((
                Application(
                    desktop_id="org.gnome.Nautilus.desktop",
                    name="Files",
                    path=Path("/usr/share/applications/org.gnome.Nautilus.desktop"),
                    fingerprint="fp1",
                    executable="/usr/bin/nautilus",
                ),
            ))
        async def launch(self, *args: Any, **kwargs: Any) -> Any:
            return Success(None)

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_application_capabilities(registry, MockAppBackend(), handles)
    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger())
    listing = normalize_request(registry.descriptor("application.list"), {"task_id": "t1", "query": "nautilus"})
    _, result = await service.invoke(listing)
    assert isinstance(result, Success)
    apps = result.unwrap().output["applications"]
    assert len(apps) == 1
    assert apps[0]["name"] == "Files"
    assert apps[0]["desktop_id"] == "org.gnome.Nautilus.desktop"


async def test_action_output_does_not_bypass_context_manifest(tmp_path: Path) -> None:
    from rai.actions.applications import Application
    from rai.actions.capabilities import register_application_capabilities
    from rai.assistant.records import AssistantCandidate, AssistantResponse
    from rai.assistant.service import AssistantService
    from rai.kernel.ports import LifecycleState

    class MockAppBackend:
        async def discover(self) -> Any:
            return Success((
                Application("files.desktop", "Files", Path("/files.desktop"), "fp1", "/files"),
            ))
        async def launch(self, *args: Any, **kwargs: Any) -> Any:
            return Success(None)

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_application_capabilities(registry, MockAppBackend(), handles)
    runtime = ActionCapabilityService(registry, PolicyEngine(), InMemoryAuditLedger(), SyntheticApprovalBroker())
    runtime.handles = handles
    runtime.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")

    class Recognizer:
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
            return Success(ActionIntent(source_turn_id=turn.record_id, language="pl", outcome="application.list"))

    actions = AssistantActions(runtime, Recognizer(), store)

    class MockModelBackend:
        is_remote = False
        model_name = "test-model"
        received_context = None

        async def start(self) -> Result[LifecycleState, ActionFailure]:
            return Success(LifecycleState.RUNNING)

        async def stop(self) -> Result[LifecycleState, ActionFailure]:
            return Success(LifecycleState.STOPPED)

        async def generate(self, req: Any, token: Any) -> Result[AssistantCandidate, ActionFailure]:
            self.received_context = req.context
            return Success(AssistantCandidate(
                text="Przeanalizowałem: aplikacja Files służy do otwierania plików.",
                tokens_in=10,
                tokens_out=20,
            ))

    backend = MockModelBackend()
    service = AssistantService(store, backend=backend, actions=actions)
    await service.start()
    try:
        turn = ConversationTurn(
            producer=ACTOR,
            session_id="analysis-sess",
            role="user",
            text="Wypisz wszystkie aplikacje jakie możesz uruchomić i przeanalizuj, która z nich może do tego służyć.",
        )
        res = (await service.accept_turn(turn, request_id="req-analysis")).unwrap()
        assert "Files" in res.text
        assert backend.received_context is None
        recent = (await store.get_recent_reply_chain(turn.session_id, limit=1)).unwrap()
        assert len(recent) == 1
        assert recent[0].metadata["action"]["action_choices"]["entries"][0]["name"] == "Files"
    finally:
        await service.stop()


async def test_intent_recognizer_handles_null_fields_and_cot_thinking() -> None:
    turn = user_turn("O czym ostatnio rozmawialiśmy?")

    class CotThinkingEngine:
        async def generate(self, **kwargs: Any) -> Result[InferenceResult, Exception]:
            raw_text = (
                "<think>The user asks about past topics. This is conversational.</think>"
                '```json\n{"source_turn_id": "' + turn.record_id + '", "outcome": "no_action", '
                '"language": "pl", "query": null, "question": null, "options": null}\n```'
            )
            return Success(InferenceResult(text=raw_text, stats=GenerationStats(10, 10, 0.1, 100)))

    recognizer = LocalIntentRecognizer(CotThinkingEngine())
    res = await recognizer.recognize(turn, CancellationToken())
    assert isinstance(res, Success)
    intent = res.unwrap()
    assert intent.outcome == "no_action"
    assert intent.query == ""
    assert intent.options == ()


async def test_intent_recognizer_filters_and_prioritizes_catalog_tokens() -> None:
    from rai.actions.applications import Application

    turn = user_turn("otwórz nautilus")

    apps = [
        Application(f"app-{i}.desktop", f"App {i}", Path(f"/app-{i}.desktop"), f"fp{i}", f"/app-{i}")
        for i in range(100)
    ]
    nautilus = Application("org.gnome.Nautilus.desktop", "Files", Path("/nautilus.desktop"), "fpN", "/usr/bin/nautilus")
    apps.append(nautilus)

    class MockAppCatalog:
        async def discover(self) -> Result[tuple[Application, ...], ActionFailure]:
            return Success(tuple(apps))

    class CapturingCatalogEngine:
        captured_catalog = None

        async def generate(self, **kwargs: Any) -> Result[InferenceResult, Exception]:
            source = json.loads(kwargs["messages"][1]["content"])
            self.captured_catalog = source["installed_applications"]
            return Success(InferenceResult(
                text=json.dumps({"source_turn_id": turn.record_id, "outcome": "application.launch",
                                "language": "pl", "query": "org.gnome.Nautilus.desktop"}),
                stats=GenerationStats(10, 10, 0.1, 100),
            ))

    engine = CapturingCatalogEngine()
    recognizer = LocalIntentRecognizer(engine, applications=MockAppCatalog())
    res = await recognizer.recognize(turn, CancellationToken())
    assert isinstance(res, Success)
    assert engine.captured_catalog is not None
    assert len(engine.captured_catalog) <= 30
    assert any(entry["desktop_id"] == "org.gnome.Nautilus.desktop" for entry in engine.captured_catalog)




@pytest.mark.parametrize("classification,profile", [("PRIVATE", "default"), ("LOCAL", "default"), ("PUBLIC", "other")])
async def test_recognizer_excludes_ineligible_dialogue(tmp_path: Path, classification: str, profile: str) -> None:
    from rai.kernel.records import DataClass
    store = SQLiteMemoryGraphStore(tmp_path / "privacy.db")
    await store.start()
    try:
        past = ConversationTurn(producer=ACTOR, session_id="privacy", role="user", text="excluded-secret",
            data_class=DataClass(classification), metadata={"profile_scope": profile}, status="COMPLETED")
        await store.accept_turn(past)
        current = ConversationTurn(producer=ACTOR, session_id="privacy", role="user", text="hello", data_class=DataClass.PUBLIC)
        class Engine:
            async def generate(self, **kwargs: Any) -> Any:
                assert "excluded-secret" not in json.dumps(kwargs["messages"])
                return Success(InferenceResult(text=json.dumps({"source_turn_id": current.record_id,
                    "outcome": "no_action", "language": "en"}), stats=GenerationStats(1, 1, 0.1, 10)))
        assert isinstance(await LocalIntentRecognizer(Engine(), store).recognize(current, CancellationToken()), Success)
    finally:
        await store.stop()
