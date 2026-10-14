# ruff: noqa
"""Independent MR !36 audit probes converted into permanent regression tests."""
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from returns.result import Failure, Success

from rai.actions.assistant import AssistantActions
from rai.actions.browser import register_browser_capabilities
from rai.actions.choices import find_active_choice_frame
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent
from rai.actions.service import ActionCapabilityService
from rai.actions.workflow import ResearchWorkflowLimits, TaskStepController
from rai.assistant.context import AssistantContextBuilder
from rai.assistant.records import ConversationTurn
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, ProducerIdentity, _utc_now
from rai.kernel.synthetic import SyntheticApprovalBroker
from tests.test_assistant_chat_regression import MockBrowserBackend, MockIntentRecognizer

ACTOR = ProducerIdentity(producer_id="audit-user", kind="user", version="1")


class Recognizer(MockIntentRecognizer):
    async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken):
        if turn.text.startswith("przeczytaj"):
            self.calls.append(turn.text)
            return Success(
                ActionIntent(
                    source_turn_id=turn.record_id,
                    language="pl",
                    outcome="browser.read_page",
                    selection=1,
                )
            )
        return await super().recognize(turn, cancellation)


@pytest_asyncio.fixture
async def stack(tmp_path):
    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    browser = MockBrowserBackend()
    register_browser_capabilities(registry, handles, browser)
    capabilities = ActionCapabilityService(
        registry,
        PolicyEngine(),
        InMemoryAuditLedger(),
        SyntheticApprovalBroker(),
    )
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    recognizer = Recognizer()
    actions = AssistantActions(capabilities, recognizer, store, profile="default")
    service = AssistantService(store=store, actions=actions)
    await service.start()

    async def ask(text: str):
        turn = ConversationTurn(
            producer=ACTOR,
            session_id="audit",
            role="user",
            text=text,
            data_class=DataClass.PUBLIC,
        )
        result = await service.accept_turn(turn)
        assert isinstance(result, Success), result
        return result.unwrap()

    yield ask, browser, recognizer, service, capabilities
    await service.stop()


@pytest.mark.asyncio
async def test_open_then_read_keeps_selected_resource(stack):
    ask, browser, _, service, _ = stack
    await ask("Poszukaj w sieci informacji o modelach")
    await ask("1")
    assert len(browser.opened_urls) == 1
    response = await ask("przeczytaj 1")
    assert browser.read_urls == ["https://example.com/gemini-4"], response.text
    # Verify in-turn synthesis: text is synthesized by the model, not a raw dump
    assert "Odpowiedź na podstawie odczytanej strony" in response.text
    recent = (await service.store.get_recent_reply_chain("audit", limit=1)).unwrap()
    assert recent[-1].metadata.get("action", {}).get("synthesized_from_action") is True


@pytest.mark.asyncio
async def test_choice_survives_three_intermediate_exchanges(stack):
    ask, browser, recognizer, _, _ = stack
    await ask("Poszukaj w sieci informacji o modelach")
    for text in ("Co to znaczy?", "Wyjaśnij więcej.", "Dziękuję."):
        await ask(text)
    calls_before = len(recognizer.calls)
    response = await ask("5")
    assert len(browser.opened_urls) == 1, (response.text, recognizer.calls[calls_before:])


@pytest.mark.asyncio
async def test_cancelled_frame_does_not_fall_back_to_model_execution(stack):
    ask, browser, _, service, capabilities = stack
    await ask("Poszukaj w sieci informacji o modelach")
    recent = (await service.store.get_recent_reply_chain("audit", limit=2)).unwrap()
    search_turn = next(t for t in recent if t.role == "assistant")
    metadata = dict(search_turn.metadata)
    action = dict(metadata["action"])
    action["active_choice_frame"] = {
        **action["active_choice_frame"],
        "state": "CANCELLED",
    }
    metadata["action"] = action
    cancelled_turn = search_turn.model_copy(update={"metadata": metadata})
    store = AsyncMock()
    store.get_recent_reply_chain.return_value = Success((cancelled_turn,))
    recognizer = AsyncMock()
    turn = ConversationTurn(
        producer=ACTOR,
        session_id="audit",
        role="user",
        text="1",
        data_class=DataClass.PUBLIC,
    )
    recognizer.recognize.return_value = Success(
        ActionIntent(
            source_turn_id=turn.record_id,
            language="pl",
            outcome="browser.open_result",
            selection=1,
        )
    )
    actions = AssistantActions(capabilities, recognizer, store)
    assert find_active_choice_frame((cancelled_turn,), "default", _utc_now()) is None
    response = await actions.handle(turn, CancellationToken())
    assert not browser.opened_urls, (
        response.text,
        recognizer.recognize.await_count,
        browser.opened_urls,
    )


def test_workflow_completion_cannot_exceed_total_step_limit():
    controller = TaskStepController()
    workflow = controller.create_workflow(
        task_id="audit",
        session_id="audit",
        query="query",
        limits=ResearchWorkflowLimits(max_total_steps=1),
    )
    workflow = controller.transition_to_search(workflow).unwrap()
    result = controller.record_search_completed(workflow, ())
    assert isinstance(result, Failure), f"steps={len(result.unwrap().steps)}, limit=1"


def tool_turn(record_id: str, capability: str, output: dict, timestamp):
    return ConversationTurn(
        record_id=record_id,
        producer=ACTOR,
        session_id="audit",
        role="assistant",
        text="Tool result",
        timestamp=timestamp,
        metadata={
            "profile_scope": "default",
            "action": {
                "action_result": {
                    "record_type": "action_result",
                    "capability": capability,
                    "output": output,
                }
            },
        },
    )


async def build(turns):
    store = AsyncMock()
    store.get_recent_reply_chain.return_value = Success(turns)
    store.retrieve_relevant_memories.return_value = Success(())
    store.retrieve_relevant_turns.return_value = Success(())
    builder = AssistantContextBuilder(
        store,
        system_instruction="Use evidence.",
        enable_recent_route=False,
    )
    turn = ConversationTurn(
        producer=ACTOR,
        session_id="audit",
        role="user",
        text="Podsumuj ostatni wynik.",
    )
    return (await builder.build_context(turn)).unwrap()


@pytest.mark.asyncio
async def test_context_budget_keeps_latest_page_instead_of_oldest():
    now = _utc_now()
    turns = tuple(
        tool_turn(
            record_id,
            "browser.read_page",
            {
                "title": record_id,
                "observed_url": f"https://example.com/{record_id}",
                "text": "x" * 4000,
            },
            now + timedelta(seconds=i),
        )
        for i, record_id in enumerate(("older", "latest"))
    )
    package = await build(turns)
    retained = [o["source_turn_id"] for o in package.content["tool_observations"]]
    assert "latest" in retained, retained


@pytest.mark.asyncio
async def test_selected_process_details_reach_context():
    package = await build((
        tool_turn(
            "selected-process",
            "process.inspect",
            {"pid": 123, "name": "test-process", "fingerprint": "verified"},
            _utc_now(),
        ),
    ))
    content = package.content["tool_observations"][0]["content"]
    assert "123" in str(content), content


@pytest.mark.asyncio
async def test_activity_query_triggers_in_turn_synthesis(tmp_path):
    from rai.actions.activity import register_activity_capabilities
    from tests.test_action_activity import _make_episode

    ep = _make_episode(
        "ep-1",
        minutes_ago=10,
        applications=("bash", "code"),
        projects=("rai",),
        activity_types=("development",),
        data_class=DataClass.PRIVATE,
    )

    class MockHistory:
        def query(self, **_kwargs):
            return [ep]

    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: MockHistory())
    capabilities = ActionCapabilityService(
        registry,
        PolicyEngine(),
        InMemoryAuditLedger(),
        SyntheticApprovalBroker(),
    )
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")

    class ActivityRecognizer(MockIntentRecognizer):
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken):
            if "aktywność" in turn.text:
                return Success(
                    ActionIntent(
                        source_turn_id=turn.record_id,
                        language="pl",
                        outcome="activity.query",
                    )
                )
            return Success(
                ActionIntent(
                    source_turn_id=turn.record_id,
                    language="pl",
                    outcome="no_action",
                )
            )

    recognizer = ActivityRecognizer()
    actions = AssistantActions(capabilities, recognizer, store, profile="default")
    service = AssistantService(store=store, actions=actions)
    await service.start()

    turn = ConversationTurn(
        producer=ACTOR,
        session_id="audit-activity",
        role="user",
        text="Sprawdź i podsumuj moją aktywność",
        data_class=DataClass.PRIVATE,
    )
    res = await service.accept_turn(turn)
    assert isinstance(res, Success)
    response = res.unwrap()
    assert "Podsumowanie aktywności w systemie:" in response.text
    recent = (await store.get_recent_reply_chain("audit-activity", limit=1)).unwrap()
    assert recent[-1].metadata.get("action", {}).get("synthesized_from_action") is True

    # Verify that follow-up reflection request doesn't re-execute activity.query
    turn2 = ConversationTurn(
        producer=ACTOR,
        session_id="audit-activity",
        role="user",
        text="Ok, ale co dokładnie z tego wynika?",
        data_class=DataClass.PRIVATE,
    )
    res2 = await service.accept_turn(turn2)
    assert isinstance(res2, Success)
    resp2 = res2.unwrap()
    recent2 = (await store.get_recent_reply_chain("audit-activity", limit=1)).unwrap()
    assert not recent2[-1].metadata.get("action", {}).get("synthesized_from_action")

    await service.stop()


@pytest.mark.asyncio
async def test_browser_search_synthesizes_and_allows_choice_selection(stack):
    ask, browser, _, service, _ = stack
    # 1. Search request - should be synthesized into a grounded response
    res1 = await ask("Poszukaj w sieci informacji o modelach")
    assert "Odpowiedź na podstawie wyszukiwania:" in res1.text
    recent = (await service.store.get_recent_reply_chain("audit", limit=1)).unwrap()
    assert recent[-1].metadata.get("action", {}).get("synthesized_from_action") is True
    # Choice frame must still be preserved in action metadata
    assert "active_choice_frame" in recent[-1].metadata.get("action", {})

    # 2. User selects result 2 - choice frame should resolve and open result 2
    res2 = await ask("2")
    assert len(browser.opened_urls) == 1
    assert browser.opened_urls[0] == "https://example.com/gemini-4-details"


@pytest.mark.asyncio
async def test_in_turn_model_tool_call_is_intercepted_and_synthesized(tmp_path):
    from typing import Any
    from returns.result import Result, Success
    from rai.assistant.records import AssistantCandidate, InferenceRequest

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    browser = MockBrowserBackend()
    register_browser_capabilities(registry, handles, browser)
    capabilities = ActionCapabilityService(
        registry,
        PolicyEngine(),
        InMemoryAuditLedger(),
        SyntheticApprovalBroker(),
    )
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")

    class InTurnToolBackend:
        def __init__(self):
            self.step = 0
            self.model_name = "mock-gemma"
            self.is_remote = False

        async def start(self):
            return Success("RUNNING")

        async def stop(self):
            return Success("STOPPED")

        async def generate(self, req: InferenceRequest, token: CancellationToken) -> Result[AssistantCandidate, Any]:
            self.step += 1
            if self.step == 1:
                # First pass: model outputs Gemma tool call tag
                text = (
                    "Rozumiem. Szukam informacji o laureatach.\n\n"
                    '<|tool_call>call: browser.search{queries:[<|"|>Nobel 2026 physics<|"|>]}<tool_call|>'
                )
                return Success(AssistantCandidate(text=text, tokens_in=50, tokens_out=30))
            # Second pass: model synthesizes answer from observations
            return Success(AssistantCandidate(
                text="Laureatem Nagrody Nobla w 2026 roku został Francis Halzen za odkrycie neutrin.",
                tokens_in=100,
                tokens_out=25,
            ))

    class NullRecognizer(MockIntentRecognizer):
        async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken):
            return Success(
                ActionIntent(
                    source_turn_id=turn.record_id,
                    language="pl",
                    outcome="no_action",
                )
            )

    recognizer = NullRecognizer()
    actions = AssistantActions(capabilities, recognizer, store, profile="default")
    backend = InTurnToolBackend()
    service = AssistantService(store=store, actions=actions, backend=backend)
    await service.start()

    turn = ConversationTurn(
        producer=ACTOR,
        session_id="audit-tool-call",
        role="user",
        text="Pamiętaj, że mamy rok 2026.",
        data_class=DataClass.PUBLIC,
    )
    res = await service.accept_turn(turn)
    assert isinstance(res, Success)
    resp = res.unwrap()
    # Tool call should be executed:
    assert browser.searches == ["Nobel 2026 physics"]
    # Final delivered text should NOT contain raw <|tool_call> dump:
    assert "<|tool_call>" not in resp.text
    assert "Francis Halzen" in resp.text

    await service.stop()


def test_bound_content_repr_preserves_five_search_results():
    from rai.assistant.backends.local import LocalAssistantBackend

    sample_results = [
        {"name": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
        for i in range(1, 6)
    ]
    content = {"results": sample_results}
    bounded = LocalAssistantBackend._bound_content_repr(content, max_len=3000)

    # All 5 items must be present:
    for i in range(1, 6):
        assert f"Result {i}" in bounded
    assert "truncated" not in bounded


@pytest.mark.asyncio
async def test_browser_read_page_auto_opens_missing_tab(tmp_path):
    from rai.actions.browser import BrowserCapability
    from rai.actions.records import ResourceHandle
    from rai.kernel.records import CapabilityRequest
    import hashlib

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    now = _utc_now()
    url = "https://example.com/direct-read"
    handle = ResourceHandle(
        producer=ACTOR,
        timestamp=now,
        actor_id=ACTOR.producer_id,
        task_id="task-direct",
        kind="url",
        target=url,
        fingerprint=hashlib.sha256(url.encode()).hexdigest(),
        operations=("browser.read_page",),
        expires_at=now + timedelta(hours=1),
        data_class=DataClass.PUBLIC,
    )
    handle_id = (handles.issue(handle)).unwrap()

    class TabMissingBrowserBackend(MockBrowserBackend):
        def __init__(self):
            super().__init__()
            self.tabs_open = set()

        async def open(self, url: str, token: CancellationToken):
            self.tabs_open.add(url)
            return await super().open(url, token)

        async def read(self, url: str, token: CancellationToken):
            if url not in self.tabs_open:
                return Failure("AMBIGUOUS_OR_MISSING_TAB")
            return await super().read(url, token)

    backend = TabMissingBrowserBackend()
    cap = BrowserCapability("browser.read_page", handles, backend)
    registry = CapabilityRegistry()
    register_browser_capabilities(registry, handles, backend)
    descriptor = registry.descriptor("browser.read_page")
    assert descriptor is not None
    from rai.kernel.transport import normalize_request
    req = normalize_request(
        descriptor,
        {"task_id": "task-direct", "handle": handle_id},
        actor=ACTOR,
        request_id="req-direct-read",
        data_class=DataClass.PUBLIC,
    )
    result = await cap.invoke(req, CancellationToken())
    assert isinstance(result, Success)
    # The tab was automatically opened before read:
    assert url in backend.tabs_open
    assert backend.read_urls == [url]


def test_container_wires_step_controller():
    from rai.container import ApplicationContainer

    container = ApplicationContainer(config={}, testing=True)
    service = container.assistant_service
    assert service.step_controller is not None
    assert isinstance(service.step_controller, TaskStepController)

