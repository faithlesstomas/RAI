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
    ask, browser, _, _, _ = stack
    await ask("Poszukaj w sieci informacji o modelach")
    await ask("1")
    assert len(browser.opened_urls) == 1
    response = await ask("przeczytaj 1")
    assert browser.read_urls == ["https://example.com/gemini-4"], response.text


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
