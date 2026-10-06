# ruff: noqa: PLR2004, PLR0915
"""Regression test replicating multi-turn dialog search and selection flow.

Verifies that:
1. Search produces an ActiveChoiceFrame with stable indices and handles.
2. Intermediate non-action conversation turns do NOT break the active choice frame.
3. A numeric input (e.g. "5") resolves deterministically without calling LLM intent recognition.
4. An expired or consumed frame does not execute any action.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any

from returns.result import Result, Success

from rai.actions.assistant import AssistantActions
from rai.actions.browser import BrowserBackend, register_browser_capabilities
from rai.actions.choices import find_active_choice_frame
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.handles import SQLiteHandleStore
from rai.actions.intent import ActionIntent, IntentRecognizer
from rai.actions.records import ActiveChoiceFrame, ChoiceState
from rai.actions.service import ActionCapabilityService
from rai.assistant.records import ConversationTurn
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, ProducerIdentity, _utc_now
from rai.kernel.synthetic import SyntheticApprovalBroker


class MockBrowserBackend(BrowserBackend):
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.opened_urls: list[str] = []
        self.read_urls: list[str] = []

    async def search(self, query: str, token: CancellationToken) -> Result[tuple[dict[str, Any], ...], str]:
        self.searches.append(query)
        results = (
            {"name": "1. Introducing Gemini 4 Argon", "url": "https://example.com/gemini-4"},
            {"name": "2. Gemini 4 Details", "url": "https://example.com/gemini-4-details"},
            {"name": "3. Benchmarks Argon", "url": "https://example.com/benchmarks"},
            {"name": "4. Architecture Overview", "url": "https://example.com/arch"},
            {"name": "5. Gemini 4 Argon: Release and Impact", "url": "https://example.com/impact"},
        )
        return Success(results)

    async def open(self, url: str, token: CancellationToken) -> Result[dict[str, Any], str]:
        self.opened_urls.append(url)
        return Success({"target_id": "tab-1", "observed_url": url, "frame_id": "frame-1"})

    async def read(self, url: str, token: CancellationToken) -> Result[dict[str, Any], str]:
        self.read_urls.append(url)
        return Success({
            "target_id": "tab-1",
            "observed_url": url,
            "title": "Gemini 4 Argon: Release and Impact",
            "text": "Full text of the article about Gemini 4 Argon.",
            "untrusted_content": True,
        })


class MockIntentRecognizer(IntentRecognizer):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Result[ActionIntent, str]:
        self.calls.append(turn.text)
        if "Poszukaj w sieci" in turn.text:
            return Success(ActionIntent(
                source_turn_id=turn.record_id,
                language="pl",
                outcome="browser.search",
                query="nowy model LLM Gemini 4",
            ))
        return Success(ActionIntent(
            source_turn_id=turn.record_id,
            language="pl",
            outcome="no_action",
        ))


async def test_regression_search_intermediate_conversation_and_numeric_selection(tmp_path: Path) -> None:
    # 1. Setup services
    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    browser = MockBrowserBackend()
    register_browser_capabilities(registry, handles, browser)

    audit = InMemoryAuditLedger()
    capabilities = ActionCapabilityService(registry, PolicyEngine(), audit, SyntheticApprovalBroker())
    capabilities.handles = handles
    capabilities.executions = SQLiteExecutionStore(tmp_path / "execution.db")

    recognizer = MockIntentRecognizer()
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")
    actions = AssistantActions(capabilities, recognizer, store, profile="default")
    service = AssistantService(store=store, actions=actions, profile_scope="default")

    await service.start()
    try:
        actor = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
        session_id = "test-session-1"

        # --- Turn 1: Search query ---
        turn1 = ConversationTurn(
            producer=actor,
            session_id=session_id,
            role="user",
            text="Poszukaj w sieci informacji na temat nowego modelu LLM Gemini 4.",
            data_class=DataClass.PUBLIC,
        )
        resp1 = await service.accept_turn(turn1)
        assert isinstance(resp1, Success)
        assert len(browser.searches) == 1
        assert "Gemini 4 Argon" in resp1.unwrap().text

        # Verify ActiveChoiceFrame was generated in turn 1
        recent = (await store.get_recent_reply_chain(session_id, limit=3)).unwrap()
        now = _utc_now()
        active_frame = find_active_choice_frame(recent, "default", now)
        assert active_frame is not None
        assert len(active_frame.entries) == 5
        assert active_frame.state == ChoiceState.ACTIVE
        assert active_frame.intended_capability == "browser.open_result"

        # --- Turn 2: Intermediate question ("Co to znaczy?") ---
        # Recognizer will return no_action, model generates response
        turn2 = ConversationTurn(
            producer=actor,
            session_id=session_id,
            role="user",
            text="Co to znaczy? Poproszę więcej informacji.",
            data_class=DataClass.PUBLIC,
        )
        resp2 = await service.accept_turn(turn2)
        assert isinstance(resp2, Success)

        # Verify ActiveChoiceFrame is STILL active despite intermediate conversation turn!
        recent2 = (await store.get_recent_reply_chain(session_id, limit=5)).unwrap()
        active_frame2 = find_active_choice_frame(recent2, "default", now)
        assert active_frame2 is not None
        assert active_frame2.record_id == active_frame.record_id

        # --- Turn 3: User enters numeric choice "5" ---
        recognizer_calls_before = len(recognizer.calls)
        turn3 = ConversationTurn(
            producer=actor,
            session_id=session_id,
            role="user",
            text="5",
            data_class=DataClass.PUBLIC,
        )
        resp3 = await service.accept_turn(turn3)
        assert isinstance(resp3, Success)

        # CRITICAL ASSERTION: The LLM intent recognizer was NOT called for "5"!
        assert len(recognizer.calls) == recognizer_calls_before

        # Capability was executed deterministically!
        assert len(browser.opened_urls) == 1
        assert browser.opened_urls[0] == "https://example.com/impact"

        # ActiveChoiceFrame is now marked CONSUMED
        recent3 = (await store.get_recent_reply_chain(session_id, limit=6)).unwrap()
        assert find_active_choice_frame(recent3, "default", now) is None

        # --- Turn 4: User enters "5" again - frame is already consumed, so no action ---
        turn4 = ConversationTurn(
            producer=actor,
            session_id=session_id,
            role="user",
            text="5",
            data_class=DataClass.PUBLIC,
        )
        resp4 = await service.accept_turn(turn4)
        assert isinstance(resp4, Success)
        # Opened URLs count did not increase
        assert len(browser.opened_urls) == 1
    finally:
        await service.stop()
