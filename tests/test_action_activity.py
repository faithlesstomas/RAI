"""Tests for activity.query capability and Rich History intent retrieval."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import pytest
from returns.result import Success

from rai.actions.activity import ActivityQuery, register_activity_capabilities
from rai.actions.assistant import AssistantActions
from rai.actions.intent import ActionIntent
from rai.actions.service import ActionCapabilityService
from rai.assistant.evidence import RichHistoryEvidenceProvider
from rai.assistant.ports import MemoryQuery
from rai.assistant.query import MemoryQueryResolver
from rai.assistant.records import ConversationTurn
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    CapabilityRequest,
    DataClass,
    Episode,
    ProducerIdentity,
    ProvenanceReference,
    RiskClass,
)
from rai.kernel.synthetic import SyntheticApprovalBroker

PRODUCER = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")


class MockRichHistory:
    def __init__(self, episodes: tuple[Episode, ...] = ()) -> None:
        self.episodes = episodes
        self.query_calls = 0

    def query(self, **filters: Any) -> tuple[Episode, ...]:
        self.query_calls += 1
        return self.episodes


class MockRecognizer:
    def __init__(self, intent: ActionIntent) -> None:
        self.intent = intent

    async def recognize(self, turn: ConversationTurn, cancellation: CancellationToken) -> Success[ActionIntent]:
        return Success(self.intent)


def _make_episode(
    record_id: str,
    *,
    minutes_ago: int = 10,
    applications: tuple[str, ...] = ("bash",),
    projects: tuple[str, ...] = ("rai",),
    resources: tuple[str, ...] = ("main.py",),
    activity_types: tuple[str, ...] = ("coding",),
    data_class: DataClass = DataClass.LOCAL,
    confidence: float = 0.9,
) -> Episode:
    now = datetime.now(timezone.utc)
    started = now - timedelta(minutes=minutes_ago + 5)
    ended = now - timedelta(minutes=minutes_ago)
    return Episode(
        record_id=record_id,
        timestamp=ended,
        producer=PRODUCER,
        started_at=started,
        ended_at=ended,
        observation_ids=("obs-1",),
        provenance=(
            ProvenanceReference(
                source_id="obs-1",
                source_type="observation",
                source_version="1.0.0",
                relation="DERIVED_FROM",
                producer=PRODUCER,
            ),
        ),
        applications=applications,
        projects=projects,
        resources=resources,
        activity_types=activity_types,
        confidence=confidence,
        data_class=data_class,
    )


@pytest.mark.asyncio
async def test_activity_query_capability_success_and_filtering() -> None:
    ep1 = _make_episode("ep-1", minutes_ago=5, applications=("code", "bash"), projects=("rai",), data_class=DataClass.LOCAL)
    ep2 = _make_episode("ep-2", minutes_ago=20, applications=("firefox",), projects=(), data_class=DataClass.PUBLIC)
    ep3 = _make_episode("ep-3", minutes_ago=30, applications=("secret-tool",), data_class=DataClass.PRIVATE)

    history = MockRichHistory((ep1, ep2, ep3))
    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: history)
    desc = registry.descriptor("activity.query")
    assert desc is not None
    cap = registry._capabilities["activity.query"]

    from rai.kernel.transport import normalize_request

    token = CancellationToken()
    # Query with LOCAL data_class -> ep1 (LOCAL) and ep2 (PUBLIC) allowed, ep3 (PRIVATE) excluded
    req = normalize_request(desc, {"task_id": "task-1"}, data_class=DataClass.LOCAL)

    res = await cap.invoke(req, token)
    assert isinstance(res, Success)
    result = res.unwrap()
    episodes = result.output["episodes"]
    assert len(episodes) == 2
    assert episodes[0]["episode_id"] == "ep-1"
    assert episodes[1]["episode_id"] == "ep-2"

    # Query with search filter
    req_filtered = normalize_request(desc, {"task_id": "task-2", "query": "firefox"}, data_class=DataClass.LOCAL)
    res_filtered = await cap.invoke(req_filtered, token)
    assert isinstance(res_filtered, Success)
    episodes_filtered = res_filtered.unwrap().output["episodes"]
    assert len(episodes_filtered) == 1
    assert episodes_filtered[0]["episode_id"] == "ep-2"


@pytest.mark.asyncio
async def test_activity_query_registration_and_unavailable_service() -> None:
    from rai.kernel.transport import normalize_request

    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: None)
    desc = registry.descriptor("activity.query")
    assert desc is not None
    assert desc.risk_class == RiskClass.LOW

    cap = registry._capabilities["activity.query"]
    req = normalize_request(desc, {"task_id": "task-3"}, data_class=DataClass.LOCAL)
    res = await cap.invoke(req, CancellationToken())
    assert not isinstance(res, Success)
    assert res.failure().code == "HISTORY_SERVICE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_assistant_actions_routes_activity_query(tmp_path: Path) -> None:
    ep = _make_episode("ep-10", applications=("code",), projects=("rai",), activity_types=("coding",))
    history = MockRichHistory((ep,))

    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: history)
    policy = PolicyEngine(isolation_available=lambda _: True)
    capabilities = ActionCapabilityService(registry, policy, InMemoryAuditLedger(), SyntheticApprovalBroker())

    intent = ActionIntent(
        source_turn_id="turn-act-1",
        outcome="activity.query",
        language="pl",
        query="",
    )
    actions = AssistantActions(capabilities, MockRecognizer(intent))

    turn = ConversationTurn(
        record_id="turn-act-1",
        producer=PRODUCER,
        session_id="session-1",
        role="user",
        text="Co ostatnio działo się w systemie?",
        data_class=DataClass.LOCAL,
    )
    candidate = await actions.handle(turn, CancellationToken())
    assert candidate is not None
    assert "Ostatnia aktywność w systemie:" in candidate.text
    assert "code" in candidate.text
    assert "rai" in candidate.text
    assert "action_result" in candidate.metadata


@pytest.mark.asyncio
async def test_rich_history_evidence_provider_natural_queries() -> None:
    ep = _make_episode("ep-nat", applications=("gnome-shell", "bash"), projects=("rai",), confidence=0.88)
    history = MockRichHistory((ep,))
    provider = RichHistoryEvidenceProvider(history)

    # 1. Polish natural query with typo / colloquial wording
    q1 = MemoryQueryResolver.resolve("Co ostanio działo się w systemie i na pulpicie użytkownika?")
    res1 = await provider.retrieve(q1, (DataClass.LOCAL,), limit=10)
    assert isinstance(res1, Success)
    evidence1 = res1.unwrap()
    assert len(evidence1) == 1
    assert evidence1[0].source_id == "ep-nat"

    # 2. English natural query
    q2 = MemoryQueryResolver.resolve("Show recent system activity")
    res2 = await provider.retrieve(q2, (DataClass.LOCAL,), limit=10)
    assert isinstance(res2, Success)
    evidence2 = res2.unwrap()
    assert len(evidence2) == 1
    assert evidence2[0].source_id == "ep-nat"

    # 3. Polish "historia aktywności"
    q3 = MemoryQueryResolver.resolve("historia aktywności systemu")
    res3 = await provider.retrieve(q3, (DataClass.LOCAL,), limit=10)
    assert isinstance(res3, Success)
    evidence3 = res3.unwrap()
    assert len(evidence3) == 1

    # 4. Unrelated query does not pull activity
    q_unrelated = MemoryQueryResolver.resolve("Ile mam lat?")
    res_unrelated = await provider.retrieve(q_unrelated, (DataClass.LOCAL,), limit=10)
    assert isinstance(res_unrelated, Success)
    assert len(res_unrelated.unwrap()) == 0


def test_action_intent_validation_activity_query() -> None:
    intent_empty = ActionIntent(
        source_turn_id="turn-1",
        outcome="activity.query",
        language="pl",
        query="",
    )
    assert intent_empty.outcome == "activity.query"
    assert intent_empty.query == ""

    intent_filtered = ActionIntent(
        source_turn_id="turn-2",
        outcome="activity.query",
        language="en",
        query="firefox",
    )
    assert intent_filtered.outcome == "activity.query"
    assert intent_filtered.query == "firefox"


@pytest.mark.asyncio
async def test_assistant_actions_activity_query_empty_and_english() -> None:
    history_empty = MockRichHistory(())
    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: history_empty)
    policy = PolicyEngine(isolation_available=lambda _: True)
    capabilities = ActionCapabilityService(registry, policy, InMemoryAuditLedger(), SyntheticApprovalBroker())

    # English empty
    intent_en = ActionIntent(
        source_turn_id="turn-en",
        outcome="activity.query",
        language="en",
        query="",
    )
    actions_en = AssistantActions(capabilities, MockRecognizer(intent_en))
    turn_en = ConversationTurn(
        record_id="turn-en",
        producer=PRODUCER,
        session_id="session-en",
        role="user",
        text="Recent activity?",
        data_class=DataClass.LOCAL,
    )
    cand_en = await actions_en.handle(turn_en, CancellationToken())
    assert cand_en is not None
    assert cand_en.text == "No recorded activity."

    # Polish empty
    intent_pl = ActionIntent(
        source_turn_id="turn-pl",
        outcome="activity.query",
        language="pl",
        query="",
    )
    actions_pl = AssistantActions(capabilities, MockRecognizer(intent_pl))
    turn_pl = ConversationTurn(
        record_id="turn-pl",
        producer=PRODUCER,
        session_id="session-pl",
        role="user",
        text="Co się działo?",
        data_class=DataClass.LOCAL,
    )
    cand_pl = await actions_pl.handle(turn_pl, CancellationToken())
    assert cand_pl is not None
    assert cand_pl.text == "Brak zarejestrowanej aktywności."

