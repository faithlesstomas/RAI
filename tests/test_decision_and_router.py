"""Unit tests for DecisionBackend implementations and HybridRouter."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from returns.result import Failure, Success

from rai.inference.decision import (
    DecisionCalibration,
    DecisionOption,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
    DeterministicDecisionBackend,
    HostedJevDecisionBackend,
    LocalLemonadeDecisionBackend,
)
from rai.inference.governor import GovernorConfig, InferenceBudgetGovernor
from rai.inference.hybrid_router import HybridRouter
from rai.kernel.egress import EgressFirewall
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
)

TEST_PRODUCER = ProducerIdentity(
    producer_id="test.decision", kind="test", version="1.0.0"
)


def _make_context(
    task_id: str = "task-dec-1",
    data_class: DataClass = DataClass.PUBLIC,
    approved: bool = True,
) -> ContextPackage:
    manifest = ContextManifest(
        producer=TEST_PRODUCER,
        destination="test-target",
        items=(
            ContextManifestItem(
                source_id="item-1",
                source_type="text",
                data_class=data_class,
            ),
        ),
        approved=approved,
    )
    return ContextPackage(
        producer=TEST_PRODUCER,
        task_id=task_id,
        manifest=manifest,
        content={"query": "Test decision query"},
    )


def _make_budget() -> InferenceBudget:
    return InferenceBudget(
        producer=TEST_PRODUCER,
        max_input_tokens=5000,
        max_output_tokens=1000,
        max_agent_turns=3,
        max_tool_calls=5,
        max_images=1,
        max_audio_seconds=0.0,
        max_latency_seconds=30.0,
        max_provider_cost=1.0,
        max_ram_bytes=4 * 1024**3,
        max_vram_bytes=2 * 1024**3,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=60),
    )


# --- DecisionResult Schema Tests ---


def test_decision_result_validation() -> None:
    # Valid DECIDED
    valid_decided = DecisionResult(
        producer=TEST_PRODUCER,
        request_id="req-1",
        status="DECIDED",
        selected_option="opt_a",
        outcome_distribution={"opt_a": 0.8, "opt_b": 0.2},
        backend_name="test",
        model_revision="rev-1",
    )
    assert valid_decided.selected_option == "opt_a"

    # Invalid: selected_option missing when DECIDED
    with pytest.raises(ValueError, match="DECIDED status requires a non-null selected_option"):
        DecisionResult(
            producer=TEST_PRODUCER,
            request_id="req-2",
            status="DECIDED",
            selected_option=None,
            outcome_distribution={"opt_a": 0.8, "opt_b": 0.2},
            backend_name="test",
            model_revision="rev-1",
        )

    # Invalid: selected_option not None when ABSTAINED
    with pytest.raises(ValueError, match="must have selected_option=None"):
        DecisionResult(
            producer=TEST_PRODUCER,
            request_id="req-3",
            status="ABSTAINED",
            selected_option="opt_a",
            outcome_distribution={"opt_a": 0.5, "opt_b": 0.5},
            backend_name="test",
            model_revision="rev-1",
        )

    # Invalid: probabilities don't sum to 1.0
    with pytest.raises(ValueError, match="must sum to 1.0"):
        DecisionResult(
            producer=TEST_PRODUCER,
            request_id="req-4",
            status="DECIDED",
            selected_option="opt_a",
            outcome_distribution={"opt_a": 0.5, "opt_b": 0.1},
            backend_name="test",
            model_revision="rev-1",
        )


# --- DeterministicDecisionBackend Tests ---


@pytest.mark.asyncio
async def test_deterministic_decision_backend() -> None:
    backend = DeterministicDecisionBackend(default_option="opt_b")
    context = _make_context()
    budget = _make_budget()
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-det",
        task_kind="routing_hint",
        context=context,
        options=(
            DecisionOption(option_id="opt_a", description="A"),
            DecisionOption(option_id="opt_b", description="B"),
        ),
        budget=budget,
    )
    res = await backend.decide(req, CancellationToken())
    assert isinstance(res, Success)
    assert res.unwrap().selected_option == "opt_b"
    assert res.unwrap().outcome_distribution["opt_b"] == 1.0


# --- HostedJevDecisionBackend Tests ---


@pytest.mark.asyncio
async def test_hosted_jev_egress_firewall_blocks_local_and_secret() -> None:
    firewall = EgressFirewall(profile="REMOTE_ALLOWED")
    jev = HostedJevDecisionBackend(api_key="mock-key", egress_firewall=firewall)
    budget = _make_budget()

    # Context with LOCAL data class
    local_ctx = _make_context(data_class=DataClass.LOCAL)
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-jev-local",
        task_kind="routing_hint",
        context=local_ctx,
        options=(
            DecisionOption(option_id="opt_1", description="Opt 1"),
            DecisionOption(option_id="opt_2", description="Opt 2"),
        ),
        budget=budget,
    )
    res = await jev.decide(req, CancellationToken())
    assert isinstance(res, Failure)
    assert res.failure().code == "EGRESS_LOCAL_DATA_LEAK"


@pytest.mark.asyncio
async def test_hosted_jev_decision_success() -> None:
    firewall = EgressFirewall(profile="REMOTE_ALLOWED")
    jev = HostedJevDecisionBackend(
        api_key="mock-jev-api-key",
        pinned_model_revision="jev-1.13.2",
        egress_firewall=firewall,
    )
    ctx = _make_context(data_class=DataClass.PUBLIC)
    budget = _make_budget()
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-jev-success",
        task_kind="routing_hint",
        context=ctx,
        options=(
            DecisionOption(option_id="opt_cloud", description="Cloud"),
            DecisionOption(option_id="opt_local", description="Local"),
        ),
        budget=budget,
    )
    res = await jev.decide(req, CancellationToken())
    assert isinstance(res, Success)
    decision = res.unwrap()
    assert decision.backend_name == "typesafe-jev"
    assert decision.model_revision == "jev-1.13.2"
    assert decision.status == "DECIDED"
    assert sum(decision.outcome_distribution.values()) == pytest.approx(1.0, 0.01)


# --- LocalLemonadeDecisionBackend Tests ---


@pytest.mark.asyncio
async def test_local_lemonade_decision_dynamic_model() -> None:
    backend = LocalLemonadeDecisionBackend(worker_model="qwen-dynamic-eval")
    ctx = _make_context()
    budget = _make_budget()
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-lemonade",
        task_kind="routing_hint",
        context=ctx,
        options=(
            DecisionOption(option_id="opt_fast", description="Fast"),
            DecisionOption(option_id="opt_slow", description="Slow"),
        ),
        budget=budget,
    )
    res = await backend.decide(req, CancellationToken())
    assert isinstance(res, Success)
    decision = res.unwrap()
    assert decision.backend_name == "lemonade-local"
    assert decision.model_revision == "qwen-dynamic-eval"
    assert decision.status == "DECIDED"


# --- HybridRouter Tests ---


@pytest.mark.asyncio
async def test_router_deterministic_keyword_matches_local() -> None:
    router = HybridRouter()
    ctx = _make_context()
    budget = _make_budget()

    res = await router.route_inference(
        context=ctx,
        budget=budget,
        user_prompt="/offline Przetwórz te notatki",
    )
    assert isinstance(res, Success)
    assert res.unwrap().outcome == "LOCAL"
    assert "keyword match" in res.unwrap().reason


@pytest.mark.asyncio
async def test_router_privacy_firewall_forces_local_for_secret_and_local() -> None:
    router = HybridRouter(profile="REMOTE_ALLOWED")
    budget = _make_budget()

    for dc in (DataClass.LOCAL, DataClass.SECRET, DataClass.BLOCKED):
        ctx = _make_context(data_class=dc)
        res = await router.route_inference(
            context=ctx,
            budget=budget,
            user_prompt="Napisz skrypt",
        )
        assert isinstance(res, Success)
        assert res.unwrap().outcome == "LOCAL"


@pytest.mark.asyncio
async def test_router_background_task_forces_local() -> None:
    router = HybridRouter(profile="REMOTE_ALLOWED")
    ctx = _make_context(data_class=DataClass.PUBLIC)
    budget = _make_budget()

    res = await router.route_inference(
        context=ctx,
        budget=budget,
        user_prompt="Generuj raport",
        is_background=True,
    )
    assert isinstance(res, Success)
    assert res.unwrap().outcome == "LOCAL"
    assert "Background task forbidden" in res.unwrap().reason


@pytest.mark.asyncio
async def test_router_escalates_to_remote_when_decision_backend_votes_remote() -> None:
    # Deterministic backend returns opt_remote
    dec_backend = DeterministicDecisionBackend(default_option="opt_remote")
    router = HybridRouter(
        profile="REMOTE_ALLOWED",
        decision_backend=dec_backend,
    )
    ctx = _make_context(data_class=DataClass.PUBLIC)
    budget = _make_budget()

    res = await router.route_inference(
        context=ctx,
        budget=budget,
        user_prompt="Zaprojektuj architekturę mikrousług",
    )
    assert isinstance(res, Success)
    decision = res.unwrap()
    assert decision.outcome == "REMOTE"
    assert decision.target_backend == "antigravity"


@pytest.mark.asyncio
async def test_router_hybrid_approval_yields_ask() -> None:
    dec_backend = DeterministicDecisionBackend(default_option="opt_remote")
    router = HybridRouter(
        profile="HYBRID_APPROVAL",
        decision_backend=dec_backend,
    )
    ctx = _make_context(data_class=DataClass.PUBLIC)
    budget = _make_budget()

    res = await router.route_inference(
        context=ctx,
        budget=budget,
        user_prompt="Zaprojektuj architekturę",
    )
    assert isinstance(res, Success)
    assert res.unwrap().outcome == "ASK"
    assert "HYBRID_APPROVAL requires user consent" in res.unwrap().reason
