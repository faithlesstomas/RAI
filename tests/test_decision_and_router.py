"""Unit tests for DecisionBackend implementations and HybridRouter."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
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
    with pytest.raises(
        ValueError, match="DECIDED status requires a non-null selected_option"
    ):
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
    provider_payload = {
        "status": "DECIDED",
        "selected_option": "opt_local",
        "outcome_distribution": {"opt_cloud": 0.1, "opt_local": 0.9},
        "model_revision": "jev-1.13.2",
        "calibration": {
            "status": "CALIBRATED",
            "method": "provider-calibration",
            "version": "1.13.2",
            "metrics": {"ece": 0.02},
        },
        "usage": {"input_tokens": 22, "output_tokens": 5, "cost_usd": 0.01},
    }
    response = httpx.Response(
        200,
        json=provider_payload,
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/decision"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = response
    with patch("rai.inference.decision.httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = client
        client_class.return_value.__aexit__.return_value = None
        res = await jev.decide(req, CancellationToken())
    assert isinstance(res, Success)
    decision = res.unwrap()
    assert decision.backend_name == "typesafe-jev"
    assert decision.model_revision == "jev-1.13.2"
    assert decision.status == "DECIDED"
    assert decision.selected_option == "opt_local"
    assert sum(decision.outcome_distribution.values()) == pytest.approx(1.0, 0.01)
    sent_payload = client.post.await_args.kwargs["json"]
    assert sent_payload["context_content"] == {"query": "Test decision query"}


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
    completion_payload = {
        "model": "qwen-dynamic-eval",
        "choices": [
            {
                "message": {
                    "content": (
                        '{"status":"DECIDED","selected_option":"opt_slow",'
                        '"outcome_distribution":{"opt_fast":0.2,"opt_slow":0.8}}'
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 9},
    }
    response = httpx.Response(
        200,
        json=completion_payload,
        request=httpx.Request("POST", "http://127.0.0.1:13305/api/v1/chat/completions"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = response
    with patch("rai.inference.decision.httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = client
        client_class.return_value.__aexit__.return_value = None
        res = await backend.decide(req, CancellationToken())
    assert isinstance(res, Success)
    decision = res.unwrap()
    assert decision.backend_name == "lemonade-local"
    assert decision.model_revision == "qwen-dynamic-eval"
    assert decision.status == "DECIDED"
    assert decision.selected_option == "opt_slow"
    assert decision.calibration.status == "UNKNOWN"


@pytest.mark.asyncio
async def test_hosted_jev_rejects_incomplete_provider_distribution() -> None:
    jev = HostedJevDecisionBackend(
        api_key="mock-key",
        egress_firewall=EgressFirewall(profile="REMOTE_ALLOWED"),
    )
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-jev-invalid",
        task_kind="routing_hint",
        context=_make_context(data_class=DataClass.PUBLIC),
        options=(
            DecisionOption(option_id="opt_a", description="A"),
            DecisionOption(option_id="opt_b", description="B"),
        ),
        budget=_make_budget(),
    )
    response = httpx.Response(
        200,
        json={
            "status": "DECIDED",
            "selected_option": "opt_a",
            "outcome_distribution": {"opt_a": 1.0},
            "model_revision": "jev-1.13.2",
            "usage": {},
        },
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/decision"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = response
    with patch("rai.inference.decision.httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = client
        client_class.return_value.__aexit__.return_value = None
        result = await jev.decide(req, CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "JEV_DECISION_FAILED"


@pytest.mark.asyncio
async def test_hosted_jev_rejects_missing_provider_usage() -> None:
    jev = HostedJevDecisionBackend(
        api_key="mock-key",
        egress_firewall=EgressFirewall(profile="REMOTE_ALLOWED"),
    )
    req = DecisionRequest(
        producer=TEST_PRODUCER,
        request_id="req-jev-missing-usage",
        task_kind="routing_hint",
        context=_make_context(data_class=DataClass.PUBLIC),
        options=(
            DecisionOption(option_id="opt_a", description="A"),
            DecisionOption(option_id="opt_b", description="B"),
        ),
        budget=_make_budget(),
    )
    response = httpx.Response(
        200,
        json={
            "status": "DECIDED",
            "selected_option": "opt_b",
            "outcome_distribution": {"opt_a": 0.1, "opt_b": 0.9},
            "model_revision": "jev-1.13.2",
            "usage": {},
        },
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/decision"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = response
    with patch("rai.inference.decision.httpx.AsyncClient") as client_class:
        client_class.return_value.__aenter__.return_value = client
        client_class.return_value.__aexit__.return_value = None
        result = await jev.decide(req, CancellationToken())

    assert isinstance(result, Failure)
    assert result.failure().code == "JEV_DECISION_FAILED"
    assert "usage.input_tokens" in result.failure().message


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


@pytest.mark.asyncio
async def test_router_requires_approval_for_private_remote_route() -> None:
    router = HybridRouter(
        profile="REMOTE_ALLOWED",
        decision_backend=DeterministicDecisionBackend(default_option="opt_remote"),
    )

    result = await router.route_inference(
        context=_make_context(data_class=DataClass.PRIVATE, approved=False),
        budget=_make_budget(),
        user_prompt="Przeanalizuj prywatne dane",
    )

    assert isinstance(result, Success)
    assert result.unwrap().outcome == "ASK"
    assert "PRIVATE" in result.unwrap().reason


@pytest.mark.asyncio
async def test_router_enforces_budget_before_remote_route() -> None:
    router = HybridRouter(
        profile="REMOTE_ALLOWED",
        decision_backend=DeterministicDecisionBackend(default_option="opt_remote"),
    )
    budget = _make_budget().model_copy(update={"max_output_tokens": 0})

    result = await router.route_inference(
        context=_make_context(data_class=DataClass.PUBLIC),
        budget=budget,
        user_prompt="Zaprojektuj architekturę",
    )

    assert isinstance(result, Failure)
    assert result.failure().code == "TASK_OUTPUT_FORBIDDEN"
