"""Unit tests for WorkspaceEngine, specialists competition, and GWT blackboard deliberation."""
# ruff: noqa: PLR2004, ANN401, PLC0415

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, NoReturn

import pytest
from returns.result import Failure, Result, Success

from rai.assistant.cognitive.arbiter import SalienceArbiter
from rai.assistant.cognitive.models import (
    WorkspaceContribution,
    WorkspaceContributionKind,
    WorkspaceState,
)
from rai.assistant.cognitive.probe import JLensProbe
from rai.assistant.cognitive.workspace import WorkspaceEngine
from rai.assistant.ports import AssistantModelBackend, InteractionController
from rai.container import ApplicationContainer
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
)
from rai.kernel.capabilities import CapabilityDescriptor
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    RiskClass,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.skills.defaults import create_default_skill_registry


class MockModelBackend(AssistantModelBackend):
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or ["Conversational answer."])
        self.call_count = 0
        self.model_name = "mock-workspace-model"

    @property
    def state(self) -> LifecycleState:
        return LifecycleState.RUNNING

    @property
    def is_remote(self) -> bool:
        return False

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        return Success(LifecycleState.RUNNING)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        return Success(LifecycleState.STOPPED)

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        del request, cancellation
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return Success(
            AssistantCandidate(
                text=response,
                tokens_in=15,
                tokens_out=20,
            )
        )

    def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
        del request, cancellation
        raise NotImplementedError


def _make_context_package(
    turn: ConversationTurn,
    evidence_required: bool = False,
    routing_decision: str = "recent_conversation",
) -> AssistantContextPackage:
    producer = ProducerIdentity(producer_id="test", kind="test", version="1.0.0")
    manifest = AssistantContextManifest(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_required=evidence_required,
        routing_decision=routing_decision,  # type: ignore[arg-type]
    )
    return AssistantContextPackage(
        producer=producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content={"recent_turns": [], "tool_observations": []},
    )


def _make_budget() -> InferenceBudget:
    now = _utc_now()
    return InferenceBudget(
        record_id=_new_id(),
        timestamp=now,
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        max_input_tokens=2048,
        max_output_tokens=512,
        max_agent_turns=4,
        max_tool_calls=5,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=60,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=60),
    )


def test_workspace_engine_protocol_conformance() -> None:
    engine = WorkspaceEngine()
    assert isinstance(engine, InteractionController)


@pytest.mark.asyncio
async def test_workspace_engine_direct_answer_convergence() -> None:
    backend = MockModelBackend(responses=["Here is the direct answer to your question."])
    engine = WorkspaceEngine(backend=backend)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-workspace-1",
        role="user",
        text="What is functional programming?",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.text == "Here is the direct answer to your question."
    assert candidate.metadata.get("workspace_converged") is True
    assert candidate.metadata.get("winning_specialist") == "answer_draft"
    assert candidate.metadata.get("workspace_cycles") == 1
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_workspace_engine_action_deliberation_and_observation() -> None:
    # Cycle 1: ActionReasoner proposes tool call; Cycle 2: AnswerDraft synthesizes observed apps
    backend = MockModelBackend(
        responses=[
            "<tool_call>call: application.list{}</tool_call>",
            "Found 2 applications: Files and Terminal.",
        ]
    )

    class MockCapabilities(CapabilityService):
        def __init__(self) -> None:
            class MockRegistry:
                def __init__(self) -> None:
                    self._desc = CapabilityDescriptor(
                        name="application.list",
                        description="List applications",
                        input_schema={"type": "object", "properties": {}},
                        risk_class=RiskClass.LOW,
                        side_effects=(),
                        isolation="process",
                        verification_plan=("verify_read_only",),
                    )

                def descriptor(self, name: str) -> CapabilityDescriptor | None:
                    return self._desc if name == "application.list" else None

                def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
                    return (self._desc,)

            self.registry = MockRegistry()

        async def invoke(
            self, request: CapabilityRequest, cancellation: CancellationToken
        ) -> tuple[Any, Result[ActionResult, ActionFailure]]:
            del cancellation
            return None, Success(
                ActionResult(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
                    request_id=request.record_id,
                    capability="application.list",
                    output={"apps": ["Files.desktop", "Terminal.desktop"]},
                    verification={"status": "verified"},
                )
            )

    capabilities = MockCapabilities()
    skills = create_default_skill_registry()
    engine = WorkspaceEngine(
        backend=backend,
        capabilities=capabilities,
        skills=skills,
        max_cycles=4,
    )

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-workspace-action",
        role="user",
        text="Count applications in the system",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert "Found 2 applications" in candidate.text
    assert candidate.metadata.get("tool_calls_executed") == 1
    assert candidate.metadata.get("observations_count") == 1
    assert candidate.metadata.get("workspace_converged") is True
    expected_llm_calls = 2
    assert backend.call_count == expected_llm_calls


@pytest.mark.asyncio
async def test_workspace_engine_emits_unexecuted_action_proposal_without_capabilities() -> None:
    from rai.assistant.cognitive.specialists import (  # noqa: PLC0415
        ActionReasonerSpecialist,
        AnswerDraftSpecialist,
        GroundingEvaluationSpecialist,
        RetrieverSpecialist,
    )

    backend = MockModelBackend(
        responses=[
            "<tool_call>call: application.list{}</tool_call>",
            "Proposed application discovery action.",
        ]
    )
    engine = WorkspaceEngine(
        backend=backend,
        capabilities=None,
        specialists=(
            RetrieverSpecialist(),
            ActionReasonerSpecialist(
                backend=backend,
                capability_names=("application.list",),
            ),
            AnswerDraftSpecialist(backend=backend),
            GroundingEvaluationSpecialist(),
        ),
        max_cycles=3,
    )

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-workspace-proposal",
        role="user",
        text="List installed applications",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    proposals = candidate.metadata.get("action_proposals")
    assert isinstance(proposals, list)
    assert len(proposals) == 1
    assert proposals[0]["capability"] == "application.list"


@pytest.mark.asyncio
async def test_workspace_engine_deterministic_abstention() -> None:
    backend = MockModelBackend()
    engine = WorkspaceEngine(backend=backend)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-workspace-abstain",
        role="user",
        text="What is my secret code?",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn, evidence_required=True, routing_decision="no_evidence")
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.metadata.get("deterministic_abstention") is True
    assert candidate.metadata.get("reason") == "memory_evidence_required_but_unavailable"


def test_salience_arbiter_selection_and_critique_penalty() -> None:
    arbiter = SalienceArbiter(convergence_threshold=0.8)
    workspace = WorkspaceState(
        cycle=1,
        goal="Find files",
        session_id="s1",
        turn_id="t1",
        history=(
            WorkspaceContribution(
                cycle=0,
                specialist_name="retriever",
                kind=WorkspaceContributionKind.CRITIQUE,
                salience=0.5,
            ),
        ),
    )

    c1 = WorkspaceContribution(
        cycle=1,
        specialist_name="retriever",
        kind=WorkspaceContributionKind.RETRIEVAL,
        content="Retrieved data",
        salience=0.8,  # Discounted to 0.4 due to critique
    )
    c2 = WorkspaceContribution(
        cycle=1,
        specialist_name="answer_draft",
        kind=WorkspaceContributionKind.ANSWER_DRAFT,
        content="Draft answer",
        salience=0.6,
    )

    winner = arbiter.select_winner([c1, c2], workspace)
    assert winner is not None
    assert winner.specialist_name == "answer_draft"


def test_container_wires_workspace_engine() -> None:
    config = {
        "assistant": {
            "interaction_controller": "workspace",
        }
    }
    container = ApplicationContainer(config=config, testing=True)
    service = container.assistant_service
    assert isinstance(service.controller, WorkspaceEngine)


@pytest.mark.asyncio
async def test_workspace_engine_jlens_probe_latent_drift_detection() -> None:
    class MockDriftInspector:
        def inspect_drift(self, goal: str, content: str) -> float:
            del goal, content
            return 0.95  # Exceeds divergence threshold 0.85

    backend = MockModelBackend(responses=["Potentially divergent answer"])
    probe = JLensProbe(inspector=MockDriftInspector(), divergence_threshold=0.85)
    engine = WorkspaceEngine(backend=backend, probe=probe)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-drift",
        role="user",
        text="Ambiguous query about unknown topic",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert "conceptual drift" in candidate.text
    assert candidate.metadata.get("latent_drift_detected") is True


@pytest.mark.asyncio
async def test_workspace_engine_grounding_critique_on_empty_draft() -> None:
    backend = MockModelBackend(responses=["   ", "Grounded non-empty response."])
    engine = WorkspaceEngine(backend=backend, max_cycles=3)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-grounding",
        role="user",
        text="Explain grounding evaluation",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.text == "Grounded non-empty response."


@pytest.mark.asyncio
async def test_workspace_engine_fails_closed_on_backend_failure() -> None:
    class FailingBackend(MockModelBackend):
        async def generate(
            self, request: InferenceRequest, cancellation: CancellationToken
        ) -> Result[AssistantCandidate, ActionFailure]:
            del request, cancellation
            return Failure(
                ActionFailure(
                    request_id="req-fail",
                    capability="model",
                    code="MODEL_ERROR",
                    message="Backend crashed",
                    producer=ProducerIdentity(
                        producer_id="test", kind="test", version="1.0.0"
                    ),
                )
            )

    engine = WorkspaceEngine(
        backend=FailingBackend(),
        specialists=(
            WorkspaceEngine(backend=FailingBackend()).specialists[2],
        ),
    )
    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-fail",
        role="user",
        text="Trigger backend failure",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Failure)
    assert result.failure().code == "MODEL_ERROR"


@pytest.mark.asyncio
async def test_workspace_engine_defaults_max_cycles_safely() -> None:
    backend = MockModelBackend(responses=["Cycle response."])
    engine = WorkspaceEngine(backend=backend, max_cycles=None)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-default-cycles",
        role="user",
        text="Check safe max cycles",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget().model_copy(update={"max_agent_turns": 0})

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)


@pytest.mark.asyncio
async def test_workspace_engine_enforces_latency_deadline() -> None:
    backend = MockModelBackend(responses=["Delayed response."])
    engine = WorkspaceEngine(backend=backend, max_cycles=4)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-timeout",
        role="user",
        text="Enforce deadline",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    # Deadline in the past
    past = _utc_now() - timedelta(seconds=10)
    budget = _make_budget().model_copy(update={"cancellation_deadline": past})

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Failure)
    assert result.failure().code == "DEADLINE_EXCEEDED"


@pytest.mark.asyncio
async def test_workspace_engine_enforces_token_budget_exhaustion() -> None:
    backend = MockModelBackend(responses=["Response using tokens."])
    engine = WorkspaceEngine(backend=backend, max_cycles=4)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-tokens",
        role="user",
        text="Exhaust token budget",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    # 1 input token ceiling
    budget = _make_budget().model_copy(update={"max_input_tokens": 1})

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Failure)
    assert result.failure().code == "BUDGET_EXCEEDED"


@pytest.mark.asyncio
async def test_workspace_engine_critique_circuit_breaker() -> None:
    # Model produces the same draft every cycle
    backend = MockModelBackend(
        responses=[
            "Identical ungrounded draft.",
            "Identical ungrounded draft.",
            "Identical ungrounded draft.",
        ]
    )
    engine = WorkspaceEngine(backend=backend, max_cycles=10)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-circuit-breaker",
        role="user",
        text="Test circuit breaker",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.text == "Identical ungrounded draft."


@pytest.mark.asyncio
async def test_action_reasoner_tokens_accounted_even_without_tool_call() -> None:
    from rai.assistant.cognitive.specialists import ActionReasonerSpecialist

    class TokenCountingBackend:
        def __init__(self) -> None:
            self.model_name = "test-token-model"

        async def generate(self, request: Any, token: CancellationToken) -> Result[AssistantCandidate, ActionFailure]:
            return Success(
                AssistantCandidate(
                    text="NO_ACTION",
                    tokens_in=42,
                    tokens_out=7,
                )
            )

    backend = TokenCountingBackend()
    specialist = ActionReasonerSpecialist(backend=backend, capability_names=("browser.search",))
    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-tokens-no-action",
        role="user",
        text="Hello world",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    state = WorkspaceState(
        goal="Hello world",
        session_id="session-tokens-no-action",
        turn_id=turn.record_id,
        context=context,
        budget=_make_budget(),
    )
    res = await specialist.evaluate(state, CancellationToken())
    assert isinstance(res, Success)
    contrib = res.unwrap()
    assert contrib is not None
    assert contrib.tokens_in == 42
    assert contrib.tokens_out == 7
    assert contrib.salience == 0.0


@pytest.mark.asyncio
async def test_workspace_engine_binds_trusted_session_id_overriding_model_input() -> None:
    backend = MockModelBackend(
        responses=[
            '<tool_call>call: run_python_code{"code": "x = 1", "session_id": "malicious-victim-session"}</tool_call>',
            "Executed code successfully.",
        ]
    )

    captured_requests: list[CapabilityRequest] = []

    class MockCapabilities(CapabilityService):
        def __init__(self) -> None:
            class MockRegistry:
                def __init__(self) -> None:
                    self._desc = CapabilityDescriptor(
                        name="run_python_code",
                        description="Run python",
                        input_schema={"type": "object", "properties": {"code": {"type": "string"}, "session_id": {"type": "string"}}},
                        risk_class=RiskClass.LOW,
                        side_effects=(),
                        isolation="process",
                        verification_plan=("verify_read_only",),
                    )

                def descriptor(self, name: str) -> CapabilityDescriptor | None:
                    return self._desc if name == "run_python_code" else None

                def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
                    return (self._desc,)

            self.registry = MockRegistry()

        async def invoke(
            self, request: CapabilityRequest, cancellation: CancellationToken
        ) -> tuple[Any, Result[ActionResult, ActionFailure]]:
            del cancellation
            captured_requests.append(request)
            return None, Success(
                ActionResult(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
                    request_id=request.record_id,
                    capability="run_python_code",
                    output={"result": "ok"},
                    verification={"status": "verified"},
                )
            )

    capabilities = MockCapabilities()
    skills = create_default_skill_registry()
    engine = WorkspaceEngine(
        backend=backend,
        capabilities=capabilities,
        skills=skills,
        max_cycles=4,
    )

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="trusted-workspace-session-id",
        role="user",
        text="Run x = 1",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    assert len(captured_requests) == 1
    assert captured_requests[0].arguments["session_id"] == "trusted-workspace-session-id"


@pytest.mark.asyncio
async def test_workspace_engine_persists_observation_artifact_when_store_provided(
    tmp_path: Path,
) -> None:
    from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415

    db_path = tmp_path / "obs_workspace.sqlite3"
    store = SQLiteMemoryGraphStore(path=db_path)

    backend = MockModelBackend(
        responses=[
            "<tool_call>call: application.list{}</tool_call>",
            "Found applications.",
        ]
    )

    class MockAppCapabilities(CapabilityService):
        def __init__(self) -> None:
            class MockRegistry:
                def __init__(self) -> None:
                    self._desc = CapabilityDescriptor(
                        name="application.list",
                        description="List applications",
                        input_schema={"type": "object", "properties": {}},
                        risk_class=RiskClass.LOW,
                        side_effects=(),
                        isolation="process",
                        verification_plan=("verify_read_only",),
                    )

                def descriptor(self, name: str) -> CapabilityDescriptor | None:
                    return self._desc if name == "application.list" else None

                def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
                    return (self._desc,)

            self.registry = MockRegistry()

        async def invoke(
            self, request: CapabilityRequest, cancellation: CancellationToken
        ) -> tuple[Any, Result[ActionResult, ActionFailure]]:
            del cancellation
            return None, Success(
                ActionResult(
                    record_id=_new_id(),
                    timestamp=_utc_now(),
                    producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
                    request_id=request.record_id,
                    capability="application.list",
                    output={"apps": ["Files.desktop", "Terminal.desktop"]},
                    verification={"status": "verified"},
                )
            )

    engine = WorkspaceEngine(
        backend=backend,
        capabilities=MockAppCapabilities(),
        skills=create_default_skill_registry(),
        store=store,
        max_cycles=4,
    )

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-ws-store-test",
        role="user",
        text="List apps",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await engine.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.metadata.get("observations_count") == 1

    stored_list_res = await store.list_for_session("session-ws-store-test")
    assert isinstance(stored_list_res, Success)
    stored_list = stored_list_res.unwrap()
    assert len(stored_list) == 1
    assert stored_list[0].artifact_id == "obs_1"
    assert stored_list[0].capability == "application.list"
    assert list(stored_list[0].content["apps"]) == ["Files.desktop", "Terminal.desktop"]
