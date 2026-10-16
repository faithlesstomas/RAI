"""Unit tests for WorkspaceEngine, specialists competition, and GWT blackboard deliberation."""

from __future__ import annotations

from datetime import timedelta
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
        text="Policz aplikacje w systemie",
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
    assert "Nie mam wystarczających" in candidate.text
    assert candidate.metadata.get("deterministic_abstention") is True


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
