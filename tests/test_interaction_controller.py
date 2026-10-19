"""Unit tests for InteractionController protocol, DirectController, and BoundedStepController integration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, NoReturn

import pytest
from returns.result import Result, Success

from rai.actions.workflow import (
    BoundedStepController,
    BoundedWorkflowLimits,
)
from rai.assistant.controllers.direct import DirectController
from rai.assistant.ports import (
    AssistantActionHandler,
    AssistantModelBackend,
    InteractionController,
)
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
)
from rai.assistant.service import AssistantService
from rai.assistant.store import SQLiteMemoryGraphStore
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


class MockModelBackend(AssistantModelBackend):
    def __init__(self, response_text: str = "Hello from mock model") -> None:
        self.response_text = response_text
        self.model_name = "mock-model"
        self._state = LifecycleState.RUNNING

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def is_remote(self) -> bool:
        return False

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        return Success(LifecycleState.STOPPED)

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]:
        return Success(
            AssistantCandidate(
                text=self.response_text,
                tokens_in=10,
                tokens_out=15,
            )
        )

    def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
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
        max_agent_turns=5,
        max_tool_calls=5,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=60,
        max_provider_cost=0,
        max_ram_bytes=2 * 1024 * 1024 * 1024,
        max_vram_bytes=0,
        cancellation_deadline=now + timedelta(seconds=60),
    )


def test_interaction_controller_protocol_conformance() -> None:
    backend = MockModelBackend()
    direct = DirectController(backend=backend)
    assert isinstance(direct, InteractionController)

    bounded = BoundedStepController()
    assert isinstance(bounded, InteractionController)


@pytest.mark.asyncio
async def test_direct_controller_conversational_turn() -> None:
    backend = MockModelBackend(response_text="Direct conversational response.")
    controller = DirectController(backend=backend)

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-1",
        role="user",
        text="Hello assistant",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await controller.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.text == "Direct conversational response."
    expected_tokens_in = 10
    expected_tokens_out = 15
    assert candidate.tokens_in == expected_tokens_in
    assert candidate.tokens_out == expected_tokens_out


@pytest.mark.asyncio
async def test_direct_controller_action_passthrough_and_synthesis() -> None:
    backend = MockModelBackend(response_text="Synthesized application list.")

    class MockActionHandler(AssistantActionHandler):
        async def handle(
            self, turn: ConversationTurn, cancellation: CancellationToken
        ) -> AssistantCandidate | None:
            del cancellation
            if "applications" in turn.text.lower():
                return AssistantCandidate(
                    text="Files\nTerminal\nCalculator",
                    metadata={
                        "requires_synthesis": True,
                        "action_result": {
                            "capability": "application.list",
                            "output": {"entries": ["Files", "Terminal", "Calculator"]},
                        },
                    },
                )
            return None

    controller = DirectController(backend=backend, actions=MockActionHandler())
    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-1",
        role="user",
        text="List installed applications",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await controller.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.text == "Synthesized application list."
    assert candidate.metadata.get("synthesized_from_action") is True
    assert "updated_manifest" in candidate.metadata


@pytest.mark.asyncio
async def test_direct_controller_evidence_abstention() -> None:
    backend = MockModelBackend()
    controller = DirectController(backend=backend)
    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-1",
        role="user",
        text="What is my secret code?",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn, evidence_required=True, routing_decision="no_evidence")
    budget = _make_budget()

    result = await controller.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert candidate.metadata.get("deterministic_abstention") is True
    assert candidate.metadata.get("reason") == "memory_evidence_required_but_unavailable"


@pytest.mark.asyncio
async def test_bounded_step_controller_process_turn_agentic_loop() -> None:
    class StepModelBackend(AssistantModelBackend):
        def __init__(self) -> None:
            self.model_name = "step-model"
            self.turn_count = 0

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
            del cancellation
            self.turn_count += 1
            if self.turn_count == 1:
                # Deliberate: emit tool call to discover applications
                return Success(
                    AssistantCandidate(
                        text='<tool_call>call: application.list{}</tool_call>',
                        tokens_in=20,
                        tokens_out=10,
                    )
                )
            # Synthesis: observe results and synthesize answer
            assert request.context.content.get("tool_observations")
            assert "Files.desktop" in (request.system_instruction or "")
            return Success(
                AssistantCandidate(
                    text="Found 2 applications in the system: Files and Calculator.",
                    tokens_in=30,
                    tokens_out=15,
                )
            )

        def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
            del request, cancellation
            raise NotImplementedError

    class MockCapabilities(CapabilityService):
        def __init__(self) -> None:
            class MockRegistry:
                def __init__(self) -> None:
                    self._desc = CapabilityDescriptor(
                        name="application.list",
                        description="List installed desktop applications",
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
                    output={"applications": ["Files.desktop", "Calculator.desktop"]},
                    verification={"status": "verified"},
                )
            )

    step_backend = StepModelBackend()
    capabilities = MockCapabilities()
    controller = BoundedStepController(
        capabilities=capabilities,
        backend=step_backend,
        limits=BoundedWorkflowLimits(max_steps=10, max_tool_calls=2),
    )

    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
    turn = ConversationTurn(
        producer=producer,
        session_id="session-agentic",
        role="user",
        text="Count all applications in the system.",
        data_class=DataClass.PUBLIC,
    )
    context = _make_context_package(turn)
    budget = _make_budget()

    result = await controller.process_turn(turn, context, budget, CancellationToken())
    assert isinstance(result, Success)
    candidate = result.unwrap()
    assert "Found 2 applications in the system" in candidate.text
    assert candidate.metadata.get("tool_calls_executed") == 1
    min_expected_steps = 2
    assert candidate.metadata.get("steps_count") >= min_expected_steps


@pytest.mark.asyncio
async def test_assistant_service_with_custom_controller(tmp_path: Path) -> None:
    store = SQLiteMemoryGraphStore(tmp_path / "memory.db")

    class CustomInteractionController(InteractionController):
        async def process_turn(
            self,
            turn: ConversationTurn,
            context: AssistantContextPackage,
            budget: InferenceBudget,
            token: CancellationToken,
        ) -> Result[AssistantCandidate, ActionFailure]:
            del context, budget, token
            return Success(
                AssistantCandidate(
                    text=f"Custom response to: {turn.text}",
                    metadata={"custom_controller_active": True},
                )
            )

    custom_controller = CustomInteractionController()
    service = AssistantService(store=store, controller=custom_controller)
    await service.start()
    try:
        producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")
        turn = ConversationTurn(
            producer=producer,
            session_id="session-custom",
            role="user",
            text="Testing custom controller injection",
            data_class=DataClass.PUBLIC,
        )
        resp = await service.accept_turn(turn)
        assert isinstance(resp, Success)
        response = resp.unwrap()
        assert response.text == "Custom response to: Testing custom controller injection"
    finally:
        await service.stop()


def test_cli_controller_option_wires_bounded_step_by_default() -> None:
    import inspect  # noqa: PLC0415

    from rai.commands.assistant import ask_command, chat_command  # noqa: PLC0415
    from rai.commands.settings import _assistant_config  # noqa: PLC0415
    from rai.container import ApplicationContainer  # noqa: PLC0415

    ask_sig = inspect.signature(ask_command)
    chat_sig = inspect.signature(chat_command)
    assert ask_sig.parameters["controller"].default.default == "bounded_step"
    assert chat_sig.parameters["controller"].default.default == "bounded_step"

    cfg = _assistant_config(backend=None, model=None, controller="bounded_step")
    assistant_cfg = cfg.get("assistant")
    assert isinstance(assistant_cfg, dict)
    assert assistant_cfg.get("interaction_controller") == "bounded_step"

    container = ApplicationContainer(config=cfg, testing=True)
    assert isinstance(container.assistant_service.controller, BoundedStepController)


@pytest.mark.asyncio
async def test_bounded_step_controller_synthesizes_application_count_and_capabilities() -> None:
    total_apps = 123
    apps_list = [
        {"name": f"App {i}", "desktop_id": f"app-{i}.desktop", "handle": f"h-{i}"}
        for i in range(1, total_apps + 1)
    ]

    class PreRoutedActionHandler(AssistantActionHandler):
        async def handle(
            self, turn: ConversationTurn, cancellation: CancellationToken
        ) -> AssistantCandidate | None:
            del cancellation
            if "count" in turn.text.lower():
                return AssistantCandidate(
                    text="W Twoim systemie mogę uruchomić zainstalowane aplikacje... (oraz 108 innych)",
                    metadata={
                        "action_choices": {
                            "kind": "application",
                            "task_id": "t-1",
                            "entries": apps_list,
                        },
                        "action_result": {
                            "capability": "application.list",
                            "status": "SUCCEEDED",
                            "output": {
                                "total_count": total_apps,
                                "application_names": [a["name"] for a in apps_list],
                                "applications": apps_list,
                            },
                        },
                    },
                )
            if "what can you do" in turn.text.lower():
                return AssistantCandidate(
                    text="Registered Linux actions:\n- application.list: Discover installed desktop applications.",
                    metadata={
                        "action_result": {
                            "capability": "capabilities.list",
                            "status": "SUCCEEDED",
                            "output": {
                                "total_count": 1,
                                "capabilities": [
                                    {
                                        "name": "application.list",
                                        "description": "Discover installed desktop applications.",
                                    }
                                ],
                            },
                        },
                    },
                )
            return None

    class InspectingBackend(AssistantModelBackend):
        def __init__(self) -> None:
            self.model_name = "inspecting-model"
            self.seen_instructions: list[str] = []

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
            del cancellation
            sys_inst = request.system_instruction or ""
            self.seen_instructions.append(sys_inst)
            if '"total_count": 123' in sys_inst:
                return Success(
                    AssistantCandidate(
                        text="There are 123 installed applications in your system.",
                        tokens_in=40,
                        tokens_out=12,
                        metadata={"reasoning_content": "Counted 123 apps from total_count."},
                    )
                )
            return Success(
                AssistantCandidate(
                    text="I can list and launch Linux desktop applications for you.",
                    tokens_in=35,
                    tokens_out=14,
                    metadata={"reasoning_content": "Summarized capabilities."},
                )
            )

        def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
            del request, cancellation
            raise NotImplementedError

    class DummyCapabilities(CapabilityService):
        def __init__(self) -> None:
            class EmptyRegistry:
                def descriptor(self, name: str) -> None:
                    del name

                def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
                    return ()

            self.registry = EmptyRegistry()

    backend = InspectingBackend()
    controller = BoundedStepController(
        capabilities=DummyCapabilities(),
        backend=backend,
        actions=PreRoutedActionHandler(),
    )
    producer = ProducerIdentity(producer_id="user", kind="user", version="1.0.0")

    turn_count = ConversationTurn(
        producer=producer,
        session_id="session-synth",
        role="user",
        text="Count all applications in the system.",
        data_class=DataClass.LOCAL,
    )
    res_count = await controller.process_turn(
        turn_count, _make_context_package(turn_count), _make_budget(), CancellationToken()
    )
    assert isinstance(res_count, Success)
    cand_count = res_count.unwrap()
    assert cand_count.text == "There are 123 installed applications in your system."
    assert cand_count.metadata.get("synthesized_from_action") is True
    assert cand_count.metadata.get("reasoning_content") == "Counted 123 apps from total_count."

    turn_caps = ConversationTurn(
        producer=producer,
        session_id="session-synth",
        role="user",
        text="What can you do for me?",
        data_class=DataClass.LOCAL,
    )
    res_caps = await controller.process_turn(
        turn_caps, _make_context_package(turn_caps), _make_budget(), CancellationToken()
    )
    assert isinstance(res_caps, Success)
    cand_caps = res_caps.unwrap()
    assert cand_caps.text == "I can list and launch Linux desktop applications for you."
    assert cand_caps.metadata.get("synthesized_from_action") is True

