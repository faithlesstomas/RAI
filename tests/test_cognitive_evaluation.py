"""Unit tests for InteractionController and WorkspaceEngine comparative evaluation benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NoReturn

import pytest
from returns.result import Result, Success

from rai.actions.workflow import BoundedStepController, BoundedWorkflowLimits
from rai.assistant.cognitive.evaluation import (
    ControllerBenchmarkReport,
    create_default_benchmark_scenarios,
    evaluate_controller,
    format_benchmark_report_markdown,
    run_controller_comparative_benchmark,
    save_benchmark_report,
)
from rai.assistant.cognitive.workspace import WorkspaceEngine
from rai.assistant.controllers.direct import DirectController
from rai.assistant.ports import AssistantModelBackend
from rai.assistant.records import AssistantCandidate, InferenceRequest
from rai.kernel.capabilities import CapabilityDescriptor
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    ProducerIdentity,
    RiskClass,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.skills.defaults import create_default_skill_registry

_MIN_DEFAULT_SCENARIOS: int = 4
_TOTAL_BENCHMARK_SCENARIOS: int = 4
_TOTAL_BENCHMARK_METRICS: int = 12
_EXPECTED_DIRECT_TOKENS: int = 55


class MockBenchmarkBackend(AssistantModelBackend):
    def __init__(self, responses: list[str] | None = None, scenario_aware: bool = False) -> None:
        self.responses = list(
            responses or ["Rich AI (RAI) is the secure local-first integration layer."]
        )
        self.call_count = 0
        self.scenario_aware = scenario_aware
        self.model_name = "mock-benchmark-backend"

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
        if self.scenario_aware:
            sys_inst = request.system_instruction or ""
            obs = request.context.content.get("tool_observations", [])
            sid = request.session_id
            if "action_system_discovery" in sid:
                if "Action Reasoning Specialist" in sys_inst and not obs:
                    text = "<tool_call>call: application.list{}</tool_call>"
                elif "Action Reasoning Specialist" in sys_inst:
                    text = "NO_ACTION"
                else:
                    text = "Installed applications: Terminal and Files."
            elif "multi_turn_context_retention" in sid:
                text = (
                    "NO_ACTION"
                    if "Action Reasoning Specialist" in sys_inst
                    else "The Apollo project is developed in Rust."
                )
            elif "hallucination_abstention_vault" in sid:
                text = (
                    "NO_ACTION"
                    if "Action Reasoning Specialist" in sys_inst
                    else "No evidence available in memory."
                )
            else:
                text = (
                    "NO_ACTION"
                    if "Action Reasoning Specialist" in sys_inst
                    else "Rich AI (RAI) is a local-first Linux assistant."
                )
        else:
            text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return Success(
            AssistantCandidate(
                text=text,
                tokens_in=30,
                tokens_out=25,
            )
        )

    def stream(self, request: InferenceRequest, cancellation: CancellationToken) -> NoReturn:
        del request, cancellation
        raise NotImplementedError


class MockBenchmarkCapabilities(CapabilityService):
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
                output={"apps": ["Terminal.desktop", "Files.desktop"]},
                verification={"status": "verified"},
            )
        )


def test_default_scenarios_composition() -> None:
    scenarios = create_default_benchmark_scenarios()
    assert len(scenarios) >= _MIN_DEFAULT_SCENARIOS
    categories = {s.category for s in scenarios}
    assert "action_research" in categories
    assert "context_retention" in categories
    assert "hallucination_abstention" in categories
    assert "conversational" in categories


@pytest.mark.asyncio
async def test_evaluate_single_scenario_metric() -> None:
    backend = MockBenchmarkBackend(responses=["Rich AI (RAI) provides desktop capabilities."])
    direct = DirectController(backend=backend)
    scenarios = create_default_benchmark_scenarios()
    conv_scenario = next(s for s in scenarios if s.category == "conversational")

    metric = await evaluate_controller("direct", direct, conv_scenario)
    assert metric.controller_name == "direct"
    assert metric.completed is True
    assert metric.hallucinated is False
    assert metric.tokens_total == _EXPECTED_DIRECT_TOKENS
    assert metric.latency_ms > 0


@pytest.mark.asyncio
async def test_run_controller_comparative_benchmark_suite() -> None:
    capabilities = MockBenchmarkCapabilities()
    skills = create_default_skill_registry()

    direct_backend = MockBenchmarkBackend(
        responses=[
            "Found applications: Terminal and Files.",
            "The Apollo project is written in Rust.",
            "Insufficient admitted evidence in memory to answer.",
            "Rich AI (RAI) is a local AI assistant for Linux.",
        ]
    )
    direct_ctrl = DirectController(backend=direct_backend)

    bounded_backend = MockBenchmarkBackend(
        responses=[
            "<tool_call>call: application.list{}</tool_call>",
            "Here are the applications in the system: Terminal and Files.",
            "The Apollo project is written in Rust.",
            "Insufficient admitted evidence in memory to answer.",
            "Rich AI (RAI) is a local assistant.",
        ]
    )
    bounded_ctrl = BoundedStepController(
        capabilities=capabilities,
        backend=bounded_backend,
        limits=BoundedWorkflowLimits(max_steps=8),
    )

    workspace_backend = MockBenchmarkBackend(scenario_aware=True)
    workspace_engine = WorkspaceEngine(
        backend=workspace_backend,
        capabilities=capabilities,
        skills=skills,
        max_cycles=3,
    )

    controllers = {
        "direct": direct_ctrl,
        "bounded_step": bounded_ctrl,
        "workspace_engine": workspace_engine,
    }

    result = await run_controller_comparative_benchmark(controllers)
    assert isinstance(result, Success)
    report: ControllerBenchmarkReport = result.unwrap()

    assert len(report.scenarios) == _TOTAL_BENCHMARK_SCENARIOS
    assert len(report.metrics) == _TOTAL_BENCHMARK_METRICS  # 3 controllers * 4 scenarios

    assert "direct" in report.aggregates
    assert "bounded_step" in report.aggregates
    assert "workspace_engine" in report.aggregates
    assert report.aggregates["bounded_step"].tool_call_efficiency > 0.0
    assert report.aggregates["workspace_engine"].tool_call_efficiency > 0.0
    assert report.aggregates["workspace_engine"].task_completion_rate == 1.0

    # Verify report formatting
    md = format_benchmark_report_markdown(report)
    assert "# InteractionController Comparative Benchmark Report" in md
    assert "| `direct` |" in md
    assert "| `bounded_step` |" in md
    assert "| `workspace_engine` |" in md

    # Verify JSON persistence
    with TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "benchmark_report.json"
        saved = save_benchmark_report(report, json_path)
        assert saved.exists()
        loaded = json.loads(saved.read_text(encoding="utf-8"))
        assert loaded["report_id"] == report.report_id
        assert "aggregates" in loaded
