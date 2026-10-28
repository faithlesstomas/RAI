"""Tests for max_iterations, max_full_observations, and dynamic context scaling configuration."""

# ruff: noqa: PLR2004, ANN001, ANN201, ANN401, PLC0415

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
import pytest
from returns.result import Failure, Result, Success

from rai.actions.workflow import (
    BoundedStepController,
    BoundedWorkflowLimits,
    _format_older_observation_summary,
)
from rai.assistant.backends.deterministic import DeterministicAssistantBackend
from rai.assistant.cognitive.models import WorkspaceState
from rai.assistant.cognitive.workspace import WorkspaceEngine
from rai.assistant.records import AssistantCandidate, ConversationTurn
from rai.assistant.runtime import AssistantRuntimeConfig, resolve_assistant_config
from rai.commands.settings import _assistant_config
from rai.configuration.models import ModelSettings
from rai.configuration.resolution import effective_model_settings
from rai.kernel.capabilities import CapabilityDescriptor
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionResult, ProducerIdentity, RiskClass, _new_id


def test_model_settings_max_iterations_and_full_obs() -> None:
    # Defaults
    s = ModelSettings()
    assert s.max_iterations is None
    assert s.max_full_observations == 3

    # Various ways to express unbounded
    for inf_val in ("inf", "infinity", "none", "null", "0", "unlimited", float("inf"), None):
        s_inf = ModelSettings(max_iterations=inf_val)
        assert s_inf.max_iterations is None

    # Positive integer
    s_val = ModelSettings(max_iterations=5, max_full_observations=4)
    assert s_val.max_iterations == 5
    assert s_val.max_full_observations == 4


def test_resolution_env_and_cli() -> None:
    # Test env vars
    env = {
        "RAI_MAX_ITERATIONS": "7",
        "RAI_MAX_FULL_OBSERVATIONS": "5",
        "RAI_MAX_CONTEXT_CHARACTERS": "12000",
    }
    vals, sources = effective_model_settings(config={}, environment=env)
    assert vals["max_iterations"] == 7
    assert sources["max_iterations"] == "environment"
    assert vals["max_full_observations"] == 5
    assert sources["max_full_observations"] == "environment"
    assert vals["max_context_characters"] == 12000
    assert sources["max_context_characters"] == "environment"

    # Test CLI overrides with None (unbounded) overriding config
    vals, sources = effective_model_settings(
        config={"assistant": {"max_iterations": 10}},
        overrides={"max_iterations": None},
    )
    assert vals["max_iterations"] is None
    assert sources["max_iterations"] == "CLI"


def test_dynamic_vs_manual_context_characters() -> None:
    # Default settings -> default characters (32000)
    cfg = resolve_assistant_config(
        config={"assistant": {"backend": "lemonade", "model": "mock"}},
    )
    assert cfg.max_context_characters == 32000

    # Custom context window without manual max_context_characters -> dynamic scaling
    cfg_dyn = resolve_assistant_config(
        config={"assistant": {"backend": "lemonade", "model": "mock", "context_window": 4096, "max_output_tokens": 512}},
    )
    # usable_tokens = max(512, 4096 - 512 - 256) = 3328; 3328 * 3.5 = 11648
    assert cfg_dyn.max_context_characters == int((4096 - 512 - 256) * 3.5)

    # Explicit manual override takes precedence
    cfg_man = resolve_assistant_config(
        config={
            "assistant": {
                "backend": "lemonade",
                "model": "mock",
                "context_window": 4096,
                "max_output_tokens": 512,
                "max_context_characters": 15000,
            },
            "_cli_overrides": {"max_context_characters": 15000},
        },
    )
    assert cfg_man.max_context_characters == 15000


def test_format_older_observation_summary() -> None:
    obs_dict = {
        "status": "SUCCESS",
        "total_count": 10,
        "title": "Search Results",
        "results": [{"title": "Riemman"}, {"title": "Hypothesis"}],
    }
    summary = _format_older_observation_summary(obs_dict)
    assert "status=SUCCESS" in summary
    assert "total_count=10" in summary
    assert "title='Search Results'" in summary
    assert "\n" not in summary

    # String obs
    assert _format_older_observation_summary("Simple text output") == "Simple text output"


def test_cli_assistant_config_parsing() -> None:
    # String "inf" -> None
    cfg = _assistant_config(
        backend="local",
        model="test-model",
        max_iterations="inf",
        max_full_observations=2,
        max_context_characters=8000,
    )
    assert cfg["assistant"]["max_iterations"] is None
    assert cfg["_cli_overrides"]["max_iterations"] is None
    assert cfg["assistant"]["max_full_observations"] == 2
    assert cfg["_cli_overrides"]["max_full_observations"] == 2
    assert cfg["assistant"]["max_context_characters"] == 8000
    assert cfg["_cli_overrides"]["max_context_characters"] == 8000

    # Explicit integer
    cfg_int = _assistant_config(
        backend="local",
        model="test-model",
        max_iterations="4",
    )
    assert cfg_int["assistant"]["max_iterations"] == 4
    assert cfg_int["_cli_overrides"]["max_iterations"] == 4


@pytest.mark.asyncio
async def test_bounded_step_controller_observation_compaction() -> None:
    """Verify that older observations are compacted and only the last max_full_observations are kept in full."""
    captured_prompts: list[str] = []

    class MockBackend(DeterministicAssistantBackend):
        async def generate(self, request: Any, token: CancellationToken) -> Any:
            sys_inst = request.system_instruction or ""
            captured_prompts.append(sys_inst)
            step_count = len(captured_prompts)
            if step_count == 1:
                return Success(
                    AssistantCandidate(
                        text='<tool_call>{"name": "tool_a", "arguments": {"x": 1}}</tool_call>'
                    )
                )
            if step_count == 2:
                return Success(
                    AssistantCandidate(
                        text='<tool_call>{"name": "tool_b", "arguments": {"y": 2}}</tool_call>'
                    )
                )
            if step_count == 3:
                return Success(
                    AssistantCandidate(
                        text='<tool_call>{"name": "tool_c", "arguments": {"z": 3}}</tool_call>'
                    )
                )
            return Success(AssistantCandidate(text="Final answer based on all tools."))

    class MockCapabilities:
        class Registry:
            def descriptors(self) -> list[Any]:
                class Desc:
                    name = "tool_a"
                    description = "A"
                return [Desc()]

            def descriptor(self, name: str) -> Any:
                return CapabilityDescriptor(
                    name=name,
                    description="desc",
                    input_schema={"type": "object", "properties": {}},
                    risk_class=RiskClass.LOW,
                    side_effects=(),
                    isolation="host-api",
                    verification_plan=("observed",),
                )

        registry = Registry()

        async def invoke(self, req: Any, cancellation: Any) -> Any:
            res = ActionResult(
                record_id=_new_id(),
                timestamp=datetime.now(timezone.utc),
                producer=req.producer,
                request_id=req.record_id,
                capability=req.capability,
                output={"status": "SUCCESS", "step": req.capability},
                verification={"status": "dispatched"},
            )
            return req, Success(res)

    controller = BoundedStepController(
        capabilities=MockCapabilities(),  # type: ignore[arg-type]
        backend=MockBackend(),
        limits=BoundedWorkflowLimits(max_steps=None, max_iterations=None, max_tool_calls=None),
        max_full_observations=2,
    )

    from rai.assistant.records import AssistantContextManifest, AssistantContextPackage

    turn = ConversationTurn(
        record_id=_new_id(),
        session_id=_new_id(),
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        role="user",
        text="Run three tools and answer",
    )
    manifest = AssistantContextManifest(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_required=False,
        routing_decision="no_evidence",
    )
    ctx = AssistantContextPackage(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content={"recent_turns": [], "memories": []},
    )

    from rai.kernel.records import InferenceBudget
    budget = InferenceBudget(
        record_id=_new_id(),
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        max_input_tokens=1000,
        max_output_tokens=200,
        max_agent_turns=0,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=60),
    )

    res = await controller.process_turn(turn, ctx, budget, CancellationToken())
    assert isinstance(res, Success)
    cand = res.unwrap()
    assert "Final answer" in cand.text

    # Check that in step 4 prompt, Observation 1 is compacted as (summary), and Observations 2 and 3 are full
    last_prompt = captured_prompts[-1]
    assert "Observation 1 (summary):" in last_prompt
    assert "Observation 2: {" in last_prompt
    assert "Observation 3: {" in last_prompt


@pytest.mark.asyncio
async def test_bounded_step_controller_max_iterations_enforced() -> None:
    """Verify that when max_iterations is reached, the controller stops tool loops and forces synthesis."""
    call_count = 0

    class InfiniteToolBackend(DeterministicAssistantBackend):
        async def generate(self, request: Any, token: CancellationToken) -> Any:
            nonlocal call_count
            call_count += 1
            sys_inst = request.system_instruction or ""
            if "Provide the final comprehensive answer" in sys_inst:
                return Success(AssistantCandidate(text="Forced synthesis completed."))
            return Success(
                AssistantCandidate(
                    text=f'<tool_call>{{"name": "tool_x", "arguments": {{"val": {call_count}}}}}</tool_call>'
                )
            )

    class MockCapabilities:
        class Registry:
            def descriptors(self) -> list[Any]:
                return []

            def descriptor(self, name: str) -> Any:
                return CapabilityDescriptor(
                    name=name,
                    description="desc",
                    input_schema={"type": "object", "properties": {}},
                    risk_class=RiskClass.LOW,
                    side_effects=(),
                    isolation="host-api",
                    verification_plan=("observed",),
                )

        registry = Registry()

        async def invoke(self, req: Any, cancellation: Any) -> Any:
            res = ActionResult(
                record_id=_new_id(),
                timestamp=datetime.now(timezone.utc),
                producer=req.producer,
                request_id=req.record_id,
                capability=req.capability,
                output={"status": "SUCCESS", "val": req.arguments.get("val")},
                verification={"status": "dispatched"},
            )
            return req, Success(res)

    controller = BoundedStepController(
        capabilities=MockCapabilities(),  # type: ignore[arg-type]
        backend=InfiniteToolBackend(),
        limits=BoundedWorkflowLimits(max_steps=None, max_iterations=2, max_tool_calls=None),
    )

    from rai.assistant.records import AssistantContextManifest, AssistantContextPackage

    turn = ConversationTurn(
        record_id=_new_id(),
        session_id=_new_id(),
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        role="user",
        text="Infinite loop task",
    )
    manifest = AssistantContextManifest(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_required=False,
        routing_decision="no_evidence",
    )
    ctx = AssistantContextPackage(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content={"recent_turns": [], "memories": []},
    )

    from rai.kernel.records import InferenceBudget
    budget = InferenceBudget(
        record_id=_new_id(),
        producer=ProducerIdentity(producer_id="test", kind="test", version="1.0.0"),
        max_input_tokens=1000,
        max_output_tokens=200,
        max_agent_turns=0,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=60),
    )

    res = await controller.process_turn(turn, ctx, budget, CancellationToken())
    assert isinstance(res, Success)
    cand = res.unwrap()
    assert "Forced synthesis completed" in cand.text


def test_workspace_state_unbounded_max_cycles() -> None:
    ws = WorkspaceState(
        cycle=0,
        max_cycles=None,
        goal="Test goal",
        session_id="s1",
        turn_id="t1",
    )
    assert ws.max_cycles is None


def test_model_settings_max_observation_characters_validation() -> None:
    # Default is auto
    s = ModelSettings()
    assert s.max_observation_characters == "auto"

    # None and 'auto' resolve to 'auto'
    assert ModelSettings(max_observation_characters=None).max_observation_characters == "auto"
    assert ModelSettings(max_observation_characters="auto").max_observation_characters == "auto"
    assert ModelSettings(max_observation_characters="AUTO").max_observation_characters == "auto"

    # Invariants for infinity/unlimited
    for inf_cand in ("inf", "infinity", "unlimited", "none", "null", float("inf")):
        assert ModelSettings(max_observation_characters=inf_cand).max_observation_characters == "inf"

    # Valid positive integers
    assert ModelSettings(max_observation_characters=5000).max_observation_characters == 5000
    assert ModelSettings(max_observation_characters="12000").max_observation_characters == 12000

    # Invalid values raise ValueError
    for bad in (0, -100, "0", "-5", "invalid", "xyz"):
        with pytest.raises(ValueError):
            ModelSettings(max_observation_characters=bad)


def test_resolve_assistant_config_max_observation_characters() -> None:
    # 1. Default config
    cfg1 = {"backend": "deterministic", "max_output_tokens": 100, "context_window": 1000}
    res1 = resolve_assistant_config(cfg1)
    assert res1.max_observation_characters == "auto"

    # 2. Explicit integer
    cfg2 = {
        "backend": "deterministic",
        "max_output_tokens": 100,
        "context_window": 1000,
        "assistant": {"max_observation_characters": 8000},
    }
    res2 = resolve_assistant_config(cfg2)
    assert res2.max_observation_characters == 8000

    # 3. Explicit inf
    cfg3 = {
        "backend": "deterministic",
        "max_output_tokens": 100,
        "context_window": 1000,
        "assistant": {"max_observation_characters": "inf"},
    }
    res3 = resolve_assistant_config(cfg3)
    assert res3.max_observation_characters == "inf"


@pytest.mark.asyncio
async def test_bounded_step_controller_dynamic_observation_budgeting() -> None:
    from rai.actions.workflow import BoundedStepController  # noqa: PLC0415
    from rai.assistant.ports import AssistantModelBackend  # noqa: PLC0415
    from rai.assistant.records import AssistantContextManifest, AssistantContextPackage  # noqa: PLC0415
    from rai.kernel.records import InferenceBudget  # noqa: PLC0415

    class SmallContextBuilder:
        max_context_characters = 2000

    observed_prompts: list[str] = []

    class CapturingBackend(AssistantModelBackend):
        model_name = "capturing-backend"

        async def generate(self, request: Any, token: CancellationToken) -> Result[AssistantCandidate, Any]:
            del token
            observed_prompts.append(str(request.system_instruction or ""))
            return Success(
                AssistantCandidate(
                    text="Completed after observing.",
                    tokens_in=50,
                    tokens_out=20,
                )
            )

    class MockCapService:
        class Registry:
            def descriptors(self) -> list[Any]:
                return []

            def descriptor(self, name: str) -> Any:
                return None

        registry = Registry()

        async def invoke(self, req: Any, cancellation: Any) -> Any:
            return req, Success(None)

    ctrl = BoundedStepController(
        capabilities=MockCapService(),  # type: ignore[arg-type]
        backend=CapturingBackend(),
        context_builder=SmallContextBuilder(),
        max_observation_characters="auto",
    )

    turn = ConversationTurn(
        record_id=_new_id(),
        session_id="session-dyn-budget",
        producer=ProducerIdentity(producer_id="user", kind="user", version="1.0.0"),
        role="user",
        text="Task needing observation",
    )

    # Initial observations with large content (e.g. 100 items)
    large_obs = {
        "items": [{"id": i, "name": f"Item {i} with long description to expand payload"} for i in range(100)],
        "_artifact_id": "obs_1",
    }
    manifest = AssistantContextManifest(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        evidence_character_budget=2000,
    )
    ctx = AssistantContextPackage(
        producer=turn.producer,
        session_id=turn.session_id,
        turn_id=turn.record_id,
        manifest=manifest,
        content={"tool_observations": [large_obs]},
    )
    budget = InferenceBudget(
        record_id=_new_id(),
        producer=turn.producer,
        max_input_tokens=500,
        max_output_tokens=100,
        max_agent_turns=0,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=30,
        max_provider_cost=0,
        max_ram_bytes=0,
        max_vram_bytes=0,
        cancellation_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    res = await ctrl.process_turn(turn, ctx, budget, CancellationToken())
    assert isinstance(res, Success)
    assert len(observed_prompts) == 1
    # Verify that observation in prompt was reduced to fit within small budget
    prompt = observed_prompts[0]
    assert len(prompt) < 2000
    assert "Tool Observations:" in prompt
