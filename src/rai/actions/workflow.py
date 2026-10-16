"""Bounded agentic workflows and deterministic step controller.

Enforces:
1. Strict finite state transitions:
   PLANNING -> EVALUATING -> EXECUTING -> OBSERVING -> SYNTHESIZING -> COMPLETE
   (alongside backwards-compatible ResearchWorkflow lifecycle).
2. Hard limits: max tool calls, max total steps, timeouts.
3. Multi-stage approval isolation: approval tokens are scoped and single-use per action.
4. Unified token, latency, and tool-call accounting recorded into UsageLedger.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.assistant.ports import (
    AssistantActionHandler,
    AssistantModelBackend,
    InteractionController,
)
from rai.assistant.records import (
    AssistantCandidate,
    AssistantContextPackage,
    ConversationTurn,
    InferenceRequest,
    ToolObservation,
)
from rai.assistant.tool_calling import extract_model_tool_call
from rai.kernel.ports import CancellationToken, UsageLedger
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    DataClass,
    InferenceBudget,
    KernelRecord,
    ProducerIdentity,
    UsageRecord,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from rai.skills.models import SkillRegistry

PRODUCER = ProducerIdentity(producer_id="bounded-step-controller", kind="service", version="1.0.0")
_MEMORY_ABSTENTION_TEXT: str = (
    "Nie mam wystarczających, dopuszczonych dowodów w pamięci, aby odpowiedzieć."
)


class AgentWorkflowState(str, Enum):
    """Generic finite state machine states for universal agentic workflows."""

    PLANNING = "PLANNING"
    EVALUATING = "EVALUATING"
    EXECUTING = "EXECUTING"
    OBSERVING = "OBSERVING"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ALLOWED_GENERIC_TRANSITIONS: dict[AgentWorkflowState, set[AgentWorkflowState]] = {
    AgentWorkflowState.PLANNING: {
        AgentWorkflowState.EVALUATING,
        AgentWorkflowState.SYNTHESIZING,
        AgentWorkflowState.COMPLETE,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.EVALUATING: {
        AgentWorkflowState.EXECUTING,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.EXECUTING: {
        AgentWorkflowState.OBSERVING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.OBSERVING: {
        AgentWorkflowState.SYNTHESIZING,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.SYNTHESIZING: {
        AgentWorkflowState.COMPLETE,
        AgentWorkflowState.PLANNING,
        AgentWorkflowState.FAILED,
        AgentWorkflowState.CANCELLED,
    },
    AgentWorkflowState.COMPLETE: set(),
    AgentWorkflowState.FAILED: set(),
    AgentWorkflowState.CANCELLED: set(),
}


_MAX_OBSERVATION_CHARS: int = 1000


def _state_str(state: object) -> str:
    return state.value if isinstance(state, Enum) else str(state)


class BoundedWorkflowLimits(BaseModel):
    """Upper bounds preventing unbounded agent loops across any capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=5, ge=0, le=50)
    max_tokens_budget: int = Field(default=16384, ge=256)
    max_wall_time_seconds: float = Field(default=120.0, ge=1.0)


class BoundedWorkflowStep(BaseModel):
    """Audited step in a generic bounded workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    state_before: AgentWorkflowState
    state_after: AgentWorkflowState
    action_taken: str = Field(min_length=1)
    capability: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] | None = None
    approval_id: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class BoundedWorkflow(KernelRecord):
    """Durable record of a generic bounded agentic workflow."""

    record_type: Literal["bounded_workflow"] = "bounded_workflow"
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    state: AgentWorkflowState = AgentWorkflowState.PLANNING
    limits: BoundedWorkflowLimits = Field(default_factory=BoundedWorkflowLimits)
    tool_calls_executed: int = Field(default=0, ge=0)
    steps: tuple[BoundedWorkflowStep, ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    approvals_used: tuple[str, ...] = ()
    final_synthesis: str | None = None
    data_class: DataClass = DataClass.PUBLIC


# Backwards-compatible specialized research workflow records
class ResearchWorkflowState(str, Enum):
    """Explicit lifecycle states for bounded research flows."""

    PLANNING = "PLANNING"
    SEARCHING = "SEARCHING"
    SELECTING_SOURCES = "SELECTING_SOURCES"
    READING_PAGE = "READING_PAGE"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ResearchWorkflowLimits(BaseModel):
    """Hard upper bounds preventing unbounded agent loops in research workflows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_searches: int = Field(default=1, ge=1, le=5)
    max_page_reads: int = Field(default=2, ge=0, le=10)
    max_total_steps: int = Field(default=8, ge=1, le=20)


class ResearchWorkflowStep(BaseModel):
    """Single audited step within the research workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_index: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=_utc_now)
    state_before: ResearchWorkflowState
    state_after: ResearchWorkflowState
    action_taken: str = Field(min_length=1)
    capability: str | None = None
    target_resource: str | None = None
    approval_id: str | None = None
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class ResearchWorkflow(KernelRecord):
    """Durable record of a bounded research task."""

    record_type: Literal["research_workflow"] = "research_workflow"
    task_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    state: ResearchWorkflowState = ResearchWorkflowState.PLANNING
    limits: ResearchWorkflowLimits = Field(default_factory=ResearchWorkflowLimits)
    searches_executed: int = Field(default=0, ge=0)
    page_reads_executed: int = Field(default=0, ge=0)
    steps: tuple[ResearchWorkflowStep, ...] = ()
    search_approval_id: str | None = None
    read_approval_ids: tuple[str, ...] = ()
    found_results: tuple[dict[str, Any], ...] = ()
    read_pages: tuple[dict[str, Any], ...] = ()
    final_synthesis: str | None = None
    data_class: DataClass = DataClass.PUBLIC


class BoundedStepController(InteractionController):
    """Universal controller governing bounded agentic workflows across any capabilities."""

    def __init__(  # noqa: PLR0913
        self,
        ledger: UsageLedger | None = None,
        capabilities: CapabilityService | None = None,
        backend: AssistantModelBackend | None = None,
        actions: AssistantActionHandler | None = None,
        limits: BoundedWorkflowLimits | None = None,
        skills: SkillRegistry | None = None,
    ) -> None:
        self.ledger = ledger
        self.capabilities = capabilities
        self.backend = backend
        self.actions = actions
        self.limits = limits or BoundedWorkflowLimits()
        self.skills = skills

    async def process_turn(  # noqa: PLR0915
        self,
        turn: ConversationTurn,
        context: AssistantContextPackage,
        budget: InferenceBudget,
        token: CancellationToken,
    ) -> Result[AssistantCandidate, ActionFailure]:
        """Execute a turn under the bounded agentic loop state machine."""
        manifest = context.manifest
        initial_observations: list[dict[str, Any]] = []
        if self.actions is not None and hasattr(self.actions, "handle"):
            action_cand = await self.actions.handle(turn, token)
            if action_cand is not None:
                if (
                    not action_cand.metadata.get("requires_synthesis")
                    and not action_cand.metadata.get("untrusted_content")
                ):
                    return Success(
                        action_cand.model_copy(
                            update={
                                "metadata": {
                                    **action_cand.metadata,
                                    "updated_manifest": manifest,
                                    "latency_ms": 0.0,
                                }
                            }
                        )
                    )
                act_res = action_cand.metadata.get("action_result")
                if isinstance(act_res, dict) and isinstance(act_res.get("output"), dict):
                    initial_observations.append(dict(act_res["output"]))
                elif action_cand.text:
                    initial_observations.append({"formatted_text": action_cand.text})

        if manifest.evidence_required and manifest.routing_decision == "no_evidence":
            return Success(
                AssistantCandidate(
                    text=_MEMORY_ABSTENTION_TEXT,
                    metadata={
                        "deterministic_abstention": True,
                        "reason": "memory_evidence_required_but_unavailable",
                        "updated_manifest": manifest,
                        "latency_ms": 0.0,
                    },
                )
            )

        if self.capabilities is None or self.backend is None:
            return Failure(
                ActionFailure(
                    request_id=f"bounded:{turn.record_id}",
                    capability="bounded_step_controller",
                    code="BACKEND_UNAVAILABLE",
                    message="Capabilities service or backend is not connected to BoundedStepController",
                    producer=PRODUCER,
                )
            )

        task_id = f"task:{turn.record_id}"

        async def step_decider(
            workflow: BoundedWorkflow,
            cancellation: CancellationToken,
        ) -> Result[dict[str, Any], ActionFailure]:
            descriptors = self.capabilities.registry.descriptors() if self.capabilities else ()
            actions_desc = "\n".join(f"- {d.name}: {d.description}" for d in descriptors)
            skills_section = ""
            if self.skills is not None:
                avail_caps = tuple(d.name for d in descriptors)
                skills_section = self.skills.format_skills_instruction(avail_caps)

            all_obs = (*initial_observations, *workflow.observations)
            obs_lines = []
            tool_obs_records = list(context.content.get("tool_observations", []))
            obs_ids = list(manifest.tool_observation_ids)
            for i, obs in enumerate(all_obs, 1):
                obs_summary = json.dumps(obs, ensure_ascii=False)
                if len(obs_summary) > _MAX_OBSERVATION_CHARS:
                    obs_summary = obs_summary[:_MAX_OBSERVATION_CHARS] + "... (truncated)"
                obs_lines.append(f"Observation {i}: {obs_summary}")
                obs_id = f"{workflow.task_id}:obs:{i}"
                obs_ids.append(obs_id)
                tool_obs_records.append(
                    ToolObservation(
                        source_id=obs_id,
                        source_turn_id=turn.record_id,
                        capability="workflow_step",
                        status="SUCCESS",
                        summary=obs_summary[:120],
                        untrusted_content=True,
                        content=obs if isinstance(obs, dict) else {"value": obs},
                    ).model_dump(mode="json")
                )
            obs_text = "\n".join(obs_lines) if obs_lines else "None yet."
            prompt_content = f"User Goal: {workflow.goal}\n\nTool Observations:\n{obs_text}"

            system_prompt = (
                "You are an agent executing user tasks step by step.\n"
                "Available tools:\n"
                f"{actions_desc}"
                f"{skills_section}\n\n"
                "Tool calling format:\n"
                "To invoke a tool, output:\n"
                "<tool_call>call: tool_name{\"param\": \"value\"}</tool_call>\n"
                "Or in JSON format:\n"
                "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>\n\n"
                f"{prompt_content}\n\n"
                "When you have collected all required information to answer the user request, "
                "or if no tools are needed, provide the final comprehensive answer directly "
                "without tool tags."
            )

            step_manifest = manifest.model_copy(
                update={"tool_observation_ids": tuple(obs_ids)}
            )
            step_context = context.model_copy(
                update={
                    "manifest": step_manifest,
                    "content": {
                        **context.content,
                        "tool_observations": tool_obs_records,
                    },
                }
            )

            req = InferenceRequest(
                record_id=_new_id(),
                timestamp=_utc_now(),
                producer=PRODUCER,
                session_id=turn.session_id,
                turn_id=turn.record_id,
                request_id=_new_id(),
                context=step_context,
                budget=budget,
                strategy="BOUNDED_STEP",
                system_instruction=system_prompt,
                model_name=str(getattr(self.backend, "model_name", "deterministic")),
            )

            start = time.perf_counter()
            gen_res = await self.backend.generate(req, cancellation)
            latency = (time.perf_counter() - start) * 1000
            if isinstance(gen_res, Failure):
                return Failure(gen_res.failure())

            cand = gen_res.unwrap()
            tool_call = extract_model_tool_call(cand.text)
            if tool_call:
                _, cap_name, tool_args = tool_call
                return Success({
                    "action_type": "tool",
                    "capability": cap_name,
                    "arguments": tool_args,
                    "tokens_in": cand.tokens_in,
                    "tokens_out": cand.tokens_out,
                    "latency_ms": latency,
                })

            return Success({
                "action_type": "finish",
                "text": cand.text,
                "tokens_in": cand.tokens_in,
                "tokens_out": cand.tokens_out,
                "latency_ms": latency,
            })

        loop_res = await self.run_agentic_loop(
            task_id=task_id,
            session_id=turn.session_id,
            goal=turn.text,
            capabilities=self.capabilities,
            step_decider=step_decider,
            cancellation=token,
            limits=self.limits,
            actor=turn.producer,
        )
        if isinstance(loop_res, Failure):
            return Failure(loop_res.failure())

        workflow = loop_res.unwrap()
        total_latency = sum(step.latency_ms for step in workflow.steps)
        total_tokens_in = sum(step.tokens_in for step in workflow.steps)
        total_tokens_out = sum(step.tokens_out for step in workflow.steps)

        candidate = AssistantCandidate(
            text=workflow.final_synthesis or "",
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            metadata={
                "bounded_workflow_id": workflow.task_id,
                "steps_count": len(workflow.steps),
                "tool_calls_executed": workflow.tool_calls_executed,
                "updated_manifest": context.manifest,
                "latency_ms": total_latency,
            },
        )
        return Success(candidate)

    # -------------------------------------------------------------------------
    # Generic Bounded Workflow Orchestration
    # -------------------------------------------------------------------------

    def create_generic_workflow(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        goal: str,
        limits: BoundedWorkflowLimits | None = None,
        data_class: DataClass = DataClass.PUBLIC,
        actor: ProducerIdentity | None = None,
    ) -> BoundedWorkflow:
        """Initialize a new bounded workflow in the PLANNING state."""
        return BoundedWorkflow(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=actor or PRODUCER,
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            state=AgentWorkflowState.PLANNING,
            limits=limits or BoundedWorkflowLimits(),
            data_class=data_class,
        )

    def transition_generic(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow,
        target_state: AgentWorkflowState,
        action_taken: str,
        capability: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[BoundedWorkflow, str]:
        """Transition generic workflow to a target state, validating against transition rules and limits."""
        allowed = ALLOWED_GENERIC_TRANSITIONS.get(workflow.state, set())
        if target_state not in allowed:
            return Failure(
                f"INVALID_TRANSITION: cannot transition from {_state_str(workflow.state)} to {_state_str(target_state)}"
            )
        if len(workflow.steps) >= workflow.limits.max_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = BoundedWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=target_state,
            action_taken=action_taken,
            capability=capability,
            arguments=arguments or {},
            observation=observation,
            approval_id=approval_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": target_state,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    async def execute_generic_step(  # noqa: PLR0911, PLR0913
        self,
        workflow: BoundedWorkflow,
        capabilities: CapabilityService,
        capability_name: str,
        arguments: dict[str, Any],
        cancellation: CancellationToken,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[tuple[BoundedWorkflow, ActionResult], ActionFailure]:
        """Invoke a capability under generic state machine supervision: EVALUATING -> EXECUTING -> OBSERVING."""
        step_index = len(workflow.steps) + 1
        req_id = f"{workflow.task_id}:step:{step_index}"

        if workflow.tool_calls_executed >= workflow.limits.max_tool_calls:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="TOOL_CALL_LIMIT_EXCEEDED",
                    message="Maximum allowed tool calls limit reached",
                    producer=PRODUCER,
                )
            )

        if len(workflow.steps) >= workflow.limits.max_steps:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="STEP_LIMIT_EXCEEDED",
                    message="Maximum allowed steps limit reached",
                    producer=PRODUCER,
                )
            )

        if approval_id is not None and approval_id in workflow.approvals_used:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="APPROVAL_ALREADY_USED",
                    message=f"Approval token '{approval_id}' has already been consumed",
                    producer=PRODUCER,
                )
            )

        # 1. PLANNING -> EVALUATING
        eval_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.EVALUATING,
            action_taken=f"evaluate:{capability_name}",
            capability=capability_name,
            arguments=arguments,
            approval_id=approval_id,
        )
        if isinstance(eval_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=eval_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = eval_tx.unwrap()

        # 2. EVALUATING -> EXECUTING
        exec_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.EXECUTING,
            action_taken=f"execute:{capability_name}",
            capability=capability_name,
            arguments=arguments,
            approval_id=approval_id,
        )
        if isinstance(exec_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=exec_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = exec_tx.unwrap()

        descriptor = capabilities.registry.descriptor(capability_name)
        if descriptor is None:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="CAPABILITY_NOT_FOUND",
                    message=f"Capability '{capability_name}' is not registered",
                    producer=PRODUCER,
                )
            )

        request = normalize_request(
            descriptor,
            arguments,
            actor=workflow.producer,
            request_id=req_id,
            data_class=workflow.data_class,
        )
        _, executed = await capabilities.invoke(request, cancellation)
        if isinstance(executed, Failure):
            _ = self.transition_generic(
                workflow,
                AgentWorkflowState.FAILED,
                action_taken=f"execution_failed:{executed.failure().message}",
                capability=capability_name,
            )
            return Failure(executed.failure())

        action_res = executed.unwrap()

        # 3. EXECUTING -> OBSERVING
        obs_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.OBSERVING,
            action_taken=f"observed:{capability_name}",
            capability=capability_name,
            observation=action_res.output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(obs_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability=capability_name,
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=obs_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = obs_tx.unwrap()

        approvals_now = (*workflow.approvals_used, approval_id) if approval_id else workflow.approvals_used
        updated_wf = workflow.model_copy(
            update={
                "tool_calls_executed": workflow.tool_calls_executed + 1,
                "observations": (*workflow.observations, action_res.output),
                "approvals_used": approvals_now,
            }
        )

        if self.ledger is not None:
            await self.record_usage(
                updated_wf,
                processor=f"step_{capability_name}",
                backend="capability",
                model=capability_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((updated_wf, action_res))

    async def execute_generic_synthesis(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[BoundedWorkflow, ActionFailure]:
        """Advance workflow to SYNTHESIZING (if needed) and COMPLETE with final text."""
        req_id = f"{workflow.task_id}:synthesis"
        if workflow.state in {AgentWorkflowState.OBSERVING, AgentWorkflowState.PLANNING}:
            synth_tx = self.transition_generic(
                workflow,
                AgentWorkflowState.SYNTHESIZING,
                action_taken="begin_synthesis",
            )
            if isinstance(synth_tx, Failure):
                return Failure(
                    ActionFailure(
                        request_id=req_id,
                        capability="assistant.synthesis",
                        code="WORKFLOW_TRANSITION_FAILED",
                        message=synth_tx.failure(),
                        producer=PRODUCER,
                    )
                )
            workflow = synth_tx.unwrap()

        complete_tx = self.transition_generic(
            workflow,
            AgentWorkflowState.COMPLETE,
            action_taken="synthesis_completed",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(complete_tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="assistant.synthesis",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=complete_tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = complete_tx.unwrap()

        updated = workflow.model_copy(update={"final_synthesis": final_synthesis})

        if self.ledger is not None:
            await self.record_usage(
                updated,
                processor="assistant_synthesizer",
                backend="assistant_model",
                model="synthesizer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=0,
                latency_ms=latency_ms,
            )

        return Success(updated)

    async def run_agentic_loop(  # noqa: PLR0911, PLR0912, PLR0913
        self,
        task_id: str,
        session_id: str,
        goal: str,
        capabilities: CapabilityService,
        step_decider: Callable[
            [BoundedWorkflow, CancellationToken],
            Awaitable[Result[dict[str, Any], ActionFailure]],
        ],
        cancellation: CancellationToken,
        limits: BoundedWorkflowLimits | None = None,
        actor: ProducerIdentity | None = None,
    ) -> Result[BoundedWorkflow, ActionFailure]:
        """Run an end-to-end universal agentic loop governed by BoundedWorkflow state machine."""
        workflow = self.create_generic_workflow(
            task_id=task_id,
            session_id=session_id,
            goal=goal,
            limits=limits,
            actor=actor,
        )

        while workflow.state not in {
            AgentWorkflowState.COMPLETE,
            AgentWorkflowState.FAILED,
            AgentWorkflowState.CANCELLED,
        }:
            if cancellation.cancelled:
                cancel_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.CANCELLED,
                    action_taken="cancelled_by_token",
                )
                if isinstance(cancel_tx, Success):
                    workflow = cancel_tx.unwrap()
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:cancel",
                        capability="agent_loop",
                        code="TASK_CANCELLED",
                        message="Workflow cancelled by token",
                        producer=PRODUCER,
                    )
                )

            if len(workflow.steps) >= workflow.limits.max_steps:
                fail_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.FAILED,
                    action_taken="step_limit_exceeded",
                )
                if isinstance(fail_tx, Success):
                    workflow = fail_tx.unwrap()
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:limit",
                        capability="agent_loop",
                        code="STEP_LIMIT_EXCEEDED",
                        message=f"Step limit {workflow.limits.max_steps} exceeded",
                        producer=PRODUCER,
                    )
                )

            decision_res = await step_decider(workflow, cancellation)
            if isinstance(decision_res, Failure):
                fail_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.FAILED,
                    action_taken=f"decision_failed:{decision_res.failure().message}",
                )
                if isinstance(fail_tx, Success):
                    workflow = fail_tx.unwrap()
                return Failure(decision_res.failure())

            decision = decision_res.unwrap()
            action_type = decision.get("action_type")

            if action_type == "finish":
                final_text = decision.get("text", "")
                return await self.execute_generic_synthesis(
                    workflow,
                    final_synthesis=final_text,
                    tokens_in=decision.get("tokens_in", 0),
                    tokens_out=decision.get("tokens_out", 0),
                    latency_ms=decision.get("latency_ms", 0.0),
                )

            if action_type == "tool":
                cap_name = decision.get("capability")
                if not cap_name:
                    return Failure(
                        ActionFailure(
                            request_id=f"{workflow.task_id}:step",
                            capability="agent_loop",
                            code="MISSING_CAPABILITY",
                            message="Decision specified 'tool' without capability name",
                            producer=PRODUCER,
                        )
                    )
                step_res = await self.execute_generic_step(
                    workflow=workflow,
                    capabilities=capabilities,
                    capability_name=cap_name,
                    arguments=decision.get("arguments", {}),
                    cancellation=cancellation,
                    approval_id=decision.get("approval_id"),
                    tokens_in=decision.get("tokens_in", 0),
                    tokens_out=decision.get("tokens_out", 0),
                    latency_ms=decision.get("latency_ms", 0.0),
                )
                if isinstance(step_res, Failure):
                    return Failure(step_res.failure())

                workflow, _ = step_res.unwrap()

                # Deliberate next step: OBSERVING -> PLANNING
                plan_tx = self.transition_generic(
                    workflow,
                    AgentWorkflowState.PLANNING,
                    action_taken=f"deliberate_after:{cap_name}",
                )
                if isinstance(plan_tx, Failure):
                    return Failure(
                        ActionFailure(
                            request_id=f"{workflow.task_id}:loop",
                            capability="agent_loop",
                            code="WORKFLOW_TRANSITION_FAILED",
                            message=plan_tx.failure(),
                            producer=PRODUCER,
                        )
                    )
                workflow = plan_tx.unwrap()
            else:
                return Failure(
                    ActionFailure(
                        request_id=f"{workflow.task_id}:decision",
                        capability="agent_loop",
                        code="INVALID_DECISION_TYPE",
                        message=f"Unknown decision action_type: {action_type}",
                        producer=PRODUCER,
                    )
                )

        return Success(workflow)

    # -------------------------------------------------------------------------
    # Specialized Research Workflow Orchestration (Backwards Compatibility)
    # -------------------------------------------------------------------------

    def create_workflow(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        query: str,
        limits: ResearchWorkflowLimits | None = None,
        data_class: DataClass = DataClass.PUBLIC,
        actor: ProducerIdentity | None = None,
    ) -> ResearchWorkflow:
        """Initialize a new bounded research workflow in the PLANNING state."""
        return ResearchWorkflow(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=actor or PRODUCER,
            task_id=task_id,
            session_id=session_id,
            query=query,
            state=ResearchWorkflowState.PLANNING,
            limits=limits or ResearchWorkflowLimits(),
            data_class=data_class,
        )

    async def record_usage(  # noqa: PLR0913
        self,
        workflow: BoundedWorkflow | ResearchWorkflow,
        processor: str,
        backend: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        tool_calls: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[UsageRecord, ActionFailure]:
        """Record structured usage metrics into the ledger."""
        if self.ledger is None:
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:usage",
                    capability="usage_ledger",
                    code="LEDGER_UNAVAILABLE",
                    message="No UsageLedger configured",
                    producer=PRODUCER,
                )
            )
        record = UsageRecord(
            task_id=workflow.task_id,
            processor=processor,
            backend=backend,
            model=model,
            provider="local",
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            tool_calls=tool_calls,
            images=0,
            audio_seconds=0.0,
            latency_seconds=latency_ms / 1000.0,
            producer=PRODUCER,
        )
        return await self.ledger.append(record)

    def verify_multistage_approval(
        self,
        workflow: ResearchWorkflow,
        stage: Literal["search", "read_page"],
        approval_id: str,
    ) -> Result[bool, str]:
        """Verify that approvals for searching and page reading are distinct."""
        if not approval_id:
            return Failure("MISSING_APPROVAL_ID")

        if stage == "search":
            return Success(True)

        if stage == "read_page":
            if workflow.search_approval_id and approval_id == workflow.search_approval_id:
                return Failure("APPROVAL_REUSE_FORBIDDEN: cannot reuse search approval for page read")
            if approval_id in workflow.read_approval_ids:
                return Failure("APPROVAL_ALREADY_USED: duplicate approval token for page read")
            return Success(True)

        return Failure("INVALID_STAGE")

    def transition_to_search(
        self,
        workflow: ResearchWorkflow,
        approval_id: str | None = None,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from PLANNING to SEARCHING."""
        if workflow.state != ResearchWorkflowState.PLANNING:
            return Failure(f"INVALID_TRANSITION: cannot transition to SEARCHING from {_state_str(workflow.state)}")
        if workflow.searches_executed >= workflow.limits.max_searches:
            return Failure("SEARCH_LIMIT_EXCEEDED")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.SEARCHING,
            action_taken="initiate_search",
            capability="browser.search",
            approval_id=approval_id,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.SEARCHING,
                "search_approval_id": approval_id or workflow.search_approval_id,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def record_search_completed(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        results: tuple[dict[str, Any], ...],
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SEARCHING to SELECTING_SOURCES with recorded results."""
        if workflow.state != ResearchWorkflowState.SEARCHING:
            return Failure(f"INVALID_TRANSITION: cannot complete search from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.SELECTING_SOURCES,
            action_taken=f"search_completed: found {len(results)} items",
            capability="browser.search",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.SELECTING_SOURCES,
                "searches_executed": workflow.searches_executed + 1,
                "found_results": results,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def transition_to_reading_page(
        self,
        workflow: ResearchWorkflow,
        target_url: str,
        approval_id: str,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SELECTING_SOURCES to READING_PAGE with verified page read approval."""
        if workflow.state != ResearchWorkflowState.SELECTING_SOURCES:
            return Failure(f"INVALID_TRANSITION: cannot transition to READING_PAGE from {_state_str(workflow.state)}")
        if workflow.page_reads_executed >= workflow.limits.max_page_reads:
            return Failure("PAGE_READ_LIMIT_EXCEEDED")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        approval_check = self.verify_multistage_approval(workflow, "read_page", approval_id)
        if isinstance(approval_check, Failure):
            return Failure(approval_check.failure())

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.READING_PAGE,
            action_taken=f"read_page_target: {target_url}",
            capability="browser.read_page",
            target_resource=target_url,
            approval_id=approval_id,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.READING_PAGE,
                "read_approval_ids": (*workflow.read_approval_ids, approval_id),
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def record_page_read_completed(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        page_content: dict[str, Any],
        more_sources_needed: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Record completed page read and advance to SELECTING_SOURCES or SYNTHESIZING."""
        if workflow.state != ResearchWorkflowState.READING_PAGE:
            return Failure(f"INVALID_TRANSITION: cannot complete page read from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        reads_now = workflow.page_reads_executed + 1
        can_read_more = reads_now < workflow.limits.max_page_reads and more_sources_needed
        next_state = ResearchWorkflowState.SELECTING_SOURCES if can_read_more else ResearchWorkflowState.SYNTHESIZING

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=next_state,
            action_taken=f"page_read_completed: {page_content.get('title', 'untitled')}",
            capability="browser.read_page",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": next_state,
                "page_reads_executed": reads_now,
                "read_pages": (*workflow.read_pages, page_content),
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    def transition_to_complete(
        self,
        workflow: ResearchWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, str]:
        """Transition from SYNTHESIZING to COMPLETE with final answer."""
        if workflow.state not in {ResearchWorkflowState.SYNTHESIZING, ResearchWorkflowState.SELECTING_SOURCES}:
            return Failure(f"INVALID_TRANSITION: cannot complete from {_state_str(workflow.state)}")
        if len(workflow.steps) >= workflow.limits.max_total_steps:
            return Failure("STEP_LIMIT_EXCEEDED")

        step = ResearchWorkflowStep(
            step_index=len(workflow.steps) + 1,
            state_before=workflow.state,
            state_after=ResearchWorkflowState.COMPLETE,
            action_taken="synthesis_completed",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        updated = workflow.model_copy(
            update={
                "state": ResearchWorkflowState.COMPLETE,
                "final_synthesis": final_synthesis,
                "steps": (*workflow.steps, step),
            }
        )
        return Success(updated)

    async def execute_bounded_search(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        capabilities: CapabilityService,
        query: str,
        cancellation: CancellationToken,
        approval_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[tuple[ResearchWorkflow, ActionResult], ActionFailure]:
        """Orchestrate search transition, capability invocation, and usage ledger entry."""
        tx = self.transition_to_search(workflow, approval_id)
        if isinstance(tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = tx.unwrap()

        descriptor = capabilities.registry.descriptor("browser.search")
        if descriptor is None:
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="CAPABILITY_NOT_FOUND",
                    message="browser.search capability is not registered",
                    producer=PRODUCER,
                )
            )

        request = normalize_request(
            descriptor,
            {"task_id": workflow.task_id, "query": query},
            actor=workflow.producer,
            request_id=f"{workflow.task_id}:search",
            data_class=workflow.data_class,
        )
        _, executed = await capabilities.invoke(request, cancellation)
        if isinstance(executed, Failure):
            return Failure(executed.failure())

        action_res = executed.unwrap()
        results = action_res.output.get("results", ())

        completed = self.record_search_completed(
            workflow,
            tuple(results),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:search",
                    capability="browser.search",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="browser_search",
                backend="browser",
                model="web_search",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((workflow, action_res))

    async def execute_bounded_read(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        capabilities: CapabilityService,
        handle: str,
        target_url: str,
        approval_id: str,
        cancellation: CancellationToken,
        more_sources_needed: bool = False,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[tuple[ResearchWorkflow, ActionResult], ActionFailure]:
        """Orchestrate page read transition, capability invocation, and usage ledger entry."""
        step_num = workflow.page_reads_executed + 1
        req_id = f"{workflow.task_id}:read_page:{step_num}"
        tx = self.transition_to_reading_page(workflow, target_url=target_url, approval_id=approval_id)
        if isinstance(tx, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=tx.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = tx.unwrap()

        descriptor = capabilities.registry.descriptor("browser.read_page")
        if descriptor is None:
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="CAPABILITY_NOT_FOUND",
                    message="browser.read_page capability is not registered",
                    producer=PRODUCER,
                )
            )

        request = normalize_request(
            descriptor,
            {"task_id": workflow.task_id, "handle": handle},
            actor=workflow.producer,
            request_id=req_id,
            data_class=workflow.data_class,
        )
        _, executed = await capabilities.invoke(request, cancellation)
        if isinstance(executed, Failure):
            return Failure(executed.failure())

        action_res = executed.unwrap()
        page_content = action_res.output

        completed = self.record_page_read_completed(
            workflow,
            page_content,
            more_sources_needed=more_sources_needed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=req_id,
                    capability="browser.read_page",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="browser_read_page",
                backend="browser",
                model="web_reader",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=1,
                latency_ms=latency_ms,
            )

        return Success((workflow, action_res))

    async def execute_workflow_synthesis(  # noqa: PLR0913
        self,
        workflow: ResearchWorkflow,
        final_synthesis: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
    ) -> Result[ResearchWorkflow, ActionFailure]:
        """Transition workflow to COMPLETE and record synthesis usage metrics."""
        completed = self.transition_to_complete(
            workflow,
            final_synthesis=final_synthesis,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        if isinstance(completed, Failure):
            return Failure(
                ActionFailure(
                    request_id=f"{workflow.task_id}:synthesis",
                    capability="assistant.synthesis",
                    code="WORKFLOW_TRANSITION_FAILED",
                    message=completed.failure(),
                    producer=PRODUCER,
                )
            )
        workflow = completed.unwrap()

        if self.ledger is not None:
            await self.record_usage(
                workflow,
                processor="assistant_synthesizer",
                backend="assistant_model",
                model="synthesizer",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=0,
                latency_ms=latency_ms,
            )

        return Success(workflow)

    async def run_bounded_research(  # noqa: PLR0913
        self,
        task_id: str,
        session_id: str,
        query: str,
        capabilities: CapabilityService,
        synthesizer: Callable[[ResearchWorkflow, CancellationToken], Awaitable[Result[str, ActionFailure]]],
        cancellation: CancellationToken,
        search_approval_id: str | None = None,
        read_approval_factory: Callable[[int, str], str] | None = None,
        limits: ResearchWorkflowLimits | None = None,
        actor: ProducerIdentity | None = None,
    ) -> Result[ResearchWorkflow, ActionFailure]:
        """Run an end-to-end bounded research task: search -> read -> synthesize -> complete."""
        workflow = self.create_workflow(
            task_id=task_id,
            session_id=session_id,
            query=query,
            limits=limits,
            actor=actor,
        )

        # 1. Bounded search
        search_res = await self.execute_bounded_search(
            workflow, capabilities, query, cancellation, approval_id=search_approval_id
        )
        if isinstance(search_res, Failure):
            return Failure(search_res.failure())
        workflow, _ = search_res.unwrap()

        # 2. Bounded page reads (up to max_page_reads)
        results = workflow.found_results
        max_reads = min(len(results), workflow.limits.max_page_reads)
        for i in range(max_reads):
            item = results[i]
            target_url = item.get("url", "")
            handle = item.get("handle", "")
            approval_id = (
                read_approval_factory(i + 1, target_url)
                if read_approval_factory
                else f"{task_id}:read:{i + 1}"
            )
            more_needed = (i + 1) < max_reads
            read_res = await self.execute_bounded_read(
                workflow,
                capabilities,
                handle=handle,
                target_url=target_url,
                approval_id=approval_id,
                cancellation=cancellation,
                more_sources_needed=more_needed,
            )
            if isinstance(read_res, Failure):
                return Failure(read_res.failure())
            workflow, _ = read_res.unwrap()

        # 3. Model synthesis
        synth_res = await synthesizer(workflow, cancellation)
        if isinstance(synth_res, Failure):
            return Failure(synth_res.failure())
        final_text = synth_res.unwrap()

        # 4. Complete and record usage
        return await self.execute_workflow_synthesis(workflow, final_synthesis=final_text)


class TaskStepController(BoundedStepController):
    """Backwards-compatible subclass alias for BoundedStepController."""


WorkflowLimits = BoundedWorkflowLimits
WorkflowStep = BoundedWorkflowStep

__all__ = [
    "ALLOWED_GENERIC_TRANSITIONS",
    "AgentWorkflowState",
    "BoundedStepController",
    "BoundedWorkflow",
    "BoundedWorkflowLimits",
    "BoundedWorkflowStep",
    "ResearchWorkflow",
    "ResearchWorkflowLimits",
    "ResearchWorkflowState",
    "ResearchWorkflowStep",
    "TaskStepController",
    "WorkflowLimits",
    "WorkflowStep",
]
