"""Bounded research workflow and deterministic task step controller.

Enforces:
1. Strict finite state transitions:
   PLANNING -> SEARCHING -> SELECTING_SOURCES -> READING_PAGE -> SYNTHESIZING -> COMPLETE
2. Hard limits: max searches (default 1), max page reads (default 2), max total steps.
3. Multi-stage approval: search approval is distinct from external page read approval.
4. Unified token, latency, and tool-call accounting recorded into UsageLedger.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, UsageLedger
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    DataClass,
    KernelRecord,
    PolicyDecision,
    ProducerIdentity,
    UsageRecord,
    _new_id,
    _utc_now,
)
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request

PRODUCER = ProducerIdentity(producer_id="task-step-controller", kind="service", version="1.0.0")


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


def _state_str(state: ResearchWorkflowState | str) -> str:
    return state.value if isinstance(state, Enum) else str(state)


class ResearchWorkflowLimits(BaseModel):
    """Hard upper bounds preventing unbounded agent loops."""

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


class TaskStepController:
    """Deterministic, finite state machine controller for research workflows."""

    def __init__(self, ledger: UsageLedger | None = None) -> None:
        self.ledger = ledger

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
        now = _utc_now()
        return ResearchWorkflow(
            record_id=_new_id(),
            timestamp=now,
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
        workflow: ResearchWorkflow,
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
