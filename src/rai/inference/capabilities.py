"""Capability registry composition for bounded local inference tasks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import (
    CapabilityDescriptor,
    CapabilityRegistry,
    RegisteredCapability,
)
from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    ContextManifest,
    ContextManifestItem,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
    RiskClass,
    Task,
    _new_id,
    _utc_now,
    max_data_class,
)

from .supervisor import ProcessorSupervisor
from .tasks import BoundedTaskKind

PRODUCER = ProducerIdentity(
    producer_id="rai.inference", kind="capability", version="1.0.0"
)

BOUNDED_TASK_CAPABILITY = "inference.bounded_task"


def bounded_task_descriptor() -> CapabilityDescriptor:
    """Return the transport-neutral schema and policy metadata for bounded inference."""
    return CapabilityDescriptor(
        name=BOUNDED_TASK_CAPABILITY,
        description=(
            "Execute a schema-constrained, policy-bounded local inference task "
            "(episode summarization, intent classification, entity extraction, "
            "salience estimation, privacy risk elevation, routing hints)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [k.value for k in BoundedTaskKind],
                },
                "objective": {"type": "string", "minLength": 1},
                "content": {"type": "object"},
                "task_id": {"type": "string"},
                "data_class": {
                    "type": "string",
                    "enum": ["PUBLIC", "LOCAL", "PRIVATE", "SECRET", "BLOCKED"],
                },
                "max_input_tokens": {"type": "integer", "minimum": 1},
                "max_output_tokens": {"type": "integer", "minimum": 1},
                "max_latency_seconds": {"type": "number", "minimum": 0.1},
            },
            "required": ["kind", "objective"],
            "additionalProperties": False,
        },
        risk_class=RiskClass.LOW,
        side_effects=(),
        isolation="in-process",
        verification_plan=("claim-schema-validation",),
    )


class BoundedTaskCapability:
    """Invokes ProcessorSupervisor bounded tasks through the common capability path."""

    name = BOUNDED_TASK_CAPABILITY

    def __init__(
        self,
        supervisor: ProcessorSupervisor | Callable[[], ProcessorSupervisor],
    ) -> None:
        self._supervisor = supervisor

    @property
    def supervisor(self) -> ProcessorSupervisor:
        if isinstance(self._supervisor, ProcessorSupervisor):
            return self._supervisor
        return self._supervisor()

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        if cancellation.cancelled:
            return Failure(
                ActionFailure(
                    record_id=f"failure:{request.record_id}:CANCELLED",
                    producer=PRODUCER,
                    correlation_id=request.correlation_id,
                    request_id=request.record_id,
                    capability=self.name,
                    code="CANCELLED",
                    message="Bounded inference task was cancelled",
                )
            )

        args = request.arguments
        kind_raw = args.get("kind")
        objective = args.get("objective")
        if not kind_raw or not objective:
            return Failure(
                ActionFailure(
                    record_id=f"failure:{request.record_id}:INVALID_ARGUMENT",
                    producer=PRODUCER,
                    correlation_id=request.correlation_id,
                    request_id=request.record_id,
                    capability=self.name,
                    code="INVALID_ARGUMENT",
                    message="Missing required 'kind' or 'objective'",
                )
            )

        try:
            task_kind = BoundedTaskKind(kind_raw)
        except ValueError:
            return Failure(
                ActionFailure(
                    record_id=f"failure:{request.record_id}:INVALID_TASK_KIND",
                    producer=PRODUCER,
                    correlation_id=request.correlation_id,
                    request_id=request.record_id,
                    capability=self.name,
                    code="INVALID_TASK_KIND",
                    message=f"Unsupported bounded task kind '{kind_raw}'",
                )
            )

        content: dict[str, Any] = args.get("content") or {}
        task_id = args.get("task_id") or request.record_id

        data_class_val = args.get("data_class")
        data_class = max_data_class(request.data_class, data_class_val)

        task = Task(
            record_id=f"task:{task_id}",
            producer=request.actor,
            correlation_id=request.correlation_id,
            objective=objective,
            status="RUNNING",
        )

        manifest_items = (
            tuple(
                ContextManifestItem(
                    source_id=key,
                    source_type="input_field",
                    data_class=data_class,
                    fields=(key,),
                )
                for key in content
            )
            if content
            else (
                ContextManifestItem(
                    source_id="direct_input",
                    source_type="direct_input",
                    data_class=data_class,
                ),
            )
        )

        manifest = ContextManifest(
            record_id=f"manifest:{_new_id()}",
            producer=request.actor,
            correlation_id=request.correlation_id,
            destination="local_processor",
            items=manifest_items,
            approved=True,
        )

        context = ContextPackage(
            record_id=f"ctx:{_new_id()}",
            producer=request.actor,
            correlation_id=request.correlation_id,
            task_id=task.record_id,
            manifest=manifest,
            content=content,
        )

        now = _utc_now()
        if request.budget is not None:
            max_in = min(
                request.budget.max_input_tokens,
                int(args.get("max_input_tokens", request.budget.max_input_tokens)),
            )
            max_out = min(
                request.budget.max_output_tokens,
                int(args.get("max_output_tokens", request.budget.max_output_tokens)),
            )
            max_lat = min(
                request.budget.max_latency_seconds,
                float(args.get("max_latency_seconds", request.budget.max_latency_seconds)),
            )
            arg_deadline = now + timedelta(seconds=max_lat)
            cancellation_deadline = (
                min(request.budget.cancellation_deadline, arg_deadline)
                if request.budget.cancellation_deadline.tzinfo is not None
                else request.budget.cancellation_deadline
            )
            budget = request.budget.model_copy(
                update={
                    "max_input_tokens": max_in,
                    "max_output_tokens": max_out,
                    "max_latency_seconds": max_lat,
                    "cancellation_deadline": cancellation_deadline,
                }
            )
        else:
            max_in = int(args.get("max_input_tokens", 2048))
            max_out = int(args.get("max_output_tokens", 512))
            max_lat = float(args.get("max_latency_seconds", 30.0))
            budget = InferenceBudget(
                record_id=f"budget:{_new_id()}",
                producer=request.actor,
                correlation_id=request.correlation_id,
                max_input_tokens=max_in,
                max_output_tokens=max_out,
                max_agent_turns=1,
                max_tool_calls=0,
                max_images=0,
                max_audio_seconds=0.0,
                max_latency_seconds=max_lat,
                max_provider_cost=0.0,
                max_ram_bytes=0,
                max_vram_bytes=0,
                cancellation_deadline=now + timedelta(seconds=max_lat),
            )

        if self.supervisor.state != LifecycleState.RUNNING:
            start_result = await self.supervisor.start()
            if isinstance(start_result, Failure):
                start_failure = start_result.failure()
                return Failure(
                    ActionFailure(
                        record_id=f"failure:{request.record_id}:{start_failure.code}",
                        producer=PRODUCER,
                        correlation_id=request.correlation_id,
                        request_id=request.record_id,
                        capability=self.name,
                        code=start_failure.code,
                        message=start_failure.message,
                        retryable=start_failure.retryable,
                    )
                )

        result = await self.supervisor.process_bounded(
            task_kind, task, context, budget, cancellation
        )
        if isinstance(result, Failure):
            failure = result.failure()
            return Failure(
                ActionFailure(
                    record_id=f"failure:{request.record_id}:{failure.code}",
                    producer=PRODUCER,
                    correlation_id=request.correlation_id,
                    request_id=request.record_id,
                    capability=self.name,
                    code=failure.code,
                    message=failure.message,
                    retryable=failure.retryable,
                )
            )

        claim = result.unwrap()
        return Success(
            ActionResult(
                record_id=f"result:{request.record_id}",
                producer=PRODUCER,
                correlation_id=request.correlation_id,
                request_id=request.record_id,
                capability=self.name,
                output={
                    "statement": claim.statement,
                    "confidence": claim.confidence,
                    "epistemic_status": claim.epistemic_status,
                    "data_class": (
                        claim.data_class.value
                        if hasattr(claim.data_class, "value")
                        else str(claim.data_class)
                    ),
                },
                verification={
                    "claim_id": claim.record_id,
                    "provenance_count": len(claim.provenance),
                },
            )
        )


def register_bounded_inference_capabilities(
    registry: CapabilityRegistry,
    supervisor: ProcessorSupervisor | Callable[[], ProcessorSupervisor],
) -> None:
    """Register bounded inference task capabilities into the CapabilityRegistry."""
    capability = BoundedTaskCapability(supervisor)
    registry.register(
        RegisteredCapability(
            descriptor=bounded_task_descriptor(),
            implementation=capability,
        )
    )
