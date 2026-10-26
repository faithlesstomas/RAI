"""Single policy-controlled capability invocation service."""

from __future__ import annotations

import inspect
from typing import Any, Protocol

from returns.result import Failure, Result, Success

from .audit import AuditEntry
from .capabilities import CapabilityRegistry, validate_arguments
from .policy import PolicyEngine
from .ports import ApprovalBroker, CancellationToken
from .records import (
    ActionFailure,
    ActionResult,
    CapabilityRequest,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
)

SERVICE_PRODUCER = ProducerIdentity(
    producer_id="rai.capability-service", kind="runtime", version="1.0.0"
)


class AuditLedger(Protocol):
    async def append(self, entry: AuditEntry) -> Result[AuditEntry, ActionFailure]: ...


class CapabilityService:
    """Resolve, authorize, execute and audit every capability request."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        policy: PolicyEngine,
        audit: AuditLedger,
        approvals: ApprovalBroker | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.approvals = approvals

    async def _invoke_with_approval(
        self,
        capability: Any,  # noqa: ANN401
        decision: PolicyDecision,
        request: CapabilityRequest,
        token: CancellationToken,
    ) -> tuple[str | None, Result[ActionResult, ActionFailure]]:
        invalid = validate_arguments(capability.descriptor.input_schema, request.arguments)
        if invalid is not None:
            return None, Failure(self._failure(request, "INVALID_ARGUMENT", invalid))
        if self.approvals is None:
            return None, Failure(
                self._failure(request, "APPROVAL_UNAVAILABLE", "approval is required")
            )
        review_action: Any = getattr(self.approvals, "request_action", None)
        if callable(review_action) and inspect.iscoroutinefunction(review_action):
            approval = await review_action(decision, request, token)  # pylint: disable=not-callable
        else:
            approval = await self.approvals.request(decision, token)
        if isinstance(approval, Failure):
            return None, Failure(approval.failure())
        approval_id = approval.unwrap()
        return approval_id, await capability.invoke(request, token)

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken | None = None
    ) -> tuple[PolicyDecision | None, Result[ActionResult, ActionFailure]]:
        token = cancellation or CancellationToken()
        resolved = self.registry.resolve(request)
        if isinstance(resolved, Failure):
            return None, Failure(resolved.failure())
        capability = resolved.unwrap()
        decision = self.policy.evaluate(request, capability.descriptor)
        audit_result = await self.audit.append(AuditEntry(stage="DECISION", decision=decision))
        if isinstance(audit_result, Failure):
            return decision, Failure(audit_result.failure())

        approval_id: str | None = None
        if decision.outcome == PolicyOutcome.ASK:
            approval_id, result = await self._invoke_with_approval(
                capability, decision, request, token
            )
        elif decision.outcome == PolicyOutcome.ALLOW:
            result = await capability.invoke(request, token)
        else:
            code = (
                "ESCALATION_REQUIRED"
                if decision.outcome == PolicyOutcome.ESCALATE
                else "POLICY_DENIED"
            )
            result = Failure(self._failure(request, code, decision.reason_codes[0]))

        terminal = result.unwrap() if isinstance(result, Success) else result.failure()
        terminal_audit = await self.audit.append(
            AuditEntry(
                stage="TERMINAL",
                decision=decision,
                approval_id=approval_id,
                result=terminal,
            )
        )
        if isinstance(terminal_audit, Failure):
            return decision, Failure(terminal_audit.failure())
        return decision, result

    @staticmethod
    def _failure(request: CapabilityRequest, code: str, message: str) -> ActionFailure:
        return ActionFailure(
            record_id=f"failure:{request.record_id}:{code}",
            timestamp=request.timestamp,
            producer=SERVICE_PRODUCER,
            correlation_id=request.correlation_id,
            request_id=request.record_id,
            capability=request.capability,
            code=code,
            message=message,
        )
