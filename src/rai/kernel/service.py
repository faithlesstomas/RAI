"""Single policy-controlled capability invocation service."""

from __future__ import annotations

from typing import Protocol

from returns.result import Failure, Result, Success

from .audit import AuditEntry
from .capabilities import CapabilityRegistry
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
            if self.approvals is None:
                result: Result[ActionResult, ActionFailure] = Failure(
                    self._failure(request, "APPROVAL_UNAVAILABLE", "approval is required")
                )
            else:
                approval = await self.approvals.request(decision, token)
                if isinstance(approval, Failure):
                    result = Failure(approval.failure())
                else:
                    approval_id = approval.unwrap()
                    result = await capability.invoke(request, token)
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
