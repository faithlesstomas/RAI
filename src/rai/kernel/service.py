"""Single policy-controlled capability invocation service."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol

from returns.result import Failure, Result, Success

from rai.assistant.ports import ObservationStore
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
from .references import (
    make_provenance_references,
    preflight_scan_references,
    resolve_arguments,
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
        observation_store: ObservationStore | Callable[[], ObservationStore | None] | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.approvals = approvals
        self._observation_store = observation_store

    @property
    def observation_store(self) -> ObservationStore | None:
        if callable(self._observation_store):
            return self._observation_store()
        return self._observation_store

    @observation_store.setter
    def observation_store(
        self, store: ObservationStore | Callable[[], ObservationStore | None] | None
    ) -> None:
        self._observation_store = store

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

        session_id = (
            getattr(request, "session_id", None)
            or request.arguments.get("session_id")
            or ""
        )
        scan = None
        store = self.observation_store
        if store is not None:
            scan_res = await preflight_scan_references(
                request.arguments, store, session_id, request.data_class
            )
            if isinstance(scan_res, Failure):
                err_code, err_msg = scan_res.failure()
                return None, Failure(self._failure(request, err_code, err_msg))
            scan = scan_res.unwrap()
            if scan.has_references:
                request = request.model_copy(update={"data_class": scan.effective_data_class})

        decision = self.policy.evaluate(request, capability.descriptor)
        audit_result = await self.audit.append(
            AuditEntry(
                stage="DECISION",
                decision=decision,
                referenced_artifact_ids=scan.referenced_ids if scan else (),
            )
        )
        if isinstance(audit_result, Failure):
            return decision, Failure(audit_result.failure())

        if scan and scan.has_references:
            resolve_res = resolve_arguments(request.arguments, scan.referenced_artifacts)
            if isinstance(resolve_res, Failure):
                err_code, err_msg = resolve_res.failure()
                return decision, Failure(self._failure(request, err_code, err_msg))
            request = request.model_copy(update={"arguments": resolve_res.unwrap()})
            invalid = validate_arguments(capability.descriptor.input_schema, request.arguments)
            if invalid is not None:
                return decision, Failure(self._failure(request, "INVALID_ARGUMENT", invalid))

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

        if scan and scan.has_references and isinstance(result, Success):
            act_res = result.unwrap()
            prov_refs = make_provenance_references(list(scan.referenced_artifacts.values()))
            combined_prov = tuple(list(act_res.provenance) + list(prov_refs))
            result = Success(act_res.model_copy(update={"provenance": combined_prov}))

        terminal = result.unwrap() if isinstance(result, Success) else result.failure()
        terminal_audit = await self.audit.append(
            AuditEntry(
                stage="TERMINAL",
                decision=decision,
                approval_id=approval_id,
                result=terminal,
                referenced_artifact_ids=scan.referenced_ids if scan else (),
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
