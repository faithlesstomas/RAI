"""Deterministic capability policy for the Stage 1 kernel."""

from __future__ import annotations

from collections.abc import Callable

from .capabilities import CapabilityDescriptor
from .records import (
    CapabilityRequest,
    DataClass,
    PolicyDecision,
    PolicyOutcome,
    ProducerIdentity,
    RiskClass,
)

POLICY_VERSION = "1.0.0"
POLICY_PRODUCER = ProducerIdentity(
    producer_id="rai.policy", kind="policy-engine", version=POLICY_VERSION
)


class PolicyEngine:
    """Pure policy evaluation; models and backend names have no authority here."""

    def __init__(
        self,
        isolation_available: Callable[[str], bool] | None = None,
        policy_version: str = POLICY_VERSION,
    ) -> None:
        self._isolation_available = isolation_available or (lambda isolation: bool(isolation))
        self.policy_version = policy_version

    def evaluate(
        self, request: CapabilityRequest, descriptor: CapabilityDescriptor
    ) -> PolicyDecision:
        outcome, reasons = self._outcome(request, descriptor)
        return PolicyDecision(
            record_id=f"policy:{self.policy_version}:{request.record_id}",
            timestamp=request.timestamp,
            producer=POLICY_PRODUCER,
            correlation_id=request.correlation_id,
            request_id=request.record_id,
            outcome=outcome,
            risk_class=descriptor.risk_class,
            policy_version=self.policy_version,
            reason_codes=reasons,
            actor=request.actor,
            data_class=request.data_class,
            target_resource=request.target_resource,
            requested_side_effects=request.requested_side_effects,
            isolation=request.isolation,
            budget_id=request.budget.record_id if request.budget else None,
            verification_plan=request.verification_plan,
        )

    def _outcome(  # noqa: PLR0911
        self, request: CapabilityRequest, descriptor: CapabilityDescriptor
    ) -> tuple[PolicyOutcome, tuple[str, ...]]:
        if request.capability != descriptor.name:
            return PolicyOutcome.DENY, ("CAPABILITY_NAME_MISMATCH",)
        if request.data_class in {DataClass.SECRET, DataClass.BLOCKED}:
            return PolicyOutcome.DENY, ("DATA_CLASS_FORBIDDEN",)
        if set(request.requested_side_effects) - set(descriptor.side_effects):
            return PolicyOutcome.DENY, ("UNDECLARED_SIDE_EFFECT",)
        if request.verification_plan != descriptor.verification_plan:
            return PolicyOutcome.DENY, ("VERIFICATION_PLAN_MISMATCH",)
        if descriptor.requires_isolation and (
            request.isolation != descriptor.isolation
            or not self._isolation_available(descriptor.isolation)
        ):
            return PolicyOutcome.DENY, ("ISOLATION_UNAVAILABLE",)
        if descriptor.risk_class == RiskClass.CRITICAL:
            return PolicyOutcome.DENY, ("CRITICAL_RISK",)
        if request.target_resource.startswith(("https://", "http://")) and (
            request.data_class == DataClass.PRIVATE
        ):
            return PolicyOutcome.ESCALATE, ("PRIVATE_DATA_EGRESS",)
        if descriptor.risk_class in {RiskClass.MODERATE, RiskClass.HIGH}:
            return PolicyOutcome.ASK, (f"{descriptor.risk_class}_RISK_APPROVAL",)
        return PolicyOutcome.ALLOW, ("LOW_RISK_LOCAL",)
