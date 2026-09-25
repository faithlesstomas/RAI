"""Multi-tier Hybrid Routing Engine for Local and Remote Assistant Operations.

Orchestrates model selection and execution paths across:
1. Deterministic rules & intent hints.
2. Privacy Firewall (LOCAL, SECRET, BLOCKED remain strictly local).
3. Budget Governor boundaries (background_remote_tokens = 0).
4. Finite DecisionBackend evaluation (Lemonade SLM or Jev).
5. Profile acceptance policy (LOCAL_ONLY, LOCAL_PREFERRED, HYBRID_APPROVAL, REMOTE_ALLOWED).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.inference.decision import (
    DecisionBackend,
    DecisionOption,
    DecisionRequest,
    DecisionResult,
    DeterministicDecisionBackend,
)
from rai.inference.governor import GovernorConfig, InferenceBudgetGovernor
from rai.kernel.egress import EgressFirewall
from rai.kernel.ports import CancellationToken
from rai.kernel.records import (
    ActionFailure,
    ContextPackage,
    DataClass,
    InferenceBudget,
    ProducerIdentity,
)

logger = logging.getLogger(__name__)

ROUTER_PRODUCER = ProducerIdentity(
    producer_id="rai.inference.hybrid_router",
    kind="router",
    version="1.0.0",
)

RoutingOutcome = Literal["LOCAL", "REMOTE", "ASK", "DENY"]


class RoutingDecision(BaseModel):
    """Immutable routing decision output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: RoutingOutcome
    target_backend: str
    target_model: str | None = None
    reason: str
    decision_result: DecisionResult | None = None


class HybridRouter:
    """Intelligent multi-tier router for dispatching inferences safely."""

    def __init__(
        self,
        governor: InferenceBudgetGovernor | None = None,
        egress_firewall: EgressFirewall | None = None,
        decision_backend: DecisionBackend | None = None,
        profile: str = "LOCAL_PREFERRED",
        default_local_model: str = "qwen-local",
        default_remote_model: str = "gemini-2.5-pro",
    ) -> None:
        self.governor = governor or InferenceBudgetGovernor(GovernorConfig(profile=profile))
        self.egress_firewall = egress_firewall or EgressFirewall(profile=profile)
        self.decision_backend = decision_backend or DeterministicDecisionBackend()
        self.profile = profile
        self.default_local_model = default_local_model
        self.default_remote_model = default_remote_model

    async def route_inference(
        self,
        context: ContextPackage,
        budget: InferenceBudget,
        user_prompt: str,
        is_background: bool = False,
        cancellation: CancellationToken | None = None,
    ) -> Result[RoutingDecision, ActionFailure]:
        """Determine whether the inference should be served locally, remotely, or denied."""
        cancel = cancellation or CancellationToken()

        # Tier 1: Deterministic rules (e.g. prompt commands, local keywords)
        prompt_strip = user_prompt.strip().lower()
        if prompt_strip.startswith(("/local", "/offline")):
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason="Deterministic keyword match (/local or /offline)",
                )
            )

        # Tier 2: Privacy boundaries - check manifest items for LOCAL, SECRET, BLOCKED
        for item in context.manifest.items:
            dc = item.data_class
            if isinstance(dc, str):
                dc = DataClass(dc)
            if dc in {DataClass.SECRET, DataClass.BLOCKED}:
                return Success(
                    RoutingDecision(
                        outcome="LOCAL",
                        target_backend="local",
                        target_model=self.default_local_model,
                        reason=f"Data class '{dc.value}' in manifest requires strict local containment",
                    )
                )
            if dc == DataClass.LOCAL:
                return Success(
                    RoutingDecision(
                        outcome="LOCAL",
                        target_backend="local",
                        target_model=self.default_local_model,
                        reason="Data class 'LOCAL' cannot leave machine trust boundary",
                    )
                )

        # Tier 3: Profile and Background Governor constraints
        if self.profile == "LOCAL_ONLY":
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason="LOCAL_ONLY profile active",
                )
            )

        if is_background and self.governor.config.background_remote_tokens == 0:
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason="Background task forbidden from remote tokens",
                )
            )

        # Tier 4: Evaluate with DecisionBackend
        options = (
            DecisionOption(
                option_id="opt_local",
                description="Task can be competently handled by fast local SLM",
            ),
            DecisionOption(
                option_id="opt_remote",
                description="Task requires advanced reasoning or external knowledge (cloud LLM)",
            ),
        )
        dec_req = DecisionRequest(
            producer=ROUTER_PRODUCER,
            request_id=f"dec-{context.task_id}",
            task_kind="routing_hint",
            context=context,
            options=options,
            budget=budget,
        )

        dec_res = await self.decision_backend.decide(dec_req, cancel)
        if isinstance(dec_res, Failure):
            logger.warning("DecisionBackend evaluation failed: %s; falling back to local", dec_res.failure().message)
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason=f"DecisionBackend failure fallback: {dec_res.failure().message}",
                )
            )

        decision = dec_res.unwrap()
        if decision.status == "ABSTAINED" or decision.selected_option is None:
            if self.governor.config.remote_on_ambiguous_input == "ask":
                return Success(
                    RoutingDecision(
                        outcome="ASK",
                        target_backend="local",
                        reason="DecisionBackend abstained and ambiguous routing policy is ASK",
                        decision_result=decision,
                    )
                )
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason="DecisionBackend abstained; defaulting to local",
                    decision_result=decision,
                )
            )

        if decision.selected_option == "opt_local":
            return Success(
                RoutingDecision(
                    outcome="LOCAL",
                    target_backend="local",
                    target_model=self.default_local_model,
                    reason="DecisionBackend classified task as local",
                    decision_result=decision,
                )
            )

        # Escalation to remote
        if self.profile == "HYBRID_APPROVAL":
            return Success(
                RoutingDecision(
                    outcome="ASK",
                    target_backend="antigravity",
                    target_model=self.default_remote_model,
                    reason="HYBRID_APPROVAL requires user consent before remote execution",
                    decision_result=decision,
                )
            )

        return Success(
            RoutingDecision(
                outcome="REMOTE",
                target_backend="antigravity",
                target_model=self.default_remote_model,
                reason="DecisionBackend escalated task to remote model",
                decision_result=decision,
            )
        )
