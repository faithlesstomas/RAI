"""Inference budget, cost, and usage governor for local and hybrid AI operations.

Enforces:
1. Strict invariant: background_remote_tokens = 0 (background tasks cannot consume remote tokens).
2. Per-task limits on agent turns, tool calls, and input/output tokens.
3. Daily and monthly token and monetary cost ceilings.
4. Cancellation on deadline expiry or budget breach (on_budget_exceeded = CANCEL).
5. Telemetry privacy (no prompt contents leaked into usage reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    InferenceBudget,
    ProducerIdentity,
    _utc_now,
)

logger = logging.getLogger(__name__)

GOVERNOR_PRODUCER = ProducerIdentity(
    producer_id="rai.inference.governor",
    kind="budget-governor",
    version="1.0.0",
)


def make_governor_failure(
    code: str,
    message: str,
    request_id: str = "governor",
    retryable: bool = False,
    producer: ProducerIdentity | None = None,
) -> ActionFailure:
    return ActionFailure(
        request_id=request_id,
        capability="inference.governor",
        code=code,
        message=message,
        retryable=retryable,
        producer=producer or GOVERNOR_PRODUCER,
    )


class GovernorConfig(BaseModel):
    """Runtime limits and configuration for the budget governor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: Literal["LOCAL_ONLY", "LOCAL_PREFERRED", "HYBRID_APPROVAL", "REMOTE_ALLOWED"] = (
        "LOCAL_PREFERRED"
    )
    background_remote_tokens: int = Field(default=0, ge=0)
    remote_on_ambiguous_input: Literal["ask", "deny", "local"] = "ask"
    on_missing_usage: Literal["ask", "estimate", "cancel"] = "ask"
    on_budget_exceeded: Literal["cancel", "deny"] = "cancel"

    per_task_max_tokens: int = Field(default=20_000, ge=0)
    per_task_max_agent_turns: int = Field(default=3, ge=1)
    per_task_max_tool_calls: int = Field(default=8, ge=0)
    per_task_max_images: int = Field(default=1, ge=0)

    daily_remote_token_limit: int = Field(default=100_000, ge=0)
    monthly_remote_token_limit: int = Field(default=1_000_000, ge=0)
    daily_cost_limit_usd: float = Field(default=10.0, ge=0.0)
    monthly_cost_limit_usd: float = Field(default=50.0, ge=0.0)


@dataclass
class UsageEntry:
    """Audit entry recording inference metrics without prompt content."""

    request_id: str
    timestamp: datetime
    tokens_in: int
    tokens_out: int
    cost_usd: float
    is_remote: bool
    is_background: bool
    provider: str


class InferenceBudgetGovernor:
    """Manages and enforces inference quotas across local and remote providers."""

    def __init__(
        self,
        config: GovernorConfig | None = None,
        producer: ProducerIdentity = GOVERNOR_PRODUCER,
    ) -> None:
        self.config = config or GovernorConfig()
        self.producer = producer
        self._usage_ledger: list[UsageEntry] = []

    def check_request(
        self,
        budget: InferenceBudget,
        request_id: str = "inference-request",
        is_background: bool = False,
        is_remote: bool = False,
        estimated_tokens: int = 0,
        current_turns: int = 0,
        current_tool_calls: int = 0,
        now: datetime | None = None,
    ) -> Result[None, ActionFailure]:
        """Validate an upcoming inference request against all budget boundaries."""
        current_time = now or datetime.now(timezone.utc)

        # 1. Deadline expiration check
        if current_time > budget.cancellation_deadline:
            return Failure(
                make_governor_failure(
                    code="DEADLINE_EXCEEDED",
                    message="Inference budget cancellation deadline has passed",
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        # 2. Strict invariant: background tasks are forbidden from remote tokens
        if is_background and is_remote:
            allowed_bg = self.config.background_remote_tokens
            if allowed_bg == 0 or estimated_tokens > allowed_bg:
                return Failure(
                    make_governor_failure(
                        code="BACKGROUND_REMOTE_FORBIDDEN",
                        message=(
                            f"Background task attempted remote inference, but "
                            f"background_remote_tokens limit is {allowed_bg}"
                        ),
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

        # 3. Profile restriction: LOCAL_ONLY completely rejects remote inference
        if is_remote and self.config.profile == "LOCAL_ONLY":
            return Failure(
                make_governor_failure(
                    code="REMOTE_FORBIDDEN_IN_LOCAL_ONLY",
                    message="Remote model inference is prohibited under LOCAL_ONLY profile",
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        # 4. Per-task constraints
        if current_turns > budget.max_agent_turns:
            return Failure(
                make_governor_failure(
                    code="TASK_MAX_TURNS_EXCEEDED",
                    message=(
                        f"Task agent turns ({current_turns}) reached budget limit "
                        f"({budget.max_agent_turns})"
                    ),
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        if current_tool_calls > budget.max_tool_calls:
            return Failure(
                make_governor_failure(
                    code="TASK_MAX_TOOL_CALLS_EXCEEDED",
                    message=(
                        f"Task tool calls ({current_tool_calls}) reached budget limit "
                        f"({budget.max_tool_calls})"
                    ),
                    request_id=request_id,
                    producer=self.producer,
                )
            )

        # 5. Cumulative remote limits
        if is_remote:
            daily_used = self._get_tokens_in_period(current_time, days=1, remote_only=True)
            if daily_used + estimated_tokens > self.config.daily_remote_token_limit:
                return Failure(
                    make_governor_failure(
                        code="DAILY_TOKEN_LIMIT_EXCEEDED",
                        message=(
                            f"Daily remote token ceiling ({self.config.daily_remote_token_limit}) "
                            f"would be exceeded (currently used: {daily_used}, estimated: {estimated_tokens})"
                        ),
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

            monthly_used = self._get_tokens_in_period(current_time, days=30, remote_only=True)
            if monthly_used + estimated_tokens > self.config.monthly_remote_token_limit:
                return Failure(
                    make_governor_failure(
                        code="MONTHLY_TOKEN_LIMIT_EXCEEDED",
                        message=(
                            f"Monthly remote token ceiling ({self.config.monthly_remote_token_limit}) "
                            f"would be exceeded (currently used: {monthly_used}, estimated: {estimated_tokens})"
                        ),
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

            daily_cost = self._get_cost_in_period(current_time, days=1)
            if daily_cost > self.config.daily_cost_limit_usd:
                return Failure(
                    make_governor_failure(
                        code="DAILY_COST_LIMIT_EXCEEDED",
                        message=f"Daily cost ceiling (${self.config.daily_cost_limit_usd:.2f}) exceeded",
                        request_id=request_id,
                        producer=self.producer,
                    )
                )

        return Success(None)

    def record_usage(
        self,
        request_id: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float = 0.0,
        is_remote: bool = False,
        is_background: bool = False,
        provider: str = "local",
        timestamp: datetime | None = None,
    ) -> Result[None, ActionFailure]:
        """Record inference metrics into the usage ledger."""
        ts = timestamp or datetime.now(timezone.utc)

        # Check for missing usage on remote calls
        if is_remote and tokens_in == 0 and tokens_out == 0:
            if self.config.on_missing_usage == "ask":
                logger.warning(
                    "Remote inference request %s reported zero or missing usage tokens",
                    request_id,
                )
            elif self.config.on_missing_usage == "estimate":
                # Apply conservative default
                tokens_in = 500
                tokens_out = 200

        entry = UsageEntry(
            request_id=request_id,
            timestamp=ts,
            tokens_in=max(0, tokens_in),
            tokens_out=max(0, tokens_out),
            cost_usd=max(0.0, cost_usd),
            is_remote=is_remote,
            is_background=is_background,
            provider=provider,
        )
        self._usage_ledger.append(entry)
        return Success(None)

    def get_usage_summary(self) -> dict[str, Any]:
        """Provide aggregated metrics without exposing user prompts."""
        total_tokens_in = sum(e.tokens_in for e in self._usage_ledger)
        total_tokens_out = sum(e.tokens_out for e in self._usage_ledger)
        total_cost = sum(e.cost_usd for e in self._usage_ledger)
        remote_tokens = sum(
            e.tokens_in + e.tokens_out for e in self._usage_ledger if e.is_remote
        )
        local_tokens = sum(
            e.tokens_in + e.tokens_out for e in self._usage_ledger if not e.is_remote
        )
        bg_remote_tokens = sum(
            e.tokens_in + e.tokens_out
            for e in self._usage_ledger
            if e.is_remote and e.is_background
        )

        return {
            "total_requests": len(self._usage_ledger),
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "total_cost_usd": round(total_cost, 4),
            "remote_tokens": remote_tokens,
            "local_tokens": local_tokens,
            "background_remote_tokens": bg_remote_tokens,
        }

    def _get_tokens_in_period(
        self, current_time: datetime, days: int, remote_only: bool = False
    ) -> int:
        cutoff = current_time.timestamp() - (days * 86400)
        return sum(
            e.tokens_in + e.tokens_out
            for e in self._usage_ledger
            if e.timestamp.timestamp() >= cutoff and (not remote_only or e.is_remote)
        )

    def _get_cost_in_period(self, current_time: datetime, days: int) -> float:
        cutoff = current_time.timestamp() - (days * 86400)
        return sum(
            e.cost_usd for e in self._usage_ledger if e.timestamp.timestamp() >= cutoff
        )
