"""System and desktop activity history query capability."""
from __future__ import annotations

import asyncio
from datetime import timedelta

from rai.diagnostics import trace
from collections.abc import Callable
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, DataClass, Episode, RiskClass, _utc_now
from .capabilities import failure, result

_MAX_EPISODES = 30
_MAX_LOOKBACK_MINUTES = 10080
_DATA_CLASS_ALLOWED = {
    DataClass.PUBLIC: {DataClass.PUBLIC},
    DataClass.LOCAL: {DataClass.PUBLIC, DataClass.LOCAL},
    DataClass.PRIVATE: {DataClass.PUBLIC, DataClass.LOCAL, DataClass.PRIVATE},
}


class ActivityQuery:
    """Read-only bounded query of encrypted local activity episodes."""

    name = "activity.query"

    def __init__(self, service_provider: Callable[[], Any]) -> None:
        self.service_provider = service_provider

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]:
        service = self.service_provider()
        if service is None:
            return Failure(failure(request, "HISTORY_SERVICE_UNAVAILABLE"))

        query_str = str(request.arguments.get("query", "")).strip()
        limit_arg = request.arguments.get("limit", 15)
        try:
            limit = min(_MAX_EPISODES, max(1, int(limit_arg)))
        except (ValueError, TypeError):
            limit = 15

        try:
            filters = {}
            if request.arguments.get("lookback_minutes") is not None:
                minutes = request.arguments["lookback_minutes"]
                if not isinstance(minutes, int) or isinstance(minutes, bool) or not 1 <= minutes <= _MAX_LOOKBACK_MINUTES:
                    return Failure(failure(request, "INVALID_ARGUMENT"))
                filters = {"since": _utc_now() - timedelta(minutes=minutes), "until": _utc_now()}
            raw_episodes = await asyncio.to_thread(service.query, **filters)
        except Exception:  # noqa: BLE001
            return Failure(failure(request, "HISTORY_QUERY_FAILED"))

        if cancellation.cancelled:
            return Failure(failure(request, "CANCELLED"))

        # Caller data class containment: fail-closed against SECRET / BLOCKED
        req_dc = (
            request.data_class
            if isinstance(request.data_class, DataClass)
            else DataClass(request.data_class)
        )
        allowed_classes = _DATA_CLASS_ALLOWED.get(req_dc, {DataClass.PUBLIC})

        matched_episodes: list[dict[str, Any]] = []
        excluded = 0
        visible = 0
        for ep in raw_episodes:
            if not isinstance(ep, Episode):
                continue
            ep_class = (
                ep.data_class
                if isinstance(ep.data_class, DataClass)
                else DataClass(ep.data_class)
            )
            if ep_class not in allowed_classes:
                excluded += 1
                continue
            visible += 1

            if query_str:
                searchable = " ".join(
                    (
                        *ep.applications,
                        *ep.projects,
                        *ep.resources,
                        *ep.activity_types,
                        *ep.outcome_signals,
                    )
                ).casefold()
                if query_str.casefold() not in searchable:
                    continue

            matched_episodes.append(
                {
                    "episode_id": ep.record_id,
                    "started_at": ep.started_at.isoformat(),
                    "ended_at": ep.ended_at.isoformat(),
                    "applications": list(ep.applications),
                    "projects": list(ep.projects),
                    "resources": list(ep.resources),
                    "activity_types": list(ep.activity_types),
                    "confidence": ep.confidence,
                }
            )

        matched_episodes.sort(key=lambda item: item["ended_at"], reverse=True)
        results = matched_episodes[:limit]

        empty_reason = None
        if not results:
            empty_reason = "classification_filtered" if excluded and not visible else (
                "no_match" if visible else "no_episodes"
            )
        trace("activity.query", request_id=request.record_id, scanned=len(raw_episodes),
              excluded=excluded, returned=len(results), empty_reason=empty_reason,
              data_class=req_dc.value, query_present=bool(query_str),
              lookback_minutes=request.arguments.get("lookback_minutes"))
        return Success(
            result(
                request,
                {"episodes": results, "count": len(results), "empty_reason": empty_reason},
                {"episodes_found": len(results), "data_class": req_dc.value},
            )
        )


def register_activity_capabilities(
    registry: CapabilityRegistry, service_provider: Callable[[], Any]
) -> None:
    """Register activity query capability with the capability registry."""
    registry.register(
        RegisteredCapability(
            CapabilityDescriptor(
                name="activity.query",
                description=(
                    "Inspect recent system and desktop activity episodes "
                    "(applications used, projects, activity types, and resources)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "query": {"type": "string"},
                        "lookback_minutes": {"type": "integer", "minimum": 1, "maximum": _MAX_LOOKBACK_MINUTES},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
                risk_class=RiskClass.LOW,
                isolation="in-process",
                verification_plan=("encrypted-history-bounded-query",),
            ),
            implementation=ActivityQuery(service_provider),
        )
    )
