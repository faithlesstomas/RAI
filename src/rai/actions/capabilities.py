"""Application capabilities with scoped resources and explicit OS evidence."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, ProducerIdentity, RiskClass, _utc_now

from .applications import Application, ApplicationBackend
from .handles import SQLiteHandleStore
from .records import ResourceHandle

PRODUCER = ProducerIdentity(producer_id="rai.actions", kind="capability", version="1.0.0")
HANDLE_TTL = timedelta(minutes=5)
MAX_RESULTS = 30
MAX_APPLICATION_RESULTS = 1024
MAX_QUERY_LENGTH = 256


def failure(request: CapabilityRequest, code: str) -> ActionFailure:
    """Do not leak backend exceptions or resource contents into failure messages."""
    return ActionFailure(
        record_id=f"failure:{request.record_id}:{code}", producer=PRODUCER,
        correlation_id=request.correlation_id, request_id=request.record_id,
        capability=request.capability, code=code, message=code,
    )


def result(request: CapabilityRequest, output: dict[str, Any], evidence: dict[str, Any]) -> ActionResult:
    """Return only postconditions actually established by the adapter."""
    return ActionResult(
        record_id=f"result:{request.record_id}", producer=PRODUCER,
        correlation_id=request.correlation_id, request_id=request.record_id,
        capability=request.capability, output=output, verification=evidence,
    )


class ApplicationList:
    """List installed entries and issue bounded authority for this caller/task."""

    name = "application.list"

    def __init__(self, backend: ApplicationBackend, handles: SQLiteHandleStore) -> None:
        self.backend = backend
        self.handles = handles

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]:
        query = request.arguments.get("query", "")
        task_id = request.arguments.get("task_id", "")
        if not task_id or len(query) > MAX_QUERY_LENGTH:
            return Failure(failure(request, "INVALID_ARGUMENT"))
        discovered = await self.backend.discover()
        if isinstance(discovered, Failure):
            return Failure(failure(request, discovered.failure()))
        def _match_app(app: Application, q: str) -> bool:
            if not q:
                return True
            q_cf = q.casefold().strip()
            targets = [
                app.name,
                app.desktop_id,
                Path(app.executable).name,
                *app.localized_names,
                getattr(app, "generic_name", ""),
                getattr(app, "comment", ""),
                *getattr(app, "categories", ()),
                *getattr(app, "mime_types", ()),
            ]
            if any(q_cf in t.casefold() for t in targets):
                return True
            terms_map = {
                "wideo": ["video", "player", "movie", "film"],
                "video": ["wideo", "player", "movie", "film"],
                "muzyka": ["audio", "music", "sound", "player"],
                "audio": ["muzyka", "audio", "sound", "player"],
                "odtwarzacz": ["player", "media", "video", "audio"],
                "player": ["odtwarzacz", "media"],
                "przeglądarka": ["browser", "web", "navigator"],
                "browser": ["przeglądarka", "web"],
            }
            extra_terms = terms_map.get(q_cf, [])
            return any(
                any(term in t.casefold() for term in extra_terms)
                for t in targets
            )

        matches = [app for app in discovered.unwrap() if _match_app(app, query)]
        output = []
        for app in matches[:MAX_APPLICATION_RESULTS]:
            if cancellation.cancelled:
                return Failure(failure(request, "CANCELLED"))
            now = _utc_now()
            handle = ResourceHandle(
                producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id,
                task_id=task_id, kind="application", target=app.desktop_id,
                fingerprint=app.fingerprint, operations=("application.launch", "document.open"),
                expires_at=now + HANDLE_TTL, data_class=request.data_class,
            )
            issued = await asyncio.to_thread(self.handles.issue, handle)
            if isinstance(issued, Failure):
                return Failure(failure(request, issued.failure()))
            app_entry: dict[str, Any] = {
                "desktop_id": app.desktop_id,
                "name": app.name,
                "handle": issued.unwrap(),
                "expires_at": handle.expires_at.isoformat(),
            }
            if getattr(app, "generic_name", ""):
                app_entry["generic_name"] = app.generic_name
            if getattr(app, "categories", ()):
                app_entry["categories"] = list(app.categories)
            if getattr(app, "comment", ""):
                app_entry["comment"] = app.comment
            output.append(app_entry)
        is_truncated = len(matches) > MAX_APPLICATION_RESULTS
        result_payload: dict[str, Any] = {
            "total_count": len(matches),
            "applications": output,
            "complete": not is_truncated,
            "truncated": is_truncated,
        }
        if is_truncated:
            result_payload["reason"] = "max_results_exceeded"
        return Success(result(request, result_payload, {"source": "installed-desktop-entries"}))



class ApplicationLaunch:
    """Re-resolve a handle immediately before an approved launch."""

    name = "application.launch"

    def __init__(self, backend: ApplicationBackend, handles: SQLiteHandleStore) -> None:
        self.backend = backend
        self.handles = handles

    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]:
        resolved = await asyncio.to_thread(
            self.handles.resolve, request.arguments["handle"],
            actor_id=request.actor.producer_id, task_id=request.arguments["task_id"],
            operation=self.name,
        )
        if isinstance(resolved, Failure):
            return Failure(failure(request, resolved.failure()))
        handle = resolved.unwrap()
        if handle.data_class != request.data_class:
            return Failure(failure(request, "RESOURCE_CLASS_MISMATCH"))
        discovered = await self.backend.discover()
        if isinstance(discovered, Failure):
            return Failure(failure(request, discovered.failure()))
        matches = [app for app in discovered.unwrap()
                   if app.desktop_id == handle.target and app.fingerprint == handle.fingerprint]
        if len(matches) != 1:
            return Failure(failure(request, "STALE_RESOURCE"))
        launched = await self.backend.launch(matches[0], cancellation)
        if isinstance(launched, Failure):
            return Failure(failure(request, launched.failure()))
        return Success(result(request, {"desktop_id": handle.target, "status": "SUCCEEDED"},
                              {"process": asdict(launched.unwrap())}))


def register_application_capabilities(
    registry: CapabilityRegistry, backend: ApplicationBackend, handles: SQLiteHandleStore,
) -> None:
    """Expose the same typed implementations to CLI, REST and MCP."""
    for implementation, properties, required, risk, effects, verification in (
        (ApplicationList(backend, handles), {"query": {"type": "string"}, "task_id": {"type": "string"}},
         ["task_id"], RiskClass.LOW, (), ("installed-desktop-entries",)),
        (ApplicationLaunch(backend, handles), {"handle": {"type": "string"}, "task_id": {"type": "string"}},
         ["handle", "task_id"], RiskClass.MODERATE, ("application-launch",), ("process-identity",)),
    ):
        descriptor = CapabilityDescriptor(
            name=implementation.name, description=f"Bounded {implementation.name} (v1).",
            input_schema={"type": "object", "properties": properties, "required": required,
                          "additionalProperties": False}, risk_class=risk,
            side_effects=effects, isolation="host-api", verification_plan=verification,
        )
        registry.register(RegisteredCapability(descriptor, implementation=implementation))
