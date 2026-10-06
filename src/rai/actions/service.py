"""Bounded durable action path layered on the common capability service."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from returns.result import Failure, Result, Success

from rai.kernel.audit import AuditEntry
from rai.kernel.capabilities import validate_arguments
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, ActionResult, CapabilityRequest, PolicyDecision, PolicyOutcome
from rai.kernel.service import CapabilityService

from rai.diagnostics import trace
from .consent import BROWSER_ACTIONS, browser_consent
from .capabilities import failure
from .execution import SQLiteExecutionStore
from .handles import SQLiteHandleStore

ACTION_NAMES = frozenset({
    "application.list", "application.launch", "file.access", "file.list", "file.search", "document.open",
    "browser.search", "browser.open_result", "browser.read_page",
    "system.volume.get", "system.volume.set", "process.inspect",
})
MAX_CONCURRENT_ACTIONS = 4
ACTION_TIMEOUT = 20.0
APPROVAL_TIMEOUT = 60.0
T = TypeVar("T")


async def bounded(awaitable: Awaitable[T], token: CancellationToken, seconds: float) -> T:
    """Cancel and reap cooperative I/O when its token or deadline fires."""
    work = asyncio.ensure_future(awaitable)
    cancelled = asyncio.create_task(token.wait())
    try:
        done, _ = await asyncio.wait((work, cancelled), timeout=seconds,
                                     return_when=asyncio.FIRST_COMPLETED)
        if cancelled in done:
            raise asyncio.CancelledError
        if work not in done:
            raise asyncio.TimeoutError
        return await work
    finally:
        for task in (work, cancelled):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, cancelled, return_exceptions=True)


class ActionCapacityError(Exception):
    """The runtime is already processing its bounded action allowance."""


class ActionCapabilityService(CapabilityService):
    """All transports share policy/audit; catalog actions also get durable replay."""

    _active_actions: int = 0
    executions: SQLiteExecutionStore
    handles: SQLiteHandleStore

    async def invoke(  # noqa: PLR0911, PLR0912, PLR0915
        self, request: CapabilityRequest, cancellation: CancellationToken | None = None,
    ) -> tuple[PolicyDecision | None, Result[ActionResult, ActionFailure]]:
        if request.capability not in ACTION_NAMES:
            return await super().invoke(request, cancellation)
        token = cancellation or CancellationToken()
        # Synchronous short SQLite transactions cannot be abandoned halfway through reservation.
        reservation = self.executions.reserve(request)
        if isinstance(reservation, Failure):
            return None, Failure(failure(request, reservation.failure()))
        previous = reservation.unwrap()
        if previous is not None:
            decision = self.executions.decision(request.record_id)
            delivered = await self._deliver_audit(request.record_id)
            if isinstance(delivered, Failure):
                return decision, Failure(failure(request, delivered.failure()))
            return decision, Success(previous) if isinstance(previous, ActionResult) else Failure(previous)
        original = request
        decision = None
        approval_id = None
        started = False
        acquired = False
        try:
            if self._active_actions >= MAX_CONCURRENT_ACTIONS:
                raise ActionCapacityError
            self._active_actions += 1
            acquired = True
            descriptor = self.registry.descriptor(request.capability)
            if descriptor is None:
                terminal = failure(request, "CAPABILITY_NOT_FOUND")
            else:
                invalid = validate_arguments(descriptor.input_schema, request.arguments)
                preflight = "INVALID_ARGUMENT" if invalid else None
                if not preflight and "handle" in request.arguments:
                    resolved = self.handles.resolve(
                        request.arguments["handle"], actor_id=request.actor.producer_id,
                        task_id=request.arguments.get("task_id", ""), operation=request.capability,
                    )
                    if isinstance(resolved, Failure):
                        preflight = resolved.failure()
                    else:
                        resource = resolved.unwrap()
                        if resource.data_class != request.data_class:
                            preflight = "RESOURCE_CLASS_MISMATCH"
                        else:
                            # This exact resolved target is shown to the approval broker.
                            request = request.model_copy(update={
                                "target_resource": f"{resource.kind}://{resource.target}",
                            })
                if not preflight and "application_handle" in request.arguments:
                    app = self.handles.resolve(request.arguments["application_handle"],
                        actor_id=request.actor.producer_id, task_id=request.arguments.get("task_id", ""),
                        operation=request.capability)
                    if isinstance(app, Failure):
                        preflight = app.failure()
                    elif app.unwrap().kind != "application" or app.unwrap().data_class != request.data_class:
                        preflight = "RESOURCE_CLASS_MISMATCH"
                    else:
                        request = request.model_copy(update={
                            "target_resource": f"{request.target_resource} using application://{app.unwrap().target}",
                        })
                decision = self.policy.evaluate(request, descriptor)
                if preflight:
                    decision = decision.model_copy(update={
                        "outcome": PolicyOutcome.DENY, "reason_codes": (preflight,),
                    })
                audited = await self.audit.append(AuditEntry(stage="DECISION", decision=decision))
                if isinstance(audited, Failure):
                    terminal = failure(request, "AUDIT_UNAVAILABLE")
                elif preflight:
                    terminal = failure(request, preflight)
                else:
                    permitted = decision.outcome == PolicyOutcome.ALLOW
                    terminal = failure(request, "POLICY_DENIED")
                    if token.cancelled:
                        raise asyncio.CancelledError
                    review_action = getattr(self.approvals, "request_action", None)
                    export_review = (
                        decision.outcome == PolicyOutcome.ESCALATE
                        and decision.reason_codes == ("PRIVATE_DATA_EGRESS",)
                        and request.capability in BROWSER_ACTIONS
                        and review_action is not None
                    )
                    if decision.outcome == PolicyOutcome.ASK or export_review:
                        if self.approvals is None:
                            terminal = failure(request, "APPROVAL_UNAVAILABLE")
                        else:
                            trace("approval.start", request_id=request.record_id, capability=request.capability)
                            pending = (review_action(decision, request, token) if review_action
                                       else self.approvals.request(decision, token))
                            approval = await bounded(pending, token, APPROVAL_TIMEOUT)
                            trace("approval.end", request_id=request.record_id,
                                  outcome="approved" if isinstance(approval, Success) else "denied")
                            if isinstance(approval, Success):
                                approval_id, permitted = approval.unwrap(), True
                            else:
                                terminal = failure(request, approval.failure().code)
                    elif decision.outcome == PolicyOutcome.ESCALATE:
                        terminal = failure(request, "ESCALATION_REQUIRED")
                    if permitted:
                        started = True
                        capability = self.registry.resolve(request).unwrap()
                        # Consent is task-local and bound to the exact immutable request.
                        # Classification and handle labels remain PRIVATE.
                        with browser_consent(request):
                            invoked = await bounded(capability.invoke(request, token), token, ACTION_TIMEOUT)
                        if isinstance(invoked, Success):
                            terminal = invoked.unwrap()
                        elif invoked.failure().code == "CAPABILITY_FAILED":
                            terminal = failure(request, "UNKNOWN")
                        else:
                            terminal = invoked.failure()
        except ActionCapacityError:
            terminal = failure(original, "ACTION_CAPACITY_EXCEEDED")
        except asyncio.CancelledError:
            terminal = failure(original, "UNKNOWN" if started else "CANCELLED")
        except asyncio.TimeoutError:
            terminal = failure(original, "UNKNOWN" if started else "TIMEOUT")
        except Exception:  # pylint: disable=broad-exception-caught
            terminal = failure(original, "UNKNOWN" if started else "ACTION_FAILED")
        finally:
            if acquired:
                self._active_actions -= 1
        # Store first so even an audit I/O failure cannot replay the OS effect.
        saved = self.executions.finish(original, terminal, decision, approval_id)
        if isinstance(saved, Failure):
            return decision, Failure(failure(original, "EXECUTION_STORE_UNAVAILABLE"))
        delivered = await self._deliver_audit(original.record_id)
        if isinstance(delivered, Failure):
            return decision, Failure(failure(original, delivered.failure()))
        return decision, Success(terminal) if isinstance(terminal, ActionResult) else Failure(terminal)

    async def _deliver_audit(self, request_id: str) -> Result[None, str]:
        pending = self.executions.pending_audit(request_id)
        if isinstance(pending, Failure):
            return Failure(pending.failure())
        entry = pending.unwrap()
        if entry is None:
            return Success(None)
        try:
            append = getattr(self.audit, "append_once", self.audit.append)
            delivered = await append(entry)
        except Exception:  # pylint: disable=broad-exception-caught
            return Failure("AUDIT_UNAVAILABLE")
        if isinstance(delivered, Failure):
            return Failure("AUDIT_UNAVAILABLE")
        return self.executions.acknowledge_audit(request_id)
