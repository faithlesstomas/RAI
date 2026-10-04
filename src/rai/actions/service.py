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

from .capabilities import failure
from .execution import SQLiteExecutionStore
from .handles import SQLiteHandleStore

ACTION_NAMES = frozenset({
    "application.list", "application.launch", "file.search", "document.open",
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
                    if decision.outcome == PolicyOutcome.ASK:
                        if self.approvals is None:
                            terminal = failure(request, "APPROVAL_UNAVAILABLE")
                        else:
                            approval = await bounded(self.approvals.request(decision, token), token, APPROVAL_TIMEOUT)
                            if isinstance(approval, Success):
                                approval_id, permitted = approval.unwrap(), True
                            else:
                                terminal = failure(request, approval.failure().code)
                    elif decision.outcome == PolicyOutcome.ESCALATE:
                        terminal = failure(request, "ESCALATION_REQUIRED")
                    if permitted:
                        started = True
                        capability = self.registry.resolve(request).unwrap()
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
        saved = self.executions.finish(original, terminal, decision)
        if isinstance(saved, Failure):
            return decision, Failure(failure(original, "EXECUTION_STORE_UNAVAILABLE"))
        if decision is not None:
            audited = await self.audit.append(AuditEntry(
                stage="TERMINAL", decision=decision, approval_id=approval_id, result=terminal,
            ))
            if isinstance(audited, Failure):
                return decision, Failure(failure(original, "AUDIT_UNAVAILABLE"))
        return decision, Success(terminal) if isinstance(terminal, ActionResult) else Failure(terminal)
