"""Interactive approvals bound to one policy decision or outbound manifest."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rai.container import ApplicationContainer
    from rai.assistant.service import AssistantService

import typer
from returns.result import Failure, Success, Result

from rai.kernel.defaults import HitlApprovalBroker
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, DataClass, PolicyDecision, _new_id
from rai.assistant.records import AssistantContextPackage, make_assistant_failure


async def confirm(prompt: str, token: CancellationToken) -> bool:
    """No approval on redirected stdin; cancellation cancels and reaps the prompt."""
    if token.cancelled or not sys.stdin.isatty():
        return False
    from prompt_toolkit import PromptSession  # noqa: PLC0415

    session = PromptSession()
    task = asyncio.create_task(session.prompt_async(prompt + " [y/N] "))
    cancelled = asyncio.create_task(token.wait())
    try:
        done, _ = await asyncio.wait(
            (task, cancelled), timeout=60, return_when=asyncio.FIRST_COMPLETED
        )
        return (
            task in done
            and not token.cancelled
            and task.result().strip().lower() in {"y", "yes"}
        )
    except (EOFError, KeyboardInterrupt):
        return False
    finally:
        for pending in (task, cancelled):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, cancelled, return_exceptions=True)


class TerminalApprovalBroker:
    async def request(
        self, decision: PolicyDecision, cancellation: CancellationToken
    ) -> Result[str, ActionFailure]:
        typer.echo(
            f"Target: {decision.target_resource}\nEffects: {', '.join(decision.requested_side_effects)}",
            err=True,
        )
        if await confirm("Approve this action?", cancellation):
            return Success(_new_id())
        return Failure(
            HitlApprovalBroker._failure(
                decision, "DENIED", "Approval denied or unavailable"
            )
        )


async def approve_egress(
    package: AssistantContextPackage, cancellation: CancellationToken
) -> Result[AssistantContextPackage, ActionFailure]:
    """Review exactly the context that will be sent; LOCAL is never reclassified."""
    manifest = package.manifest
    if any(
        item.data_class in {DataClass.LOCAL, DataClass.SECRET, DataClass.BLOCKED}
        for item in manifest.items
    ):
        return Failure(
            make_assistant_failure(
                code="EGRESS_LOCAL_DATA_LEAK",
                message="Remote context contains non-exportable data",
                request_id=manifest.turn_id,
            )
        )
    typer.echo("Outbound context manifest (remote model):", err=True)
    for item in manifest.items:
        typer.echo(f"  {item.source_id}: {item.data_class}", err=True)
    if not await confirm(
        "Approve sending this context to the remote model?", cancellation
    ):
        return Failure(
            make_assistant_failure(
                code="EGRESS_DENIED",
                message="Outbound context was not approved",
                request_id=manifest.turn_id,
            )
        )
    return Success(
        package.model_copy(
            update={"manifest": manifest.model_copy(update={"approved": True})}
        )
    )


def configure_approvals(
    container: ApplicationContainer, service: AssistantService
) -> None:
    container.capability_service.approvals = TerminalApprovalBroker()
    from rai.assistant.backends.antigravity import AntigravityAssistantModelBackend  # noqa: PLC0415

    if isinstance(service.backend, AntigravityAssistantModelBackend):
        service.context_approver = approve_egress
