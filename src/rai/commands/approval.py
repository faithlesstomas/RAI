"""Interactive approvals bound to one policy decision or outbound manifest."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rai.container import ApplicationContainer
    from rai.assistant.service import AssistantService

import typer
from returns.result import Failure, Success, Result

from rai.kernel.defaults import HitlApprovalBroker
from rai.kernel.ports import CancellationToken
from rai.kernel.records import ActionFailure, CapabilityRequest, DataClass, PolicyDecision, _new_id
from rai.actions.consent import BROWSER_ACTIONS, review_text
from rai.configuration.approvals import matching_rule, update_rule
from rai.configuration.models import ApprovalRule
from rai.configuration.storage import selected_path
from rai.diagnostics import trace
from rai.assistant.records import AssistantContextPackage, make_assistant_failure


async def choose(prompt: str, token: CancellationToken) -> str:
    """No approval on redirected stdin; cancellation cancels and reaps the prompt."""
    if token.cancelled or not sys.stdin.isatty():
        return ""
    from prompt_toolkit import PromptSession  # noqa: PLC0415

    session = PromptSession()
    task = asyncio.create_task(session.prompt_async(prompt))
    cancelled = asyncio.create_task(token.wait())
    try:
        done, _ = await asyncio.wait(
            (task, cancelled), timeout=60, return_when=asyncio.FIRST_COMPLETED
        )
        return task.result().strip().lower() if task in done and not token.cancelled else ""
    except (EOFError, KeyboardInterrupt):
        return ""
    finally:
        for pending in (task, cancelled):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, cancelled, return_exceptions=True)


async def confirm(prompt: str, token: CancellationToken) -> bool:
    return await choose(prompt + " [y/N] ", token) in {"y", "yes"}


class TerminalApprovalBroker:
    def __init__(self, profile: str | None = None, path: Path | None = None) -> None:
        self.profile = profile
        self.path = path or selected_path()

    async def request_action(
        self, decision: PolicyDecision, request: CapabilityRequest, cancellation: CancellationToken,
    ) -> Result[str, ActionFailure]:
        if request.capability not in BROWSER_ACTIONS:
            return await self.request(decision, cancellation)
        if cancellation.cancelled:
            return Failure(HitlApprovalBroker._failure(decision, "CANCELLED", "Approval cancelled"))
        rule = matching_rule(self.profile, request, self.path) if self.profile else None
        if rule and rule.decision != "ask":
            trace("approval.rule", request_id=request.record_id, rule_id=rule.id, outcome=rule.decision)
            typer.echo(f"Approval rule {rule.id}: {rule.decision} ({self.profile})", err=True)
            if rule.decision == "allow":
                return Success(f"rule:{self.profile}:{rule.id}")
            return Failure(HitlApprovalBroker._failure(decision, "DENIED", "Profile rule denied action"))
        typer.echo(review_text(request), err=True)
        remember = bool(self.profile and request.capability == "browser.search")
        prompt = ("Approve? [y=once / N=deny / a=remember exact query / d=remember denial] "
                  if remember else "Approve this outbound request? [y/N] ")
        answer = await choose(prompt, cancellation)
        if cancellation.cancelled:
            answer = ""
        if remember and answer in {"a", "d"}:
            data_class_name = request.data_class.value if hasattr(request.data_class, "value") else str(request.data_class)
            rule = ApprovalRule(id="search-" + _new_id(), data_class=data_class_name,
                                decision="allow" if answer == "a" else "deny", query=request.arguments["query"])
            update_rule(self.profile, rule.id, rule, self.path)
            typer.echo(f"Saved rule: {rule.id} (profile: {self.profile})", err=True)
        if answer in {"y", "yes"} or (remember and answer == "a"):
            return Success(f"rule:{self.profile}:{rule.id}" if remember and answer == "a" else _new_id())
        return Failure(HitlApprovalBroker._failure(decision, "DENIED", "Approval denied or unavailable"))

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
    container.capability_service.approvals = TerminalApprovalBroker(service.profile_scope)
    from rai.assistant.backends.antigravity import AntigravityAssistantModelBackend  # noqa: PLC0415

    if isinstance(service.backend, AntigravityAssistantModelBackend):
        service.context_approver = approve_egress
