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


def format_request_details(request: CapabilityRequest, decision: PolicyDecision) -> str:
    import json  # noqa: PLC0415

    arguments_json = json.dumps(request.arguments, indent=2, ensure_ascii=False)
    side_effects = ", ".join(decision.requested_side_effects) if decision.requested_side_effects else "none"
    reasons = ", ".join(decision.reason_codes) if decision.reason_codes else "none"
    return (
        f"\n--- [Action Details] ---\n"
        f"Capability:      {request.capability}\n"
        f"Target Resource: {request.target_resource}\n"
        f"Data Class:      {request.data_class}\n"
        f"Risk Class:      {decision.risk_class}\n"
        f"Side Effects:    {side_effects}\n"
        f"Policy Reasons:  {reasons}\n"
        f"Arguments:\n{arguments_json}\n"
        f"------------------------\n"
    )


def format_decision_details(decision: PolicyDecision) -> str:
    side_effects = ", ".join(decision.requested_side_effects) if decision.requested_side_effects else "none"
    reasons = ", ".join(decision.reason_codes) if decision.reason_codes else "none"
    return (
        f"\n--- [Decision Details] ---\n"
        f"Target Resource: {decision.target_resource}\n"
        f"Data Class:      {decision.data_class}\n"
        f"Risk Class:      {decision.risk_class}\n"
        f"Side Effects:    {side_effects}\n"
        f"Policy Reasons:  {reasons}\n"
        f"Request ID:      {decision.request_id}\n"
        f"--------------------------\n"
    )


async def choose(prompt: str, token: CancellationToken, timeout: float | None = None) -> str:
    """No approval on redirected stdin; cancellation cancels and reaps the prompt."""
    if token.cancelled or not sys.stdin.isatty():
        return ""
    from prompt_toolkit import PromptSession  # noqa: PLC0415

    session = PromptSession()
    task = asyncio.create_task(session.prompt_async(prompt))
    cancelled = asyncio.create_task(token.wait())
    try:
        done, _ = await asyncio.wait(
            (task, cancelled), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        return task.result().strip().lower() if task in done and not token.cancelled else ""
    except KeyboardInterrupt:
        token.cancel()
        return ""
    except EOFError:
        return ""
    finally:
        for pending in (task, cancelled):
            if not pending.done():
                pending.cancel()
        await asyncio.gather(task, cancelled, return_exceptions=True)


async def confirm(prompt: str, token: CancellationToken) -> bool:
    return await choose(prompt + " [y/N] ", token) in {"y", "yes"}


def _format_action_summary(request: CapabilityRequest, decision: PolicyDecision) -> str:
    if request.capability in BROWSER_ACTIONS:
        return review_text(request)
    effects_text = ", ".join(decision.requested_side_effects) or "none"
    reasons_text = ", ".join(decision.reason_codes) or str(decision.risk_class)
    if request.capability == "run_python_code":
        script_body = str(request.arguments.get("code") or "<empty script>")
        return (
            f"Action: run_python_code (Tier-2 Bubblewrap OS execution)\n"
            f"Risk: {decision.risk_class} ({reasons_text})\n"
            f"Target: {decision.target_resource}\n"
            f"Effects: {effects_text}\n"
            f"Script:\n{script_body}"
        )
    if request.capability == "run_shell_command":
        cmd_body = str(request.arguments.get("command") or "<empty command>")
        return (
            f"Action: run_shell_command\n"
            f"Risk: {decision.risk_class} ({reasons_text})\n"
            f"Target: {decision.target_resource}\n"
            f"Effects: {effects_text}\n"
            f"Command: {cmd_body}"
        )
    return (
        f"Action: {request.capability}\n"
        f"Target: {decision.target_resource}\n"
        f"Effects: {effects_text}"
    )


class TerminalApprovalBroker:
    def __init__(self, profile: str | None = None, path: Path | None = None) -> None:
        self.profile = profile
        self.path = path or selected_path()

    async def request_action(
        self, decision: PolicyDecision, request: CapabilityRequest, cancellation: CancellationToken,
    ) -> Result[str, ActionFailure]:
        if cancellation.cancelled:
            return Failure(HitlApprovalBroker._failure(decision, "CANCELLED", "Approval cancelled"))
        rule = matching_rule(self.profile, request, self.path) if self.profile else None
        if rule and rule.decision != "ask":
            trace("approval.rule", request_id=request.record_id, rule_id=rule.id, outcome=rule.decision)
            typer.echo(f"Approval rule {rule.id}: {rule.decision} ({self.profile})")
            if rule.decision == "allow":
                return Success(f"rule:{self.profile}:{rule.id}")
            return Failure(HitlApprovalBroker._failure(decision, "DENIED", "Profile rule denied action"))

        typer.echo(_format_action_summary(request, decision))

        remember = bool(self.profile and request.capability == "browser.search")
        prompt = ("Approve? [y=once / N=deny / a=remember exact query / d=remember denial / s=show details] "
                  if remember else "Approve this action? [y=once / N=deny / s=show details] ")

        while True:
            answer = await choose(prompt, cancellation)
            if cancellation.cancelled:
                answer = ""
                break
            if answer in {"s", "show", "details"}:
                typer.echo(format_request_details(request, decision))
                continue
            break

        if cancellation.cancelled:
            return Failure(HitlApprovalBroker._failure(decision, "CANCELLED", "Approval cancelled"))

        if remember and answer in {"a", "d"}:
            data_class_name = request.data_class.value if hasattr(request.data_class, "value") else str(request.data_class)
            rule = ApprovalRule(id="search-" + _new_id(), data_class=data_class_name,
                                decision="allow" if answer == "a" else "deny", query=request.arguments["query"])
            update_rule(self.profile, rule.id, rule, self.path)
            typer.echo(f"Saved rule: {rule.id} (profile: {self.profile})")
        if answer in {"y", "yes"} or (remember and answer == "a"):
            return Success(f"rule:{self.profile}:{rule.id}" if remember and answer == "a" else _new_id())
        return Failure(HitlApprovalBroker._failure(decision, "DENIED", "Approval denied or unavailable"))

    async def request(
        self, decision: PolicyDecision, cancellation: CancellationToken
    ) -> Result[str, ActionFailure]:
        if cancellation.cancelled:
            return Failure(HitlApprovalBroker._failure(decision, "CANCELLED", "Approval cancelled"))
        effects_text = ", ".join(decision.requested_side_effects) or "none"
        typer.echo(
            f"Target: {decision.target_resource}\nEffects: {effects_text}"
        )
        prompt = "Approve this action? [y=once / N=deny / s=show details] "
        while True:
            answer = await choose(prompt, cancellation)
            if cancellation.cancelled:
                answer = ""
                break
            if answer in {"s", "show", "details"}:
                typer.echo(format_decision_details(decision))
                continue
            break

        if cancellation.cancelled:
            return Failure(HitlApprovalBroker._failure(decision, "CANCELLED", "Approval cancelled"))

        if answer in {"y", "yes"}:
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
    if service.backend.is_remote:
        service.context_approver = approve_egress
