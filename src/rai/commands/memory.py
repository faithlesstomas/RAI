"""Model-free memory inspection and rendering."""

from __future__ import annotations
import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
import click
from .settings import _assistant_config


async def _echo_context_manifest(service: Any, manifest_id: str) -> None:  # noqa: ANN401
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.get_manifest(manifest_id)
    if isinstance(result, Success) and result.unwrap() is not None:
        manifest = result.unwrap()
        summary = {
            "manifest_id": manifest.record_id,
            "session_id": manifest.session_id,
            "recent_turn_ids": manifest.recent_turn_ids,
            "durable_memory_ids": manifest.durable_memory_ids,
            "exclusions": manifest.exclusions,
            "redactions": manifest.redactions,
            "backend": manifest.backend_name,
            "model": manifest.model_name,
            "model_artifact_version": manifest.model_artifact_version,
            "prompt_template_version": manifest.prompt_template_version,
            "estimated_tokens": manifest.actual_tokens,
        }
        entries = await service.audit_ledger.list_for_session(manifest.session_id)
        run = next(
            (
                entry
                for entry in reversed(entries)
                if entry.manifest_id == manifest.record_id
            ),
            None,
        )
        if run is not None:
            summary["run"] = {
                "latency_ms": round(run.latency_ms, 2),
                "tokens": run.tokens,
                "generation": run.generation_metadata,
            }
        package_result = await service.store.get_context_package(manifest.record_id)
        if isinstance(package_result, Success) and package_result.unwrap() is not None:
            summary["context_window"] = package_result.unwrap().content
        click.echo(f"\nContext manifest:\n{json.dumps(summary, indent=2)}", err=True)
        return
    click.echo("Context manifest not found.", err=True)


async def _echo_memories(service: Any) -> None:  # noqa: ANN401
    """Show active memories in the current local profile scope."""
    from returns.result import Success  # noqa: PLC0415
    from rai.kernel.records import DataClass  # noqa: PLC0415

    result = await service.store.retrieve_relevant_memories(
        profile_scope=service.profile_scope,
        data_classes=(DataClass.PUBLIC, DataClass.LOCAL, DataClass.PRIVATE),
        limit=20,
    )
    if not isinstance(result, Success):
        click.echo(f"Could not read memories: {result.failure().message}", err=True)
        return
    memories = result.unwrap()
    if not memories:
        click.echo("No active durable memories for this profile.")
        return
    click.echo("Active durable memories:")
    for memory, _reason in memories:
        click.echo(
            f"- [{memory.kind}] {memory.topic}: {json.dumps(memory.content, ensure_ascii=False)}"
        )


async def _echo_history(service: Any, session_id: str, limit: int = 20) -> None:  # noqa: ANN401
    """Show the persisted conversation window for a session."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.get_recent_turns(session_id, limit=limit)
    if not isinstance(result, Success):
        click.echo(f"Could not read chat history: {result.failure().message}", err=True)
        return
    turns = result.unwrap()
    if not turns:
        click.echo(f"No completed turns for session {session_id}.")
        return
    click.echo(f"Chat history for {session_id} ({len(turns)} turns):")
    for turn in turns:
        timestamp = turn.timestamp.astimezone().isoformat(timespec="seconds")
        click.echo(f"- {timestamp} {turn.role}: {turn.text}")


async def _echo_sessions(service: Any, limit: int = 20) -> None:  # noqa: ANN401
    """Show discoverable local conversation sessions."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.list_sessions(limit=limit)
    if not isinstance(result, Success):
        click.echo(f"Could not list sessions: {result.failure().message}", err=True)
        return
    sessions = result.unwrap()
    if not sessions:
        click.echo("No local assistant sessions found.")
        return
    click.echo("Assistant sessions:")
    for session in sessions:
        timestamp = session.updated_at.astimezone().isoformat(timespec="seconds")
        click.echo(
            f"- {session.session_id} | {session.turn_count} turns | {timestamp} | "
            f"{session.last_role}: {session.preview}"
        )


async def _echo_memory_operations(service: Any, limit: int = 20) -> None:  # noqa: ANN401
    """Show the append-only memory operation trace for the active profile."""
    from returns.result import Success  # noqa: PLC0415

    result = await service.store.list_memory_operations(
        profile_scope=service.profile_scope, limit=limit
    )
    if not isinstance(result, Success):
        click.echo(
            f"Could not list memory operations: {result.failure().message}", err=True
        )
        return
    operations = result.unwrap()
    if not operations:
        click.echo("No memory operations for this profile.")
        return
    click.echo("Memory operations:")
    for operation in operations:
        click.echo(
            f"- {operation.timestamp.astimezone().isoformat(timespec='seconds')} "
            f"{operation.operation} {operation.status} "
            f"targets={','.join(operation.target_memory_ids) or '-'} "
            f"results={','.join(operation.result_memory_ids) or '-'}"
        )


async def _echo_memory_diagnostics(service: Any) -> None:  # noqa: ANN401
    """Show stage-specific memory integrity diagnostics."""
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.diagnostics import diagnose_memory  # noqa: PLC0415

    result = await diagnose_memory(service.store, profile_scope=service.profile_scope)
    if not isinstance(result, Success):
        click.echo(f"Could not diagnose memory: {result.failure().message}", err=True)
        return
    report = result.unwrap()
    click.echo(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))


def _run_assistant_read(
    profile: str | None,
    action: Callable[[Any], Awaitable[None]],
) -> None:
    """Run a model-free assistant inspection command against durable local state."""
    from returns.result import Success  # noqa: PLC0415

    from rai.assistant.service import AssistantService  # noqa: PLC0415
    from rai.assistant.store import SQLiteMemoryGraphStore  # noqa: PLC0415

    config = _assistant_config(None, None, profile)
    scope = str(config.get("active_agent") or "default")
    service = AssistantService(
        store=SQLiteMemoryGraphStore(),
        profile_scope=scope,
    )

    async def _run() -> None:
        start_res = await service.start()
        if not isinstance(start_res, Success):
            error = start_res.failure()
            raise click.ClickException(f"[{error.code}] {error.message}")
        try:
            await action(service)
        finally:
            await service.stop()

    asyncio.run(_run())
