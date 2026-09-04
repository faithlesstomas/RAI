"""Composition and compatibility checks for the Stage 1 application kernel."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from returns.result import Failure, Success

from rai.backends.antigravity import AntigravityBackend
from rai.container import ApplicationContainer
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.compatibility import policy_wrapped_handlers
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import AgentBackend, CancellationToken, LifecycleState
from rai.kernel.records import (
    ContextManifest,
    ContextPackage,
    InferenceBudget,
    ProducerIdentity,
    Task,
)
from rai.kernel.service import CapabilityService
from rai.server import create_app

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PRODUCER = ProducerIdentity(
    producer_id="stage1-test", kind="test", version="1.0.0"
)


def _backend_inputs() -> tuple[Task, ContextPackage, InferenceBudget]:
    task = Task(timestamp=NOW, producer=PRODUCER, objective="answer")
    manifest = ContextManifest(
        timestamp=NOW, producer=PRODUCER, destination="antigravity", items=()
    )
    context = ContextPackage(
        timestamp=NOW,
        producer=PRODUCER,
        task_id=task.record_id,
        manifest=manifest,
        content={},
    )
    budget = InferenceBudget(
        timestamp=NOW,
        producer=PRODUCER,
        max_input_tokens=10,
        max_output_tokens=10,
        max_agent_turns=1,
        max_tool_calls=0,
        max_images=0,
        max_audio_seconds=0,
        max_latency_seconds=1,
        max_provider_cost=0,
        max_ram_bytes=1,
        max_vram_bytes=0,
        cancellation_deadline=NOW + timedelta(seconds=1),
    )
    return task, context, budget


def test_app_factory_owns_distinct_containers_and_mcp_runtimes() -> None:
    first = create_app(ApplicationContainer(config={}, testing=True))
    second = create_app(ApplicationContainer(config={}, testing=True))
    assert first.state.container is not second.state.container
    assert first.state.mcp_runtime is not second.state.mcp_runtime


@pytest.mark.asyncio
async def test_antigravity_is_a_lifecycle_managed_backend_with_typed_output() -> None:
    backend = AntigravityBackend(
        CapabilityService(
            create_default_capability_registry(),
            PolicyEngine(),
            InMemoryAuditLedger(),
        )
    )
    assert isinstance(backend, AgentBackend)
    task, context, budget = _backend_inputs()
    not_started = await backend.execute(
        task, context, (), budget, CancellationToken()
    )
    assert isinstance(not_started, Failure)
    assert not_started.failure().code == "BACKEND_NOT_RUNNING"

    assert isinstance(await backend.start(), Success)
    backend.run_chain = AsyncMock(  # type: ignore[method-assign]
        return_value=Success(
            {"content": "answer", "tool_calls": None, "session_id": "session"}
        )
    )
    result = await backend.execute(task, context, (), budget, CancellationToken())
    assert isinstance(result, Success)
    assert result.unwrap().output == {
        "content": "answer",
        "tool_calls": None,
        "session_id": "session",
    }
    assert backend.state == LifecycleState.RUNNING
    assert isinstance(await backend.stop(), Success)


@pytest.mark.asyncio
async def test_legacy_tool_signature_still_uses_policy_and_audit() -> None:
    audit = InMemoryAuditLedger()
    service = CapabilityService(
        create_default_capability_registry(), PolicyEngine(), audit
    )
    handlers = policy_wrapped_handlers(service, "CalculatorTools")
    assert len(handlers) == 1
    assert await handlers[0]("2 + 2") == "4"
    assert [entry.stage for entry in audit.entries] == ["DECISION", "TERMINAL"]


def test_chat_facade_does_not_import_backend_sdk_directly() -> None:
    source = Path("src/rai/services/chat.py").read_text(encoding="utf-8")
    assert "google.antigravity" not in source
    assert "._step_queue" not in source
