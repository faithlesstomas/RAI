"""Fail-closed edges of the durable action path: stores, handles, files and the service state machine."""
from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from returns.result import Failure, Success

import rai.actions.execution as execution
import rai.actions.files as files
import rai.actions.handles as handle_module
import rai.actions.service as action_service
from rai.actions.capabilities import PRODUCER, failure, result
from rai.actions.execution import SQLiteExecutionStore
from rai.actions.files import DocumentOpen, FileSearch, inspect_document, register_file_capabilities, search_documents
from rai.actions.handles import SQLiteHandleStore
from rai.actions.records import ResourceHandle
from rai.actions.service import ActionCapabilityService
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityDescriptor, CapabilityRegistry, RegisteredCapability
from rai.kernel.policy import PolicyEngine
from rai.kernel.ports import CancellationToken
from rai.kernel.records import DataClass, PolicyOutcome, RiskClass, _utc_now
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request

SCHEMA = {"type": "object", "properties": {"handle": {"type": "string"}, "task_id": {"type": "string"}}}


def descriptor(name: str = "application.launch", risk: RiskClass = RiskClass.MODERATE) -> CapabilityDescriptor:
    return CapabilityDescriptor(name=name, description="Edge fixture", input_schema=SCHEMA, risk_class=risk,
                                side_effects=("application-launch",) if risk != RiskClass.LOW else (),
                                isolation="host-api", verification_plan=("observed",))


class Effect:
    def __init__(self, name: str = "application.launch", error: Exception | None = None) -> None:
        self.name, self.error, self.calls = name, error, 0

    async def invoke(self, request: Any, token: CancellationToken) -> Any:
        self.calls += 1
        if self.error:
            raise self.error
        return Success(result(request, {"status": "SUCCEEDED"}, {"observed": True}))


def runtime(tmp_path: Path, effect: Effect | None = None, name: str = "application.launch",
            risk: RiskClass = RiskClass.MODERATE, audit: Any = None) -> ActionCapabilityService:
    registry = CapabilityRegistry()
    registry.register(RegisteredCapability(descriptor(name, risk), implementation=effect or Effect(name)))
    instance = ActionCapabilityService(registry, PolicyEngine(), audit or InMemoryAuditLedger(), SyntheticApprovalBroker())
    instance.executions = SQLiteExecutionStore(tmp_path / "executions.db")
    instance.handles = SQLiteHandleStore(tmp_path / "handles.db")
    return instance


def request_for(instance: ActionCapabilityService, arguments: dict | None = None, name: str = "application.launch") -> Any:
    return normalize_request(instance.registry.descriptor(name), arguments or {})


def handle_for(request: Any, **overrides: Any) -> ResourceHandle:
    now = _utc_now()
    fields = dict(producer=PRODUCER, timestamp=now, actor_id=request.actor.producer_id, task_id="t", kind="file",
                  target="/x", fingerprint="f", operations=("application.launch",),
                  expires_at=now + timedelta(minutes=5), data_class=request.data_class)
    return ResourceHandle(**{**fields, **overrides})


# --- service state machine -----------------------------------------------------------------------------------------


async def test_non_catalog_capabilities_use_the_common_path(tmp_path: Path) -> None:
    instance = runtime(tmp_path, name="demo.echo", risk=RiskClass.LOW)
    request = request_for(instance, name="demo.echo")
    assert isinstance((await instance.invoke(request))[1], Success)
    assert instance.executions.decision(request.record_id) is None  # never reserved durably


async def test_reservation_failure_blocks_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    request = request_for(instance)
    monkeypatch.setattr(instance.executions, "reserve", lambda _request: Failure("EXECUTION_STORE_UNAVAILABLE"))
    decision, outcome = await instance.invoke(request)
    assert decision is None and outcome.failure().code == "EXECUTION_STORE_UNAVAILABLE" and not effect.calls


async def test_missing_capability_and_invalid_arguments_are_denied_before_effects(tmp_path: Path) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    request = request_for(instance)
    instance.registry = CapabilityRegistry()
    assert (await instance.invoke(request))[1].failure().code == "CAPABILITY_NOT_FOUND"
    instance = runtime(tmp_path / "second", effect)
    invalid = request_for(instance).model_copy(update={"arguments": {"task_id": 7}})
    decision, outcome = await instance.invoke(invalid)
    assert outcome.failure().code == "INVALID_ARGUMENT" and decision.outcome == PolicyOutcome.DENY and not effect.calls


async def test_handle_preflight_rejects_unknown_and_cross_class_resources(tmp_path: Path) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    unknown = request_for(instance, {"task_id": "t", "handle": "missing"})
    assert (await instance.invoke(unknown))[1].failure().code == "HANDLE_NOT_FOUND"
    base = request_for(instance, {"task_id": "t"})
    other_class = DataClass.PUBLIC if base.data_class != DataClass.PUBLIC else DataClass.LOCAL
    handle_id = instance.handles.issue(handle_for(base, data_class=other_class)).unwrap()
    mismatch = request_for(instance, {"task_id": "t", "handle": handle_id})
    assert (await instance.invoke(mismatch))[1].failure().code == "RESOURCE_CLASS_MISMATCH"
    assert not effect.calls


async def test_resolved_target_is_shown_to_policy_and_audit(tmp_path: Path) -> None:
    audit = InMemoryAuditLedger()
    instance = runtime(tmp_path, audit=audit)
    base = request_for(instance, {"task_id": "t"})
    handle_id = instance.handles.issue(handle_for(base)).unwrap()
    decision, outcome = await instance.invoke(request_for(instance, {"task_id": "t", "handle": handle_id}))
    assert isinstance(outcome, Success) and decision.target_resource == "file:///x"


async def test_missing_approval_broker_and_escalation_never_execute(tmp_path: Path) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    instance.approvals = None
    assert (await instance.invoke(request_for(instance)))[1].failure().code == "APPROVAL_UNAVAILABLE"
    escalating = runtime(tmp_path / "second", effect)
    real = escalating.policy

    class Escalate:
        def evaluate(self, request: Any, descriptor: Any) -> Any:
            return real.evaluate(request, descriptor).model_copy(update={"outcome": PolicyOutcome.ESCALATE})

    escalating.policy = Escalate()  # type: ignore[assignment]
    assert (await escalating.invoke(request_for(escalating)))[1].failure().code == "ESCALATION_REQUIRED"
    assert not effect.calls


async def test_cancellation_before_approval_is_cancelled_not_unknown(tmp_path: Path) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    token = CancellationToken()
    token.cancel()
    assert (await instance.invoke(request_for(instance), token))[1].failure().code == "CANCELLED"
    assert not effect.calls


async def test_approval_timeout_before_effect_is_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = runtime(tmp_path)
    monkeypatch.setattr(action_service, "APPROVAL_TIMEOUT", 0.01)

    class Hanging:
        async def request(self, decision: Any, token: Any) -> Any:
            await asyncio.Event().wait()

    instance.approvals = Hanging()  # type: ignore[assignment]
    assert (await instance.invoke(request_for(instance)))[1].failure().code == "TIMEOUT"


async def test_unexpected_errors_distinguish_before_and_after_effect(tmp_path: Path) -> None:
    instance = runtime(tmp_path, Effect(error=RuntimeError("secret detail")))
    outcome = (await instance.invoke(request_for(instance)))[1]
    assert outcome.failure().code == "UNKNOWN" and "secret" not in outcome.failure().message

    class Broken:
        async def request(self, decision: Any, token: Any) -> Any:
            raise RuntimeError("approval ui crashed")

    early = runtime(tmp_path / "early")
    early.approvals = Broken()  # type: ignore[assignment]
    assert (await early.invoke(request_for(early)))[1].failure().code == "ACTION_FAILED"


async def test_store_or_audit_failures_are_reported_without_replaying_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    effect = Effect()
    instance = runtime(tmp_path, effect)
    request = request_for(instance)
    monkeypatch.setattr(instance.executions, "finish", lambda *_a, **_k: Failure("TERMINAL_CONFLICT"))
    assert (await instance.invoke(request))[1].failure().code == "EXECUTION_STORE_UNAVAILABLE"
    assert effect.calls == 1

    class Audit:
        def __init__(self, fail_on: int) -> None:
            self.fail_on, self.count = fail_on, 0

        async def append(self, entry: Any) -> Any:
            self.count += 1
            return Failure(failure(request, "AUDIT")) if self.count == self.fail_on else Success(entry)

    for fail_on in (1, 2):
        effect = Effect()
        audited = runtime(tmp_path / f"audit{fail_on}", effect, audit=Audit(fail_on))
        outcome = (await audited.invoke(request_for(audited)))[1]
        assert outcome.failure().code == "AUDIT_UNAVAILABLE"
        assert effect.calls == (0 if fail_on == 1 else 1)


# --- execution store ------------------------------------------------------------------------------------------------


def test_execution_store_rejects_oversized_conflicting_and_unfinished_requests(tmp_path: Path) -> None:
    store = SQLiteExecutionStore(tmp_path / "executions.db")
    base = request_for(runtime(tmp_path))
    assert store.reserve(base.model_copy(update={"arguments": {"x": "a" * 20000}})) == Failure("REQUEST_SIZE_LIMIT")
    assert store.reserve(base) == Success(None)
    assert store.reserve(base) == Failure("EXECUTION_UNKNOWN")
    changed = base.model_copy(update={"arguments": {"task_id": "other"}})
    assert store.reserve(changed) == Failure("REQUEST_ID_CONFLICT")


def test_execution_store_capacity_and_unavailable_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteExecutionStore(tmp_path / "executions.db")
    instance = runtime(tmp_path)
    monkeypatch.setattr(execution, "MAX_EXECUTION_RECORDS", 1)
    assert store.reserve(request_for(instance)) == Success(None)
    assert store.reserve(request_for(instance)) == Failure("EXECUTION_CAPACITY_EXCEEDED")
    broken = SQLiteExecutionStore(tmp_path)  # a directory cannot be opened as a database
    request = request_for(instance)
    assert broken.reserve(request) == Failure("EXECUTION_STORE_UNAVAILABLE")
    assert broken.finish(request, failure(request, "X")) == Failure("EXECUTION_STORE_UNAVAILABLE")


def test_execution_store_finish_is_exactly_once_and_replays_both_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteExecutionStore(tmp_path / "executions.db")
    instance = runtime(tmp_path)
    ok, bad = request_for(instance), request_for(instance)
    assert store.decision(ok.record_id) is None
    store.reserve(ok)
    store.reserve(bad)
    mismatch = failure(ok, "X").model_copy(update={"request_id": "another"})
    assert store.finish(ok, mismatch) == Failure("TERMINAL_ID_MISMATCH")
    success = result(ok, {"status": "SUCCEEDED"}, {"observed": True})
    assert store.finish(ok, success) == Success(None)
    assert store.finish(ok, failure(ok, "LATE")) == Failure("TERMINAL_CONFLICT")
    assert store.finish(bad, failure(bad, "DENIED")) == Success(None)
    assert store.reserve(ok) == Success(success)
    assert store.reserve(bad).unwrap().code == "DENIED"
    oversize = request_for(instance)
    store.reserve(oversize)
    monkeypatch.setattr(execution, "MAX_TERMINAL_BYTES", 10)
    assert store.finish(oversize, failure(oversize, "BIG")) == Failure("TERMINAL_SIZE_LIMIT")


# --- handle store ---------------------------------------------------------------------------------------------------


def test_handle_lifetime_expiry_revocation_and_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [_utc_now()]
    store = SQLiteHandleStore(tmp_path / "handles.db", clock=lambda: now[0])
    base = request_for(runtime(tmp_path))
    future = handle_for(base, timestamp=now[0] + timedelta(hours=1), expires_at=now[0] + timedelta(hours=2))
    assert store.issue(future) == Failure("INVALID_HANDLE_LIFETIME")
    assert store.issue(handle_for(base, timestamp=now[0] - timedelta(seconds=10), expires_at=now[0] - timedelta(seconds=1))) == Failure("INVALID_HANDLE_LIFETIME")
    issued = store.issue(handle_for(base, timestamp=now[0], expires_at=now[0] + timedelta(seconds=30))).unwrap()
    resolve = lambda: store.resolve(issued, actor_id=base.actor.producer_id, task_id="t", operation="application.launch")  # noqa: E731
    assert resolve().unwrap().target == "/x"
    now[0] += timedelta(seconds=31)
    assert resolve() == Failure("HANDLE_EXPIRED")
    monkeypatch.setattr(handle_module, "MAX_HANDLES", 1)
    fresh = handle_for(base, timestamp=now[0], expires_at=now[0] + timedelta(minutes=1))
    survivor = store.issue(fresh).unwrap()  # expired rows are purged before the capacity check
    assert store.issue(handle_for(base, timestamp=now[0], expires_at=now[0] + timedelta(minutes=1))) == Failure("HANDLE_CAPACITY")
    assert store.revoke(survivor) == Success(None)
    assert store.resolve(survivor, actor_id="a", task_id="t", operation="x") == Failure("HANDLE_NOT_FOUND")


def test_handle_store_reports_corruption_and_unavailable_storage(tmp_path: Path) -> None:
    store = SQLiteHandleStore(tmp_path / "handles.db")
    base = request_for(runtime(tmp_path))
    issued = store.issue(handle_for(base)).unwrap()
    with sqlite3.connect(tmp_path / "handles.db") as connection:
        connection.execute("UPDATE handles SET payload = 'not json' WHERE id = ?", (issued,))
    assert store.resolve(issued, actor_id="a", task_id="t", operation="x") == Failure("HANDLE_STORE_UNAVAILABLE")
    broken = SQLiteHandleStore(tmp_path)
    assert broken.issue(handle_for(base)) == Failure("HANDLE_STORE_UNAVAILABLE")
    assert broken.resolve("x", actor_id="a", task_id="t", operation="x") == Failure("HANDLE_STORE_UNAVAILABLE")
    assert broken.revoke("x") == Failure("HANDLE_STORE_UNAVAILABLE")


# --- files and documents ----------------------------------------------------------------------------------------------


def test_inspect_document_accepts_only_plain_safe_documents_inside_roots(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    good = root / "note.TXT"
    good.write_text("x")
    assert inspect_document(good, (root,)) is not None
    assert inspect_document(good, (tmp_path / "other",)) is None
    assert inspect_document(root / "missing.txt", (root,)) is None
    assert inspect_document(root, (root,)) is None
    script = root / "run.txt"
    script.write_text("x")
    script.chmod(0o755)
    assert inspect_document(script, (root,)) is None
    binary = root / "tool.exe"
    binary.write_text("x")
    assert inspect_document(binary, (root,)) is None
    link = root / "link.txt"
    link.symlink_to(good)
    assert inspect_document(link, (root,)) is None


def test_search_is_bounded_and_ignores_links_and_unreadable_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    for name in ("a-report.txt", "nested/b-report.txt", "nested/c-report.txt"):
        (root / name).write_text("x")
    (root / "link-report.txt").symlink_to(root / "a-report.txt")
    assert [d.path.name for d in search_documents((tmp_path / "gone", root), "REPORT")] == [
        "a-report.txt", "b-report.txt", "c-report.txt"]
    monkeypatch.setattr(files, "MAX_RESULTS", 1)
    assert len(search_documents((root,), "report")) == 1
    monkeypatch.setattr(files, "MAX_RESULTS", 30)
    monkeypatch.setattr(files, "MAX_VISITED", 1)
    assert len(search_documents((root,), "report")) <= 1


def file_services(tmp_path: Path, backend: Any = None) -> tuple[FileSearch, DocumentOpen, Any, Path]:
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    registry = CapabilityRegistry()
    register_file_capabilities(registry, (root,), handles, backend)
    return (registry.descriptor("file.search").implementation if False else FileSearch((root,), handles),
            DocumentOpen((root,), handles, backend), registry, root)


async def test_file_search_validates_scope_and_reports_cancellation_or_issue_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search, _, registry, root = file_services(tmp_path)
    (root / "report.txt").write_text("x")
    make = lambda **arguments: normalize_request(registry.descriptor("file.search"), {"task_id": "t", "query": "report", **arguments})  # noqa: E731
    token = CancellationToken()
    assert (await search.invoke(make(), token)).unwrap().output["documents"][0]["name"] == "report.txt"
    for bad in ({"query": " "}, {"query": "q" * 257}, {"task_id": ""}):
        assert (await search.invoke(make(**bad), token)).failure().code == "INVALID_ARGUMENT"
    no_roots = FileSearch((), search.handles)
    assert (await no_roots.invoke(make(), token)).failure().code == "FILE_ROOTS_NOT_CONFIGURED"
    cancelled = CancellationToken()
    cancelled.cancel()
    assert (await search.invoke(make(), cancelled)).failure().code == "CANCELLED"
    monkeypatch.setattr(search.handles, "issue", lambda _handle: Failure("HANDLE_CAPACITY"))
    assert (await search.invoke(make(), token)).failure().code == "HANDLE_CAPACITY"


async def test_document_open_revalidates_before_activation(tmp_path: Path) -> None:
    opened: list[Any] = []

    class Backend:
        outcome: Any = Success({"open_file_descriptor": True})

        async def open(self, document: Any, token: CancellationToken) -> Any:
            opened.append(document)
            return self.outcome

    backend = Backend()
    search, opener, registry, root = file_services(tmp_path, backend)
    document = root / "report.txt"
    document.write_text("x")
    token = CancellationToken()
    listed = await search.invoke(normalize_request(registry.descriptor("file.search"), {"task_id": "t", "query": "report"}), token)
    handle = listed.unwrap().output["documents"][0]["handle"]
    open_request = lambda task="t", item=handle: normalize_request(  # noqa: E731
        registry.descriptor("document.open"), {"task_id": task, "handle": item})
    assert (await opener.invoke(open_request("other"), token)).failure().code == "HANDLE_SCOPE_DENIED"
    assert (await opener.invoke(open_request(), token)).unwrap().output["status"] == "SUCCEEDED"
    backend.outcome = Failure("DOCUMENT_OPEN_FAILED")
    assert (await opener.invoke(open_request(), token)).failure().code == "DOCUMENT_OPEN_FAILED"
    mismatch = (await opener.invoke(open_request().model_copy(update={"data_class": DataClass.PUBLIC}), token))
    assert mismatch.failure().code == "RESOURCE_CLASS_MISMATCH"
    calls = len(opened)
    document.write_text("changed content that alters size and mtime")
    assert (await opener.invoke(open_request(), token)).failure().code == "STALE_RESOURCE"
    document.unlink()
    assert (await opener.invoke(open_request(), token)).failure().code == "STALE_RESOURCE"
    assert len(opened) == calls
    other = opener.handles.issue(handle_for(open_request(), kind="process", operations=("document.open",))).unwrap()
    assert (await opener.invoke(open_request(item=other), token)).failure().code == "STALE_RESOURCE"
