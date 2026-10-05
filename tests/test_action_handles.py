"""Authority survives restart but cannot cross scope, expiry or revocation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from returns.result import Failure, Success

from rai.actions.handles import SQLiteHandleStore
from rai.actions.records import ResourceHandle
from rai.kernel.records import ProducerIdentity

NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def handle() -> ResourceHandle:
    return ResourceHandle(
        record_id="opaque-id", timestamp=NOW,
        producer=ProducerIdentity(producer_id="runtime", kind="runtime", version="1.0.0"),
        actor_id="alice", task_id="launch-task", kind="application",
        target="org.example.Editor.desktop", fingerprint="entry-digest",
        operations=("application.launch",), expires_at=NOW + timedelta(minutes=5),
    )


def test_handle_restart_scope_and_revocation(tmp_path: Path) -> None:
    path = tmp_path / "authority.sqlite3"
    store = SQLiteHandleStore(path, lambda: NOW)
    assert isinstance(store.issue(handle()), Success)
    restarted = SQLiteHandleStore(path, lambda: NOW)
    scope = dict(actor_id="alice", task_id="launch-task", operation="application.launch")
    assert restarted.resolve("opaque-id", **scope).unwrap() == handle()
    for field, value in (("actor_id", "bob"), ("task_id", "other"), ("operation", "shell.run")):
        assert restarted.resolve("opaque-id", **(scope | {field: value})) == Failure("HANDLE_SCOPE_DENIED")
    assert isinstance(restarted.revoke("opaque-id"), Success)
    assert restarted.resolve("opaque-id", **scope) == Failure("HANDLE_NOT_FOUND")
    assert path.stat().st_mode & 0o777 == 0o600  # noqa: PLR2004


def test_expired_and_duplicate_handles_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "authority.sqlite3"
    store = SQLiteHandleStore(path, lambda: NOW)
    assert isinstance(store.issue(handle()), Success)
    substituted = handle().model_copy(update={"target": "malicious.desktop"})
    assert isinstance(store.issue(substituted), Failure)
    expired = SQLiteHandleStore(path, lambda: NOW + timedelta(minutes=6))
    assert expired.resolve("opaque-id", actor_id="alice", task_id="launch-task",
                           operation="application.launch") == Failure("HANDLE_EXPIRED")


def test_schema_rejects_ambiguous_expiry_and_unsupported_version() -> None:
    values = handle().model_dump()
    for update in ({"expires_at": NOW.replace(tzinfo=None)}, {"schema_version": "2.0.0"},
                   {"expires_at": NOW}, {"data_class": "SECRET"}):
        with pytest.raises(ValidationError):
            ResourceHandle.model_validate(values | update)
