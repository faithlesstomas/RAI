"""Restart/retry must not execute an uncertain action a second time."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from returns.result import Failure, Success

from rai.actions.capabilities import result
from rai.actions.execution import SQLiteExecutionStore
from rai.kernel.defaults import create_default_capability_registry
from rai.kernel.transport import normalize_request


def test_reservation_survives_restart_and_rejects_request_substitution(tmp_path: Path) -> None:
    path = tmp_path / "executions.db"
    request = normalize_request(create_default_capability_registry().descriptor("test.echo"),
                                {"text": "test"}, request_id="same-id")
    assert SQLiteExecutionStore(path).reserve(request) == Success(None)
    restarted = SQLiteExecutionStore(path)
    assert restarted.reserve(request) == Failure("EXECUTION_UNKNOWN")
    substituted = request.model_copy(update={"arguments": {"text": "different"}})
    assert restarted.reserve(substituted) == Failure("REQUEST_ID_CONFLICT")
    terminal = result(request, {"status": "SUCCEEDED"}, {"checked": True})
    assert restarted.finish(request, terminal) == Success(None)
    assert restarted.reserve(request) == Success(terminal)
    assert restarted.finish(request, terminal) == Failure("TERMINAL_CONFLICT")


def test_concurrent_reservations_have_one_owner(tmp_path: Path) -> None:
    path = tmp_path / "executions.db"
    request = normalize_request(create_default_capability_registry().descriptor("test.echo"),
                                {"text": "test"}, request_id="same-id")
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(lambda _: SQLiteExecutionStore(path).reserve(request), range(2)))
    assert attempts.count(Success(None)) == 1
    assert attempts.count(Failure("EXECUTION_UNKNOWN")) == 1


def test_capacity_retains_old_identity_and_blocks_only_new_requests(tmp_path, monkeypatch):
    import rai.actions.execution as execution
    monkeypatch.setattr(execution, 'MAX_EXECUTION_RECORDS', 1)
    store = SQLiteExecutionStore(tmp_path / 'executions.db')
    descriptor = create_default_capability_registry().descriptor('test.echo')
    first = normalize_request(descriptor, {'text': 'first'}, request_id='first')
    assert store.reserve(first) == Success(None)
    terminal = result(first, {'status': 'SUCCEEDED'}, {'checked': True})
    assert store.finish(first, terminal) == Success(None)
    second = normalize_request(descriptor, {'text': 'second'}, request_id='second')
    assert store.reserve(second) == Failure('EXECUTION_CAPACITY_EXCEEDED')
    assert store.reserve(first) == Success(terminal)


def test_oversized_records_cannot_expand_store_without_bound(tmp_path):
    from rai.actions.execution import MAX_REQUEST_BYTES, MAX_TERMINAL_BYTES
    store = SQLiteExecutionStore(tmp_path / 'executions.db')
    descriptor = create_default_capability_registry().descriptor('test.echo')
    oversized = normalize_request(descriptor, {'text': 'x' * MAX_REQUEST_BYTES})
    assert store.reserve(oversized) == Failure('REQUEST_SIZE_LIMIT')
    request = normalize_request(descriptor, {'text': 'normal'})
    assert store.reserve(request) == Success(None)
    terminal = result(request, {'text': 'x' * MAX_TERMINAL_BYTES}, {'checked': True})
    assert store.finish(request, terminal) == Failure('TERMINAL_SIZE_LIMIT')
    assert store.reserve(request) == Failure('EXECUTION_UNKNOWN')
