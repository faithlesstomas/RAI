"""Runtime diagnostics report outcomes without recording tool content."""
import logging
from pathlib import Path
import sqlite3

from rai.diagnostics import trace
from rai.history.diagnostics import history_storage_status

import pytest
from returns.result import Success

from rai.actions.activity import register_activity_capabilities
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.kernel.transport import normalize_request
from test_action_activity import MockRichHistory, _make_episode


@pytest.mark.asyncio
async def test_trace_excludes_history_content_and_query(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAI_TRACE", "1")
    history = MockRichHistory((_make_episode("ep", applications=("sensitive-application",)),))
    registry = CapabilityRegistry()
    register_activity_capabilities(registry, lambda: history)
    service = CapabilityService(registry, PolicyEngine(), InMemoryAuditLedger())
    request = normalize_request(registry.descriptor("activity.query"), {
        "task_id": "test", "query": "sensitive-application",
    })
    with caplog.at_level(logging.INFO, logger="rai.trace"):
        _, result = await service.invoke(request)
    assert isinstance(result, Success)
    assert "sensitive-application" not in caplog.text
    for event in ("capability.policy", "capability.start", "activity.query", "capability.end"):
        assert event in caplog.text
    assert "returned=1" in caplog.text
    assert "status=success" in caplog.text


def test_trace_is_opt_in(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.delenv("RAI_TRACE", raising=False)
    with caplog.at_level(logging.INFO, logger="rai.trace"):
        trace("hidden.event", returned=5)
    assert "hidden.event" not in caplog.text


def test_history_status_reads_only_aggregate_metadata(tmp_path: Path) -> None:

    path = tmp_path / "history.sqlite3"
    assert history_storage_status(path).unwrap() == {"exists": False}
    assert not path.exists()
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE observations(timestamp TEXT); CREATE TABLE episodes(ended_at TEXT);")
        db.execute("INSERT INTO observations VALUES (?)", ("2026-10-05T12:00:00+00:00",))
    status = history_storage_status(path).unwrap()
    assert status["observations"] == 1
    assert status["episodes"] == 0
    assert status["latest_observation"] == "2026-10-05T12:00:00+00:00"
    assert status["latest_episode"] is None
