"""Acceptance and privacy tests for the Stage 3 Rich History slice."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from returns.result import Success

from conftest import ASGITestClient
from rai.container import ApplicationContainer
from rai.history.collectors import (
    AtspiSemanticCollector,
    BrowserSemanticCollector,
    CollectorSupervisor,
    GnomeSessionCollector,
    QueueEventSource,
    SemanticCollector,
    JsonLinesSidecarSource,
    register_configured_collectors,
)
from rai.history.models import PrivacyAction, SourceEvent
from rai.history.privacy import PrivacyFirewall, PrivacyPolicy
from rai.history.service import RichHistoryService
from rai.history.storage import (
    EncryptedHistoryStore,
    KeyUnavailableError,
    RetentionPolicy,
    SecretServiceKeyProvider,
    StaticKeyProvider,
)
from rai.kernel.events import EventCursor
from rai.kernel.journal import SQLiteEventJournal
from rai.kernel.ports import CancellationToken
from rai.server import create_app

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
KEY = b"r" * 32
OK = 200
EXPECTED_OBSERVATIONS = 4
MAX_ATSPI_DEPTH = 32
EXPECTED_CHANGE_COUNT = 3
PRIVATE_FILE_MODE = 0o600


def service(tmp_path: Path, firewall: PrivacyFirewall | None = None) -> RichHistoryService:
    return RichHistoryService(
        SQLiteEventJournal(tmp_path / "journal.sqlite3"),
        EncryptedHistoryStore(
            tmp_path / "activity.sqlite3", key_provider=StaticKeyProvider(KEY)
        ),
        firewall=firewall,
        collection_enabled=True,
    )


@pytest.mark.asyncio
async def test_acceptance_slice_query_evidence_encryption_and_deletion(tmp_path: Path) -> None:
    history = service(
        tmp_path,
        PrivacyFirewall(PrivacyPolicy(allowed_origins=frozenset({"docs.example"}))),
    )
    events = (
        SourceEvent(
            source="browser", kind="navigation", timestamp=NOW,
            application_id="firefox.desktop", origin="docs.example",
            url="https://docs.example/rai", title="RAI design", resource_id="tab:7",
        ),
        SourceEvent(
            source="gnome", kind="active_window", timestamp=NOW + timedelta(seconds=10),
            application_id="org.gnome.terminal.desktop", title="RAI terminal",
            project="rai", resource_id="window:terminal",
        ),
        SourceEvent(
            source="atspi", kind="document_focus", timestamp=NOW + timedelta(seconds=20),
            application_id="code.desktop", project="rai", resource_id="file:src/rai/history.py",
            toolkit="gtk", quality=0.9,
        ),
        SourceEvent(
            source="process", kind="test_run", timestamp=NOW + timedelta(seconds=30),
            application_id="org.gnome.terminal.desktop", project="rai", resource_id="process:pytest",
            payload={"outcome": "passed"},
        ),
    )
    for event in events:
        assert isinstance(await history.ingest(event), Success)

    answer = history.answer_what_was_i_working_on()
    assert answer["remote_tokens"] == 0
    assert answer["applications"] == (
        "code.desktop", "firefox.desktop", "org.gnome.terminal.desktop",
    )
    assert answer["projects"] == ("rai",)
    assert answer["evidence"]
    assert all(item["observation_ids"] for item in answer["evidence"])
    assert b"RAI design" not in (tmp_path / "activity.sqlite3").read_bytes()

    deleted = await history.delete_range(NOW, NOW + timedelta(minutes=1))
    assert deleted["observations"] == EXPECTED_OBSERVATIONS
    assert deleted["verified"] is True
    assert history.query() == ()


@pytest.mark.asyncio
async def test_privacy_corpus_never_persists_dropped_content(tmp_path: Path) -> None:
    corpus = json.loads(
        (Path(__file__).parent / "fixtures/history/privacy-corpus.json").read_text()
    )
    firewall = PrivacyFirewall(
        PrivacyPolicy(
            excluded_applications=frozenset({"blocked.desktop"}),
            allowed_origins=frozenset({"docs.example"}),
        )
    )
    history = service(tmp_path, firewall)
    for fixture in corpus:
        event = SourceEvent(timestamp=NOW, **fixture["event"])
        decision = firewall.decide(event)
        assert decision.action == fixture["outcome"], fixture["name"]
        result = await history.ingest(event)
        assert isinstance(result, Success)
        assert (result.unwrap() is None) == (fixture["outcome"] == "DROP")

    journal = await history.journal.read(EventCursor.from_sequence(0), 100)
    assert isinstance(journal, Success)
    serialized = json.dumps(journal.unwrap().model_dump(mode="json"))
    disk = (tmp_path / "activity.sqlite3").read_bytes()
    for forbidden in ("hunter2", "private.example/secret", "confidential", "sk-example-secret", "ada@example.org"):
        assert forbidden not in serialized.casefold()
        assert forbidden.encode() not in disk.lower()
    assert "ignore previous instructions" in serialized


def test_adapter_contracts_strip_raw_input_and_private_selected_text() -> None:
    password = AtspiSemanticCollector.normalize({
        "kind": "text_change", "role": "password", "typed_text": "do-not-copy",
        "change_count": 3, "duration_ms": 500, "depth": 999,
    })
    assert "typed_text" not in password.payload
    assert password.depth == MAX_ATSPI_DEPTH
    assert password.payload["change_count"] == EXPECTED_CHANGE_COUNT

    browser = BrowserSemanticCollector.normalize({
        "tab_id": "4", "private": True, "selected_text": "secret", "user_requested": False,
    })
    assert browser.private_browsing is True
    assert browser.selected_text is None

    gnome = GnomeSessionCollector.normalize({
        "kind": "active_window", "desktop_entry_id": "Firefox.desktop", "title": "Page",
    })
    assert gnome.application_id == "firefox.desktop"


@pytest.mark.asyncio
async def test_supervisor_stops_immediately_and_isolates_failures() -> None:
    received: list[SourceEvent] = []
    source = QueueEventSource()

    async def receive(event: SourceEvent) -> None:
        received.append(event)

    supervisor = CollectorSupervisor(receive, base_backoff=0)
    supervisor.register(SemanticCollector("synthetic", source))
    await supervisor.set_controls(enabled=True)
    await source.emit(SourceEvent(source="gnome", kind="active_window"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(received) == 1
    await supervisor.set_controls(session_locked=True)
    assert supervisor.status()[0].enabled is False
    assert supervisor.status()[0].state == "STOPPED"


@pytest.mark.asyncio
async def test_production_sidecar_is_bounded_and_does_not_inherit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAI_API_TOKEN", "must-not-reach-collector")
    script = (
        "import json,os; print(json.dumps({"
        "'source':'gnome','kind':'active_window',"
        "'title':os.environ.get('RAI_API_TOKEN','not-inherited')}))"
    )
    source = JsonLinesSidecarSource((sys.executable, "-c", script), "gnome")
    token = CancellationToken()
    event = await anext(source.events(token))
    assert event.title == "not-inherited"


def test_configured_collectors_are_registered_without_starting_them() -> None:
    async def sink(_event: SourceEvent) -> None:
        return None

    supervisor = CollectorSupervisor(sink)
    register_configured_collectors(
        supervisor,
        {
            "collectors": {
                "gnome": {"command": ["gnome-sidecar"], "permission": "GRANTED"},
                "filesystem": {"command": ["fs-sidecar"], "roots": ["/tmp/project"]},
            }
        },
    )
    assert [item.name for item in supervisor.status()] == ["filesystem", "gnome"]


def test_retention_and_secret_service_unavailable_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("rai.history.storage.shutil.which", lambda _name: None)
    with pytest.raises(KeyUnavailableError, match="remains disabled"):
        SecretServiceKeyProvider().get_or_create()
    store = EncryptedHistoryStore(
        tmp_path / "retention.sqlite3", key_provider=StaticKeyProvider(KEY),
        retention=RetentionPolicy(observation_ttl=timedelta(seconds=1)),
    )
    assert store.path.stat().st_mode & 0o777 == PRIVATE_FILE_MODE
    assert store.enforce_retention(NOW) == {"observations": 0, "episodes": 0, "memories": 0}


def test_authenticated_review_api_uses_same_local_service(tmp_path: Path) -> None:
    container = ApplicationContainer(
        {}, testing=True, event_journal_path=tmp_path / "journal.sqlite3",
        rich_history_path=tmp_path / "activity.sqlite3",
        history_key_provider=StaticKeyProvider(KEY),
    )
    client = ASGITestClient(create_app(container))
    resumed = client.post("/api/v1/activity/collection", json={"action": "resume"})
    assert resumed.status_code == OK
    observed = client.post(
        "/api/v1/activity/observations",
        json={
            "source": "gnome", "kind": "active_window", "timestamp": NOW.isoformat(),
            "application_id": "code.desktop", "project": "rai",
        },
    )
    assert observed.status_code == OK
    answer = client.get(
        "/api/v1/activity/answer", params={"question": "what_was_i_working_on"}
    )
    assert answer.status_code == OK
    assert answer.json()["projects"] == ["rai"]
    deleted = client.request(
        "DELETE", "/api/v1/activity/episodes",
        json={"since": NOW.isoformat(), "until": (NOW + timedelta(seconds=1)).isoformat()},
    )
    assert deleted.status_code == OK
    assert deleted.json()["verified"] is True
