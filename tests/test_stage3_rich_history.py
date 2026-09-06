"""Acceptance and privacy tests for the Stage 3 Rich History slice."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from returns.result import Failure, Result, Success

from conftest import ASGITestClient
from rai.container import ApplicationContainer
from rai.config_manager import load_config, save_config
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
from rai.history.privacy import policy_from_config
from rai.history.service import RichHistoryService
from rai.history.storage import (
    EncryptedHistoryStore,
    KeyUnavailableError,
    RetentionPolicy,
    SecretServiceKeyProvider,
    StaticKeyProvider,
    retention_from_config,
)
from rai.history.sidecars.atspi import _metadata
from rai.history.sidecars.filesystem import (
    approved_roots,
    changed_paths,
    metadata_snapshot,
)
from rai.history.sidecars.gnome import EXTENSION_BUS, EXTENSION_PATH
from rai.history.sidecars.process import activity_kind, process_event
from rai.history.sidecars.install_gnome import install
from rai.kernel.events import EventCursor, JournalFailure
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
EXPECTED_SPLIT_EPISODES = 2


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


@pytest.mark.asyncio
async def test_untrusted_atspi_event_is_sanitized_again_at_api_boundary(
    tmp_path: Path,
) -> None:
    history = service(tmp_path)
    result = await history.ingest(
        SourceEvent(
            source="atspi", kind="text_activity", application_id="code.desktop",
            payload={
                "typed_text": "must-never-persist",
                "change_count": 4,
                "duration_ms": 100,
            },
        )
    )
    assert isinstance(result, Success)
    journal = await history.journal.read(EventCursor.from_sequence(0), 10)
    assert isinstance(journal, Success)
    serialized = json.dumps(journal.unwrap().model_dump(mode="json"))
    assert "must-never-persist" not in serialized
    assert '"change_count": 4' in serialized


def test_source_event_rejects_oversized_and_deep_payloads() -> None:
    with pytest.raises(ValueError, match="size limit"):
        SourceEvent(source="atspi", kind="focus", payload={"value": "x" * (65 * 1024)})
    nested: object = "leaf"
    for _index in range(10):
        nested = {"child": nested}
    with pytest.raises(ValueError, match="nesting limit"):
        SourceEvent(source="atspi", kind="focus", payload={"root": nested})


def test_password_role_variants_are_dropped() -> None:
    firewall = PrivacyFirewall()
    for role in ("password", "password text", "credential-field", "secret_input"):
        event = SourceEvent(source="atspi", kind="focus", field_role=role)
        assert firewall.decide(event).action == PrivacyAction.DROP


@pytest.mark.asyncio
async def test_failed_journal_erasure_leaves_history_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = service(tmp_path)
    assert isinstance(
        await history.ingest(
            SourceEvent(source="process", kind="foreground_process", timestamp=NOW)
        ),
        Success,
    )

    async def unavailable(_ids: tuple[str, ...]) -> Result[int, JournalFailure]:
        return Failure(
            JournalFailure(
                code="JOURNAL_UNAVAILABLE", message="unavailable", retryable=True
            )
        )

    monkeypatch.setattr(history.journal, "delete_observations", unavailable)
    with pytest.raises(RuntimeError, match="could not be verified"):
        await history.delete_range(NOW, NOW)
    assert len(history.store.observations()) == 1


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


@pytest.mark.asyncio
async def test_sidecar_reports_effective_unavailable_permission() -> None:
    source = JsonLinesSidecarSource(
        ("/definitely/not/a/rai-sidecar",), "gnome"
    )
    with pytest.raises(FileNotFoundError):
        await anext(source.events(CancellationToken()))
    assert source.permission == "UNAVAILABLE"


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


def test_filesystem_sidecar_reads_metadata_only_and_rejects_broad_root(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    secret_content = "content-must-never-be-read-or-emitted"
    tracked = project / "module.py"
    tracked.write_text(secret_content, encoding="utf-8")
    roots = approved_roots([str(project)])
    first = metadata_snapshot(roots)
    tracked.touch()
    second = metadata_snapshot(roots)
    assert list(changed_paths(first, second)) == [("save", tracked)]
    assert secret_content not in repr(second)
    with pytest.raises(ValueError, match="root cannot"):
        approved_roots([str(Path(Path.cwd().anchor))])


def test_filesystem_sidecar_emits_bounded_json_lines(tmp_path: Path) -> None:
    project = tmp_path / "rai"
    project.mkdir()
    source = project / "main.py"
    source.write_text("TOP_SECRET = 'never emitted'", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, "-m", "rai.history.sidecars.filesystem",
            "--once", str(project),
        ],
        check=True, capture_output=True, text=True, timeout=5,
    )
    event = json.loads(result.stdout)
    assert event["source"] == "filesystem"
    assert event["path"] == str(source)
    assert "TOP_SECRET" not in result.stdout


def test_process_sidecar_emits_no_arguments_or_environment(
    capfd: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    executable = tmp_path / "usr/bin/python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "42"
    process_dir.mkdir(parents=True)
    (process_dir / "exe").symlink_to(executable)
    (process_dir / "cmdline").write_text("python\x00--password=secret", encoding="utf-8")
    (process_dir / "environ").write_text("TOKEN=secret", encoding="utf-8")
    process_event("code.desktop", 42, proc_root)
    output = capfd.readouterr().out
    event = json.loads(output)
    assert event["payload"] == {"executable": "python"}
    assert "password" not in output
    assert "TOKEN" not in output


def test_atspi_metadata_is_bounded_without_accessible_text() -> None:
    class Application:
        @staticmethod
        def get_name() -> str:
            return "Code"

    class Accessible:
        def __init__(self, depth: int) -> None:
            self.depth = depth

        @staticmethod
        def get_application() -> Application:
            return Application()

        @staticmethod
        def get_attributes() -> dict[str, str]:
            return {"toolkit": "gtk", "DocURL": "file:///approved/main.py", "text": "secret"}

        @staticmethod
        def get_role_name() -> str:
            return "document text"

        @staticmethod
        def get_process_id() -> int:
            return 42

        def get_parent(self) -> "Accessible | None":
            return Accessible(self.depth - 1) if self.depth > 0 else None

    metadata = _metadata(Accessible(100))
    assert metadata["depth"] == MAX_ATSPI_DEPTH
    assert metadata["toolkit"] == "gtk"
    assert "text" not in metadata


def test_gnome_extension_exports_supported_focus_and_workspace_contract() -> None:
    extension = (
        Path("src/rai/history/gnome_extension/extension.js").read_text(encoding="utf-8")
    )
    assert EXTENSION_BUS in extension
    assert EXTENSION_PATH in extension
    assert "notify::focus-window" in extension
    assert "active-workspace-changed" in extension
    assert "get_core_idle_monitor" in extension
    assert "/dev/input" not in extension


def test_gnome_extension_installs_with_private_permissions(tmp_path: Path) -> None:
    target = install(tmp_path / "rai-history@tk-lab1")
    assert json.loads((target / "metadata.json").read_text())["uuid"] == "rai-history@tk-lab1"
    assert (target / "extension.js").stat().st_mode & 0o777 == PRIVATE_FILE_MODE


@pytest.mark.asyncio
async def test_native_lock_transition_suppresses_activity_until_unlock(
    tmp_path: Path,
) -> None:
    history = service(tmp_path)
    await history._collect(SourceEvent(source="gnome", kind="session_locked", session_locked=True))
    suppressed = await history.ingest(
        SourceEvent(source="atspi", kind="focus", application_id="code.desktop")
    )
    assert isinstance(suppressed, Success) and suppressed.unwrap() is None
    await history._collect(SourceEvent(source="gnome", kind="session_unlocked"))
    accepted = await history.ingest(
        SourceEvent(source="atspi", kind="focus", application_id="code.desktop")
    )
    assert isinstance(accepted, Success) and accepted.unwrap() is not None


def test_rich_history_collector_configuration_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    expected = {
        "enabled": True,
        "collectors": {"filesystem": {"command": ["rai-history-filesystem", "/work"], "roots": ["/work"]}},
    }
    save_config({"rich_history": expected}, str(path))
    assert load_config(str(path))["rich_history"] == expected


def test_privacy_and_retention_are_loaded_from_persistent_configuration(
    tmp_path: Path,
) -> None:
    policy = policy_from_config({
        "privacy": {
            "allowed_applications": ["code.desktop"],
            "excluded_origins": ["private.example"],
            "allowed_paths": [str(tmp_path)],
        }
    })
    assert policy.allowed_applications == frozenset({"code.desktop"})
    assert policy.excluded_origins == frozenset({"private.example"})
    assert policy.allowed_paths == (tmp_path,)
    retention = retention_from_config({
        "retention_days": {"raw": 1, "observations": 2, "episodes": 3, "memories": 4}
    })
    assert retention.observation_ttl == timedelta(days=2)
    assert retention.episode_ttl == timedelta(days=3)


@pytest.mark.asyncio
async def test_episode_builder_uses_delayed_application_and_resource_boundaries(
    tmp_path: Path,
) -> None:
    history = service(tmp_path)
    first = SourceEvent(
        source="gnome", kind="active_window", timestamp=NOW,
        application_id="firefox.desktop", resource_id="window:1",
    )
    second = SourceEvent(
        source="gnome", kind="active_window", timestamp=NOW + timedelta(minutes=2),
        application_id="code.desktop", resource_id="window:2",
    )
    assert isinstance(await history.ingest(first), Success)
    assert isinstance(await history.ingest(second), Success)
    assert len(history.query()) == EXPECTED_SPLIT_EPISODES


def test_process_executable_classifies_build_and_test_without_arguments() -> None:
    assert activity_kind("pytest") == "test_run"
    assert activity_kind("ninja") == "build"
    assert activity_kind("gnome-terminal-server") == "foreground_process"


def _dbus_service_available(destination: str, object_path: str) -> bool:
    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        return False
    result = subprocess.run(
        ["gdbus", "introspect", "--session", "--dest", destination,
         "--object-path", object_path],
        check=False, capture_output=True, text=True, timeout=2,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    "GNOME" not in os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    or not _dbus_service_available(EXTENSION_BUS, EXTENSION_PATH),
    reason="requires an active GNOME user session with the RAI History extension",
)
def test_live_gnome_sidecar_observes_current_session() -> None:
    result = subprocess.run(
        ["gdbus", "call", "--session", "--dest", EXTENSION_BUS,
         "--object-path", EXTENSION_PATH, "--method", f"{EXTENSION_BUS}.GetSnapshot"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    assert result.stdout.strip()


@pytest.mark.skipif(
    not Path("/usr/share/gir-1.0/Atspi-2.0.gir").exists()
    or not _dbus_service_available("org.a11y.Bus", "/org/a11y/bus"),
    reason="requires an active accessibility bus and AT-SPI introspection data",
)
def test_live_atspi_bus_is_available() -> None:
    result = subprocess.run(
        ["gdbus", "introspect", "--session", "--dest", "org.a11y.Bus",
         "--object-path", "/org/a11y/bus"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    assert "GetAddress" in result.stdout


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
            "source": "process", "kind": "foreground_process",
            "timestamp": NOW.isoformat(), "application_id": "code.desktop",
            "project": "rai", "payload": {"executable": "code"},
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
