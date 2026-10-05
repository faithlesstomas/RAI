"""Real application launch backend: bounded helper, receipt validation and process verification."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from returns.result import Failure, Success

import rai.actions.applications as applications
from rai.actions.applications import (
    Application, LaunchEvidence, LinuxApplicationBackend, desktop_roots, discover_applications,
    matching_processes, read_application,
)
from rai.actions import launch_helper
from rai.kernel.ports import CancellationToken

EVIDENCE = LaunchEvidence(321, "start", "/usr/bin/example")


def write_entry(path: Path, body: str = "") -> Application:
    path.write_text(f"[Desktop Entry]\nType=Application\nName=Example\nExec={shutil.which('true')}\n{body}")
    application = read_application(path, path.name)
    assert application is not None
    return application


class FakeProcess:
    returncode: int | None = None

    def __init__(self, code: int = 0, hang: bool = False) -> None:
        self.code, self.hang, self.killed = code, hang, False

    async def wait(self) -> int:
        if self.hang and not self.killed:
            await asyncio.Event().wait()
        self.returncode = -9 if self.killed else self.code
        return self.returncode

    def kill(self) -> None:
        self.killed = True


@pytest.fixture
def desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A graphical session with a gio executable; no real subprocess is spawned."""
    monkeypatch.setenv("DISPLAY", ":0")
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/usr/bin/gio" if name == "gio" else real(name, *a, **k))
    monkeypatch.setattr(applications, "matching_processes", lambda _executable: {EVIDENCE.pid: EVIDENCE})


def spawn(monkeypatch: pytest.MonkeyPatch, process: FakeProcess, receipt: Any = None, raw: bytes | None = None) -> list[Any]:
    calls: list[Any] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        calls.append((args, kwargs))
        if raw is not None:
            Path(args[5]).write_bytes(raw)
        elif receipt is not None:
            Path(args[5]).write_text(json.dumps(receipt))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


async def test_verified_launch_passes_only_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    monkeypatch.setenv("RAI_SECRET_TOKEN", "must-not-leak")
    calls = spawn(monkeypatch, FakeProcess(), {"accepted": True, "pids": [EVIDENCE.pid]})
    outcome = await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken())
    assert outcome == Success(EVIDENCE)
    args, kwargs = calls[0]
    assert args[1] == "-I" and args[3] == str(app.path) and args[4] == app.fingerprint
    assert "RAI_SECRET_TOKEN" not in kwargs["env"] and kwargs["env"]["DISPLAY"] == ":0"


async def test_helper_exit_failure_is_launch_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None) -> None:
    app = write_entry(tmp_path / "example.desktop")
    spawn(monkeypatch, FakeProcess(code=2))
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("LAUNCH_FAILED")


@pytest.mark.parametrize("receipt", [{"accepted": False, "pids": [1]}, {"accepted": True, "pids": []}, {}])
async def test_unaccepted_or_pidless_receipt_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None, receipt: dict,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    spawn(monkeypatch, FakeProcess(), receipt)
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("UNKNOWN")


@pytest.mark.parametrize("raw", [b"not json", None])
async def test_missing_or_corrupt_receipt_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None, raw: bytes | None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    spawn(monkeypatch, FakeProcess(), raw=raw)
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("UNKNOWN")


async def test_unobserved_pid_is_unknown_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    monkeypatch.setattr(applications, "matching_processes", lambda _executable: {})
    monkeypatch.setattr(applications, "LAUNCH_TIMEOUT", 0.05)
    monkeypatch.setattr(applications, "VERIFY_INTERVAL", 0.01)
    spawn(monkeypatch, FakeProcess(), {"accepted": True, "pids": [999999]})
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("UNKNOWN")


async def test_cancellation_while_verifying_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    token = CancellationToken()
    token.cancel()
    spawn(monkeypatch, FakeProcess(), {"accepted": True, "pids": [EVIDENCE.pid]})
    # Cancellation before the helper starts must not spawn it at all.
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, token) == Failure("CANCELLED")
    late = CancellationToken()
    real_exec = asyncio.create_subprocess_exec

    async def cancelling_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        process = await real_exec(*args, **kwargs)
        late.cancel()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", cancelling_exec)
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, late) == Failure("UNKNOWN")


async def test_hung_helper_is_killed_and_reported_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    process = FakeProcess(hang=True)
    spawn(monkeypatch, process)
    monkeypatch.setattr(applications, "LAUNCH_TIMEOUT", 0.05)
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("UNKNOWN")
    assert process.killed


async def test_missing_system_python_is_desktop_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: False if str(self) == "/usr/bin/python3" else original(self))
    calls = spawn(monkeypatch, FakeProcess())
    assert await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken()) == Failure("DESKTOP_UNAVAILABLE")
    assert not calls


async def test_missing_gio_or_changed_entry_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: None,
) -> None:
    app = write_entry(tmp_path / "example.desktop")
    calls = spawn(monkeypatch, FakeProcess())
    write_entry(tmp_path / "example.desktop", "Comment=changed\n")
    backend = LinuxApplicationBackend((tmp_path,))
    assert await backend.launch(app, CancellationToken()) == Failure("STALE_RESOURCE")
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    assert await backend.launch(app, CancellationToken()) == Failure("DESKTOP_UNAVAILABLE")
    assert not calls


async def test_discover_wraps_catalog_in_success(tmp_path: Path) -> None:
    write_entry(tmp_path / "example.desktop")
    discovered = await LinuxApplicationBackend((tmp_path,)).discover()
    assert [app.desktop_id for app in discovered.unwrap()] == ["example.desktop"]
    assert LinuxApplicationBackend().roots == desktop_roots()


def test_desktop_roots_follow_xdg_and_drop_relative_directories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "/home/test/data")
    monkeypatch.setenv("XDG_DATA_DIRS", "/opt/share:relative/share")
    assert desktop_roots() == (Path("/home/test/data/applications"), Path("/opt/share/applications"))


def test_non_launchable_entries_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.desktop"
    path.write_text("[Desktop Entry]\nType=Application\nExec=" + str(shutil.which("true")) + "\nNoDisplay=true\n")
    assert read_application(path, path.name) is None
    path.write_text("[Desktop Entry]\nType=Application\nExec=\n")
    assert read_application(path, path.name) is None
    path.write_text("[Desktop Entry]\nType=Link\nExec=true\n")
    assert read_application(path, path.name) is None
    path.write_text("not an ini file")
    assert read_application(path, path.name) is None
    path.write_bytes(b"[Desktop Entry]\n" + b"#" * (applications.MAX_ENTRY_BYTES + 1))
    assert read_application(path, path.name) is None
    assert read_application(tmp_path / "missing.desktop", "missing.desktop") is None


def test_discovery_stops_at_budget_and_ignores_unreadable_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_entry(tmp_path / "a.desktop")
    write_entry(tmp_path / "b.desktop")
    monkeypatch.setattr(applications, "MAX_ENTRIES", 1)
    assert len(discover_applications((tmp_path / "missing", tmp_path))) <= 1


def test_matching_processes_identifies_current_user_executable() -> None:
    executable = str(Path(sys.executable).resolve())
    found = matching_processes(executable)
    assert os.getpid() in found
    assert found[os.getpid()].executable == executable
    assert matching_processes("/nonexistent/program") == {}


# --- GIO helper -----------------------------------------------------------------------------------------------


class GioError(Exception):
    """Stand-in for GLib.Error."""


def install_gi(monkeypatch: pytest.MonkeyPatch, *, app: Any, bus: Any = None) -> None:
    class KeyFile:
        def load_from_data(self, payload, length, flags):
            assert payload.startswith("[Desktop Entry]")
            assert length == len(payload.encode())
        def set_boolean(self, group, name, value):
            assert (group, name, value) == ("Desktop Entry", "DBusActivatable", False)

    glib = SimpleNamespace(
        KeyFile=KeyFile, KeyFileFlags=SimpleNamespace(NONE=0),
        SpawnFlags=SimpleNamespace(SEARCH_PATH=1), Error=GioError,
        Variant=lambda *a: a, VariantType=SimpleNamespace(new=lambda signature: signature),
    )
    gio = SimpleNamespace(
        DesktopAppInfo=SimpleNamespace(new_from_filename=lambda _path: app, new_from_keyfile=lambda _keyfile: app),
        AppLaunchContext=lambda: object(), BusType=SimpleNamespace(SESSION=0),
        DBusCallFlags=SimpleNamespace(NONE=0), bus_get_sync=lambda *_a: bus,
    )
    repository = ModuleType("gi.repository")
    repository.Gio, repository.GLib = gio, glib  # type: ignore[attr-defined]
    package = ModuleType("gi")
    package.repository = repository  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gi", package)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)


class FakeDesktopApp:
    def __init__(self, accepted: bool = True, pid: int | None = 42, dbus: bool = False) -> None:
        self.accepted, self.pid, self.dbus = accepted, pid, dbus

    def launch_uris_as_manager(self, _uris: Any, _context: Any, _flags: Any, _a: Any, _b: Any,
                               callback: Any, data: Any) -> bool:
        self.uris = _uris
        if self.pid is not None:
            callback(self, self.pid, data)
        return self.accepted

    def get_boolean(self, key: str) -> bool:
        return key == "DBusActivatable" and self.dbus


def run_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, digest: str | None = None, uri: str | None = None) -> tuple[int, Path]:
    entry = tmp_path / "example.desktop"
    if not entry.exists():
        entry.write_text("[Desktop Entry]\nType=Application\nName=Example\nExec=true\n")
    import hashlib  # noqa: PLC0415
    expected = digest or hashlib.sha256(entry.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(sys, "argv", ["helper", str(entry), expected, str(receipt), *([uri] if uri else [])])
    return launch_helper.main(), receipt


def test_helper_launches_exact_revision_and_reports_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_gi(monkeypatch, app=FakeDesktopApp())
    code, receipt = run_helper(monkeypatch, tmp_path)
    assert code == 0
    assert json.loads(receipt.read_text()) == {"accepted": True, "pids": [42]}


def test_helper_rejects_changed_or_oversized_entry_without_launching(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = FakeDesktopApp()
    install_gi(monkeypatch, app=app)
    assert run_helper(monkeypatch, tmp_path, digest="0" * 64)[0] == 2
    (tmp_path / "example.desktop").write_bytes(b"#" * (launch_helper.MAX_ENTRY_BYTES + 1))
    assert run_helper(monkeypatch, tmp_path)[0] == 2
    assert not (tmp_path / "receipt.json").exists()


def test_helper_rejects_unparseable_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install_gi(monkeypatch, app=None)
    assert run_helper(monkeypatch, tmp_path)[0] == 2


def test_helper_resolves_dbus_activated_application_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Bus:
        def call_sync(self, *_a: Any) -> Any:
            return SimpleNamespace(unpack=lambda: (77,))

    install_gi(monkeypatch, app=FakeDesktopApp(pid=None, dbus=True), bus=Bus())
    code, receipt = run_helper(monkeypatch, tmp_path)
    assert code == 0
    assert json.loads(receipt.read_text())["pids"] == [77]


def test_helper_tolerates_dbus_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Bus:
        def call_sync(self, *_a: Any) -> Any:
            raise GioError

    install_gi(monkeypatch, app=FakeDesktopApp(pid=None, dbus=True), bus=Bus())
    code, receipt = run_helper(monkeypatch, tmp_path)
    assert code == 0
    assert json.loads(receipt.read_text()) == {"accepted": True, "pids": []}


def test_helper_uses_dbus_owner_even_when_launcher_pid_was_reported(monkeypatch, tmp_path):
    class Bus:
        def call_sync(self, *_args):
            return SimpleNamespace(unpack=lambda: (77,))

    install_gi(monkeypatch, app=FakeDesktopApp(pid=42, dbus=True), bus=Bus())
    code, receipt = run_helper(monkeypatch, tmp_path)
    assert code == 0
    assert json.loads(receipt.read_text())['pids'] == [77, 42]


def test_document_helper_delivers_uri_from_verified_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app = FakeDesktopApp()
    install_gi(monkeypatch, app=app)
    uri = "file:///tmp/rai-test/a%20b.txt"
    code, receipt = run_helper(monkeypatch, tmp_path, uri=uri)
    assert code == 0
    assert app.uris == [uri]
    assert json.loads(receipt.read_text())["accepted"]
