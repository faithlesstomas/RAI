"""Desktop discovery preserves XDG overrides and never runs entry commands."""
# ruff: noqa
from pathlib import Path
import pytest
import shutil
from returns.result import Failure, Success
from rai.actions.applications import LaunchEvidence
from rai.actions.capabilities import register_application_capabilities
from rai.actions.handles import SQLiteHandleStore
from rai.kernel.audit import InMemoryAuditLedger
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.policy import PolicyEngine
from rai.kernel.service import CapabilityService
from rai.kernel.synthetic import SyntheticApprovalBroker
from rai.kernel.transport import normalize_request
from returns.result import Result
from rai.actions.applications import Application
from rai.kernel.ports import CancellationToken

from rai.actions.applications import discover_applications, read_application


def write_entry(path: Path, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("true")
    path.write_text(f"[Desktop Entry]\nType=Application\nName=Example\nExec={executable}\n{extra}")


def test_hidden_user_override_masks_system_application(tmp_path: Path) -> None:
    user, system = tmp_path / "user", tmp_path / "system"
    write_entry(system / "example.desktop")
    write_entry(user / "example.desktop", "Hidden=true\n")
    assert discover_applications((user, system)) == ()


def test_entry_fingerprint_changes_when_launch_target_changes(tmp_path: Path) -> None:
    path = tmp_path / "example.desktop"
    write_entry(path)
    before = read_application(path, path.name)
    assert before is not None
    write_entry(path, "Comment=changed\n")
    after = read_application(path, path.name)
    assert after is not None
    assert before.fingerprint != after.fingerprint


def test_invalid_or_terminal_entries_are_not_launch_candidates(tmp_path: Path) -> None:
    path = tmp_path / "example.desktop"
    write_entry(path, "Terminal=true\n")
    assert read_application(path, path.name) is None
    path.write_text("[Desktop Entry]\nType=Application\nExec=/missing/program\n")
    assert read_application(path, path.name) is None


async def test_launch_requires_policy_and_rejects_changed_entry(tmp_path: Path) -> None:

    path = tmp_path / "example.desktop"
    write_entry(path)

    class Backend:
        calls = 0

        async def discover(self) -> Result[tuple[Application, ...], str]:
            return Success(discover_applications((tmp_path,)))

        async def launch(self, app: Application, cancellation: CancellationToken) -> Result[LaunchEvidence, str]:
            self.calls += 1
            return Success(LaunchEvidence(123, "start-id", app.executable))

    backend = Backend()
    registry = CapabilityRegistry()
    register_application_capabilities(registry, backend, SQLiteHandleStore(tmp_path / "handles.db"))
    audit = InMemoryAuditLedger()
    service = CapabilityService(registry, PolicyEngine(), audit)
    listing = normalize_request(registry.descriptor("application.list"), {"task_id": "t"})
    _, listed = await service.invoke(listing)
    handle_id = listed.unwrap().output["applications"][0]["handle"]
    launch = normalize_request(registry.descriptor("application.launch"), {"task_id": "t", "handle": handle_id})
    _, denied = await service.invoke(launch)
    assert isinstance(denied, Failure)
    assert backend.calls == 0
    service.approvals = SyntheticApprovalBroker()
    _, approved = await service.invoke(launch)
    assert isinstance(approved, Success)
    assert approved.unwrap().verification["process"]["start_time"] == "start-id"
    write_entry(path, "Comment=changed\n")
    _, stale = await service.invoke(launch)
    assert stale.failure().code == "STALE_RESOURCE"
    assert backend.calls == 1


def test_localized_desktop_names_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / 'calculator.desktop'
    write_entry(path, 'Name[pl]=Kalkulator\nName[de]=Taschenrechner\n')
    application = read_application(path, path.name)
    assert application.localized_names == ('Kalkulator', 'Taschenrechner')


def test_discovery_respects_zero_budget_and_avoids_directory_links(tmp_path, monkeypatch):
    import rai.actions.applications as applications
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside'
    write_entry(outside / 'hidden.desktop')
    (root / 'linked-directory').symlink_to(outside, target_is_directory=True)
    assert discover_applications((root,)) == ()
    write_entry(root / 'visible.desktop')
    assert len(discover_applications((root,))) == 1
    monkeypatch.setattr(applications, 'MAX_ENTRIES', 0)
    assert discover_applications((root,)) == ()


async def test_unavailable_desktop_does_not_spawn(tmp_path, monkeypatch):
    from rai.actions.applications import LinuxApplicationBackend
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    path = tmp_path / 'example.desktop'
    write_entry(path)
    app = read_application(path, path.name)
    outcome = await LinuxApplicationBackend((tmp_path,)).launch(app, CancellationToken())
    assert outcome == Failure('DESKTOP_UNAVAILABLE')


async def test_application_list_validation_cancellation_and_issue_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rai.actions.capabilities import ApplicationList
    path = tmp_path / "example.desktop"
    write_entry(path)

    class Backend:
        async def discover(self) -> Result[tuple[Application, ...], str]:
            return Success((read_application(path, path.name),))  # type: ignore[arg-type]

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    app_list = ApplicationList(Backend(), handles)
    registry = CapabilityRegistry()
    register_application_capabilities(registry, Backend(), handles)

    # Missing task_id or long query
    req_no_task = normalize_request(registry.descriptor("application.list"), {"task_id": "", "query": "test"})
    assert (await app_list.invoke(req_no_task, CancellationToken())).failure().code == "INVALID_ARGUMENT"
    req_long = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": "x" * 300})
    assert (await app_list.invoke(req_long, CancellationToken())).failure().code == "INVALID_ARGUMENT"

    # Cancellation during iteration
    token = CancellationToken()
    token.cancel()
    req_ok = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": ""})
    assert (await app_list.invoke(req_ok, token)).failure().code == "CANCELLED"

    # Issue failure
    monkeypatch.setattr(handles, "issue", lambda _h: Failure("HANDLE_CAPACITY"))
    assert (await app_list.invoke(req_ok, CancellationToken())).failure().code == "HANDLE_CAPACITY"


async def test_application_launch_data_class_and_backend_failure(tmp_path: Path) -> None:
    from rai.actions.capabilities import ApplicationLaunch
    from rai.kernel.records import DataClass
    path = tmp_path / "example.desktop"
    write_entry(path)
    app = read_application(path, path.name)

    class Backend:
        launch_fail = False
        discover_fail = False

        async def discover(self) -> Result[tuple[Application, ...], str]:
            if self.discover_fail:
                return Failure("DISCOVER_ERROR")
            return Success((app,))  # type: ignore[arg-type]

        async def launch(self, a: Application, cancellation: CancellationToken) -> Result[LaunchEvidence, str]:
            if self.launch_fail:
                return Failure("LAUNCH_ERROR")
            return Success(LaunchEvidence(1, "start", a.executable))

    backend = Backend()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    launcher = ApplicationLaunch(backend, handles)
    registry = CapabilityRegistry()
    register_application_capabilities(registry, backend, handles)

    req_list = normalize_request(registry.descriptor("application.list"), {"task_id": "t"})
    res_list = await registry.resolve(req_list).unwrap().invoke(req_list, CancellationToken())
    handle = res_list.unwrap().output["applications"][0]["handle"]

    # Class mismatch
    req_launch = normalize_request(registry.descriptor("application.launch"), {"task_id": "t", "handle": handle})
    req_public = req_launch.model_copy(update={"data_class": DataClass.PUBLIC})
    assert (await launcher.invoke(req_public, CancellationToken())).failure().code == "RESOURCE_CLASS_MISMATCH"

    # Backend discover error
    backend.discover_fail = True
    assert (await launcher.invoke(req_launch, CancellationToken())).failure().code == "DISCOVER_ERROR"
    backend.discover_fail = False

    # Backend launch error
    backend.launch_fail = True
    assert (await launcher.invoke(req_launch, CancellationToken())).failure().code == "LAUNCH_ERROR"


def test_desktop_entry_categories_and_comments_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "vlc.desktop"
    write_entry(
        path,
        "Categories=AudioVideo;Player;Video;\n"
        "Comment=Media player\n"
        "GenericName=Video Player\n"
    )
    app = read_application(path, path.name)
    assert app is not None
    assert app.categories == ("AudioVideo", "Player", "Video")
    assert app.comment == "Media player"
    assert app.generic_name == "Video Player"


@pytest.mark.asyncio
async def test_application_list_matches_categories_and_generic_names(tmp_path: Path) -> None:
    from rai.actions.capabilities import ApplicationList
    path = tmp_path / "totem.desktop"
    write_entry(
        path,
        "Categories=AudioVideo;Video;Player;\n"
        "GenericName=Movie Player\n"
        "Comment=Play movies and video clips\n"
    )
    app = read_application(path, path.name)
    assert app is not None

    class Backend:
        async def discover(self) -> Result[tuple[Application, ...], str]:
            return Success((app,))

    backend = Backend()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    app_list = ApplicationList(backend, handles)
    registry = CapabilityRegistry()
    register_application_capabilities(registry, backend, handles)

    # Search by category 'video' (or 'wideo') should match
    req_video = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": "video"})
    res_video = (await app_list.invoke(req_video, CancellationToken())).unwrap()
    assert res_video.output["total_count"] == 1
    matched_app = res_video.output["applications"][0]
    assert matched_app["desktop_id"] == "totem.desktop"
    assert "Video" in matched_app["categories"]
    assert matched_app["generic_name"] == "Movie Player"

    # Search by Polish synonym 'wideo' should also match
    req_wideo = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": "wideo"})
    res_wideo = (await app_list.invoke(req_wideo, CancellationToken())).unwrap()
    assert res_wideo.output["total_count"] == 1

    # Search by Polish synonym 'odtwarzacz' should match 'Player'
    req_odtwarzacz = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": "odtwarzacz"})
    res_odtwarzacz = (await app_list.invoke(req_odtwarzacz, CancellationToken())).unwrap()
    assert res_odtwarzacz.output["total_count"] == 1
    assert res_odtwarzacz.output["complete"] is True
    assert res_odtwarzacz.output["truncated"] is False
    assert "reason" not in res_odtwarzacz.output


async def test_application_list_truncation_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rai.actions.capabilities import ApplicationList, MAX_APPLICATION_RESULTS
    path = tmp_path / "example.desktop"
    write_entry(path)
    app = read_application(path, path.name)
    assert app is not None

    class Backend:
        async def discover(self) -> Result[tuple[Application, ...], str]:
            # Generate more apps than MAX_APPLICATION_RESULTS
            apps = [
                Application(
                    desktop_id=f"app{i}.desktop",
                    name=f"App {i}",
                    executable=f"/bin/app{i}",
                    fingerprint=f"fp{i}",
                    path=Path(f"/bin/app{i}"),
                )
                for i in range(MAX_APPLICATION_RESULTS + 5)
            ]
            return Success(tuple(apps))

    handles = SQLiteHandleStore(tmp_path / "handles.db")
    app_list = ApplicationList(Backend(), handles)
    registry = CapabilityRegistry()
    register_application_capabilities(registry, Backend(), handles)

    req = normalize_request(registry.descriptor("application.list"), {"task_id": "t", "query": ""})
    res = (await app_list.invoke(req, CancellationToken())).unwrap()
    out = res.output
    assert out["truncated"] is True
    assert out["complete"] is False
    assert out["reason"] == "max_results_exceeded"
    assert out["total_count"] == MAX_APPLICATION_RESULTS + 5
    assert len(out["applications"]) == MAX_APPLICATION_RESULTS
