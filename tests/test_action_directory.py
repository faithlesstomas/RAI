"""Directory metadata and chosen-application requests retain authority boundaries."""
from pathlib import Path

from returns.result import Failure, Success

from rai.actions.applications import Application
from rai.actions.capabilities import register_application_capabilities
from rai.actions.files import register_file_capabilities
from rai.actions.handles import SQLiteHandleStore
from rai.kernel.capabilities import CapabilityRegistry
from rai.kernel.ports import CancellationToken
from rai.kernel.transport import normalize_request


async def test_directory_listing_is_bounded_and_does_not_follow_links(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "subdirectory").mkdir()
    (allowed / "code.py").write_text("private content")
    (allowed / "link").symlink_to(tmp_path)
    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    register_file_capabilities(registry, (allowed,), handles, object())
    descriptor = registry.descriptor("file.list")
    # Use the registered implementation; listing has no desktop dependencies.
    async def invoke(path):
        request = normalize_request(descriptor, {"query": str(path), "task_id": "t"})
        return await registry.resolve(request).unwrap().invoke(request, CancellationToken())
    listed = (await invoke(allowed)).unwrap().output
    assert {entry["kind"] for entry in listed["entries"]} == {"directory", "file", "symlink"}
    assert "private content" not in str(listed)
    assert isinstance(await invoke(allowed / "link"), Failure)
    assert isinstance(await invoke(tmp_path), Failure)
    for index in range(40):
        (allowed / f"note-{index}.txt").touch()
    limited = (await invoke(allowed)).unwrap().output
    assert limited["truncated"]
    assert len(limited["entries"]) <= 30


async def test_selected_application_is_scoped_and_revalidated(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    document = allowed / "note.txt"
    document.touch()
    application = Application("editor.desktop", "Editor", tmp_path / "editor.desktop", "revision", "/editor")

    class Apps:
        current = application
        async def discover(self):
            return Success((self.current,))

    class Documents:
        opened = []
        async def open(self, *args):
            raise AssertionError("Selected application must not fall back to default")
        async def open_with_application(self, target, app, cancellation):
            self.opened.append((target.path, app.desktop_id))
            return Success({"open_file_descriptor": True})

    registry = CapabilityRegistry()
    handles = SQLiteHandleStore(tmp_path / "handles.db")
    apps, backend = Apps(), Documents()
    register_application_capabilities(registry, apps, handles)
    register_file_capabilities(registry, (allowed,), handles, backend, apps)
    async def invoke(name, arguments):
        request = normalize_request(registry.descriptor(name), arguments)
        return await registry.resolve(request).unwrap().invoke(request, CancellationToken())
    app = (await invoke("application.list", {"task_id": "t", "query": "Editor"})).unwrap().output["applications"][0]
    for path in (document, allowed):
        file = (await invoke("file.search", {"task_id": "t", "query": str(path)})).unwrap().output["documents"][0]
        arguments = {"task_id": "t", "handle": file["handle"], "application_handle": app["handle"]}
        assert isinstance(await invoke("document.open", arguments), Success)
        assert backend.opened[-1] == (path, "editor.desktop")
        assert (await invoke("document.open", {**arguments, "task_id": "other"})).failure().code == "HANDLE_SCOPE_DENIED"
    other_app = (await invoke("application.list", {"task_id": "other", "query": "Editor"})).unwrap().output["applications"][0]
    assert (await invoke("document.open", {**arguments, "application_handle": other_app["handle"]})).failure().code == "HANDLE_SCOPE_DENIED"
    apps.current = Application("editor.desktop", "Editor", application.path, "changed", "/editor")
    assert (await invoke("document.open", arguments)).failure().code == "STALE_RESOURCE"
    assert len(backend.opened) == 2
