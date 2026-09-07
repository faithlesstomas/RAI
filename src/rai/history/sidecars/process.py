"""Foreground-process identity sourced from the GNOME focus-window signal."""

from __future__ import annotations

from pathlib import Path

from .common import emit
from .gnome import EXTENSION_BUS, EXTENSION_PATH

SNAPSHOT_FIELDS = 6
TEST_EXECUTABLES = frozenset({"pytest", "tox", "nox", "ctest"})
BUILD_EXECUTABLES = frozenset({"make", "ninja", "cmake", "cargo", "meson", "go"})


def activity_kind(executable: str) -> str:
    if executable in TEST_EXECUTABLES:
        return "test_run"
    if executable in BUILD_EXECUTABLES:
        return "build"
    return "foreground_process"


def process_event(application_id: str, pid: int, proc_root: Path = Path("/proc")) -> None:
    """Emit only PID, desktop identity and executable basename."""
    if pid <= 0:
        return
    try:
        executable = (proc_root / str(pid) / "exe").resolve(strict=True).name
    except OSError:
        executable = "unknown"
    emit(
        "process", activity_kind(executable), application_id=application_id or None,
        resource_id=f"process:{pid}", payload={"executable": executable},
    )


def run() -> None:
    try:
        import pydbus  # type: ignore[import-not-found]  # noqa: PLC0415
        from gi.repository import GLib  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("GNOME D-Bus bindings are unavailable") from exc
    shell = pydbus.SessionBus().get(EXTENSION_BUS, EXTENSION_PATH)

    def focused(application_id: str, _title: str, pid: int) -> None:
        process_event(application_id, int(pid))

    shell.ActiveWindowChanged.connect(focused)
    snapshot = tuple(shell.GetSnapshot())
    if len(snapshot) == SNAPSHOT_FIELDS:
        process_event(str(snapshot[3]), int(snapshot[5]))
    GLib.MainLoop().run()


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
