"""GNOME Shell and session-state D-Bus event producer."""

from __future__ import annotations

from typing import Any

from .common import emit

EXTENSION_BUS = "org.richai.History"
EXTENSION_PATH = "/org/richai/History"
IDLE_THRESHOLD_MS = 60_000
SNAPSHOT_FIELDS = 6


def _window(application_id: str, title: str, pid: int) -> None:
    emit(
        "gnome", "active_window", application_id=application_id or None,
        title=title or None, resource_id=f"window:{pid}" if pid > 0 else None,
    )


def _workspace(index: int) -> None:
    emit("gnome", "workspace_changed", payload={"workspace": int(index)})


def _locked(active: bool) -> None:
    emit("gnome", "session_locked" if active else "session_unlocked", session_locked=active)


def run() -> None:
    """Subscribe only to supported user-session D-Bus interfaces."""
    try:
        import pydbus  # type: ignore[import-not-found]  # noqa: PLC0415
        from gi.repository import GLib  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("GNOME D-Bus bindings are unavailable") from exc

    bus = pydbus.SessionBus()
    screen_saver = bus.get("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver")
    shell = bus.get(EXTENSION_BUS, EXTENSION_PATH)
    idle_monitor = bus.get(
        "org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core"
    )
    screen_saver.ActiveChanged.connect(_locked)
    shell.ActiveWindowChanged.connect(_window)
    shell.WorkspaceChanged.connect(_workspace)
    last_idle: list[bool | None] = [None]

    def poll_idle() -> bool:
        idle = int(idle_monitor.GetIdletime()) >= IDLE_THRESHOLD_MS
        if idle != last_idle[0]:
            emit("gnome", "idle" if idle else "active", payload={"idle": idle})
            last_idle[0] = idle
        return True

    snapshot: tuple[Any, ...] = tuple(shell.GetSnapshot())
    if len(snapshot) == SNAPSHOT_FIELDS:
        locked, _idle, workspace, app_id, title, pid = snapshot
        _locked(bool(locked))
        _workspace(int(workspace))
        _window(str(app_id), str(title), int(pid))
    GLib.timeout_add(1000, poll_idle)
    poll_idle()
    GLib.MainLoop().run()


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
