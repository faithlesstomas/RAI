"""Optional system-Python/GIO helper; emits PID identity, never action authority.

Executed with Python isolated mode and a sanitized desktop environment. GIO
interprets desktop-entry syntax, so model output is never a shell command.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

MAX_ENTRY_BYTES = 65536


def main() -> int:
    """Launch exactly the approved desktop entry revision using GIO."""
    from gi.repository import Gio, GLib  # noqa: PLC0415  # pylint: disable=import-error

    path, expected = Path(sys.argv[1]), sys.argv[2]
    with path.open("rb") as source:
        payload = source.read(MAX_ENTRY_BYTES + 1)
    if len(payload) > MAX_ENTRY_BYTES or hashlib.sha256(payload).hexdigest() != expected:
        return 2
    app = Gio.DesktopAppInfo.new_from_filename(str(path))
    if app is None:
        return 2
    pids: list[int] = []

    def launched(_app: object, pid: int, _data: object) -> None:
        pids.append(pid)

    accepted = app.launch_uris_as_manager(
        [], Gio.AppLaunchContext(), GLib.SpawnFlags.SEARCH_PATH,
        None, None, launched, None,
    )
    if accepted and app.get_boolean("DBusActivatable"):
        try:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            reply = bus.call_sync(
                "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
                "GetConnectionUnixProcessID", GLib.Variant("(s)", (path.stem,)),
                GLib.VariantType.new("(u)"), Gio.DBusCallFlags.NONE, 3000, None,
            )
            owner_pid = reply.unpack()[0]
            if owner_pid not in pids:
                pids.insert(0, owner_pid)
        except GLib.Error:
            pass
    Path(sys.argv[3]).write_text(json.dumps({"accepted": bool(accepted), "pids": pids[:8]}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
