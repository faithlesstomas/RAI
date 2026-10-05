"""Read only a match for one requested URI; never emit other desktop locations."""
from __future__ import annotations

import json
import sys


def main() -> int:
    """Query FileManager1 via the system Python's optional GI bindings."""
    from gi.repository import Gio, GLib  # noqa: PLC0415  # pylint: disable=import-error

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    owner = bus.call_sync(
        "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
        "GetNameOwner", GLib.Variant("(s)", ("org.freedesktop.FileManager1",)),
        GLib.VariantType.new("(s)"), Gio.DBusCallFlags.NONE, 1000, None,
    ).unpack()[0]
    locations = bus.call_sync(
        owner, "/org/freedesktop/FileManager1", "org.freedesktop.DBus.Properties",
        "Get", GLib.Variant("(ss)", ("org.freedesktop.FileManager1", "OpenLocations")),
        GLib.VariantType.new("(v)"), Gio.DBusCallFlags.NONE, 1000, None,
    ).unpack()[0]
    if sys.argv[1].rstrip("/") not in {uri.rstrip("/") for uri in locations}:
        return 1
    pid = bus.call_sync(
        "org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus",
        "GetConnectionUnixProcessID", GLib.Variant("(s)", (owner,)),
        GLib.VariantType.new("(u)"), Gio.DBusCallFlags.NONE, 1000, None,
    ).unpack()[0]
    print(json.dumps({"pid": pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
