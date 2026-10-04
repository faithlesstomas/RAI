"""Desktop document activation with an independently observed file descriptor."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from .files import Document

OPEN_TIMEOUT = 10.0
POLL_INTERVAL = 0.1
MAX_PROCESSES = 4096
MAX_FDS = 1024
DESKTOP_ENVIRONMENT = frozenset({"HOME", "PATH", "LANG", "LANGUAGE", "LC_ALL", "DISPLAY", "WAYLAND_DISPLAY",
                               "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "XDG_DATA_DIRS", "XDG_DATA_HOME",
                               "XDG_CURRENT_DESKTOP", "XAUTHORITY", "XDG_SESSION_TYPE"})


def document_reader(document: Document) -> dict | None:
    """Bounded metadata-only check: a current-user process holds this exact inode."""
    try:
        target = document.path.stat()
        for index, process in enumerate(Path("/proc").iterdir()):
            if index >= MAX_PROCESSES:
                break
            if not process.name.isdigit():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                for count, descriptor in enumerate((process / "fd").iterdir()):
                    if count >= MAX_FDS:
                        break
                    opened = descriptor.stat()
                    if (opened.st_dev, opened.st_ino) == (target.st_dev, target.st_ino):
                        start_time = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
                        return {"pid": int(process.name), "start_time": start_time,
                                "document_fingerprint": document.fingerprint, "open_file_descriptor": True}
            except (OSError, IndexError):
                continue
    except OSError:
        return None
    return None


class LinuxDocumentBackend:
    """Fail as UNKNOWN when the desktop does not expose verifiable open state."""

    async def open(self, document: Document, cancellation: CancellationToken) -> Result[dict, str]:
        gio = shutil.which("gio")
        if not gio or not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
            return Failure("DESKTOP_UNAVAILABLE")
        if cancellation.cancelled:
            return Failure("CANCELLED")
        process = await asyncio.create_subprocess_exec(
            gio, "open", document.path.as_uri(), stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={key: value for key, value in os.environ.items() if key in DESKTOP_ENVIRONMENT},
        )
        try:
            deadline = asyncio.get_running_loop().time() + OPEN_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                if cancellation.cancelled:
                    return Failure("UNKNOWN")
                evidence = await asyncio.to_thread(document_reader, document)
                if evidence is not None:
                    return Success(evidence)
                if process.returncode is not None and process.returncode != 0:
                    return Failure("DOCUMENT_OPEN_FAILED")
                await asyncio.sleep(POLL_INTERVAL)
            return Failure("UNKNOWN")
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
