"""Desktop document activation with an independently observed file descriptor."""
from __future__ import annotations

import asyncio
import os
import json
from pathlib import Path
import shutil
import tempfile

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken
from .files import Document
from .applications import Application, read_application

OPEN_TIMEOUT = 10.0
POLL_INTERVAL = 0.1
MAX_PROCESSES = 4096
MAX_FDS = 1024
DESKTOP_ENVIRONMENT = frozenset({"HOME", "PATH", "LANG", "LANGUAGE", "LC_ALL", "DISPLAY", "WAYLAND_DISPLAY",
                               "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "XDG_DATA_DIRS", "XDG_DATA_HOME",
                               "XDG_CURRENT_DESKTOP", "XAUTHORITY", "XDG_SESSION_TYPE"})


def document_reader(document: Document, executable: str | None = None) -> dict | None:
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
                if executable is not None and str((process / "exe").resolve(strict=True)) != executable:
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
        return await self._open(document, cancellation)

    async def open_with_application(self, document: Document, application: Application,
                                    cancellation: CancellationToken) -> Result[dict, str]:
        current = await asyncio.to_thread(read_application, application.path, application.desktop_id)
        if current != application:
            return Failure("STALE_RESOURCE")
        return await self._open(document, cancellation, application)

    async def _open(self, document: Document, cancellation: CancellationToken,
                    application: Application | None = None) -> Result[dict, str]:
        gio = shutil.which("gio")
        if not gio or not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
            return Failure("DESKTOP_UNAVAILABLE")
        if cancellation.cancelled:
            return Failure("CANCELLED")
        receipt = tempfile.TemporaryDirectory(prefix="rai-document-")
        command = (("/usr/bin/python3", "-I", str(Path(__file__).with_name("launch_helper.py")),
                    str(application.path), application.fingerprint, str(Path(receipt.name) / "receipt.json"), document.path.as_uri())
                   if application else (gio, "open", document.path.as_uri()))
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env={key: value for key, value in os.environ.items() if key in DESKTOP_ENVIRONMENT},
        )
        try:
            deadline = asyncio.get_running_loop().time() + OPEN_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                if cancellation.cancelled:
                    return Failure("UNKNOWN")
                evidence = (await asyncio.to_thread(document_reader, document, application.executable)
                            if application else await asyncio.to_thread(document_reader, document))
                if evidence is None and document.path.is_dir():
                    evidence = await directory_reader(document, application, cancellation)
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
            receipt.cleanup()


async def directory_reader(document: Document, application: Application | None,
                           cancellation: CancellationToken) -> dict | None:
    """Require the requested location and independently identify its bus owner."""
    from .commands import run_command  # noqa: PLC0415

    response = await run_command("/usr/bin/python3", ("-I", str(Path(__file__).with_name("directory_helper.py")),
        document.path.as_uri()), cancellation)
    if isinstance(response, Failure):
        return None
    try:
        pid = int(json.loads(response.unwrap())["pid"])
        process = Path("/proc") / str(pid)
        if process.stat().st_uid != os.getuid():
            return None
        executable = str((process / "exe").resolve(strict=True))
        if application is not None and executable != application.executable:
            return None
        start = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
        return {"pid": pid, "start_time": start, "directory_location_observed": True,
                "document_fingerprint": document.fingerprint}
    except (ValueError, KeyError, OSError, IndexError):
        return None
