"""Desktop-entry discovery and bounded, independently verified application launch."""

from __future__ import annotations

import asyncio
from configparser import ConfigParser, Error as ConfigError
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Protocol
from collections.abc import Iterator

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken

MAX_ENTRIES = 2048
MAX_ENTRY_BYTES = 65536
LAUNCH_TIMEOUT = 10.0
VERIFY_INTERVAL = 0.1


@dataclass(frozen=True)
class Application:
    """An installed desktop entry, never a model-generated command line."""

    desktop_id: str
    name: str
    path: Path
    fingerprint: str
    executable: str
    localized_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchEvidence:
    """Process identity independently observed after desktop activation."""

    pid: int
    start_time: str
    executable: str


class ApplicationBackend(Protocol):
    """Replaceable OS boundary, including real postcondition verification."""

    async def discover(self) -> Result[tuple[Application, ...], str]: ...

    async def launch(
        self, application: Application, cancellation: CancellationToken,
    ) -> Result[LaunchEvidence, str]: ...


def desktop_roots() -> tuple[Path, ...]:
    """Honor XDG precedence without searching arbitrary filesystem roots."""
    home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    system = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    return tuple(root / "applications" for root in (home, *(Path(p) for p in system.split(":")))
                 if root.is_absolute())


def read_application(path: Path, desktop_id: str) -> Application | None:  # noqa: PLR0911
    """Read bounded metadata; execution is delegated to desktop tooling."""
    try:
        with path.open("rb") as source:
            payload = source.read(MAX_ENTRY_BYTES + 1)
        if len(payload) > MAX_ENTRY_BYTES:
            return None
        parser = ConfigParser(interpolation=None, strict=True)
        parser.read_string(payload.decode("utf-8"))
        entry = parser["Desktop Entry"]
        if entry.get("Type") != "Application" or entry.getboolean("Hidden", False):
            return None
        if entry.getboolean("NoDisplay", False) or entry.getboolean("Terminal", False):
            return None
        argv = shlex.split(entry.get("Exec", ""))
        if not argv:
            return None
        executable = shutil.which(argv[0])
        if executable is None:
            return None
        return Application(desktop_id, entry.get("Name", desktop_id)[:256], path,
                           hashlib.sha256(payload).hexdigest(), str(Path(executable).resolve()),
                           tuple(value[:256] for key, value in entry.items() if key.startswith("name[")))
    except (OSError, UnicodeError, ConfigError, KeyError, ValueError):
        return None


def desktop_entries(roots: tuple[Path, ...]) -> Iterator[tuple[Path, str]]:
    """Bound all inspected entries, including directories and unrelated files."""
    remaining = MAX_ENTRIES
    for root in roots:
        pending = [root]
        while pending and remaining > 0:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        remaining -= 1
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.name.endswith(".desktop") and entry.is_file():
                            path = Path(entry.path)
                            yield path, str(path.relative_to(root)).replace("/", "-")
                        if remaining <= 0:
                            break
            except OSError:
                continue


def discover_applications(roots: tuple[Path, ...]) -> tuple[Application, ...]:
    """First XDG entry wins, including hidden overrides."""
    found: list[Application] = []
    seen: set[str] = set()
    for path, desktop_id in desktop_entries(roots):
        if desktop_id in seen:
            continue
        seen.add(desktop_id)
        application = read_application(path, desktop_id)
        if application is not None:
            found.append(application)
    return tuple(sorted(found, key=lambda app: app.desktop_id))


def matching_processes(executable: str) -> dict[int, LaunchEvidence]:
    """Read only current-user process identity, never cmdline or environment."""
    found: dict[int, LaunchEvidence] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            if str((path / "exe").resolve(strict=True)) != executable:
                continue
            # comm may contain spaces or parentheses; fields after its final ')' are stable.
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            start = fields[19]
            pid = int(path.name)
            found[pid] = LaunchEvidence(pid, start, executable)
        except (OSError, ValueError, IndexError):
            continue
    return found


class LinuxApplicationBackend:
    """Activate installed entries with gio and verify the matching process."""

    def __init__(self, roots: tuple[Path, ...] | None = None) -> None:
        self.roots = roots if roots is not None else desktop_roots()

    async def discover(self) -> Result[tuple[Application, ...], str]:
        return Success(await asyncio.to_thread(discover_applications, self.roots))

    async def launch(  # noqa: PLR0911, PLR0912
        self, application: Application, cancellation: CancellationToken,
    ) -> Result[LaunchEvidence, str]:
        gio = shutil.which("gio")
        if gio is None or not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
            return Failure("DESKTOP_UNAVAILABLE")
        current = await asyncio.to_thread(read_application, application.path, application.desktop_id)
        if current != application:
            return Failure("STALE_RESOURCE")
        if cancellation.cancelled:
            return Failure("CANCELLED")
        python = Path("/usr/bin/python3")
        if not python.is_file():
            return Failure("DESKTOP_UNAVAILABLE")
        allowed = {"HOME", "PATH", "LANG", "LANGUAGE", "LC_ALL", "DISPLAY", "WAYLAND_DISPLAY",
                   "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "XDG_DATA_DIRS", "XDG_DATA_HOME",
                   "XDG_CURRENT_DESKTOP", "XAUTHORITY", "XDG_SESSION_TYPE"}
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        receipt_dir = tempfile.TemporaryDirectory(prefix="rai-launch-")
        receipt_path = Path(receipt_dir.name) / "receipt.json"
        process = await asyncio.create_subprocess_exec(
            str(python), "-I", str(Path(__file__).with_name("launch_helper.py")),
            str(application.path), application.fingerprint, str(receipt_path),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env=environment, limit=4096,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=LAUNCH_TIMEOUT)
            if process.returncode != 0:
                return Failure("LAUNCH_FAILED")
            with receipt_path.open("rb") as receipt_file:
                receipt = json.loads(receipt_file.read(4096))
            pids = receipt.get("pids", [])
            if not receipt.get("accepted") or not pids:
                return Failure("UNKNOWN")
            deadline = asyncio.get_running_loop().time() + LAUNCH_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                if cancellation.cancelled:
                    return Failure("UNKNOWN")
                after = await asyncio.to_thread(matching_processes, application.executable)
                for pid in pids:
                    if pid in after:
                        return Success(after[pid])
                await asyncio.sleep(VERIFY_INTERVAL)
            return Failure("UNKNOWN")
        except (OSError, ValueError, asyncio.TimeoutError):
            return Failure("UNKNOWN")
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
            receipt_dir.cleanup()
