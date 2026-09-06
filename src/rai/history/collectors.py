"""Bounded platform adapters and a failure-isolating collector supervisor."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from returns.result import Failure, Result, Success

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, Observation, ProducerIdentity

from .models import CollectorHealth, SourceEvent, utc_now

MAX_SIDECAR_EVENT_BYTES = 64 * 1024

COLLECTOR_PRODUCER = ProducerIdentity(
    producer_id="rai.collector-supervisor", kind="runtime", version="1.0.0"
)


class SemanticEventSource(Protocol):
    """Unprivileged semantic source implemented by DBus/native adapters."""

    @property
    def permission(self) -> str: ...

    def events(self, cancellation: CancellationToken) -> AsyncIterator[SourceEvent]: ...


class SemanticCollector:
    """Collector port adapter around an unprivileged semantic event source."""

    def __init__(self, name: str, source: SemanticEventSource) -> None:
        self.name = name
        self.source = source
        self._state = LifecycleState.CREATED

    @property
    def state(self) -> LifecycleState:
        return self._state

    @property
    def permission(self) -> str:
        return self.source.permission

    async def start(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.RUNNING
        return Success(self._state)

    async def stop(self) -> Result[LifecycleState, ActionFailure]:
        self._state = LifecycleState.STOPPED
        return Success(self._state)

    async def source_events(self, cancellation: CancellationToken) -> AsyncIterator[SourceEvent]:
        async for event in self.source.events(cancellation):
            sanitized = self.sanitize(event)
            if sanitized is not None:
                yield sanitized

    def sanitize(self, event: SourceEvent) -> SourceEvent | None:
        """Re-validate an untrusted sidecar event at the daemon boundary."""
        return event

    async def events(
        self, cancellation: CancellationToken
    ) -> AsyncIterator[Result[Observation, ActionFailure]]:
        """The supervisor uses ``source_events`` so privacy runs before Observation."""
        del cancellation
        if False:  # pragma: no cover - protocol compatibility generator
            yield Failure(self._failure("UNAVAILABLE", "use the privacy pipeline"))

    def _failure(self, code: str, message: str) -> ActionFailure:
        return ActionFailure(
            producer=COLLECTOR_PRODUCER,
            request_id=f"collector:{self.name}",
            capability=f"collector.{self.name}",
            code=code,
            message=message,
            retryable=True,
        )


class QueueEventSource:
    """Deterministic source used by tests and native adapter bridges."""

    def __init__(self, permission: str = "GRANTED") -> None:
        self.permission = permission
        self.queue: asyncio.Queue[SourceEvent | None] = asyncio.Queue()

    async def emit(self, event: SourceEvent) -> None:
        await self.queue.put(event)

    async def close(self) -> None:
        await self.queue.put(None)

    async def events(self, cancellation: CancellationToken) -> AsyncIterator[SourceEvent]:
        while not cancellation.cancelled:
            event = await self.queue.get()
            if event is None:
                return
            yield event


class JsonLinesSidecarSource:
    """Run an unprivileged platform bridge and validate its bounded JSON lines."""

    def __init__(
        self,
        command: tuple[str, ...],
        expected_source: str,
        *,
        permission: str = "UNKNOWN",
    ) -> None:
        if not command:
            raise ValueError("sidecar command must not be empty")
        self.command = command
        self.expected_source = expected_source
        self.permission = permission

    async def events(self, cancellation: CancellationToken) -> AsyncIterator[SourceEvent]:
        environment = {
            name: value for name in (
                "PATH", "LANG", "LC_ALL", "DISPLAY", "WAYLAND_DISPLAY",
                "XDG_CURRENT_DESKTOP", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
            ) if (value := os.environ.get(name)) is not None
        }
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, env=environment,
                limit=MAX_SIDECAR_EVENT_BYTES + 1,
            )
        except FileNotFoundError:
            self.permission = "UNAVAILABLE"
            raise
        except PermissionError:
            self.permission = "DENIED"
            raise
        self.permission = "GRANTED"
        assert process.stdout is not None
        try:
            while not cancellation.cancelled:
                line = await process.stdout.readline()
                if not line:
                    if process.returncode not in (None, 0):
                        self.permission = "UNAVAILABLE"
                        raise RuntimeError("collector sidecar failed")
                    return
                if len(line) > MAX_SIDECAR_EVENT_BYTES:
                    raise ValueError("collector event exceeds size limit")
                event = SourceEvent.model_validate(json.loads(line))
                if event.source != self.expected_source:
                    raise ValueError("collector source identity mismatch")
                yield event
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()


class GnomeSessionCollector(SemanticCollector):
    """GNOME shell/session semantic bridge; never reads ``/dev/input``."""

    def __init__(self, source: SemanticEventSource) -> None:
        super().__init__("gnome", source)

    @staticmethod
    def normalize(payload: dict[str, Any]) -> SourceEvent:
        desktop_id = payload.get("desktop_entry_id") or payload.get("wm_class")
        return SourceEvent(
            source="gnome", kind=str(payload["kind"]), timestamp=payload.get("timestamp", utc_now()),
            application_id=str(desktop_id).casefold() if desktop_id else None,
            title=payload.get("title"), session_locked=bool(payload.get("session_locked", False)),
            payload={key: payload[key] for key in ("workspace", "idle") if key in payload},
        )

    @staticmethod
    def sanitize(event: SourceEvent) -> SourceEvent:
        return SourceEvent(
            source="gnome", kind=event.kind, timestamp=event.timestamp,
            application_id=event.application_id.casefold() if event.application_id else None,
            title=event.title, resource_id=event.resource_id,
            session_locked=event.session_locked,
            payload={key: event.payload[key] for key in ("workspace", "idle") if key in event.payload},
        )


class AtspiSemanticCollector(SemanticCollector):
    """Bounded AT-SPI bridge that stores activity facts, never typed characters."""

    MAX_TEXT = 8192

    def __init__(self, source: SemanticEventSource) -> None:
        super().__init__("atspi", source)

    @classmethod
    def normalize(cls, payload: dict[str, Any]) -> SourceEvent:
        role = str(payload.get("role", "unknown"))
        # text-change values are deliberately represented only by counts/duration.
        semantic_payload = {
            "role": role,
            "states": tuple(payload.get("states", ()))[:32],
            "change_count": min(int(payload.get("change_count", 0)), 10000),
            "duration_ms": min(int(payload.get("duration_ms", 0)), 3_600_000),
        }
        return SourceEvent(
            source="atspi", kind=str(payload.get("kind", "focus")),
            timestamp=payload.get("timestamp", utc_now()), application_id=payload.get("application_id"),
            field_role=role, resource_id=payload.get("document_uri"), toolkit=payload.get("toolkit"),
            quality=float(payload.get("quality", 0.5)), depth=min(int(payload.get("depth", 0)), 32),
            selected_text=(str(payload["selected_text"])[: cls.MAX_TEXT] if payload.get("user_requested") and payload.get("selected_text") else None),
            payload=semantic_payload,
        )

    @staticmethod
    def sanitize(event: SourceEvent) -> SourceEvent:
        semantic = event.payload
        return SourceEvent(
            source="atspi", kind=event.kind, timestamp=event.timestamp,
            application_id=event.application_id, field_role=event.field_role,
            resource_id=event.resource_id, toolkit=event.toolkit, quality=event.quality,
            depth=min(event.depth, 32),
            selected_text=event.selected_text if event.selection_requested else None,
            selection_requested=event.selection_requested,
            payload={
                "role": event.field_role or semantic.get("role", "unknown"),
                "states": tuple(semantic.get("states", ()))[:32],
                "change_count": min(int(semantic.get("change_count", 0)), 10000),
                "duration_ms": min(int(semantic.get("duration_ms", 0)), 3_600_000),
            },
        )


class ProcessContextCollector(SemanticCollector):
    def __init__(self, source: SemanticEventSource) -> None:
        super().__init__("process", source)

    @staticmethod
    def normalize(pid: int, executable: str, desktop_id: str | None = None) -> SourceEvent:
        return SourceEvent(
            source="process", kind="foreground_process", application_id=desktop_id,
            resource_id=f"process:{pid}", payload={"executable": Path(executable).name},
        )

    @staticmethod
    def sanitize(event: SourceEvent) -> SourceEvent:
        executable = Path(str(event.payload.get("executable", "unknown"))).name
        return SourceEvent(
            source="process", kind=event.kind, timestamp=event.timestamp,
            application_id=event.application_id, resource_id=event.resource_id,
            project=event.project, payload={"executable": executable},
        )


class FilesystemProjectCollector(SemanticCollector):
    def __init__(self, source: SemanticEventSource, roots: tuple[Path, ...]) -> None:
        super().__init__("filesystem", source)
        self.roots = tuple(root.resolve(strict=False) for root in roots)

    def normalize(self, path: Path, kind: str) -> SourceEvent | None:
        resolved = path.resolve(strict=False)
        if not any(resolved.is_relative_to(root) for root in self.roots):
            return None
        project_root = next((parent for parent in (resolved.parent, *resolved.parents) if (parent / ".git").exists()), None)
        return SourceEvent(
            source="filesystem", kind=kind, path=str(resolved),
            resource_id=f"file:{resolved}", project=project_root.name if project_root else None,
            payload={"extension": resolved.suffix},
        )

    def sanitize(self, event: SourceEvent) -> SourceEvent | None:
        if event.path is None:
            return None
        return self.normalize(Path(event.path), event.kind)


class BrowserSemanticCollector(SemanticCollector):
    """Extension/native-messaging contract for semantic browser metadata."""

    def __init__(self, source: SemanticEventSource) -> None:
        super().__init__("browser", source)

    @staticmethod
    def normalize(payload: dict[str, Any]) -> SourceEvent:
        return SourceEvent(
            source="browser", kind=str(payload.get("kind", "navigation")),
            application_id=payload.get("application_id", "browser.desktop"),
            resource_id=f"tab:{payload['tab_id']}", origin=payload.get("origin"),
            url=payload.get("url"), title=payload.get("title"),
            selected_text=payload.get("selected_text") if payload.get("user_requested") else None,
            selection_requested=bool(payload.get("user_requested", False)),
            private_browsing=bool(payload.get("private", False)),
        )

    @staticmethod
    def sanitize(event: SourceEvent) -> SourceEvent:
        return SourceEvent(
            source="browser", kind=event.kind, timestamp=event.timestamp,
            application_id=event.application_id, resource_id=event.resource_id,
            origin=event.origin, url=event.url, title=event.title,
            selected_text=event.selected_text if event.selection_requested else None,
            selection_requested=event.selection_requested,
            private_browsing=event.private_browsing,
        )


@dataclass
class _Runtime:
    collector: SemanticCollector
    cancellation: CancellationToken
    task: asyncio.Task[None] | None = None
    last_event_at: Any = None
    last_error: str | None = None
    restart_count: int = 0


class CollectorSupervisor:
    """Own collector lifecycle; individual failures never escape into the daemon."""

    def __init__(
        self,
        sink: Callable[
            [SourceEvent],
            Awaitable[Result[Observation | None, ActionFailure] | None],
        ],
        *,
        base_backoff: float = 0.1,
        max_backoff: float = 30.0,
    ) -> None:
        self._sink = sink
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._runtimes: dict[str, _Runtime] = {}
        self.enabled = False
        self.session_locked = False
        self.emergency_stopped = False

    def register(self, collector: SemanticCollector) -> None:
        if collector.name in self._runtimes:
            raise ValueError(f"collector already registered: {collector.name}")
        self._runtimes[collector.name] = _Runtime(collector, CancellationToken())

    async def start(self) -> None:
        if not self._may_collect():
            return
        for runtime in self._runtimes.values():
            if runtime.task is None or runtime.task.done():
                runtime.cancellation = CancellationToken()
                runtime.task = asyncio.create_task(self._run(runtime))

    async def stop(self) -> None:
        tasks = []
        for runtime in self._runtimes.values():
            runtime.cancellation.cancel()
            if runtime.task is not None:
                runtime.task.cancel()
                tasks.append(runtime.task)
                runtime.task = None
            await runtime.collector.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def set_controls(
        self, *, enabled: bool | None = None, session_locked: bool | None = None,
        emergency_stop: bool | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = enabled
        if session_locked is not None:
            self.session_locked = session_locked
        if emergency_stop is not None:
            self.emergency_stopped = emergency_stop
        if self._may_collect():
            await self.start()
        else:
            await self.stop()

    async def observe_session_state(self, locked: bool) -> None:
        """Keep only the minimal GNOME lock monitor alive while locked."""
        self.session_locked = locked
        if not locked:
            await self.start()
            return
        current = asyncio.current_task()
        tasks: list[asyncio.Task[None]] = []
        for name, runtime in self._runtimes.items():
            if name == "gnome":
                continue
            runtime.cancellation.cancel()
            if runtime.task is not None and runtime.task is not current:
                runtime.task.cancel()
                tasks.append(runtime.task)
                runtime.task = None
            await runtime.collector.stop()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> tuple[CollectorHealth, ...]:
        return tuple(
            CollectorHealth(
                name=name, state=runtime.collector.state, enabled=self._may_collect(),
                permission=runtime.collector.permission, last_event_at=runtime.last_event_at,
                last_error=runtime.last_error, restart_count=runtime.restart_count,
            )
            for name, runtime in sorted(self._runtimes.items())
        )

    async def _run(self, runtime: _Runtime) -> None:
        while self._may_collect() and not runtime.cancellation.cancelled:
            try:
                started = await runtime.collector.start()
                if isinstance(started, Failure):
                    raise RuntimeError(started.failure().code)
                async for event in runtime.collector.source_events(runtime.cancellation):
                    if not self.enabled or self.emergency_stopped:
                        return
                    if self.session_locked and not (
                        runtime.collector.name == "gnome"
                        and event.kind in {"session_locked", "session_unlocked"}
                    ):
                        continue
                    delivered = await self._sink(event)
                    if isinstance(delivered, Failure):
                        raise RuntimeError(delivered.failure().code)
                    runtime.last_event_at = event.timestamp
                    runtime.last_error = None
                return
            except asyncio.CancelledError:
                return
            except Exception as exc:  # collector isolation boundary
                runtime.last_error = type(exc).__name__
                runtime.restart_count += 1
                await asyncio.sleep(min(self._base_backoff * (2 ** (runtime.restart_count - 1)), self._max_backoff))

    def _may_collect(self) -> bool:
        return self.enabled and not self.session_locked and not self.emergency_stopped


def platform_support() -> dict[str, bool]:
    """Report sidecar prerequisites without importing optional GUI libraries."""
    return {
        "dbus": shutil.which("dbus-send") is not None,
        "secret_service": shutil.which("secret-tool") is not None,
    }


def register_configured_collectors(
    supervisor: CollectorSupervisor, config: dict[str, Any]
) -> None:
    """Register explicitly configured production sidecars with no shell access."""
    collector_config = config.get("collectors", {})
    if not isinstance(collector_config, dict):
        raise ValueError("rich_history.collectors must be a mapping")
    types: dict[str, type[SemanticCollector]] = {
        "gnome": GnomeSessionCollector,
        "atspi": AtspiSemanticCollector,
        "process": ProcessContextCollector,
        "browser": BrowserSemanticCollector,
    }
    for name, item in sorted(collector_config.items()):
        if name not in {*types, "filesystem"} or not isinstance(item, dict):
            raise ValueError(f"unsupported collector configuration: {name}")
        command = item.get("command")
        if not isinstance(command, list) or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError(f"collector {name} requires a command array")
        source = JsonLinesSidecarSource(
            tuple(command), name, permission=str(item.get("permission", "UNKNOWN"))
        )
        if name == "filesystem":
            roots = tuple(Path(root) for root in item.get("roots", ()))
            collector = FilesystemProjectCollector(source, roots)
        else:
            collector = types[name](source)
        supervisor.register(collector)
