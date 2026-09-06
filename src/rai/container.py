"""Explicit application composition root for RAI runtime services."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
from pathlib import Path
import tempfile
from typing import Any

from .kernel.audit import InMemoryAuditLedger, JsonlAuditLedger
from .kernel.capabilities import CapabilityRegistry
from .kernel.defaults import HitlApprovalBroker, create_default_capability_registry, isolation_available
from .kernel.policy import PolicyEngine
from .kernel.event_service import EventService
from .kernel.dispatch import DeterministicEventDispatcher
from .kernel.journal import SQLiteEventJournal
from .kernel.ports import EventJournal
from .kernel.socket_transport import EventSocketServer
from .kernel.service import AuditLedger, CapabilityService
from .services.history import HistoryService
from .services.model_registry import ModelRegistry
from .history.service import RichHistoryService
from .history.collectors import register_configured_collectors
from .history.privacy import PrivacyFirewall, policy_from_config
from .history.storage import (
    EncryptedHistoryStore,
    KeyProvider,
    KeyUnavailableError,
    retention_from_config,
)


@dataclass
class ApplicationContainer:
    """Own application-scoped services and their lifecycle."""

    config: dict[str, Any]
    capability_registry: CapabilityRegistry = field(default_factory=create_default_capability_registry)
    audit_ledger: AuditLedger | None = None
    policy_engine: PolicyEngine | None = None
    testing: bool = False
    history_path: Path | None = None
    event_journal: EventJournal | None = None
    event_journal_path: Path | None = None
    rich_history_path: Path | None = None
    history_key_provider: KeyProvider | None = None
    _history_service: HistoryService | None = field(default=None, init=False)
    _model_registry: ModelRegistry | None = field(default=None, init=False)
    _dispatcher_task: asyncio.Task[None] | None = field(default=None, init=False)
    _event_socket: EventSocketServer | None = field(default=None, init=False)
    _rich_history_service: RichHistoryService | None = field(default=None, init=False)
    _rich_history_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.audit_ledger is None:
            self.audit_ledger = InMemoryAuditLedger() if self.testing else JsonlAuditLedger()
        if self.policy_engine is None:
            self.policy_engine = PolicyEngine(isolation_available=isolation_available)
        self.capability_service = CapabilityService(
            self.capability_registry,
            self.policy_engine,
            self.audit_ledger,
            HitlApprovalBroker(),
        )
        if self.event_journal is None:
            path = self.event_journal_path
            if self.testing and path is None:
                path = Path(tempfile.mkdtemp(prefix="rai-event-tests-")) / "journal.sqlite3"
            self.event_journal = SQLiteEventJournal(path)
        self.event_service = EventService(self.event_journal)
        self.event_dispatcher = DeterministicEventDispatcher(
            self.event_journal, self.capability_service
        )

    async def start(self) -> None:
        """Start local event transports and the deterministic subscriber."""
        if self.testing:
            return
        self._event_socket = EventSocketServer(self.event_service, self.event_journal)
        await self._event_socket.start()
        self._dispatcher_task = asyncio.create_task(self._dispatch_events())
        history_config = self.config.get("rich_history", {})
        if isinstance(history_config, dict) and history_config.get("enabled"):
            try:
                await self.rich_history_service.set_controls(enabled=True)
            except KeyUnavailableError:
                # History fails closed without taking down the core daemon.
                self._rich_history_error = "KEY_UNAVAILABLE"

    async def _dispatch_events(self) -> None:
        while True:
            await self.event_dispatcher.process_next()
            await asyncio.sleep(0.05)

    @property
    def history_service(self) -> HistoryService:
        if self._history_service is None:
            self._history_service = HistoryService(
                str(self.history_path) if self.history_path is not None else None
            )
        return self._history_service

    @property
    def model_registry(self) -> ModelRegistry:
        if self._model_registry is None:
            self._model_registry = ModelRegistry(self.config)
        return self._model_registry

    @property
    def rich_history_service(self) -> RichHistoryService:
        if self._rich_history_service is None:
            assert self.event_journal is not None
            history_config = self.config.get("rich_history", {})
            config = history_config if isinstance(history_config, dict) else {}
            collector_config = config.get("collectors", {})
            filesystem_config = (
                collector_config.get("filesystem", {})
                if isinstance(collector_config, dict)
                else {}
            )
            filesystem_roots = tuple(
                Path(root)
                for root in (
                    filesystem_config.get("roots", ())
                    if isinstance(filesystem_config, dict)
                    else ()
                )
            )
            self._rich_history_service = RichHistoryService(
                self.event_journal,
                EncryptedHistoryStore(
                    self.rich_history_path,
                    key_provider=self.history_key_provider,
                    retention=retention_from_config(config),
                    backup_enabled=bool(config.get("backup_enabled", False)),
                ),
                firewall=PrivacyFirewall(policy_from_config(config)),
                collection_enabled=bool(config.get("enabled", False)),
                filesystem_roots=filesystem_roots,
            )
            register_configured_collectors(
                self._rich_history_service.supervisor, config
            )
        return self._rich_history_service

    async def close(self) -> None:
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            await asyncio.gather(self._dispatcher_task, return_exceptions=True)
            self._dispatcher_task = None
        if self._event_socket is not None:
            await self._event_socket.stop()
            self._event_socket = None
        if self._model_registry is not None:
            await self._model_registry.close()
            self._model_registry = None
        self._history_service = None
        if self._rich_history_service is not None:
            await self._rich_history_service.supervisor.stop()
            self._rich_history_service = None
