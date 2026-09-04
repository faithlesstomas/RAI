"""Explicit application composition root for RAI runtime services."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .kernel.audit import InMemoryAuditLedger, JsonlAuditLedger
from .kernel.capabilities import CapabilityRegistry
from .kernel.defaults import HitlApprovalBroker, create_default_capability_registry, isolation_available
from .kernel.policy import PolicyEngine
from .kernel.service import AuditLedger, CapabilityService
from .services.history import HistoryService
from .services.model_registry import ModelRegistry


@dataclass
class ApplicationContainer:
    """Own application-scoped services and their lifecycle."""

    config: dict[str, Any]
    capability_registry: CapabilityRegistry = field(default_factory=create_default_capability_registry)
    audit_ledger: AuditLedger | None = None
    policy_engine: PolicyEngine | None = None
    testing: bool = False
    history_path: Path | None = None
    _history_service: HistoryService | None = field(default=None, init=False)
    _model_registry: ModelRegistry | None = field(default=None, init=False)

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

    async def close(self) -> None:
        if self._model_registry is not None:
            await self._model_registry.close()
            self._model_registry = None
        self._history_service = None
