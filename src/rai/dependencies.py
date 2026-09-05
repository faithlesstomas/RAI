"""FastAPI dependency adapters backed by the application container."""

from typing import Any

from fastapi import Request

from .container import ApplicationContainer
from .kernel.service import CapabilityService
from .kernel.event_service import EventService
from .kernel.ports import EventJournal
from .services.history import HistoryService
from .services.model_registry import ModelRegistry


def get_container(request: Request) -> ApplicationContainer:
    """Resolve the container owned by the current FastAPI application."""
    return request.app.state.container


def get_config(request: Request) -> dict[str, Any]:
    return get_container(request).config


def get_model_registry(request: Request) -> ModelRegistry:
    return get_container(request).model_registry


def get_history_service(request: Request) -> HistoryService:
    return get_container(request).history_service


async def get_capability_service(request: Request) -> CapabilityService:
    return get_container(request).capability_service


async def get_event_service(request: Request) -> EventService:
    return get_container(request).event_service


async def get_event_journal(request: Request) -> EventJournal:
    journal = get_container(request).event_journal
    assert journal is not None
    return journal


async def close_dependencies(container: ApplicationContainer) -> None:
    """Close application-scoped resources without touching module globals."""
    await container.close()
