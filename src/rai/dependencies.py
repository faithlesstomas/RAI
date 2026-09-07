"""FastAPI dependency adapters backed by the application container."""

from typing import Any

from fastapi import HTTPException, Request

from .container import ApplicationContainer
from .kernel.service import CapabilityService
from .kernel.event_service import EventService
from .kernel.ports import EventJournal
from .services.history import HistoryService
from .services.model_registry import ModelRegistry
from .history.service import RichHistoryService
from .history.storage import KeyUnavailableError


def get_container(request: Request) -> ApplicationContainer:
    """Resolve the container owned by the current FastAPI application."""
    return request.app.state.container


def get_config(request: Request) -> dict[str, Any]:
    return get_container(request).config


def get_model_registry(request: Request) -> ModelRegistry:
    return get_container(request).model_registry


def get_history_service(request: Request) -> HistoryService:
    return get_container(request).history_service


async def get_rich_history_service(request: Request) -> RichHistoryService:
    try:
        return get_container(request).rich_history_service
    except KeyUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "KEY_UNAVAILABLE", "message": str(exc)},
        ) from exc


async def get_capability_service(request: Request) -> CapabilityService:
    return get_container(request).capability_service


async def get_event_service(request: Request) -> EventService:
    return get_container(request).event_service


async def get_event_journal(request: Request) -> EventJournal:
    journal = get_container(request).event_journal
    if journal is None:
        raise HTTPException(status_code=503, detail="event journal is unavailable")
    return journal


async def close_dependencies(container: ApplicationContainer) -> None:
    """Close application-scoped resources without touching module globals."""
    await container.close()
