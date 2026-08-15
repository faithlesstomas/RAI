"""
FastAPI dependencies for the RAI application.
"""
from typing import Dict, Any, Optional
from fastapi import Depends

from . import config_manager
from .services.model_registry import ModelRegistry
from .services.history import HistoryService

_MODEL_REGISTRY: Optional[ModelRegistry] = None
_HISTORY_SERVICE: Optional[HistoryService] = None

def get_config() -> Dict[str, Any]:
    """Dependency to load and provide the application configuration."""
    return config_manager.load_config()

def get_model_registry(
    config: Dict[str, Any] = Depends(get_config)
) -> ModelRegistry:
    """Dependency to provide the ModelRegistry service as a singleton."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        _MODEL_REGISTRY = ModelRegistry(config)
    return _MODEL_REGISTRY


def get_history_service() -> HistoryService:
    """Provide the history repository lazily, without import-time filesystem I/O."""
    global _HISTORY_SERVICE
    if _HISTORY_SERVICE is None:
        _HISTORY_SERVICE = HistoryService()
    return _HISTORY_SERVICE

async def close_dependencies() -> None:
    """Closes and cleans up resource connections in dependencies."""
    global _MODEL_REGISTRY, _HISTORY_SERVICE
    if _MODEL_REGISTRY is not None:
        await _MODEL_REGISTRY.close()
        _MODEL_REGISTRY = None
    _HISTORY_SERVICE = None
