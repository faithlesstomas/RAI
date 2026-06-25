"""
FastAPI dependencies for the RAI application.
"""
from typing import Dict, Any, Optional
from fastapi import Depends

from . import config_manager
from .services.model_registry import ModelRegistry

_MODEL_REGISTRY: Optional[ModelRegistry] = None

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

async def close_dependencies() -> None:
    """Closes and cleans up resource connections in dependencies."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is not None:
        await _MODEL_REGISTRY.close()
        _MODEL_REGISTRY = None
