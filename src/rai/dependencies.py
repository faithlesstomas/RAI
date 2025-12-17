"""
FastAPI dependencies for the RAI application.
"""
from typing import Dict, Any
from fastapi import Depends

from . import config_manager
from .services.model_registry import ModelRegistry

def get_config() -> Dict[str, Any]:
    """Dependency to load and provide the application configuration."""
    return config_manager.load_config()

def get_model_registry(
    config: Dict[str, Any] = Depends(get_config)
) -> ModelRegistry:
    """Dependency to provide the ModelRegistry service."""
    return ModelRegistry(config)
