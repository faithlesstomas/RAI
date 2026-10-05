"""Completion from runtime-supported backends and configured profiles/models."""

from rai.configuration.resolution import SUPPORTED_BACKENDS
from rai.configuration.storage import ConfigurationError, load_settings


def backends(incomplete: str) -> list[str]:
    return [name for name in SUPPORTED_BACKENDS if name.startswith(incomplete)]


def profiles(incomplete: str) -> list[str]:
    try:
        return [name for name in load_settings().agents if name.startswith(incomplete)]
    except ConfigurationError:
        return []


def models(incomplete: str) -> list[str]:
    try:
        settings = load_settings()
        values = {
            item.model
            for item in (
                *settings.agents.values(),
                settings.assistant,
                settings.local_ai,
            )
        }
        return sorted(
            value for value in values if value and value.startswith(incomplete)
        )
    except ConfigurationError:
        return []
