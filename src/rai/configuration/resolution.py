"""One pure precedence rule, shared by runtime and configuration inspection."""

from __future__ import annotations

import os
from typing import Any, Mapping
from .models import ModelSettings

SUPPORTED_BACKENDS = (
    "auto",
    "llama",
    "ollama",
    "lemonade",
    "antigravity",
    "deterministic",
)


def effective_model_settings(
    config: dict[str, Any],
    *,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """CLI > environment > assistant > local_ai > selected profile > defaults."""
    env = os.environ if environment is None else environment
    selected = profile or config.get("active_agent", "default")
    values = ModelSettings().model_dump(exclude_none=True)
    sources = dict.fromkeys(values, "default")
    layers = [
        (f"agents.{selected}", config.get("agents", {}).get(selected, {})),
        ("local_ai", config.get("local_ai", {})),
        ("assistant", config.get("assistant", {})),
        (
            "environment",
            {
                key: env[name]
                for key, name in {
                    "backend": "RAI_ASSISTANT_BACKEND",
                    "model": "RAI_ASSISTANT_MODEL",
                    "lemonade_host": "LEMONADE_HOST",
                    "lemonade_api_key": "LEMONADE_API_KEY",
                }.items()
                if env.get(name)
            },
        ),
        ("CLI", overrides or config.get("_cli_overrides", {})),
    ]
    for source, layer in layers:
        for key, value in layer.items():
            if key in ModelSettings.model_fields and value is not None:  # pylint: disable=unsupported-membership-test
                values[key] = value
                sources[key] = source
    return values, sources
