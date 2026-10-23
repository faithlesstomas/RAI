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
SUPPORTED_CONTROLLERS = (
    "bounded_step",
    "direct",
    "workspace",
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
                key: (
                    float(env[name])
                    if key == "lemonade_timeout"
                    else (
                        int(env[name])
                        if key in ("max_full_observations", "max_context_characters")
                        else (
                            None
                            if key == "max_iterations"
                            and str(env[name]).strip().lower()
                            in ("none", "null", "inf", "infinity", "0", "unlimited")
                            else (
                                int(env[name])
                                if key == "max_iterations"
                                else (
                                    str(env[name]).strip().lower()
                                    in ("true", "1", "yes", "on")
                                    if key == "bypass_intent_recognizer"
                                    else env[name]
                                )
                            )
                        )
                    )
                )
                for key, name in {
                    "backend": "RAI_ASSISTANT_BACKEND",
                    "model": "RAI_ASSISTANT_MODEL",
                    "lemonade_host": "LEMONADE_HOST",
                    "lemonade_api_key": "LEMONADE_API_KEY",
                    "lemonade_timeout": "LEMONADE_TIMEOUT",
                    "interaction_controller": "RAI_INTERACTION_CONTROLLER",
                    "max_iterations": "RAI_MAX_ITERATIONS",
                    "max_full_observations": "RAI_MAX_FULL_OBSERVATIONS",
                    "max_context_characters": "RAI_MAX_CONTEXT_CHARACTERS",
                    "bypass_intent_recognizer": "RAI_BYPASS_INTENT_RECOGNIZER",
                }.items()
                if env.get(name) is not None
            },
        ),
        ("CLI", overrides or config.get("_cli_overrides", {})),
    ]
    for source, layer in layers:
        for key, value in layer.items():
            if key in ModelSettings.model_fields:  # pylint: disable=unsupported-membership-test
                if value is not None:
                    values[key] = value
                    sources[key] = source
                elif key == "max_iterations" and source in (
                    "CLI",
                    "environment",
                    "assistant",
                ):
                    values[key] = None
                    sources[key] = source
    return values, sources
