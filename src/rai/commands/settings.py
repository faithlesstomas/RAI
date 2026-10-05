"""Explicit CLI overrides for the shared configuration resolver."""

from __future__ import annotations
import click


def _assistant_config(  # noqa: PLR0913
    backend: str | None,
    model: str | None,
    profile: str | None = None,
    system: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> dict[str, object]:
    from rai import config_manager  # noqa: PLC0415

    config = config_manager.load_config()
    if profile:
        profiles = config.get("agents", {})
        if not isinstance(profiles, dict) or profile not in profiles:
            raise click.ClickException(f"Unknown assistant profile: {profile}")
        config["active_agent"] = profile
    assistant = dict(config.get("assistant", {}))
    if backend and backend != "auto":
        assistant["backend"] = backend
        profiles = config.get("agents", {})
        active_profile = str(config.get("active_agent") or "default")
        profile_config = (
            profiles.get(active_profile, {}) if isinstance(profiles, dict) else {}
        )
        if (
            not model
            and isinstance(profile_config, dict)
            and profile_config.get("backend") != backend
        ):
            copied_profiles = dict(profiles)
            copied_profile = dict(profile_config)
            copied_profile.pop("model", None)
            copied_profiles[active_profile] = copied_profile
            config["agents"] = copied_profiles
    if model:
        assistant["model"] = model
    if system:
        assistant["system"] = system
    if thinking is not None:
        assistant["enable_thinking"] = thinking
    if thinking_budget is not None:
        assistant["thinking_budget"] = thinking_budget
    config["assistant"] = assistant
    config["_cli_overrides"] = {
        k: v
        for k, v in {
            "backend": backend if backend != "auto" else None,
            "model": model,
            "system": system,
            "enable_thinking": thinking,
            "thinking_budget": thinking_budget,
        }.items()
        if v is not None
    }
    return config
