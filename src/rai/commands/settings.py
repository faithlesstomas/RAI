"""Explicit CLI overrides for the shared configuration resolver."""

from __future__ import annotations
import click


def _assistant_config(  # noqa: PLR0912, PLR0913, PLR0915
    backend: str | None,
    model: str | None,
    profile: str | None = None,
    system: str | None = None,
    thinking: bool | None = None,
    thinking_budget: int | None = None,
    max_output_tokens: int | None = None,
    context_window: int | None = None,
    timeout: float | None = None,
    controller: str | None = None,
    max_iterations: str | int | None = None,
    max_full_observations: int | None = None,
    max_context_characters: int | None = None,
) -> dict[str, object]:
    from rai import config_manager  # noqa: PLC0415
    from rai.configuration.resolution import SUPPORTED_CONTROLLERS  # noqa: PLC0415

    if controller is not None and controller not in SUPPORTED_CONTROLLERS:
        raise click.ClickException(
            f"Unsupported interaction controller {controller!r}; choose {', '.join(SUPPORTED_CONTROLLERS)}."
        )

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
    if max_output_tokens is not None:
        assistant["max_output_tokens"] = max_output_tokens
    if context_window is not None:
        assistant["context_window"] = context_window
    if timeout is not None:
        assistant["lemonade_timeout"] = timeout
    if controller is not None:
        assistant["interaction_controller"] = controller

    parsed_iterations: int | None = None
    has_explicit_iterations = False
    if max_iterations is not None:
        has_explicit_iterations = True
        clean_iter = str(max_iterations).strip().lower()
        if clean_iter in ("none", "null", "inf", "infinity", "0", "unlimited"):
            parsed_iterations = None
        else:
            parsed_iterations = int(clean_iter)
        assistant["max_iterations"] = parsed_iterations

    if max_full_observations is not None:
        assistant["max_full_observations"] = int(max_full_observations)
    if max_context_characters is not None:
        assistant["max_context_characters"] = int(max_context_characters)

    config["assistant"] = assistant

    cli_overrides: dict[str, object] = {
        "backend": backend if backend != "auto" else None,
        "model": model,
        "system": system,
        "enable_thinking": thinking,
        "thinking_budget": thinking_budget,
        "max_output_tokens": max_output_tokens,
        "context_window": context_window,
        "lemonade_timeout": timeout,
        "interaction_controller": controller,
    }
    if has_explicit_iterations:
        cli_overrides["max_iterations"] = parsed_iterations
    if max_full_observations is not None:
        cli_overrides["max_full_observations"] = int(max_full_observations)
    if max_context_characters is not None:
        cli_overrides["max_context_characters"] = int(max_context_characters)

    config["_cli_overrides"] = {
        k: v
        for k, v in cli_overrides.items()
        if v is not None or (k == "max_iterations" and has_explicit_iterations)
    }
    return config
