"""Configuration and composition helpers for the local Rich Assistant runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from returns.result import Failure

from rai.configuration.models import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_CONTEXT_CHARACTERS,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from rai.inference.engines.llama import AsyncLlamaEngine
from rai.inference.factory import load_local_model

from .backends.deterministic import DeterministicAssistantBackend
from .backends.local import LocalAssistantBackend
from .context import DEFAULT_SYSTEM_INSTRUCTION
from .ports import AssistantModelBackend


class AssistantConfigurationError(ValueError):
    """Raised when no usable, explicit assistant model configuration exists."""


@dataclass(frozen=True)
class AssistantRuntimeConfig:
    """Resolved settings for one provider-neutral assistant backend."""

    backend: str
    model: str
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    temperature: float = 0.2
    context_window: int = DEFAULT_CONTEXT_WINDOW
    model_artifact_version: str | None = None
    ollama_host: str = "http://127.0.0.1:11434"
    lemonade_host: str = "http://127.0.0.1:13305"
    lemonade_api_key: str | None = None
    lemonade_timeout: float = 300.0
    profile_scope: str = "default"
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    max_context_characters: int = DEFAULT_MAX_CONTEXT_CHARACTERS
    max_recent_turns: int = 10
    max_memories: int = 5
    max_episodic_turns: int = 5
    max_external_evidence: int = 5
    memory_sufficiency_threshold: float = 0.75
    enable_thinking: bool = False
    thinking_budget: int | None = None
    thinking_level: str | None = None
    interaction_controller: str = "bounded_step"
    max_iterations: int | None = None
    max_full_observations: int = 3
    bypass_intent_recognizer: bool = True


def _discover_gguf(search_root: Path) -> Path | None:
    model_dir = search_root / "models"
    if not model_dir.is_dir():
        return None
    candidates = sorted(model_dir.glob("*.gguf"))
    if not candidates:
        return None
    chat_models = [path for path in candidates if "chat" in path.name.lower()]
    return (chat_models or candidates)[0]


def _local_artifact_version(model: str) -> str | None:
    path = Path(model)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_assistant_config(  # noqa: PLR0912, PLR0913, PLR0915
    config: dict[str, Any],
    *,
    backend_override: str | None = None,
    model_override: str | None = None,
    profile_override: str | None = None,
    search_root: Path | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
) -> AssistantRuntimeConfig:
    """Resolve CLI, environment and persisted config into one runtime config."""
    from rai.configuration.resolution import effective_model_settings  # noqa: PLC0415

    profile_name = str(profile_override or config.get("active_agent") or "default")
    overrides = dict(config.get("_cli_overrides", {}))
    if backend_override is not None:
        overrides["backend"] = backend_override
    if model_override is not None:
        overrides["model"] = model_override
    assistant, _sources = effective_model_settings(
        config, profile=profile_name, overrides=overrides
    )
    local_ai = {}
    profile = {}

    backend = (
        backend_override
        or assistant.get("backend")
        or local_ai.get("backend")
        or profile.get("backend")
        or "auto"
    )
    model = (
        model_override
        or assistant.get("model")
        or local_ai.get("model")
        or profile.get("model")
    )
    backend = str(backend).lower()

    # Dynamic vs manual context character budget
    if (
        _sources.get("max_context_characters") != "default"
        and assistant.get("max_context_characters") is not None
    ):
        resolved_context_chars = int(assistant["max_context_characters"])
    elif backend == "deterministic":
        resolved_context_chars = int(
            assistant.get("max_context_characters", DEFAULT_MAX_CONTEXT_CHARACTERS)
        )
    elif (
        int(assistant.get("context_window", DEFAULT_CONTEXT_WINDOW)) != DEFAULT_CONTEXT_WINDOW
        or int(assistant.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)) != DEFAULT_MAX_OUTPUT_TOKENS
    ):
        cw = int(assistant.get("context_window", DEFAULT_CONTEXT_WINDOW))
        mot = int(assistant.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
        usable_tokens = max(512, cw - mot - 256)
        resolved_context_chars = int(usable_tokens * 3.5)
    else:
        resolved_context_chars = int(
            assistant.get("max_context_characters", DEFAULT_MAX_CONTEXT_CHARACTERS)
        )

    raw_iterations = assistant.get("max_iterations")
    if raw_iterations is not None:
        if isinstance(raw_iterations, str) and raw_iterations.strip().lower() in (
            "none", "null", "inf", "infinity", "0", "unlimited"
        ):
            resolved_max_iterations = None
        else:
            resolved_max_iterations = int(raw_iterations)
    else:
        resolved_max_iterations = None

    resolved_max_full_obs = int(assistant.get("max_full_observations", 3))

    if backend == "deterministic":
        return AssistantRuntimeConfig(
            backend=backend,
            model="deterministic-conformance",
            max_output_tokens=int(assistant["max_output_tokens"]),
            context_window=int(assistant["context_window"]),
            profile_scope=profile_name,
            system_instruction=str(
                assistant.get("system")
                or profile.get("system")
                or DEFAULT_SYSTEM_INSTRUCTION
            ),
            max_context_characters=resolved_context_chars,
            max_recent_turns=int(assistant.get("max_recent_turns", 10)),
            max_memories=int(assistant.get("max_memories", 5)),
            max_episodic_turns=int(assistant.get("max_episodic_turns", 5)),
            max_external_evidence=int(assistant.get("max_external_evidence", 5)),
            memory_sufficiency_threshold=float(
                assistant.get("memory_sufficiency_threshold", 0.75)
            ),
            interaction_controller=str(
                assistant.get("interaction_controller")
                or local_ai.get("interaction_controller")
                or os.environ.get("RAI_INTERACTION_CONTROLLER")
                or "bounded_step"
            ),
            max_iterations=resolved_max_iterations,
            max_full_observations=resolved_max_full_obs,
        )

    if backend == "antigravity":
        if not model:
            model = (
                os.environ.get("RAI_ANTIGRAVITY_MODEL")
                or os.environ.get("GEMINI_MODEL")
                or "gemini-2.5-flash"
            )
    elif not model:
        discovered = _discover_gguf(search_root or Path.cwd())
        if discovered is not None:
            model = str(discovered)
            backend = "llama" if backend == "auto" else backend

    if not model:
        raise AssistantConfigurationError(
            "No local assistant model is configured. Pass --model PATH --backend llama "
            "for a GGUF file, or --model NAME --backend ollama for a local Ollama model."
        )

    model = str(model)
    if backend == "auto":
        backend = "llama" if Path(model).suffix.lower() == ".gguf" else "ollama"
    from rai.configuration.resolution import SUPPORTED_BACKENDS  # noqa: PLC0415
    if backend not in SUPPORTED_BACKENDS:
        raise AssistantConfigurationError(
            f"Unsupported assistant backend {backend!r}; choose llama, ollama, lemonade, or antigravity."
        )

    return AssistantRuntimeConfig(
        backend=backend,
        model=model,
        max_output_tokens=int(assistant.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
        temperature=float(assistant.get("temperature", 0.2)),
        context_window=int(assistant.get("context_window", DEFAULT_CONTEXT_WINDOW)),
        model_artifact_version=(
            str(assistant["model_artifact_version"])
            if assistant.get("model_artifact_version")
            else _local_artifact_version(model)
        ),
        ollama_host=str(
            assistant.get("ollama_host")
            or local_ai.get("ollama_host")
            or profile.get("ollama_host")
            or "http://127.0.0.1:11434"
        ),
        lemonade_host=str(
            assistant.get("lemonade_host")
            or local_ai.get("lemonade_host")
            or profile.get("lemonade_host")
            or os.environ.get("LEMONADE_HOST")
            or "http://127.0.0.1:13305"
        ),
        lemonade_api_key=(
            str(
                assistant.get("lemonade_api_key")
                or local_ai.get("lemonade_api_key")
                or profile.get("lemonade_api_key")
                or os.environ.get("LEMONADE_API_KEY")
            )
            if (
                assistant.get("lemonade_api_key")
                or local_ai.get("lemonade_api_key")
                or profile.get("lemonade_api_key")
                or os.environ.get("LEMONADE_API_KEY")
            )
            else None
        ),
        lemonade_timeout=float(
            assistant.get("lemonade_timeout")
            or local_ai.get("lemonade_timeout")
            or profile.get("lemonade_timeout")
            or os.environ.get("LEMONADE_TIMEOUT")
            or 300.0
        ),
        profile_scope=profile_name,
        system_instruction=str(
            assistant.get("system")
            or profile.get("system")
            or DEFAULT_SYSTEM_INSTRUCTION
        ),
        max_context_characters=resolved_context_chars,
        max_recent_turns=int(assistant.get("max_recent_turns", 10)),
        max_memories=int(assistant.get("max_memories", 5)),
        max_episodic_turns=int(assistant.get("max_episodic_turns", 5)),
        max_external_evidence=int(assistant.get("max_external_evidence", 5)),
        memory_sufficiency_threshold=float(
            assistant.get("memory_sufficiency_threshold", 0.75)
        ),
        enable_thinking=bool(
            enable_thinking
            if enable_thinking is not None
            else assistant.get("enable_thinking", False)
        ),
        thinking_budget=(
            thinking_budget
            if thinking_budget is not None
            else assistant.get("thinking_budget")
        ),
        thinking_level=(
            thinking_level
            if thinking_level is not None
            else (assistant.get("thinking_level") or local_ai.get("thinking_level"))
        ),
        interaction_controller=str(
            assistant.get("interaction_controller")
            or local_ai.get("interaction_controller")
            or os.environ.get("RAI_INTERACTION_CONTROLLER")
            or "bounded_step"
        ),
        max_iterations=resolved_max_iterations,
        max_full_observations=resolved_max_full_obs,
        bypass_intent_recognizer=bool(
            assistant.get("bypass_intent_recognizer", True)
            if assistant.get("bypass_intent_recognizer") is not None
            else True
        ),
    )


def build_assistant_backend(runtime: AssistantRuntimeConfig) -> AssistantModelBackend:
    """Build a lifecycle-aware backend without a silent deterministic fallback."""
    if runtime.backend == "deterministic":
        return DeterministicAssistantBackend()

    if runtime.backend == "antigravity":
        from .backends.antigravity import AntigravityAssistantModelBackend  # noqa: PLC0415

        return AntigravityAssistantModelBackend(
            model_name=runtime.model,
            max_output_tokens=runtime.max_output_tokens,
            temperature=runtime.temperature,
        )

    engine_res = load_local_model(runtime.model, backend=runtime.backend)
    if isinstance(engine_res, Failure):
        raise AssistantConfigurationError(str(engine_res.failure()))
    engine = engine_res.unwrap()
    if runtime.backend == "llama":
        engine.n_ctx = runtime.context_window
        engine = AsyncLlamaEngine(engine, n_ctx=runtime.context_window)
    elif runtime.backend == "ollama":
        engine.host = runtime.ollama_host
    elif runtime.backend == "lemonade":
        engine.host = runtime.lemonade_host
        engine.timeout_seconds = runtime.lemonade_timeout
        if runtime.lemonade_api_key:
            engine.api_key = runtime.lemonade_api_key

    return LocalAssistantBackend(
        engine=engine,
        model_name=runtime.model,
        backend_name=runtime.backend,
        max_output_tokens=runtime.max_output_tokens,
        temperature=runtime.temperature,
        model_artifact_version=runtime.model_artifact_version,
        enable_thinking=runtime.enable_thinking,
        thinking_budget=runtime.thinking_budget,
        thinking_level=runtime.thinking_level,
        max_observation_characters=max(64000, runtime.max_context_characters),
    )
