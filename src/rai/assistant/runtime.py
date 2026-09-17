"""Configuration and composition helpers for the local Rich Assistant runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

from returns.result import Failure

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
    max_output_tokens: int = 256
    temperature: float = 0.2
    context_window: int = 2048
    model_artifact_version: str | None = None
    ollama_host: str = "http://127.0.0.1:11434"
    lemonade_host: str = "http://127.0.0.1:13305"
    lemonade_api_key: str | None = None
    profile_scope: str = "default"
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    max_context_characters: int = 8_000
    max_recent_turns: int = 10
    max_memories: int = 5
    max_episodic_turns: int = 5
    max_external_evidence: int = 5
    memory_sufficiency_threshold: float = 0.75


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


def resolve_assistant_config(
    config: dict[str, Any],
    *,
    backend_override: str | None = None,
    model_override: str | None = None,
    profile_override: str | None = None,
    search_root: Path | None = None,
) -> AssistantRuntimeConfig:
    """Resolve CLI, environment and persisted config into one runtime config."""
    raw_assistant = config.get("assistant", {})
    assistant = raw_assistant if isinstance(raw_assistant, dict) else {}
    raw_local = config.get("local_ai", {})
    local_ai = raw_local if isinstance(raw_local, dict) else {}
    profile_name = str(profile_override or config.get("active_agent") or "default")
    raw_profiles = config.get("agents", {})
    profiles = raw_profiles if isinstance(raw_profiles, dict) else {}
    raw_profile = profiles.get(profile_name, {})
    profile = raw_profile if isinstance(raw_profile, dict) else {}

    backend = (
        backend_override
        or os.environ.get("RAI_ASSISTANT_BACKEND")
        or assistant.get("backend")
        or local_ai.get("backend")
        or profile.get("backend")
        or "auto"
    )
    model = (
        model_override
        or os.environ.get("RAI_ASSISTANT_MODEL")
        or assistant.get("model")
        or local_ai.get("model")
        or profile.get("model")
    )
    backend = str(backend).lower()

    if backend == "deterministic":
        return AssistantRuntimeConfig(
            backend=backend,
            model="deterministic-conformance",
            profile_scope=profile_name,
            system_instruction=str(
                assistant.get("system")
                or profile.get("system")
                or DEFAULT_SYSTEM_INSTRUCTION
            ),
            max_context_characters=int(assistant.get("max_context_characters", 8_000)),
            max_recent_turns=int(assistant.get("max_recent_turns", 10)),
            max_memories=int(assistant.get("max_memories", 5)),
            max_episodic_turns=int(assistant.get("max_episodic_turns", 5)),
            max_external_evidence=int(assistant.get("max_external_evidence", 5)),
            memory_sufficiency_threshold=float(
                assistant.get("memory_sufficiency_threshold", 0.75)
            ),
        )

    if not model:
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
    if backend not in {"llama", "ollama", "lemonade"}:
        raise AssistantConfigurationError(
            f"Unsupported assistant backend {backend!r}; choose llama, ollama, or lemonade."
        )

    return AssistantRuntimeConfig(
        backend=backend,
        model=model,
        max_output_tokens=int(assistant.get("max_output_tokens", 256)),
        temperature=float(assistant.get("temperature", 0.2)),
        context_window=int(assistant.get("context_window", 2048)),
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
        profile_scope=profile_name,
        system_instruction=str(
            assistant.get("system")
            or profile.get("system")
            or DEFAULT_SYSTEM_INSTRUCTION
        ),
        max_context_characters=int(assistant.get("max_context_characters", 8_000)),
        max_recent_turns=int(assistant.get("max_recent_turns", 10)),
        max_memories=int(assistant.get("max_memories", 5)),
        max_episodic_turns=int(assistant.get("max_episodic_turns", 5)),
        max_external_evidence=int(assistant.get("max_external_evidence", 5)),
        memory_sufficiency_threshold=float(
            assistant.get("memory_sufficiency_threshold", 0.75)
        ),
    )


def build_assistant_backend(runtime: AssistantRuntimeConfig) -> AssistantModelBackend:
    """Build a lifecycle-aware backend without a silent deterministic fallback."""
    if runtime.backend == "deterministic":
        return DeterministicAssistantBackend()

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
        if runtime.lemonade_api_key:
            engine.api_key = runtime.lemonade_api_key

    return LocalAssistantBackend(
        engine=engine,
        model_name=runtime.model,
        backend_name=runtime.backend,
        max_output_tokens=runtime.max_output_tokens,
        temperature=runtime.temperature,
        model_artifact_version=runtime.model_artifact_version,
    )
