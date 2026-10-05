"""Validated user settings, independent of transports and mutable session state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SettingsRecord(BaseModel):
    """Reject misspellings and coercions such as the string 'false'."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ModelSettings(SettingsRecord):
    backend: str | None = None
    model: str | None = None
    system: str | None = None
    model_artifact_version: str | None = None
    ollama_host: str | None = None
    lemonade_host: str | None = None
    lemonade_api_key: str | None = None
    max_output_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    context_window: int = Field(default=2048, gt=0)
    max_context_characters: int = Field(default=8000, gt=0)
    max_recent_turns: int = Field(default=10, ge=0)
    max_memories: int = Field(default=5, ge=0)
    max_episodic_turns: int = Field(default=5, ge=0)
    max_external_evidence: int = Field(default=5, ge=0)
    memory_sufficiency_threshold: float = Field(default=0.75, ge=0, le=1)
    enable_thinking: bool = False
    thinking_budget: int | None = Field(default=None, ge=0)
    thinking_level: str | None = None


class TtsSettings(SettingsRecord):
    data_dir: str | None = None
    default_voice: str = "pl_PL-gosia-medium"


class ProfileSettings(ModelSettings):
    # These fields remain consumed by the legacy REST/TUI endpoints (#43).
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    tts: TtsSettings = Field(default_factory=TtsSettings)


class CacheSettings(SettingsRecord):
    enabled: bool = True
    path: str | None = None
    ttl_seconds: float = Field(default=300, gt=0)
    max_entries: int = Field(default=128, gt=0)


class LocalSettings(ModelSettings):
    idle_unload_seconds: float = Field(default=300, ge=0)
    max_concurrency: int = Field(default=2, gt=0)
    result_cache: CacheSettings = Field(default_factory=CacheSettings)


class ActionSettings(SettingsRecord):
    allowed_file_roots: list[str] = Field(default_factory=list)
    browser_endpoint: str | None = None

    @field_validator("allowed_file_roots")
    @classmethod
    def validate_roots(cls, roots: list[str]) -> list[str]:
        for root in roots:
            path = Path(root).expanduser()
            if not path.is_absolute() or Path(os.path.abspath(path)) == Path("/"):
                raise ValueError("document roots must be absolute non-root paths")
        return roots


class CollectorSettings(SettingsRecord):
    command: list[str] = Field(default_factory=list)
    permission: str = "UNKNOWN"
    roots: list[str] = Field(default_factory=list)


class PrivacySettings(SettingsRecord):
    allowed_sources: list[str] = Field(default_factory=list)
    excluded_sources: list[str] = Field(default_factory=list)
    allowed_applications: list[str] = Field(default_factory=list)
    excluded_applications: list[str] = Field(default_factory=list)
    allowed_origins: list[str] = Field(default_factory=list)
    excluded_origins: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    redact_private_text: bool = True
    version: str = "1.0.0"


class RetentionSettings(SettingsRecord):
    raw: float = Field(default=10 / (24 * 60), gt=0)
    observations: float = Field(default=30, gt=0)
    episodes: float = Field(default=90, gt=0)
    memories: float = Field(default=365, gt=0)


class HistorySettings(SettingsRecord):
    enabled: bool = False
    backup_enabled: bool = False
    retention_interval_seconds: float = Field(default=300, gt=0)
    collectors: dict[str, CollectorSettings] = Field(default_factory=dict)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    retention_days: RetentionSettings = Field(default_factory=RetentionSettings)


class LegacySettings(SettingsRecord):
    enabled: bool = False


class AppSettings(SettingsRecord):
    schema_version: Literal[1] = 1
    active_agent: str = "default"
    agents: dict[str, ProfileSettings] = Field(
        default_factory=lambda: {"default": ProfileSettings()}
    )
    assistant: ModelSettings = Field(default_factory=ModelSettings)
    local_ai: LocalSettings = Field(default_factory=LocalSettings)
    actions: ActionSettings = Field(default_factory=ActionSettings)
    tts: TtsSettings = Field(default_factory=TtsSettings)
    rich_history: HistorySettings = Field(default_factory=HistorySettings)
    legacy_chat: LegacySettings = Field(default_factory=LegacySettings)

    def runtime_mapping(self) -> dict[str, Any]:
        """Preserve omission semantics; defaults are resolved at the runtime boundary."""
        result = self.model_dump(exclude_unset=True)
        result.setdefault("agents", {"default": {}})
        result.setdefault("active_agent", "default")
        return result
