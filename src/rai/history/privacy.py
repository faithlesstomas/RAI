"""Deterministic pre-persistence privacy firewall."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rai.kernel.records import DataClass

from .models import FilteredEvent, PrivacyAction, PrivacyDecision, SourceEvent

_SECRET_ROLES = frozenset({"password", "secret", "pin", "credential", "token"})
_SENSITIVE_APPS = frozenset({
    "1password.desktop", "bitwarden.desktop", "keepassxc.desktop",
    "org.gnome.seahorse.application", "auth-dialog", "polkit-agent",
})
_COMMUNICATION_MARKERS = ("signal", "telegram", "slack", "discord", "teams", "mail")
_SENSITIVE_ORIGIN_MARKERS = ("bank", "health", "medical", "patient", "login", "auth")
_REDACTIONS = (
    (re.compile(r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_NUMBER]"),
)


@dataclass(frozen=True)
class PrivacyPolicy:
    """Allow-only and exclusion configuration evaluated before persistence."""

    allowed_sources: frozenset[str] = frozenset({
        "gnome", "atspi", "process", "filesystem", "browser"
    })
    excluded_sources: frozenset[str] = frozenset()
    allowed_applications: frozenset[str] = frozenset()
    excluded_applications: frozenset[str] = frozenset()
    allowed_origins: frozenset[str] = frozenset()
    excluded_origins: frozenset[str] = frozenset()
    allowed_paths: tuple[Path, ...] = ()
    excluded_paths: tuple[Path, ...] = ()
    redact_private_text: bool = True
    version: str = "1.0.0"


class PrivacyFirewall:
    """Classify and transform untrusted collector data without logging it."""

    def __init__(self, policy: PrivacyPolicy | None = None) -> None:
        self.policy = policy or PrivacyPolicy()

    def apply(self, event: SourceEvent) -> FilteredEvent | None:
        decision = self.decide(event)
        if decision.action == PrivacyAction.DROP:
            return None
        if decision.action == PrivacyAction.METADATA_ONLY:
            event = event.model_copy(update={
                "title": None, "url": None, "selected_text": None, "payload": {},
            })
        elif decision.action == PrivacyAction.REDACT:
            event = event.model_copy(update={
                "title": self._redact(event.title),
                "selected_text": self._redact(event.selected_text),
                "payload": self._redact_value(event.payload),
            })
        return FilteredEvent(event=event, decision=decision)

    def decide(self, event: SourceEvent) -> PrivacyDecision:
        reasons: list[str] = []
        if event.source in self.policy.excluded_sources or (
            self.policy.allowed_sources and event.source not in self.policy.allowed_sources
        ):
            reasons.append("SOURCE_NOT_ALLOWED")
        if event.session_locked:
            reasons.append("SESSION_LOCKED")
        if event.private_browsing:
            reasons.append("PRIVATE_BROWSING")
        role_words = set(
            re.split(r"[^a-z0-9]+", (event.field_role or "").casefold())
        )
        if role_words & _SECRET_ROLES:
            reasons.append("SECRET_FIELD_ROLE")
        app = (event.application_id or "").casefold()
        if (
            app in _SENSITIVE_APPS
            or any(
                marker in app
                for marker in ("1password", "bitwarden", "keepass", "seahorse", "polkit")
            )
            or app
            in {item.casefold() for item in self.policy.excluded_applications}
        ):
            reasons.append("EXCLUDED_APPLICATION")
        if self.policy.allowed_applications and app not in {
            item.casefold() for item in self.policy.allowed_applications
        }:
            reasons.append("APPLICATION_NOT_ALLOWED")
        origin = self._origin(event.origin or event.url)
        if origin and origin in {item.casefold() for item in self.policy.excluded_origins}:
            reasons.append("EXCLUDED_ORIGIN")
        if event.source == "browser" and self.policy.allowed_origins and origin not in {
            item.casefold() for item in self.policy.allowed_origins
        }:
            reasons.append("ORIGIN_NOT_ALLOWED")
        if event.path and not self._path_allowed(Path(event.path)):
            reasons.append("PATH_NOT_ALLOWED")
        if reasons:
            return self._decision(PrivacyAction.DROP, DataClass.BLOCKED, reasons)
        if any(marker in app for marker in _COMMUNICATION_MARKERS) or any(
            marker in origin for marker in _SENSITIVE_ORIGIN_MARKERS
        ):
            return self._decision(
                PrivacyAction.METADATA_ONLY, DataClass.PRIVATE, ["BUILTIN_SENSITIVE_PROFILE"]
            )
        if self.policy.redact_private_text and self._contains_sensitive(event):
            return self._decision(PrivacyAction.REDACT, DataClass.PRIVATE, ["SENSITIVE_PATTERN"])
        classification = DataClass.PRIVATE if event.title or event.selected_text else DataClass.LOCAL
        return self._decision(PrivacyAction.ALLOW, classification, ["POLICY_ALLOWED"])

    def _decision(
        self, action: PrivacyAction, data_class: DataClass, reasons: list[str]
    ) -> PrivacyDecision:
        return PrivacyDecision(
            action=action, data_class=data_class, reason_codes=tuple(reasons),
            policy_version=self.policy.version,
        )

    def _path_allowed(self, path: Path) -> bool:
        resolved = path.expanduser().resolve(strict=False)
        if any(resolved.is_relative_to(root.expanduser().resolve(strict=False)) for root in self.policy.excluded_paths):
            return False
        return not self.policy.allowed_paths or any(
            resolved.is_relative_to(root.expanduser().resolve(strict=False))
            for root in self.policy.allowed_paths
        )

    @staticmethod
    def _origin(value: str | None) -> str:
        if not value:
            return ""
        parsed = urlsplit(value if "://" in value else f"https://{value}")
        return (parsed.hostname or "").casefold()

    @staticmethod
    def _redact(value: str | None) -> str | None:
        if value is None:
            return None
        for pattern, replacement in _REDACTIONS:
            value = pattern.sub(replacement, value)
        return value

    def _redact_value(self, value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact_value(item) for item in value]
        return value

    @staticmethod
    def _contains_sensitive(event: SourceEvent) -> bool:
        content = " ".join(
            item for item in (event.title, event.selected_text, str(event.payload)) if item
        )
        return any(pattern.search(content) for pattern, _replacement in _REDACTIONS)


def policy_from_config(config: dict[str, Any]) -> PrivacyPolicy:
    """Build a strict policy from persisted JSON configuration."""
    privacy = config.get("privacy", {})
    if not isinstance(privacy, dict):
        raise ValueError("rich_history.privacy must be a mapping")

    def strings(name: str) -> tuple[str, ...]:
        value = privacy.get(name, ())
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            if value == ():
                return ()
            raise ValueError(f"rich_history.privacy.{name} must be a string array")
        return tuple(value)

    allowed_sources = strings("allowed_sources")
    return PrivacyPolicy(
        allowed_sources=(
            frozenset(allowed_sources)
            if allowed_sources
            else PrivacyPolicy.allowed_sources
        ),
        excluded_sources=frozenset(strings("excluded_sources")),
        allowed_applications=frozenset(strings("allowed_applications")),
        excluded_applications=frozenset(strings("excluded_applications")),
        allowed_origins=frozenset(strings("allowed_origins")),
        excluded_origins=frozenset(strings("excluded_origins")),
        allowed_paths=tuple(Path(item) for item in strings("allowed_paths")),
        excluded_paths=tuple(Path(item) for item in strings("excluded_paths")),
        redact_private_text=bool(privacy.get("redact_private_text", True)),
        version=str(privacy.get("version", "1.0.0")),
    )
