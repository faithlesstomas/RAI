"""Side-effect-free reads and private atomic writes for JSON and YAML settings."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import yaml
from pydantic import ValidationError

from rai.paths import config_dir, state_dir
from .models import AppSettings

_selected: ContextVar[str | None] = ContextVar("rai_config_path", default=None)
STATE_KEYS = frozenset({"active_session_id", "session_conversation_ids"})


class ConfigurationError(ValueError):
    """Invalid or ambiguous configuration; never silently fall back."""


def selected_path(path: str | None = None) -> Path:
    explicit = path or _selected.get() or os.environ.get("RAI_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser().absolute()
    candidates = [
        config_dir() / name for name in ("config.yaml", "config.yml", "config.json")
    ]
    present = [p for p in candidates if p.exists()]
    if len(present) > 1:
        raise ConfigurationError(
            "Multiple configuration files found; select one with --config or RAI_CONFIG_FILE"
        )
    return present[0] if present else candidates[0]


@contextmanager
def configuration_path(path: str | None) -> Iterator[None]:
    token = _selected.set(path)
    try:
        yield
    finally:
        _selected.reset(token)


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently changing permissions."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ConfigurationError("Configuration keys must be unique strings")
        result[key] = value
    return result


def _yaml_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, Any]:
    return _unique_pairs(
        [
            (loader.construct_object(k), loader.construct_object(v))
            for k, v in node.value
        ]
    )


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def read_mapping(path: Path, *, missing_ok: bool = True) -> dict[str, Any]:
    if not path.exists() and missing_ok:
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.load(stream, Loader=UniqueKeyLoader)  # noqa: S506 -- SafeLoader subclass
            elif path.suffix.lower() == ".json":
                data = json.load(stream, object_pairs_hook=_unique_pairs)
            else:
                raise ConfigurationError(
                    "Configuration extension must be .yaml, .yml or .json"
                )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # Parser messages may contain credentials from the offending line.
        mark = getattr(exc, "problem_mark", None)
        line = getattr(exc, "lineno", None) or (mark.line + 1 if mark else None)
        location = f" at line {line}" if line else ""
        raise ConfigurationError(
            f"Cannot parse configuration {path}: {type(exc).__name__}{location}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration {path} must be a mapping")
    return data


def validate(data: dict[str, Any]) -> AppSettings:
    try:
        return AppSettings.model_validate(data)
    except ValidationError as exc:
        fields = "; ".join(
            ".".join(map(str, e["loc"])) + ": " + e["type"] for e in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {fields}") from exc


def load_settings(path: str | None = None) -> AppSettings:
    source = selected_path(path)
    raw = read_mapping(
        source,
        missing_ok=not bool(
            path or _selected.get() or os.environ.get("RAI_CONFIG_FILE")
        ),
    )
    settings = {k: v for k, v in raw.items() if k not in STATE_KEYS}
    if "sessions" in settings:
        raise ConfigurationError(
            "Legacy 'sessions' configuration requires explicit migration"
        )
    # Narrow support for the existing split config; no writes or tool/model migrations.
    if "agents" not in settings:
        profiles = source.parent / "agents.yaml"
        if profiles.exists():
            settings["agents"] = read_mapping(profiles)
    return validate(settings)


def atomic_write(path: Path, data: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    text = (
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        if path.suffix.lower() in {".yaml", ".yml"}
        else json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(name, path)
        else:
            os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def state_path(path: str | None = None) -> Path:
    source = str(selected_path(path))
    key = hashlib.sha256(source.encode()).hexdigest()[:24]
    return state_dir() / f"session-{key}.json"


def load_runtime_state(path: str | None = None) -> dict[str, Any]:
    source = read_mapping(selected_path(path))
    state = {k: v for k, v in source.items() if k in STATE_KEYS}
    state.update(read_mapping(state_path(path)))
    return state


def save_runtime_state(data: dict[str, Any], path: str | None = None) -> None:
    target = state_path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(str(target) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = load_runtime_state(path)
        current.update(data)
        atomic_write(target, current)


def save_settings(data: dict[str, Any], path: str | None = None) -> None:
    target = selected_path(path)
    settings = validate({k: v for k, v in data.items() if k not in STATE_KEYS})
    # Persist the existing legacy state before removing it from the settings file.
    state = load_runtime_state(path)
    state.update({k: v for k, v in data.items() if k in STATE_KEYS})
    if state:
        save_runtime_state(state, path)
    atomic_write(target, settings.runtime_mapping())


def redact(value: Any) -> Any:  # noqa: ANN401 -- recursively redact untyped configuration data
    if isinstance(value, dict):
        return {
            k: "<redacted>"
            if any(
                s in k.lower()
                for s in (
                    "api_key",
                    "password",
                    "secret",
                    "credential",
                    "authorization",
                )
            )
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
