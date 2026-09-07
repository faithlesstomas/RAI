"""Bounded AT-SPI semantic focus/state/document activity producer."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .common import emit

MAX_EVENTS_PER_SECOND = 20
MAX_TREE_DEPTH = 32
COALESCE_MS = 500


@dataclass
class _TextActivity:
    count: int
    started: float
    application_id: str | None
    role: str
    toolkit: str | None
    resource_id: str | None


def _safe(callable_: Any, default: Any = None) -> Any:  # noqa: ANN401
    try:
        return callable_()
    except Exception:  # accessibility objects can disappear between calls
        return default


def _metadata(source: Any) -> dict[str, Any]:  # noqa: ANN401
    application = _safe(source.get_application)
    attributes = _safe(source.get_attributes, {}) or {}
    if not isinstance(attributes, dict):
        attributes = {}
    role = str(_safe(source.get_role_name, "unknown"))[:128]
    pid = int(_safe(source.get_process_id, 0) or 0)
    app_name = str(_safe(application.get_name, "")) if application else ""
    parent = source
    depth = 0
    while parent is not None and depth < MAX_TREE_DEPTH:
        parent = _safe(parent.get_parent)
        depth += 1
    document_uri = attributes.get("DocURL") or attributes.get("document-uri")
    return {
        "application_id": app_name.casefold() or None,
        "field_role": role,
        "resource_id": str(document_uri)[:512] if document_uri else (f"process:{pid}" if pid else None),
        "toolkit": str(attributes.get("toolkit", ""))[:128] or None,
        "quality": 1.0 if document_uri else 0.7,
        "depth": depth,
    }


def run() -> None:  # noqa: PLR0915
    try:
        import gi  # type: ignore[import-not-found]  # noqa: PLC0415

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi, GLib  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit("AT-SPI bindings are unavailable") from exc

    text_activity: dict[tuple[str | None, str], _TextActivity] = {}
    rate: dict[int, int] = defaultdict(int)

    def allowed() -> bool:
        second = int(time.monotonic())
        rate[second] += 1
        for expired in tuple(key for key in rate if key < second):
            del rate[expired]
        return rate[second] <= MAX_EVENTS_PER_SECOND

    def on_event(event: Any) -> None:  # noqa: ANN401
        if not allowed():
            return
        metadata = _metadata(event.source)
        event_type = str(event.type)
        if "text-changed" in event_type:
            key = (metadata["application_id"], metadata["field_role"])
            activity = text_activity.get(key)
            if activity is None:
                text_activity[key] = _TextActivity(
                    1, time.monotonic(), metadata["application_id"],
                    metadata["field_role"], metadata["toolkit"], metadata["resource_id"],
                )
            else:
                activity.count += 1
            return
        kind = (
            "focus" if "focused" in event_type
            else "document_context" if "document" in event_type
            else "state_changed"
        )
        emit("atspi", kind, **metadata, payload={"states": [event_type]})

    def flush_text() -> bool:
        now = time.monotonic()
        for key, activity in tuple(text_activity.items()):
            duration = int((now - activity.started) * 1000)
            emit(
                "atspi", "text_activity", application_id=activity.application_id,
                field_role=activity.role, toolkit=activity.toolkit,
                resource_id=activity.resource_id,
                payload={"change_count": activity.count, "duration_ms": duration},
            )
            del text_activity[key]
        return True

    listener = Atspi.EventListener.new(on_event)
    for event_type in (
        "object:state-changed:focused", "object:state-changed:selected",
        "object:document:load-complete", "object:document:reload",
        "object:text-changed:insert", "object:text-changed:delete",
    ):
        listener.register(event_type)
    GLib.timeout_add(COALESCE_MS, flush_text)
    GLib.MainLoop().run()


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
