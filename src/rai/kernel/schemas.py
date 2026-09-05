"""Language-neutral JSON Schema export for kernel records."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from .records import AnyKernelRecord, CURRENT_SCHEMA_VERSION
from .events import EVENT_SCHEMA_VERSION, EventBatch, EventEnvelope

SCHEMA_ID = "https://tk-lab1.gitlab.io/rai/schemas/rai.kernel.v1.schema.json"
EVENT_SCHEMA_ID = "https://tk-lab1.gitlab.io/ai/rai/schemas/rai.events.v1.schema.json"


def kernel_json_schema() -> dict[str, Any]:
    """Return the canonical schema for every Stage 1 domain record."""
    schema = TypeAdapter(AnyKernelRecord).json_schema(mode="validation")
    schema["$id"] = SCHEMA_ID
    schema["title"] = "RAI embodiment kernel records"
    schema["x-rai-schema-version"] = CURRENT_SCHEMA_VERSION
    return schema


def event_json_schema() -> dict[str, Any]:
    """Return the canonical schema for a Stage 2 envelope or replay batch."""
    schema = TypeAdapter(EventEnvelope | EventBatch).json_schema(mode="validation")
    schema["$id"] = EVENT_SCHEMA_ID
    schema["title"] = "RAI Event Plane v1"
    schema["x-rai-schema-version"] = EVENT_SCHEMA_VERSION
    return schema
