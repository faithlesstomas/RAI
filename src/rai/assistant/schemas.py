"""Language-neutral JSON Schema export for Rich Assistant records."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from rai.kernel.records import CURRENT_SCHEMA_VERSION

from .records import AnyAssistantRecord

ASSISTANT_SCHEMA_ID = (
    "https://tk-lab1.gitlab.io/ai/rai/schemas/rai.assistant.v1.schema.json"
)


def assistant_json_schema() -> dict[str, Any]:
    """Return the canonical schema for every Rich Assistant domain record."""
    schema = TypeAdapter(AnyAssistantRecord).json_schema(mode="validation")
    schema["$id"] = ASSISTANT_SCHEMA_ID
    schema["title"] = "RAI assistant domain records v1"
    schema["x-rai-schema-version"] = CURRENT_SCHEMA_VERSION
    return schema
