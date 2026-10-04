"""Versioned wire contracts for proposals, resource authority and terminal results."""
from typing import Any

from pydantic import TypeAdapter

from rai.kernel.records import ActionFailure, ActionResult, CURRENT_SCHEMA_VERSION
from .records import ActionProposal, ResourceHandle


def action_json_schema() -> dict[str, Any]:
    """Export runtime records; resource targets remain internal to trusted storage."""
    schema = TypeAdapter(ActionProposal | ResourceHandle | ActionResult | ActionFailure).json_schema(mode="validation")
    schema["$id"] = "https://tk-lab1.gitlab.io/ai/rai/schemas/rai.actions.v1.schema.json"
    schema["title"] = "RAI Linux action records v1"
    schema["x-rai-schema-version"] = CURRENT_SCHEMA_VERSION
    return schema
