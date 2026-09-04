"""Stable machine-readable rendering for kernel CLI commands."""

from __future__ import annotations

import json

from .kernel.transport import InvocationEnvelope


def render_envelope(envelope: InvocationEnvelope) -> str:
    return json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)


def render_capabilities(capabilities: list[dict[str, object]]) -> str:
    return json.dumps({"capabilities": capabilities}, indent=2, sort_keys=True)
