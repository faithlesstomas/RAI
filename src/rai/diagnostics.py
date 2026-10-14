"""Opt-in runtime traces with configurable verbosity levels.

Level 1 (default --trace): Stage timing, operational metadata, no prompt/payload content.
Level 2 (--trace --trace / -tt / RAI_TRACE=2): Full prompt/messages payloads, context breakdown,
token usage, and budget accounting.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

LOGGER = logging.getLogger("rai.trace")


def configure_trace(enabled: bool | int = False) -> None:
    """Enable stderr traces in CLI and daemon workers via RAI_TRACE."""
    if isinstance(enabled, int):
        level = enabled
    elif enabled:
        level = 1
    else:
        env_val = os.environ.get("RAI_TRACE", "0")
        level = int(env_val) if env_val.isdigit() else (1 if env_val == "1" else 0)

    if level <= 0:
        return
    os.environ["RAI_TRACE"] = str(level)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = True


def trace_level() -> int:
    """Return the active trace verbosity level (0=off, 1=metadata, 2=full payload)."""
    env_val = os.environ.get("RAI_TRACE", "0")
    return int(env_val) if env_val.isdigit() else (1 if env_val == "1" else 0)


_METADATA_TRACE_LEVEL = 1
_PAYLOAD_TRACE_LEVEL = 2


def trace(event: str, **fields: object) -> None:
    """Callers supply only explicit, non-content diagnostic fields (active at level >= 1)."""
    if trace_level() < _METADATA_TRACE_LEVEL:
        return
    LOGGER.info("%s %s", event, " ".join(f"{key}={value}" for key, value in fields.items()))


def trace_payload(event: str, payload: object, **fields: object) -> None:
    """Log full payload, prompt, or context contents when trace level >= 2."""
    if trace_level() < _PAYLOAD_TRACE_LEVEL:
        return
    fields_str = (" " + " ".join(f"{key}={value}" for key, value in fields.items())) if fields else ""
    header = f"{event} [PAYLOAD/DEBUG]{fields_str}"
    LOGGER.info(header)
    if isinstance(payload, str):
        content_str = payload
    else:
        try:
            content_str = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:  # noqa: BLE001
            content_str = str(payload)
    LOGGER.info("--- BEGIN %s ---\n%s\n--- END %s ---", event, content_str, event)
