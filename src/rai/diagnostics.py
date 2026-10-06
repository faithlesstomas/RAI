"""Opt-in runtime traces containing metadata only, never prompts or tool payloads."""
from __future__ import annotations

import logging
import os

LOGGER = logging.getLogger("rai.trace")


def configure_trace(enabled: bool = False) -> None:
    """Enable stderr traces in CLI and daemon workers via RAI_TRACE."""
    if not enabled and os.environ.get("RAI_TRACE") != "1":
        return
    os.environ["RAI_TRACE"] = "1"
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def trace(event: str, **fields: object) -> None:
    """Callers supply only explicit, non-content diagnostic fields."""
    if os.environ.get("RAI_TRACE") != "1":
        return
    LOGGER.info("%s %s", event, " ".join(f"{key}={value}" for key, value in fields.items()))
