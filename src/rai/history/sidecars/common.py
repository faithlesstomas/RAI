"""Sidecar output helpers with bounded, content-free failures."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

MAX_OUTPUT_BYTES = 64 * 1024


def emit(source: str, kind: str, **fields: Any) -> None:  # noqa: ANN401
    """Write one bounded semantic record without ever using stderr."""
    record = {
        "source": source,
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    encoded = json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        return
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()
