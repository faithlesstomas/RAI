"""Read-only aggregate history diagnostics; no decryption or collector startup."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from returns.result import Failure, Result, Success


def history_storage_status(path: Path) -> Result[dict[str, object], str]:
    """Report freshness without creating a database or obtaining an encryption key."""
    if not path.exists():
        return Success({"exists": False})
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            observations, latest_observation = connection.execute(
                "SELECT count(*), max(timestamp) FROM observations"
            ).fetchone()
            episodes, latest_episode = connection.execute(
                "SELECT count(*), max(ended_at) FROM episodes"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return Failure("HISTORY_STATUS_UNAVAILABLE")
    return Success({
        "exists": True, "observations": observations, "episodes": episodes,
        "latest_observation": latest_observation, "latest_episode": latest_episode,
    })
