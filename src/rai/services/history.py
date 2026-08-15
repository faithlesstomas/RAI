"""SQLite-backed conversation history for the local RAI runtime."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from returns.result import Failure, Result, Success

from rai.paths import data_dir

logger = logging.getLogger(__name__)


class HistoryService:
    """Persist short conversation records independently of any model backend."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = db_path
        else:
            history_dir = data_dir()
            history_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(history_dir / "history.db")
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)"
            )

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Result[None, Exception]:
        """Add one message. The async API keeps callers backend-agnostic."""
        try:
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            with self._connect() as database:
                database.execute(
                    "INSERT INTO messages "
                    "(session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, tool_calls_json),
                )
            return Success(None)
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to add message to history: %s", error)
            return Failure(error)

    async def get_session_history(
        self, session_id: str
    ) -> Result[List[Dict[str, Any]], Exception]:
        """Return messages in deterministic insertion order."""
        try:
            with self._connect() as database:
                rows = database.execute(
                    "SELECT role, content, tool_calls, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()

            history: List[Dict[str, Any]] = []
            for row in rows:
                message: Dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["created_at"],
                }
                if row["tool_calls"]:
                    try:
                        message["tool_calls"] = json.loads(row["tool_calls"])
                    except json.JSONDecodeError:
                        logger.warning("Ignoring malformed tool_calls history value")
                history.append(message)
            return Success(history)
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to get history for session %s: %s", session_id, error)
            return Failure(error)

    async def clear_history(self, session_id: str) -> Result[None, Exception]:
        """Remove all messages belonging to a conversation session."""
        try:
            with self._connect() as database:
                database.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )
            return Success(None)
        except Exception as error:  # pylint: disable=broad-exception-caught
            return Failure(error)

    async def delete_session(self, session_id: str) -> Result[None, Exception]:
        """Alias for :meth:`clear_history`."""
        return await self.clear_history(session_id)

    async def list_sessions(self) -> Result[List[Dict[str, Any]], Exception]:
        """List sessions ordered by their latest stored message."""
        try:
            with self._connect() as database:
                rows = database.execute(
                    """
                    SELECT session_id, MAX(created_at) AS last_active,
                           COUNT(*) AS msg_count, MAX(id) AS last_id
                    FROM messages
                    GROUP BY session_id
                    ORDER BY last_id DESC
                    """
                ).fetchall()
            return Success(
                [
                    {
                        "id": row["session_id"],
                        "last_active": row["last_active"],
                        "message_count": row["msg_count"],
                    }
                    for row in rows
                ]
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            return Failure(error)
