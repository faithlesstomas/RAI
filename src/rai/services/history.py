"""
History Service for managing persistent chat history using SQLite.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite
from returns.result import Failure, Result, Success

logger = logging.getLogger(__name__)

class HistoryService:
    """
    Manages chat history persistence using SQLite.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path:
            self.db_path = db_path
        else:
            config_dir = os.path.expanduser("~/.config/rai")
            os.makedirs(config_dir, exist_ok=True)
            self.db_path = os.path.join(config_dir, "history.db")
        
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        """Synchronous initialization for simpler setup, though actual usage is async."""
        # We'll rely on lazy initialization or explicit async init if strictly needed.
        # But aiosqlite is async, so we'll do the schema check on first connection or separate method.
        pass

    async def _ensure_schema(self) -> None:
        """Ensures the database schema exists."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON messages (session_id)")
            await db.commit()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Result[None, Exception]:
        """Adds a message to the history."""
        try:
            await self._ensure_schema()
            
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, tool_calls_json)
                )
                await db.commit()
            return Success(None)
        except Exception as e:
            logger.error(f"Failed to add message to history: {e}")
            return Failure(e)

    async def get_session_history(self, session_id: str) -> Result[List[Dict[str, Any]], Exception]:
        """Retrieves history for a session."""
        try:
            await self._ensure_schema()
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT role, content, tool_calls, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    
            history = []
            for row in rows:
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["created_at"]
                }
                if row["tool_calls"]:
                     try:
                         msg["tool_calls"] = json.loads(row["tool_calls"])
                     except json.JSONDecodeError:
                         pass
                history.append(msg)
                
            return Success(history)
        except Exception as e:
            logger.error(f"Failed to get history for session {session_id}: {e}")
            return Failure(e)

    async def clear_history(self, session_id: str) -> Result[None, Exception]:
        """Clears history for a specific session."""
        try:
            await self._ensure_schema()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                await db.commit()
            return Success(None)
        except Exception as e:
            return Failure(e)

    async def delete_session(self, session_id: str) -> Result[None, Exception]:
        """Alias for clear_history."""
        return await self.clear_history(session_id)

    async def list_sessions(self) -> Result[List[Dict[str, Any]], Exception]:
         """Lists all sessions with their last update time."""
         try:
            await self._ensure_schema()
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                # Group by session_id and get max timestamp
                async with db.execute(
                    """
                    SELECT session_id, MAX(created_at) as last_active, COUNT(*) as msg_count 
                    FROM messages 
                    GROUP BY session_id 
                    ORDER BY last_active DESC
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
            
            sessions = []
            for row in rows:
                sessions.append({
                    "id": row["session_id"],
                    "last_active": row["last_active"],
                    "message_count": row["msg_count"]
                })
            return Success(sessions)
         except Exception as e:
             return Failure(e)
