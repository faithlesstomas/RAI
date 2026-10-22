"""REPL session state persistence mixin for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
import sqlite3


class ReplStoreMixin:
    """Session REPL heap snapshots and execution state persistence."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError
