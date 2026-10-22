"""Transactional SQLite reference implementation of MemoryGraphStore."""

from __future__ import annotations

import asyncio
import sqlite3

from .deletion import DeletionStoreMixin
from .graph import GraphStoreMixin
from .helpers import (
    dependent_memory_ids,
    domain_sql_filter,
    fts_query,
    memory_from_row,
    relation_metadata,
    response_from_row,
    searchable_text,
    semantic_memory_content,
    to_data_class_str,
    to_relation_kind_str,
    turn_from_row,
)
from .lifecycle import (
    _apply_forget_operations,
    _apply_memories_and_lifecycle,
)
from .repl import ReplStoreMixin
from .responses import ResponsesStoreMixin
from .schema import SchemaStoreMixin
from .search import SearchStoreMixin
from .terminal import TerminalStoreMixin
from .turns import TurnsStoreMixin


class SQLiteMemoryGraphStore(
    SchemaStoreMixin,
    ResponsesStoreMixin,
    DeletionStoreMixin,
    TurnsStoreMixin,
    TerminalStoreMixin,
    GraphStoreMixin,
    SearchStoreMixin,
    ReplStoreMixin,
):
    """Graph-aware SQLite store with strict referential integrity via central nodes."""


__all__ = [
    "SQLiteMemoryGraphStore",
    "asyncio",
    "sqlite3",
    "to_data_class_str",
    "to_relation_kind_str",
    "searchable_text",
    "fts_query",
    "domain_sql_filter",
    "relation_metadata",
    "semantic_memory_content",
    "memory_from_row",
    "turn_from_row",
    "response_from_row",
    "dependent_memory_ids",
]
