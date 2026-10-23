"""Search, retrieval, and lexical indexing for SQLiteMemoryGraphStore."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass

from ..ports import MemoryQuery
from ..query import domain_scope_matches
from ..records import (
    ConversationTurn,
    MemoryRecord,
    make_assistant_failure,
)
from .helpers import (
    MIN_STEM_LEN,
    domain_sql_filter,
    fts_query,
    memory_from_row,
    searchable_text,
    to_data_class_str,
    turn_from_row,
)


class SearchStoreMixin:
    """FTS5 and lexical retrieval queries for memories and turns."""

    _lock: asyncio.Lock

    def _connect(self) -> sqlite3.Connection:
        raise NotImplementedError

    def _init_db(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError

    async def retrieve_relevant_memories(
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_retrieve_memories, profile_scope, query, data_classes, limit
            )

    def _sync_retrieve_memories(  # noqa: PLR0912, PLR0915
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            cur = conn.cursor()
            allowed_classes = [to_data_class_str(dc) for dc in data_classes]
            placeholders = ",".join("?" for _ in allowed_classes)
            now = datetime.now(timezone.utc)
            transaction_at = query.transaction_at if query else None
            valid_at = query.valid_at if query else None
            transaction_iso = (transaction_at or now).isoformat()
            valid_iso = (valid_at or transaction_at or now).isoformat()

            built_fts_query = fts_query(query.keywords if query else ())
            use_fts = bool(built_fts_query and not (query and query.topic))
            select_prefix = (
                "SELECT m.*, l.status AS lifecycle_status, "
                "bm25(memories_fts) AS bm25_score "
                "FROM memories_fts "
                "JOIN memories m ON m.memory_id = memories_fts.memory_id "
                if use_fts
                else "SELECT m.*, l.status AS lifecycle_status, 0.0 AS bm25_score FROM memories m "
            )
            sql = (
                select_prefix  # noqa: S608
                + "JOIN memory_lifecycle l ON m.memory_id = l.memory_id "
                "WHERE m.profile_scope = ? "
                f"  AND m.data_class IN ({placeholders}) "
                "  AND m.recorded_at <= ? "
                "  AND (m.expired_at IS NULL OR m.expired_at > ?) "
                "  AND m.valid_from <= ? "
                "  AND (m.valid_until IS NULL OR m.valid_until > ?) "
            )
            params: list[Any] = [
                profile_scope,
                *allowed_classes,
                transaction_iso,
                transaction_iso,
                valid_iso,
                valid_iso,
            ]
            if query is not None:
                if query.domain_scopes:
                    domain_sql, domain_params = domain_sql_filter(
                        "m.domain_scope", query.domain_scopes
                    )
                    sql += domain_sql
                    params.extend(domain_params)
                sql += " AND m.purpose = ? "
                params.append(query.purpose)
            if use_fts:
                sql += " AND memories_fts MATCH ? AND memories_fts.profile_scope = ? "
                params.extend((built_fts_query, profile_scope))
                sql += " ORDER BY bm25_score ASC, m.created_at DESC"
            else:
                sql += " ORDER BY m.created_at DESC"
            cur.execute(sql, params)
            rows = cur.fetchall()

            scored: list[tuple[float, MemoryRecord, str]] = []
            target_topic = query.topic if query else None
            keywords = query.keywords if query else ()

            for row in rows:
                record = memory_from_row(row)
                if query is not None and not domain_scope_matches(
                    record.domain_scope, query.domain_scopes
                ):
                    continue
                content = record.content

                score = 0.0
                reason = "active durable memory"

                if target_topic and record.topic == target_topic:
                    score = 1.0
                    reason = f"topic exact match: '{record.topic}' (score: 1.0)"
                elif keywords:
                    content_str = searchable_text(
                        f"{record.topic} {json.dumps(content, ensure_ascii=False)}"
                    )
                    matched = [
                        kw for kw in keywords if searchable_text(kw) in content_str
                    ]
                    if matched:
                        score = 0.5 + 0.1 * len(matched)
                        reason = (
                            f"FTS5/BM25 claim match on {matched} "
                            f"(bm25: {float(row['bm25_score']):.4f})"
                        )

                if target_topic is None and not keywords:
                    score = 0.1

                if score > 0.0:
                    scored.append((score, record, reason))

            if not scored and use_fts and keywords:
                stems = [
                    (kw, searchable_text(kw)[:MIN_STEM_LEN])
                    for kw in keywords
                    if len(searchable_text(kw)) >= MIN_STEM_LEN
                ]
                if stems:
                    fallback_sql = (
                        "SELECT m.*, l.status AS lifecycle_status, 0.0 AS bm25_score FROM memories m "  # noqa: S608
                        "JOIN memory_lifecycle l ON m.memory_id = l.memory_id "
                        "WHERE m.profile_scope = ? "
                        "  AND l.status = 'ACTIVE' "
                        f"  AND m.data_class IN ({placeholders}) "
                        "  AND m.recorded_at <= ? "
                        "  AND (m.expired_at IS NULL OR m.expired_at > ?) "
                        "  AND m.valid_from <= ? "
                        "  AND (m.valid_until IS NULL OR m.valid_until > ?) "
                        "  AND m.purpose = ? "
                        "ORDER BY m.created_at DESC LIMIT 50"
                    )
                    fallback_params: list[Any] = [
                        profile_scope,
                        *allowed_classes,
                        transaction_iso,
                        transaction_iso,
                        valid_iso,
                        valid_iso,
                        query.purpose if query else "assistant",
                    ]
                    cur.execute(fallback_sql, fallback_params)
                    for row in cur.fetchall():
                        record = memory_from_row(row)
                        if query is not None and not domain_scope_matches(
                            record.domain_scope, query.domain_scopes
                        ):
                            continue
                        content_str = searchable_text(
                            f"{record.kind} {record.topic} {json.dumps(record.content, ensure_ascii=False)}"
                        )
                        matched_stems = [
                            kw for kw, stem in stems if stem in content_str
                        ]
                        if matched_stems:
                            scored.append(
                                (
                                    0.4 + 0.1 * len(matched_stems),
                                    record,
                                    f"stem claim match on {matched_stems}",
                                )
                            )

            scored.sort(key=lambda item: item[0], reverse=True)
            result = tuple((rec, reason) for _, rec, reason in scored[:limit])
            return Success(result)
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RETRIEVE_MEMORIES_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
                )
            )
        finally:
            conn.close()

    async def retrieve_relevant_turns(
        self,
        profile_scope: str,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        exclude_turn_ids: tuple[str, ...] = (),
        limit: int = 5,
    ) -> Result[tuple[tuple[ConversationTurn, str], ...], ActionFailure]:
        """Retrieve source turns independently of the durable claim projection."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_retrieve_turns,
                profile_scope,
                query,
                data_classes,
                exclude_turn_ids,
                limit,
            )

    def _sync_retrieve_turns(
        self,
        profile_scope: str,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        exclude_turn_ids: tuple[str, ...],
        limit: int,
    ) -> Result[tuple[tuple[ConversationTurn, str], ...], ActionFailure]:
        conn = self._connect()
        try:
            self._init_db(conn)
            allowed_classes = [to_data_class_str(dc) for dc in data_classes]
            placeholders = ",".join("?" for _ in allowed_classes)
            built_fts_query = fts_query(query.keywords)
            excluded = set(exclude_turn_ids)
            query_terms = tuple(
                dict.fromkeys(searchable_text(term) for term in query.keywords if term)
            )
            broad_recall = any(
                marker in searchable_text(query.raw_text)
                for marker in (
                    "co wiesz o mnie",
                    "co pamietasz",
                    "what do you know about me",
                    "what do you remember",
                )
            )
            domain_clause, domain_params = domain_sql_filter(
                "t.domain_scope", query.domain_scopes
            )
            if built_fts_query:
                rows = conn.execute(
                    (
                        "SELECT t.*, bm25(turns_fts) AS bm25_score "  # noqa: S608
                        "FROM turns_fts JOIN turns AS t "
                        "ON t.turn_id = turns_fts.turn_id "
                        "WHERE turns_fts MATCH ? AND turns_fts.profile_scope = ? "
                        "AND t.role = 'user' AND t.status = 'COMPLETED' "
                        f"AND t.data_class IN ({placeholders}) "
                        + domain_clause
                        + "AND t.purpose = ? "
                        + "ORDER BY bm25_score ASC, t.timestamp DESC LIMIT 200"
                    ),
                    (
                        built_fts_query,
                        profile_scope,
                        *allowed_classes,
                        *domain_params,
                        query.purpose,
                    ),
                ).fetchall()
            elif broad_recall:
                rows = conn.execute(
                    (
                        "SELECT t.*, 0.0 AS bm25_score FROM turns AS t "  # noqa: S608
                        "WHERE t.role = 'user' AND t.status = 'COMPLETED' "
                        f"AND t.data_class IN ({placeholders}) "
                        + domain_clause
                        + "AND t.purpose = ? "
                        + "ORDER BY t.timestamp DESC LIMIT 200"
                    ),
                    (*allowed_classes, *domain_params, query.purpose),
                ).fetchall()
            else:
                rows = []
            scored: list[tuple[float, ConversationTurn, str]] = []
            for recency, row in enumerate(rows):
                turn = turn_from_row(row)
                if turn.record_id in excluded:
                    continue
                turn_scope = str(
                    getattr(turn, "metadata", {}).get("profile_scope", "default")
                )
                if turn_scope != profile_scope:
                    continue
                if not domain_scope_matches(turn.domain_scope, query.domain_scopes):
                    continue
                searchable = searchable_text(turn.text)
                matched = tuple(term for term in query_terms if term in searchable)
                if not matched and not broad_recall:
                    continue
                lexical = len(matched) / max(1, len(query_terms))
                bm25_score = float(row["bm25_score"])
                score = lexical + max(0.0, 0.1 - recency / 10_000)
                reason = (
                    "FTS5/BM25 source-turn match on "
                    f"{list(matched)} (bm25: {bm25_score:.4f})"
                    if matched
                    else f"source-turn broad recall (score: {score:.2f})"
                )
                scored.append((score, turn, reason))
            scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
            bounded_limit = max(1, min(limit, 50))
            return Success(
                tuple((turn, reason) for _, turn, reason in scored[:bounded_limit])
            )
        except Exception as exc:  # noqa: BLE001
            return Failure(
                make_assistant_failure(
                    code="RETRIEVE_TURNS_FAILED",
                    message=str(exc),
                    request_id=profile_scope,
                )
            )
        finally:
            conn.close()
