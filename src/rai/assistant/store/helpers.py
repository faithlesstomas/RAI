"""Row mappers, text tokenizers, and serialization helpers for assistant store."""

from __future__ import annotations

from datetime import datetime
import json
import re
import sqlite3
from typing import Any
import unicodedata

from rai.kernel.records import (
    DataClass,
    ProducerIdentity,
    ProvenanceReference,
)

from ..query import (
    GLOBAL_DOMAIN_SCOPE,
    restrictive_domain_scopes,
)
from ..records import (
    AssistantResponse,
    ConversationTurn,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
)


def to_data_class_str(dc: DataClass | str) -> str:
    return dc.value if isinstance(dc, DataClass) else str(dc)


def to_relation_kind_str(rk: MemoryRelationKind | str) -> str:
    return rk.value if isinstance(rk, MemoryRelationKind) else str(rk)


def searchable_text(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )


MIN_STEM_LEN: int = 6


def fts_query(terms: tuple[str, ...]) -> str:
    """Build a bounded FTS5 OR query from resolver-produced lexical terms."""
    tokens = tuple(
        dict.fromkeys(
            token
            for term in terms
            for token in re.findall(r"[a-z0-9]+", searchable_text(term))
            if token
        )
    )
    return " OR ".join(f'"{token}"' for token in tokens[:16])


def domain_sql_filter(
    column: str, query_scopes: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Build a bounded SQL prefilter equivalent to domain_scope_matches()."""
    restrictive = restrictive_domain_scopes(query_scopes)
    if not restrictive:
        return "", ()

    exact = {GLOBAL_DOMAIN_SCOPE}
    descendant_patterns: set[str] = set()
    for scope in restrictive:
        parts = scope.split(":")
        exact.update(":".join(parts[:index]) for index in range(1, len(parts) + 1))
        descendant_patterns.add(f"{scope}:%")
    exact.update(
        scope for scope in query_scopes if scope in {"conversation", "activity"}
    )
    ordered_exact = tuple(sorted(exact))
    ordered_patterns = tuple(sorted(descendant_patterns))
    clauses = []
    params: list[str] = []
    if ordered_exact:
        placeholders = ",".join("?" for _ in ordered_exact)
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(ordered_exact)
    clauses.extend(f"{column} LIKE ?" for _ in ordered_patterns)
    params.extend(ordered_patterns)
    return f" AND ({' OR '.join(clauses)}) ", tuple(params)


def relation_metadata(relation: MemoryRelation) -> dict[str, Any]:
    """Persist relation qualifiers inside its inspectable metadata payload."""
    policy_outcome = getattr(relation.policy_outcome, "value", relation.policy_outcome)
    return {
        **relation.metadata,
        "confidence": relation.confidence,
        "epistemic_status": relation.epistemic_status,
        "provenance": [item.model_dump(mode="json") for item in relation.provenance],
        "policy_outcome": str(policy_outcome),
        "eligible": relation.eligible,
    }


def semantic_memory_content(content: dict[str, Any]) -> str:
    keys = ("subject", "predicate", "value", "preference", "plan", "fact", "attribute")
    semantic = {key: content[key] for key in keys if key in content}
    return json.dumps(semantic or content, sort_keys=True, ensure_ascii=False)


def memory_from_row(row: sqlite3.Row) -> MemoryRecord:
    provenance = tuple(
        ProvenanceReference.model_validate(item)
        for item in json.loads(row["provenance_json"])
    )
    return MemoryRecord(
        record_id=row["memory_id"],
        timestamp=datetime.fromisoformat(row["created_at"]),
        producer=ProducerIdentity.model_validate_json(row["producer_json"]),
        kind=row["kind"],
        topic=row["topic"],
        content=json.loads(row["content_json"]),
        source_turn_id=row["source_turn_id"],
        source_type=row["source_type"],
        data_class=DataClass(row["data_class"]),
        profile_scope=row["profile_scope"],
        domain_scope=row["domain_scope"],
        purpose=row["purpose"],
        epistemic_status=row["epistemic_status"],
        valid_from=datetime.fromisoformat(row["valid_from"]),
        valid_until=(
            datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None
        ),
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
        expired_at=(
            datetime.fromisoformat(row["expired_at"]) if row["expired_at"] else None
        ),
        provenance=provenance,
        correlation_id=row["correlation_id"],
    )


def turn_from_row(row: sqlite3.Row) -> ConversationTurn:
    """Restore one immutable turn from its SQLite representation."""
    return ConversationTurn(
        record_id=row["turn_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        producer=ProducerIdentity.model_validate_json(row["producer_json"]),
        session_id=row["session_id"],
        role=row["role"],
        text=row["text"],
        reply_to_turn_id=row["reply_to_turn_id"],
        data_class=DataClass(row["data_class"]),
        domain_scope=row["domain_scope"],
        purpose=row["purpose"],
        status=row["status"],
        correlation_id=row["correlation_id"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


def response_from_row(row: sqlite3.Row) -> AssistantResponse:
    """Restore one terminal response from its authoritative store row."""
    admitted = tuple(json.loads(row["admitted_memories_json"]))
    operation_ids = tuple(json.loads(row["memory_operations_json"] or "[]"))
    provenance = tuple(
        ProvenanceReference.model_validate(item)
        for item in json.loads(row["provenance_json"])
    )
    return AssistantResponse(
        record_id=row["response_id"],
        timestamp=datetime.fromisoformat(row["created_at"]),
        producer=(
            ProducerIdentity.model_validate_json(row["producer_json"])
            if row["producer_json"]
            else ProducerIdentity(
                producer_id="assistant-store", kind="store", version="1.0.0"
            )
        ),
        correlation_id=row["correlation_id"],
        schema_version=row["schema_version"] or "1.0.0",
        reasoning_content=row["reasoning_content"],
        session_id=row["session_id"],
        turn_id=row["assistant_turn_id"] or row["user_turn_id"],
        user_turn_id=row["user_turn_id"],
        request_id=row["request_id"],
        manifest_id=row["manifest_id"],
        text=row["text"],
        status=row["status"],
        error_message=row["error_message"],
        admitted_memory_ids=admitted,
        memory_operation_ids=operation_ids,
        provenance=provenance,
    )


def dependent_memory_ids(
    conn: sqlite3.Connection,
    memory_ids: tuple[str, ...],
    profile_scope: str | None = None,
) -> tuple[str, ...]:
    if not memory_ids:
        return ()
    placeholders = ",".join("?" for _ in memory_ids)
    profile_base = ""
    profile_recursive = ""
    profile_params: tuple[str, ...] = ()
    if profile_scope is not None:
        profile_base = "AND target_lifecycle.profile_scope = ?"
        profile_recursive = "AND recursive_lifecycle.profile_scope = ?"
        profile_params = (profile_scope, profile_scope)
    query = f"""
        WITH RECURSIVE dependent(memory_id) AS (
            SELECT relation.target_id
            FROM relations AS relation
            JOIN memory_lifecycle AS target_lifecycle
              ON target_lifecycle.memory_id = relation.target_id
            WHERE relation.source_id IN ({placeholders})
              AND relation.kind = ?
              AND json_extract(relation.metadata_json, '$.eligible') = 1
              AND json_extract(
                    relation.metadata_json, '$.policy_outcome'
                  ) = 'ALLOW'
              AND target_lifecycle.status = 'ACTIVE'
              {profile_base}
            UNION
            SELECT relation.target_id
            FROM relations AS relation
            JOIN dependent ON relation.source_id = dependent.memory_id
            JOIN memory_lifecycle AS recursive_lifecycle
              ON recursive_lifecycle.memory_id = relation.target_id
            WHERE relation.kind = ?
              AND json_extract(relation.metadata_json, '$.eligible') = 1
              AND json_extract(
                    relation.metadata_json, '$.policy_outcome'
                  ) = 'ALLOW'
              AND recursive_lifecycle.status = 'ACTIVE'
              {profile_recursive}
        )
        SELECT dependent.memory_id
        FROM dependent
        JOIN memories ON memories.memory_id = dependent.memory_id
        ORDER BY dependent.memory_id
        """  # noqa: S608 - placeholders cover every dynamic value
    rows = conn.execute(
        query,
        (
            *memory_ids,
            MemoryRelationKind.SUPPORTS.value,
            *profile_params[:1],
            MemoryRelationKind.SUPPORTS.value,
            *profile_params[1:],
        ),
    ).fetchall()
    source_ids = set(memory_ids)
    return tuple(
        str(row["memory_id"])
        for row in rows
        if str(row["memory_id"]) not in source_ids
    )
