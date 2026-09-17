"""Source-covered deterministic summaries used as a retrieval baseline."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass

from .ports import AssistantEvidence, MemoryGraphStore, MemoryQuery
from .records import MemoryRecord, make_assistant_failure

_DATA_CLASS_RANK = {
    DataClass.PUBLIC: 0,
    DataClass.LOCAL: 1,
    DataClass.PRIVATE: 2,
}
_EPISTEMIC_RANK = {
    "asserted": 0,
    "observed": 0,
    "inferred": 1,
    "uncertain": 2,
    "contested": 3,
}
_SEMANTIC_FIELDS = (
    "subject",
    "predicate",
    "value",
    "preference",
    "plan",
    "fact",
    "attribute",
)
_EVIDENCE_FIELDS = {
    "confidence",
    "modality",
    "proposal_id",
    "scope",
    "source_span",
    "source_type",
    "span_end",
    "span_start",
    "statement_type",
}


def _summary_id(memories: tuple[MemoryRecord, ...]) -> str:
    material = "\0".join(sorted(memory.record_id for memory in memories))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"grounded-summary:{digest}"


def _semantic_content(memory: MemoryRecord) -> dict[str, object]:
    semantic = {
        key: memory.content[key] for key in _SEMANTIC_FIELDS if key in memory.content
    }
    return (
        semantic
        or {
            key: value
            for key, value in memory.content.items()
            if key not in _EVIDENCE_FIELDS
        }
        or {"topic": memory.topic}
    )


class GroundedClaimSummaryProvider:
    """Build a non-persistent summary projection from eligible durable claims.

    The projection is deliberately deterministic and retains every contributing
    memory/source ID, modality and epistemic status. It is an evaluation channel,
    not an admission authority and not enabled in the default context router.
    """

    def __init__(
        self,
        store: MemoryGraphStore,
        *,
        max_claims_per_summary: int = 12,
    ) -> None:
        if max_claims_per_summary < 1:
            raise ValueError("max_claims_per_summary must be positive")
        self.store = store
        self.max_claims_per_summary = max_claims_per_summary

    async def retrieve(
        self,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        limit: int,
    ) -> Result[tuple[AssistantEvidence, ...], ActionFailure]:
        bounded_limit = max(1, min(limit, 50))
        retrieved = await self.store.retrieve_relevant_memories(
            profile_scope=query.profile_scope,
            query=query,
            data_classes=data_classes,
            limit=min(100, bounded_limit * self.max_claims_per_summary),
        )
        if isinstance(retrieved, Failure):
            return Failure(retrieved.failure())

        verified: list[tuple[MemoryRecord, str]] = []
        for memory, reason in retrieved.unwrap():
            source = await self.store.get_turn(memory.source_turn_id)
            if isinstance(source, Failure):
                return Failure(source.failure())
            if source.unwrap() is None:
                continue
            verified.append((memory, reason))

        grouped: dict[tuple[str, str], list[tuple[MemoryRecord, str]]] = defaultdict(
            list
        )
        for item in verified:
            memory, _reason = item
            grouped[(memory.domain_scope, memory.purpose)].append(item)

        evidence: list[AssistantEvidence] = []
        for (domain_scope, purpose), items in grouped.items():
            bounded = tuple(
                memory for memory, _reason in items[: self.max_claims_per_summary]
            )
            if not bounded:
                continue
            memory_ids = tuple(memory.record_id for memory in bounded)
            source_turn_ids = tuple(
                dict.fromkeys(memory.source_turn_id for memory in bounded)
            )
            source_confidences = tuple(
                memory.content.get("confidence") for memory in bounded
            )
            confidences = tuple(
                float(value)
                for value in source_confidences
                if isinstance(value, (float, int)) and not isinstance(value, bool)
            )
            summary_confidence = (
                min(confidences) if len(confidences) == len(bounded) else None
            )
            modalities = tuple(
                dict.fromkeys(
                    str(memory.content.get("modality", "unknown")) for memory in bounded
                )
            )
            epistemic_statuses = tuple(
                dict.fromkeys(memory.epistemic_status for memory in bounded)
            )
            summary_status = max(
                epistemic_statuses,
                key=lambda status: _EPISTEMIC_RANK.get(status, 3),
            )
            summary_class = max(
                (DataClass(memory.data_class) for memory in bounded),
                key=lambda value: _DATA_CLASS_RANK.get(value, 3),
            )
            source_claims = tuple(
                {
                    "memory_id": memory.record_id,
                    "topic": memory.topic,
                    "content": _semantic_content(memory),
                    "source_turn_id": memory.source_turn_id,
                    "source_span": memory.content.get("source_span"),
                    "modality": str(memory.content.get("modality", "unknown")),
                    "epistemic_status": memory.epistemic_status,
                    "confidence": memory.content.get("confidence"),
                    "valid_from": memory.valid_from.isoformat(),
                    "valid_until": (
                        memory.valid_until.isoformat() if memory.valid_until else None
                    ),
                }
                for memory in bounded
            )
            summary_text = " | ".join(
                f"{memory.topic}: "
                f"{json.dumps(_semantic_content(memory), ensure_ascii=False, sort_keys=True)}"
                for memory in bounded
            )
            evidence.append(
                AssistantEvidence(
                    source_id=_summary_id(bounded),
                    source_type="grounded_claim_summary",
                    timestamp=max(memory.timestamp for memory in bounded),
                    content={
                        "summary": summary_text,
                        "source_claims": source_claims,
                        "source_memory_ids": memory_ids,
                        "source_turn_ids": source_turn_ids,
                        "source_coverage": 1.0,
                        "confidence": summary_confidence,
                        "modalities": modalities,
                        "epistemic_statuses": epistemic_statuses,
                        "epistemic_status": summary_status,
                        "projection": "deterministic-query-time-v1",
                    },
                    data_class=summary_class,
                    domain_scope=domain_scope,
                    purpose=purpose,
                    ranking_reason=(
                        "deterministic summary of eligible claims: "
                        + ", ".join(memory_ids)
                    ),
                )
            )

        evidence.sort(key=lambda item: item.timestamp, reverse=True)
        return Success(tuple(evidence[:bounded_limit]))


def validate_grounded_summary(evidence: AssistantEvidence) -> ActionFailure | None:
    """Return a typed failure when a purported summary lost its source coverage."""
    source_ids = evidence.content.get("source_memory_ids")
    source_claims = evidence.content.get("source_claims")
    if evidence.source_type != "grounded_claim_summary":
        return make_assistant_failure(
            code="INVALID_SUMMARY_TYPE",
            message="summary evidence has an unsupported source type",
            request_id=evidence.source_id,
        )
    if not isinstance(source_ids, (tuple, list)) or not source_ids:
        return make_assistant_failure(
            code="MISSING_SUMMARY_SOURCES",
            message="grounded summary has no contributing memory IDs",
            request_id=evidence.source_id,
        )
    if not isinstance(source_claims, (tuple, list)) or len(source_claims) != len(
        source_ids
    ):
        return make_assistant_failure(
            code="INCOMPLETE_SUMMARY_COVERAGE",
            message="grounded summary does not cover every contributing memory",
            request_id=evidence.source_id,
        )
    claim_ids = tuple(
        claim.get("memory_id") for claim in source_claims if isinstance(claim, dict)
    )
    if tuple(source_ids) != claim_ids or evidence.content.get("source_coverage") != 1.0:
        return make_assistant_failure(
            code="INCOMPLETE_SUMMARY_COVERAGE",
            message="grounded summary source coverage is inconsistent",
            request_id=evidence.source_id,
        )
    return None
