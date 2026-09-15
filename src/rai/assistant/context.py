"""Context construction and manifest assembly for assistant inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from .ports import AssistantEvidenceProvider, MemoryGraphStore
from .query import MemoryQueryResolver
from .records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    ConversationTurn,
    make_assistant_failure,
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Rich AI (RAI), an intelligent and secure local assistant for the GNU/Linux desktop. "
    "You respect user preferences stored in durable memory and provide concise, accurate answers."
)


class AssistantContextBuilder:
    """Selects recent dialogue and durable memories independently to build a ContextPackage."""

    def __init__(  # noqa: PLR0913
        self,
        store: MemoryGraphStore,
        query_resolver: MemoryQueryResolver | None = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        max_recent_turns: int = 10,
        max_memories: int = 5,
        max_episodic_turns: int = 5,
        evidence_providers: tuple[AssistantEvidenceProvider, ...] = (),
        max_external_evidence: int = 5,
        memory_sufficiency_threshold: float = 0.75,
        max_context_characters: int = 8_000,
        profile_scope: str = "default",
        producer: ProducerIdentity | None = None,
    ) -> None:
        self.store = store
        self.query_resolver = query_resolver or MemoryQueryResolver()
        self.system_instruction = system_instruction
        self.max_recent_turns = max_recent_turns
        self.max_memories = max_memories
        self.max_episodic_turns = max_episodic_turns
        self.evidence_providers = evidence_providers
        self.max_external_evidence = max_external_evidence
        if not 0.0 <= memory_sufficiency_threshold <= 1.0:
            raise ValueError("memory_sufficiency_threshold must be between 0 and 1")
        self.memory_sufficiency_threshold = memory_sufficiency_threshold
        self.max_context_characters = max_context_characters
        self.profile_scope = profile_scope
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-context-builder", kind="service", version="1.0.0"
        )

    async def build_context(  # noqa: PLR0912, PLR0915
        self, turn: ConversationTurn
    ) -> Result[AssistantContextPackage, ActionFailure]:
        """Assemble a bounded, inspectable AssistantContextPackage."""
        query = self.query_resolver.resolve(turn.text)

        recent_res = await self.store.get_recent_reply_chain(
            session_id=turn.session_id,
            limit=self.max_recent_turns,
            before_turn_id=turn.record_id,
        )
        if isinstance(recent_res, Failure):
            return Failure(recent_res.failure())
        recent_turns = recent_res.unwrap()

        mem_res = await self.store.retrieve_relevant_memories(
            profile_scope=self.profile_scope,
            query=query,
            data_classes=query.data_classes,
            limit=self.max_memories,
        )
        if isinstance(mem_res, Failure):
            return Failure(mem_res.failure())
        memories_with_reasons = mem_res.unwrap()

        items: list[AssistantContextManifestItem] = []
        recent_ids: list[str] = []
        for rt in recent_turns:
            recent_ids.append(rt.record_id)
            items.append(
                AssistantContextManifestItem(
                    source_id=rt.record_id,
                    source_type="conversation_turn",
                    layer="recent_conversation",
                    data_class=rt.data_class,
                    ranking_reason="reply-chain chronological window",
                    fields=("text", "role"),
                )
            )

        durable_ids: list[str] = []
        ranking_reasons: dict[str, str] = {}
        durable_content: list[dict[str, object]] = []
        exclusions: list[str] = []
        now = datetime.now(timezone.utc)
        for mem, reason in memories_with_reasons:
            data_class = (
                mem.data_class
                if isinstance(mem.data_class, DataClass)
                else DataClass(mem.data_class)
            )
            valid_source = await self.store.get_turn(mem.source_turn_id)
            invalid_reasons = []
            if mem.profile_scope != query.profile_scope:
                invalid_reasons.append("profile_scope")
            if data_class not in query.data_classes:
                invalid_reasons.append("privacy_class")
            if mem.valid_from > now or (
                mem.valid_until is not None and mem.valid_until <= now
            ):
                invalid_reasons.append("temporal_validity")
            if isinstance(valid_source, Failure) or valid_source.unwrap() is None:
                invalid_reasons.append("missing_provenance_source")
            if invalid_reasons:
                exclusions.append(f"{mem.record_id}:{','.join(invalid_reasons)}")
                continue
            durable_ids.append(mem.record_id)
            ranking_reasons[mem.record_id] = reason
            durable_content.append(
                {
                    "record_id": mem.record_id,
                    "kind": mem.kind,
                    "topic": mem.topic,
                    "content": mem.content,
                }
            )
            items.append(
                AssistantContextManifestItem(
                    source_id=mem.record_id,
                    source_type="memory_record",
                    layer="durable_memory",
                    data_class=mem.data_class,
                    ranking_reason=reason,
                    fields=("topic", "content"),
                )
            )

        confidence_values = tuple(
            float(item["content"].get("confidence", 1.0))
            for item in durable_content
            if isinstance(item.get("content"), dict)
        )
        sufficiency_score = max(confidence_values, default=0.0)
        compact_memory_sufficient = (
            sufficiency_score >= self.memory_sufficiency_threshold
        )
        rejected_routes: list[str] = []
        episodic_with_reasons = ()
        external_evidence = []
        if compact_memory_sufficient:
            rejected_routes.append("raw_evidence:not_needed")
        else:
            rejected_routes.append("compact_memory:insufficient")
            episodic_res = await self.store.retrieve_relevant_turns(
                profile_scope=self.profile_scope,
                query=query,
                data_classes=query.data_classes,
                exclude_turn_ids=(
                    turn.record_id,
                    *(item.record_id for item in recent_turns),
                ),
                limit=self.max_episodic_turns,
            )
            if isinstance(episodic_res, Failure):
                return Failure(episodic_res.failure())
            episodic_with_reasons = episodic_res.unwrap()

            for provider in self.evidence_providers:
                evidence_res = await provider.retrieve(
                    query=query,
                    data_classes=query.data_classes,
                    limit=self.max_external_evidence,
                )
                if isinstance(evidence_res, Failure):
                    rejected_routes.append(
                        f"{type(provider).__name__}:{evidence_res.failure().code}"
                    )
                    continue
                external_evidence.extend(evidence_res.unwrap())
            external_evidence.sort(key=lambda item: item.timestamp, reverse=True)
            external_evidence = external_evidence[: self.max_external_evidence]

        episodic_ids: list[str] = []
        episodic_content: list[dict[str, object]] = []
        for source_turn, reason in episodic_with_reasons:
            episodic_ids.append(source_turn.record_id)
            ranking_reasons[source_turn.record_id] = reason
            episodic_content.append(
                {
                    "record_id": source_turn.record_id,
                    "session_id": source_turn.session_id,
                    "timestamp": source_turn.timestamp.isoformat(),
                    "role": source_turn.role,
                    "text": source_turn.text,
                }
            )
            items.append(
                AssistantContextManifestItem(
                    source_id=source_turn.record_id,
                    source_type="conversation_turn",
                    layer="episodic_evidence",
                    data_class=source_turn.data_class,
                    ranking_reason=reason,
                    fields=("text", "role", "timestamp", "session_id"),
                )
            )

        external_ids: list[str] = []
        external_content: list[dict[str, object]] = []
        for evidence in external_evidence:
            external_ids.append(evidence.source_id)
            ranking_reasons[evidence.source_id] = evidence.ranking_reason
            external_content.append(
                {
                    "source_id": evidence.source_id,
                    "source_type": evidence.source_type,
                    "timestamp": evidence.timestamp.isoformat(),
                    "content": evidence.content,
                }
            )
            items.append(
                AssistantContextManifestItem(
                    source_id=evidence.source_id,
                    source_type=evidence.source_type,
                    layer="external_evidence",
                    data_class=evidence.data_class,
                    ranking_reason=evidence.ranking_reason,
                    fields=("timestamp", "content"),
                )
            )

        recent_content = [
            {"record_id": t.record_id, "role": t.role, "text": t.text}
            for t in recent_turns
        ]
        content: dict[str, object] = {
            "system_instruction": self.system_instruction,
            "current_turn": {
                "record_id": turn.record_id,
                "role": turn.role,
                "text": turn.text,
            },
            "recent_turns": recent_content,
            "episodic_evidence": episodic_content,
            "external_evidence": external_content,
            "durable_memories": durable_content,
        }

        serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and recent_content:
            removed = recent_content.pop(0)
            removed_id = str(removed["record_id"])
            recent_ids.remove(removed_id)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and episodic_content:
            removed = episodic_content.pop()
            removed_id = str(removed["record_id"])
            episodic_ids.remove(removed_id)
            ranking_reasons.pop(removed_id, None)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and external_content:
            removed = external_content.pop()
            removed_id = str(removed["source_id"])
            external_ids.remove(removed_id)
            ranking_reasons.pop(removed_id, None)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        while len(serialized) > self.max_context_characters and durable_content:
            removed = durable_content.pop()
            removed_id = str(removed["record_id"])
            durable_ids.remove(removed_id)
            ranking_reasons.pop(removed_id, None)
            items = [item for item in items if item.source_id != removed_id]
            exclusions.append(f"{removed_id}:character_budget")
            serialized = json.dumps(content, ensure_ascii=False)
        if len(serialized) > self.max_context_characters:
            return Failure(
                make_assistant_failure(
                    code="CONTEXT_BUDGET_EXCEEDED",
                    message=(
                        "the current turn and required instructions exceed the "
                        f"{self.max_context_characters}-character context budget"
                    ),
                    request_id=turn.record_id,
                )
            )
        total_chars = len(serialized)
        est_tokens = max(1, total_chars // 4)
        routing_decision = (
            "compact_memory"
            if compact_memory_sufficient
            else "raw_evidence_fallback"
            if episodic_ids or external_ids
            else "no_evidence"
        )
        if routing_decision == "no_evidence":
            rejected_routes.append("raw_evidence:no_match")

        manifest = AssistantContextManifest(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            recent_turn_ids=tuple(recent_ids),
            episodic_turn_ids=tuple(episodic_ids),
            external_evidence_ids=tuple(external_ids),
            durable_memory_ids=tuple(durable_ids),
            ranking_reasons=ranking_reasons,
            exclusions=tuple(exclusions),
            redactions=(),
            routing_decision=routing_decision,
            route_candidates=(
                "recent_conversation",
                "compact_memory",
                "fts5_raw_turns",
                *(type(provider).__name__ for provider in self.evidence_providers),
            ),
            rejected_routes=tuple(rejected_routes),
            sufficiency_score=sufficiency_score,
            fallback_used=routing_decision == "raw_evidence_fallback",
            evidence_character_budget=self.max_context_characters,
            retriever_version="2.0.0",
            policy_version="1.0.0",
            actual_tokens=est_tokens,
            actual_characters=total_chars,
            items=tuple(items),
        )

        package = AssistantContextPackage(
            record_id=_new_id(),
            timestamp=_utc_now(),
            producer=self.producer,
            session_id=turn.session_id,
            turn_id=turn.record_id,
            manifest=manifest,
            content=content,
        )

        return Success(package)
