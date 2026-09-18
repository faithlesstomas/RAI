"""Context construction and manifest assembly for assistant inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from datetime import datetime, timezone
import re
import unicodedata

from returns.result import Failure, Result, Success

from rai.kernel.records import (
    ActionFailure,
    DataClass,
    ProducerIdentity,
    _new_id,
    _utc_now,
)

from .ports import (
    AdvancedMemoryRetriever,
    AssistantEvidenceProvider,
    MemoryGraphStore,
    MemoryQuery,
)
from .query import MemoryQueryResolver, domain_scope_matches
from .records import (
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    ConversationTurn,
    MemoryRecord,
    make_assistant_failure,
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are Rich AI (RAI), an intelligent and secure local assistant for the GNU/Linux desktop. "
    "You respect user preferences stored in durable memory and provide concise, accurate answers."
)

_EPISTEMIC_QUALITY = {
    "asserted": 1.0,
    "observed": 1.0,
    "inferred": 0.8,
    "uncertain": 0.5,
    "contested": 0.0,
}
_MODALITY_QUALITY = {
    "direct": 1.0,
    "hedged": 0.7,
    "quoted": 0.0,
    "hearsay": 0.0,
}
_SUFFICIENCY_STOP_WORDS = {
    "czy",
    "do",
    "i",
    "is",
    "jak",
    "jaki",
    "jaka",
    "jakie",
    "jakim",
    "jest",
    "nad",
    "o",
    "the",
    "w",
    "what",
}
_SUFFICIENCY_CANONICAL_TERMS = {
    "age": "age",
    "budzecie": "budzet",
    "budzetu": "budzet",
    "deadline": "termin",
    "deadlines": "termin",
    "embeddings": "embedding",
    "embeddingow": "embedding",
    "gdzie": "location",
    "imie": "name",
    "imienia": "name",
    "lat": "age",
    "live": "location",
    "mieszkam": "location",
    "mieszkasz": "location",
    "nazywam": "name",
    "old": "age",
    "projekcie": "projekt",
    "projektem": "projekt",
    "projektu": "projekt",
    "terminie": "termin",
    "terminu": "termin",
    "wiek": "age",
}
SUFFICIENCY_POLICY_VERSION = "lexical-evidence-v2"
MAX_RECENT_ROUTE_TURNS = 4


def _data_class(value: DataClass | str) -> DataClass:
    return value if isinstance(value, DataClass) else DataClass(value)


@dataclass(frozen=True)
class MemorySufficiencyAssessment:
    """Inspectable evidence coverage and quality behind a context route."""

    score: float
    query_coverage: float
    evidence_quality: float
    maximum_confidence: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    reasons: tuple[str, ...]
    policy_version: str = SUFFICIENCY_POLICY_VERSION


def _search_terms(value: str) -> tuple[str, ...]:
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )
    return tuple(
        dict.fromkeys(
            _SUFFICIENCY_CANONICAL_TERMS.get(term, term)
            for term in re.findall(r"[a-z0-9]+", normalized)
            if term not in _SUFFICIENCY_STOP_WORDS
        )
    )


def assess_memory_sufficiency(
    query: MemoryQuery, memories: tuple[MemoryRecord, ...]
) -> MemorySufficiencyAssessment:
    """Score whether compact claims cover this query with trustworthy evidence."""
    query_terms = tuple(
        dict.fromkeys(
            term for keyword in query.keywords for term in _search_terms(keyword)
        )
    )
    if not memories:
        return MemorySufficiencyAssessment(
            score=0.0,
            query_coverage=0.0,
            evidence_quality=0.0,
            maximum_confidence=0.0,
            matched_terms=(),
            missing_terms=query_terms,
            reasons=("no eligible durable memory",),
        )

    matched: set[str] = set()
    term_qualities: dict[str, float] = {}
    topic_qualities: list[float] = []
    confidences: list[float] = []
    exact_topic = False
    for memory in memories:
        searchable = set(
            _search_terms(
                f"{memory.topic} {json.dumps(memory.content, ensure_ascii=False)}"
            )
        )
        record_matches = {term for term in query_terms if term in searchable}
        topic_matches = query.topic is not None and memory.topic == query.topic
        if not record_matches and not topic_matches:
            continue
        raw_confidence = memory.content.get("confidence")
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (float, int))
            and not isinstance(raw_confidence, bool)
            else 0.0
        )
        confidence = min(1.0, max(0.0, confidence))
        confidences.append(confidence)
        modality = str(memory.content.get("modality", "unknown"))
        quality = (
            confidence
            * _MODALITY_QUALITY.get(modality, 0.0)
            * _EPISTEMIC_QUALITY.get(memory.epistemic_status, 0.0)
        )
        matched.update(record_matches)
        exact_topic = exact_topic or topic_matches
        for term in record_matches:
            term_qualities[term] = max(term_qualities.get(term, 0.0), quality)
        if topic_matches:
            topic_qualities.append(quality)

    if query_terms:
        coverage = len(matched) / len(query_terms)
        if exact_topic and query.topic_is_complete:
            coverage = 1.0
    else:
        coverage = 1.0 if exact_topic else 0.0
    supporting_qualities = tuple(term_qualities.values()) or tuple(topic_qualities)
    quality = (
        sum(supporting_qualities) / len(supporting_qualities)
        if supporting_qualities
        else 0.0
    )
    maximum_confidence = max(confidences, default=0.0)
    score = min(1.0, max(0.0, 0.6 * coverage + 0.4 * quality))
    missing = tuple(term for term in query_terms if term not in matched)
    reasons = (
        f"query coverage={coverage:.3f}",
        f"evidence quality={quality:.3f}",
        f"maximum source confidence={maximum_confidence:.3f}",
        f"policy version={SUFFICIENCY_POLICY_VERSION}",
        (
            "exact topic match"
            if exact_topic
            else f"matched terms={','.join(sorted(matched)) or 'none'}"
        ),
        f"missing terms={','.join(missing) or 'none'}",
    )
    return MemorySufficiencyAssessment(
        score=score,
        query_coverage=coverage,
        evidence_quality=quality,
        maximum_confidence=maximum_confidence,
        matched_terms=tuple(sorted(matched)),
        missing_terms=missing,
        reasons=reasons,
    )


def assess_recent_sufficiency(
    query: MemoryQuery, turns: tuple[ConversationTurn, ...]
) -> MemorySufficiencyAssessment:
    """Score direct user statements in a short reply chain as primary evidence."""
    query_terms = tuple(
        dict.fromkeys(
            term for keyword in query.keywords for term in _search_terms(keyword)
        )
    )
    user_turns = tuple(
        turn for turn in turns if turn.role == "user" and "?" not in turn.text
    )
    if not user_turns:
        return MemorySufficiencyAssessment(
            score=0.0,
            query_coverage=0.0,
            evidence_quality=0.0,
            maximum_confidence=0.0,
            matched_terms=(),
            missing_terms=query_terms,
            reasons=("no eligible user statement in recent reply chain",),
        )

    searchable = set(_search_terms(" ".join(turn.text for turn in user_turns)))
    matched = tuple(term for term in query_terms if term in searchable)
    coverage = len(matched) / len(query_terms) if query_terms else 0.0
    quality = 1.0 if matched else 0.0
    score = min(1.0, max(0.0, 0.6 * coverage + 0.4 * quality))
    missing = tuple(term for term in query_terms if term not in searchable)
    return MemorySufficiencyAssessment(
        score=score,
        query_coverage=coverage,
        evidence_quality=quality,
        maximum_confidence=1.0 if matched else 0.0,
        matched_terms=matched,
        missing_terms=missing,
        reasons=(
            f"recent user-statement coverage={coverage:.3f}",
            f"matched terms={','.join(matched) or 'none'}",
            f"missing terms={','.join(missing) or 'none'}",
            f"policy version={SUFFICIENCY_POLICY_VERSION}",
        ),
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
        advanced_retriever: AdvancedMemoryRetriever | None = None,
        max_external_evidence: int = 5,
        memory_sufficiency_threshold: float = 0.75,
        max_context_characters: int = 8_000,
        enable_recent_route: bool = True,
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
        self.advanced_retriever = advanced_retriever
        self.max_external_evidence = max_external_evidence
        if not 0.0 <= memory_sufficiency_threshold <= 1.0:
            raise ValueError("memory_sufficiency_threshold must be between 0 and 1")
        self.memory_sufficiency_threshold = memory_sufficiency_threshold
        self.max_context_characters = max_context_characters
        self.enable_recent_route = enable_recent_route
        self.profile_scope = profile_scope
        self.producer = producer or ProducerIdentity(
            producer_id="assistant-context-builder", kind="service", version="1.0.0"
        )

    async def build_context(  # noqa: PLR0912, PLR0915
        self, turn: ConversationTurn
    ) -> Result[AssistantContextPackage, ActionFailure]:
        """Assemble a bounded, inspectable AssistantContextPackage."""
        query = self.query_resolver.resolve(turn.text, self.profile_scope)

        recent_res = await self.store.get_recent_reply_chain(
            session_id=turn.session_id,
            limit=self.max_recent_turns,
            before_turn_id=turn.record_id,
            profile_scope=self.profile_scope,
        )
        if isinstance(recent_res, Failure):
            return Failure(recent_res.failure())
        all_recent_turns = recent_res.unwrap()
        recent_turns = tuple(
            item
            for item in all_recent_turns
            if domain_scope_matches(item.domain_scope, query.domain_scopes)
            and item.purpose == query.purpose
            and _data_class(item.data_class) in query.data_classes
        )
        recent_sufficiency = assess_recent_sufficiency(query, recent_turns)
        recent_route_sufficient = (
            self.enable_recent_route
            and len(recent_turns) <= MAX_RECENT_ROUTE_TURNS
            and recent_sufficiency.score > 0.0
            and recent_sufficiency.score >= self.memory_sufficiency_threshold
        )

        retrieval_channel_ids: dict[str, tuple[str, ...]] = {}
        graph_paths: tuple[dict[str, object], ...] = ()
        if recent_route_sufficient:
            mem_res = Success(())
        elif self.advanced_retriever is None:
            mem_res = await self.store.retrieve_relevant_memories(
                profile_scope=self.profile_scope,
                query=query,
                data_classes=query.data_classes,
                limit=self.max_memories,
            )
        else:
            advanced_res = await self.advanced_retriever.retrieve(
                query=query,
                data_classes=query.data_classes,
                limit=self.max_memories,
            )
            if isinstance(advanced_res, Failure):
                return Failure(advanced_res.failure())
            selection = advanced_res.unwrap()
            retrieval_channel_ids = dict(selection.channel_ids)
            graph_paths = tuple(
                {
                    "node_ids": path.node_ids,
                    "relation_ids": path.relation_ids,
                    "score": path.score,
                }
                for path in selection.graph_paths
            )
            mem_res = Success(selection.memories)
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
                    domain_scope=rt.domain_scope,
                    purpose=rt.purpose,
                    ranking_reason="reply-chain chronological window",
                    fields=("text", "role"),
                )
            )

        durable_ids: list[str] = []
        durable_memories: list[MemoryRecord] = []
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
            if not domain_scope_matches(mem.domain_scope, query.domain_scopes):
                invalid_reasons.append("domain_scope")
            if mem.purpose != query.purpose:
                invalid_reasons.append("purpose")
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
            durable_memories.append(mem)
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
                    domain_scope=mem.domain_scope,
                    purpose=mem.purpose,
                    ranking_reason=reason,
                    fields=("topic", "content"),
                )
            )

        sufficiency = assess_memory_sufficiency(query, tuple(durable_memories))
        sufficiency_score = sufficiency.score
        compact_memory_sufficient = (
            sufficiency_score > 0.0
            and sufficiency_score >= self.memory_sufficiency_threshold
        )
        rejected_routes: list[str] = []
        episodic_with_reasons = ()
        external_evidence = []
        if recent_route_sufficient:
            rejected_routes.extend(
                ("compact_memory:recent_sufficient", "raw_evidence:not_needed")
            )
        elif compact_memory_sufficient:
            rejected_routes.append("raw_evidence:not_needed")
        else:
            rejected_routes.append(
                "compact_memory:insufficient"
                f"(coverage={sufficiency.query_coverage:.3f},"
                f"quality={sufficiency.evidence_quality:.3f})"
            )
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
                    domain_scope=source_turn.domain_scope,
                    purpose=source_turn.purpose,
                    ranking_reason=reason,
                    fields=("text", "role", "timestamp", "session_id"),
                )
            )

        external_ids: list[str] = []
        external_content: list[dict[str, object]] = []
        for evidence in external_evidence:
            if not domain_scope_matches(evidence.domain_scope, query.domain_scopes):
                exclusions.append(f"{evidence.source_id}:domain_scope")
                continue
            if evidence.purpose != query.purpose:
                exclusions.append(f"{evidence.source_id}:purpose")
                continue
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
                    domain_scope=evidence.domain_scope,
                    purpose=evidence.purpose,
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
        kept_recent_turns = tuple(
            item for item in recent_turns if item.record_id in recent_ids
        )
        kept_durable_memories = tuple(
            item for item in durable_memories if item.record_id in durable_ids
        )
        final_recent_sufficiency = assess_recent_sufficiency(query, kept_recent_turns)
        final_memory_sufficiency = assess_memory_sufficiency(
            query, kept_durable_memories
        )
        final_recent_sufficient = (
            recent_route_sufficient
            and final_recent_sufficiency.score >= self.memory_sufficiency_threshold
        )
        final_recent_evidence_sufficient = (
            final_recent_sufficiency.score >= self.memory_sufficiency_threshold
        )
        final_compact_sufficient = (
            final_memory_sufficiency.score > 0.0
            and final_memory_sufficiency.score >= self.memory_sufficiency_threshold
        )
        routing_decision = (
            "recent_conversation"
            if final_recent_sufficient
            else "compact_memory"
            if final_compact_sufficient
            else "raw_evidence_fallback"
            if episodic_ids or external_ids or final_recent_evidence_sufficient
            else "no_evidence"
        )
        route_sufficiency = (
            final_recent_sufficiency
            if routing_decision == "recent_conversation"
            or (
                routing_decision == "raw_evidence_fallback"
                and final_recent_evidence_sufficient
            )
            else final_memory_sufficiency
        )
        if routing_decision == "no_evidence":
            rejected_routes.append("raw_evidence:no_match")
            if compact_memory_sufficient and not final_compact_sufficient:
                rejected_routes.append("compact_memory:removed_by_character_budget")
            if recent_route_sufficient and not final_recent_sufficient:
                rejected_routes.append(
                    "recent_conversation:removed_by_character_budget"
                )

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
            sufficiency_score=route_sufficiency.score,
            sufficiency_factors={
                "query_coverage": route_sufficiency.query_coverage,
                "evidence_quality": route_sufficiency.evidence_quality,
                "maximum_confidence": route_sufficiency.maximum_confidence,
            },
            sufficiency_reasons=route_sufficiency.reasons,
            fallback_used=routing_decision == "raw_evidence_fallback",
            evidence_required=query.evidence_required,
            retrieval_channel_ids=retrieval_channel_ids,
            graph_paths=graph_paths,
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
