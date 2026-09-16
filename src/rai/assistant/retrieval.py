"""Optional local dense, weighted-RRF and authenticated graph retrieval."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
import unicodedata
from typing import Protocol

from returns.result import Failure, Result, Success

from rai.kernel.records import ActionFailure, DataClass

from .ports import (
    GraphEvidencePath,
    MemoryGraphStore,
    MemoryQuery,
    MemoryRetrievalSelection,
)
from .query import domain_scope_matches
from .records import MemoryRecord, MemoryRelation, MemoryRelationKind, make_assistant_failure

_MINIMUM_DENSE_DIMENSIONS = 32


class DenseEmbeddingProvider(Protocol):
    """Local deterministic embedding boundary used only by evaluated retrieval."""

    @property
    def version(self) -> str: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


class FeatureHashingDenseEmbedder:
    """Dependency-free local dense baseline using signed lexical feature hashing."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < _MINIMUM_DENSE_DIMENSIONS:
            raise ValueError("dense embedding dimensions must be at least 32")
        self.dimensions = dimensions
        self.version = f"feature-hashing-word-char-v1-d{dimensions}"

    @staticmethod
    def _features(text: str) -> tuple[str, ...]:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        words = re.findall(r"[a-z0-9]+", normalized.casefold())
        word_features = [f"w:{word}" for word in words]
        bigrams = [f"b:{left}_{right}" for left, right in zip(words, words[1:])]
        compact = "_".join(words)
        character = [
            f"c:{compact[index:index + 3]}"
            for index in range(max(0, len(compact) - 2))
        ]
        return tuple((*word_features, *bigrams, *character))

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            vector[raw % self.dimensions] += -1.0 if raw & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


def weighted_reciprocal_rank_fusion(
    rankings: tuple[tuple[str, tuple[str, ...], float], ...],
    *,
    rank_constant: int = 60,
) -> tuple[tuple[str, float], ...]:
    """Fuse named rankings without allowing raw channel scores to distort budgets."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    for _channel, identifiers, weight in rankings:
        if weight < 0.0:
            raise ValueError("RRF channel weights cannot be negative")
        for rank, identifier in enumerate(identifiers, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (
                rank_constant + rank
            )
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0])))


def _memory_text(memory: MemoryRecord) -> str:
    return f"{memory.topic} {json.dumps(memory.content, ensure_ascii=False, sort_keys=True)}"


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _policy_value(relation: MemoryRelation) -> str:
    value = relation.policy_outcome
    return value.value if hasattr(value, "value") else str(value)


def _relation_kind_value(relation: MemoryRelation) -> str:
    value = relation.kind
    return value.value if hasattr(value, "value") else str(value)


_GRAPH_KINDS = {
    MemoryRelationKind.ABOUT.value,
    MemoryRelationKind.SUPPORTS.value,
    MemoryRelationKind.UPDATES.value,
}


@dataclass(frozen=True)
class MultiChannelMemoryRetriever:
    """Benchmarkable retriever; it is opt-in and never changes canonical storage."""

    store: MemoryGraphStore
    embedder: DenseEmbeddingProvider = field(default_factory=FeatureHashingDenseEmbedder)
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    graph_weight: float = 0.25
    candidate_limit: int = 200
    graph_depth: int = 2
    minimum_dense_score: float = 0.05
    minimum_relation_confidence: float = 0.75

    async def retrieve(
        self,
        query: MemoryQuery,
        data_classes: tuple[DataClass, ...],
        limit: int,
    ) -> Result[MemoryRetrievalSelection, ActionFailure]:
        bounded_limit = max(1, min(limit, 50))
        candidate_started = time.perf_counter()
        all_result = await self.store.retrieve_relevant_memories(
            profile_scope=query.profile_scope,
            query=None,
            data_classes=data_classes,
            limit=max(self.candidate_limit, bounded_limit),
        )
        if isinstance(all_result, Failure):
            return Failure(all_result.failure())
        eligible = {
            memory.record_id: memory
            for memory, _reason in all_result.unwrap()
            if domain_scope_matches(memory.domain_scope, query.domain_scopes)
            and memory.purpose == query.purpose
        }
        candidate_latency_ms = (time.perf_counter() - candidate_started) * 1_000

        lexical_started = time.perf_counter()
        lexical_result = await self.store.retrieve_relevant_memories(
            profile_scope=query.profile_scope,
            query=query,
            data_classes=data_classes,
            limit=max(self.candidate_limit, bounded_limit),
        )
        if isinstance(lexical_result, Failure):
            return Failure(lexical_result.failure())
        lexical = tuple(
            memory.record_id
            for memory, _reason in lexical_result.unwrap()
            if memory.record_id in eligible
        )
        lexical_latency_ms = (time.perf_counter() - lexical_started) * 1_000

        dense_started = time.perf_counter()
        query_vector = self.embedder.embed(query.raw_text or " ".join(query.keywords))
        dense_scored = tuple(
            sorted(
                (
                    (_dot(query_vector, self.embedder.embed(_memory_text(memory))), memory_id)
                    for memory_id, memory in eligible.items()
                ),
                key=lambda item: (-item[0], item[1]),
            )
        )
        dense = tuple(
            memory_id
            for score, memory_id in dense_scored
            if score >= self.minimum_dense_score
        )
        dense_latency_ms = (
            candidate_latency_ms + (time.perf_counter() - dense_started) * 1_000
        )
        fusion_started = time.perf_counter()
        fused = weighted_reciprocal_rank_fusion(
            (
                ("claims_bm25", lexical, self.lexical_weight),
                ("dense_local", dense, self.dense_weight),
            )
        )
        fusion_latency_ms = (time.perf_counter() - fusion_started) * 1_000

        graph_started = time.perf_counter()
        relations_result = await self.store.get_relations()
        if isinstance(relations_result, Failure):
            return Failure(relations_result.failure())
        authenticated_relations = tuple(
            relation
            for relation in relations_result.unwrap()
            if relation.eligible
            and _policy_value(relation) == "ALLOW"
            and relation.confidence >= self.minimum_relation_confidence
            and relation.provenance
            and _relation_kind_value(relation) in _GRAPH_KINDS
            and relation.source_id in eligible
            and relation.target_id in eligible
        )
        graph_paths = self._bounded_paths(
            tuple(memory_id for memory_id, _score in fused[:bounded_limit]),
            authenticated_relations,
        )
        graph_latency_ms = (time.perf_counter() - graph_started) * 1_000
        fused_scores = dict(fused)
        for path in graph_paths:
            endpoint = path.node_ids[-1]
            fused_scores[endpoint] = fused_scores.get(endpoint, 0.0) + (
                self.graph_weight * path.score
            )
        final_ids = tuple(
            memory_id
            for memory_id, _score in sorted(
                fused_scores.items(), key=lambda item: (-item[1], item[0])
            )[:bounded_limit]
        )
        reason_by_id = {
            memory_id: (
                "weighted RRF over claims_bm25 and dense_local"
                + (" with authenticated bounded graph support" if any(
                    path.node_ids[-1] == memory_id for path in graph_paths
                ) else "")
            )
            for memory_id in final_ids
        }
        return Success(
            MemoryRetrievalSelection(
                memories=tuple(
                    (eligible[memory_id], reason_by_id[memory_id])
                    for memory_id in final_ids
                ),
                channel_ids=(
                    ("claims_bm25", lexical[:bounded_limit]),
                    ("dense_local", dense[:bounded_limit]),
                    ("rrf_fused", tuple(memory_id for memory_id, _ in fused[:bounded_limit])),
                    (
                        "graph_bounded",
                        final_ids,
                    ),
                ),
                channel_latency_ms=(
                    ("claims_bm25", lexical_latency_ms),
                    ("dense_local", dense_latency_ms),
                    (
                        "rrf_fused",
                        lexical_latency_ms + dense_latency_ms + fusion_latency_ms,
                    ),
                    (
                        "graph_bounded",
                        lexical_latency_ms
                        + dense_latency_ms
                        + fusion_latency_ms
                        + graph_latency_ms,
                    ),
                ),
                graph_paths=graph_paths,
            )
        )

    def _bounded_paths(
        self,
        seeds: tuple[str, ...],
        relations: tuple[MemoryRelation, ...],
    ) -> tuple[GraphEvidencePath, ...]:
        adjacency: dict[str, list[tuple[str, str]]] = {}
        for relation in relations:
            adjacency.setdefault(relation.source_id, []).append(
                (relation.target_id, relation.relation_id)
            )
            adjacency.setdefault(relation.target_id, []).append(
                (relation.source_id, relation.relation_id)
            )
        found: list[GraphEvidencePath] = []
        for seed in seeds:
            queue = deque([(seed, (seed,), ())])
            visited = {seed}
            while queue:
                current, nodes, edge_ids = queue.popleft()
                if len(edge_ids) >= self.graph_depth:
                    continue
                for neighbor, relation_id in sorted(adjacency.get(current, ())):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_nodes = (*nodes, neighbor)
                    next_edges = (*edge_ids, relation_id)
                    found.append(
                        GraphEvidencePath(
                            node_ids=next_nodes,
                            relation_ids=next_edges,
                            score=1.0 / len(next_edges),
                        )
                    )
                    queue.append((neighbor, next_nodes, next_edges))
        return tuple(
            sorted(
                found,
                key=lambda path: (-path.score, path.node_ids, path.relation_ids),
            )
        )
