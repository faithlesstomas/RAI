"""Provider-neutral runtime ports for the Rich Assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from returns.result import Result

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, DataClass

from .records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
)


@dataclass(frozen=True)
class MemoryQuery:
    """Structured query for epistemically neutral memory retrieval."""

    topic: str | None = None
    keywords: tuple[str, ...] = ()
    profile_scope: str = "default"
    data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL)
    limit: int = 10


@runtime_checkable
class MemoryGraphStore(Protocol):
    """Transactional, graph-aware persistent store for assistant memory."""

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...

    async def accept_turn(
        self, turn: ConversationTurn
    ) -> Result[ConversationTurn, ActionFailure]: ...

    async def commit_terminal(
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
    ) -> Result[AssistantResponse, ActionFailure]: ...

    async def get_response_by_request_id(
        self, request_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]: ...

    async def get_manifest(
        self, manifest_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]: ...

    async def get_turn(
        self, turn_id: str
    ) -> Result[ConversationTurn | None, ActionFailure]: ...

    async def get_recent_reply_chain(
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]: ...

    async def retrieve_relevant_memories(
        self,
        profile_scope: str = "default",
        query: MemoryQuery | None = None,
        data_classes: tuple[DataClass, ...] = (DataClass.PUBLIC, DataClass.LOCAL),
        limit: int = 10,
    ) -> Result[tuple[tuple[MemoryRecord, str], ...], ActionFailure]: ...

    async def delete_turn(self, turn_id: str) -> Result[int, ActionFailure]: ...

    async def get_relations(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: MemoryRelationKind | None = None,
    ) -> Result[tuple[MemoryRelation, ...], ActionFailure]: ...


@runtime_checkable
class AssistantModelBackend(Protocol):
    """Provider-neutral LLM reasoning backend for assistant inference."""

    @property
    def state(self) -> LifecycleState: ...

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...

    async def generate(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> Result[AssistantCandidate, ActionFailure]: ...

    def stream(
        self, request: InferenceRequest, cancellation: CancellationToken
    ) -> AsyncIterator[Result[str, ActionFailure]]: ...
