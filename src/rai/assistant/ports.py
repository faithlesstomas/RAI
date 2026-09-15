"""Provider-neutral runtime ports for the Rich Assistant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Protocol, runtime_checkable

from returns.result import Result

from rai.kernel.ports import CancellationToken, LifecycleState
from rai.kernel.records import ActionFailure, DataClass

from .records import (
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextPackage,
    AssistantResponse,
    ConversationTurn,
    InferenceRequest,
    MemoryOperation,
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


@dataclass(frozen=True)
class AssistantSessionSummary:
    """Bounded metadata for discovering an existing local conversation."""

    session_id: str
    started_at: datetime
    updated_at: datetime
    turn_count: int
    last_role: str
    preview: str


@runtime_checkable
class MemoryGraphStore(Protocol):
    """Transactional, graph-aware persistent store for assistant memory."""

    async def start(self) -> Result[LifecycleState, ActionFailure]: ...

    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...

    async def accept_turn(
        self, turn: ConversationTurn
    ) -> Result[ConversationTurn, ActionFailure]: ...

    async def commit_terminal(  # noqa: PLR0913
        self,
        response: AssistantResponse,
        manifest: AssistantContextManifest,
        assistant_turn: ConversationTurn | None,
        memories: tuple[MemoryRecord, ...],
        relations: tuple[MemoryRelation, ...],
        operations: tuple[MemoryOperation, ...] = (),
        context: AssistantContextPackage | None = None,
    ) -> Result[AssistantResponse, ActionFailure]: ...

    async def get_response_by_request_id(
        self, request_id: str
    ) -> Result[AssistantResponse | None, ActionFailure]: ...

    async def get_manifest(
        self, manifest_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]: ...

    async def get_context_package(
        self, manifest_id: str
    ) -> Result[AssistantContextPackage | None, ActionFailure]: ...

    async def get_latest_manifest_for_session(
        self, session_id: str
    ) -> Result[AssistantContextManifest | None, ActionFailure]: ...

    async def get_turn(
        self, turn_id: str
    ) -> Result[ConversationTurn | None, ActionFailure]: ...

    async def get_memory(
        self, memory_id: str
    ) -> Result[tuple[MemoryRecord, str] | None, ActionFailure]: ...

    async def get_recent_reply_chain(
        self,
        session_id: str,
        limit: int = 10,
        before_turn_id: str | None = None,
    ) -> Result[tuple[ConversationTurn, ...], ActionFailure]: ...

    async def list_sessions(
        self, limit: int = 50
    ) -> Result[tuple[AssistantSessionSummary, ...], ActionFailure]: ...

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

    async def list_memory_operations(
        self, profile_scope: str = "default", limit: int = 100
    ) -> Result[tuple[MemoryOperation, ...], ActionFailure]: ...

    async def replay_memory_projection(
        self, profile_scope: str = "default"
    ) -> Result[tuple[str, ...], ActionFailure]: ...


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
