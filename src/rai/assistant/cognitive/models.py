"""Data models for the Global Workspace Theory (GWT) cognitive engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rai.assistant.records import AssistantCandidate


class WorkspaceContributionKind(str, Enum):
    """Classification of specialist contributions to the global workspace."""

    RETRIEVAL = "retrieval"
    ACTION_PROPOSAL = "action_proposal"
    TOOL_CALL = "tool_call"
    ANSWER_DRAFT = "answer_draft"
    CRITIQUE = "critique"
    STATUS = "status"


class WorkspaceContribution(BaseModel):
    """One candidate contribution or broadcast focus from a specialist."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    cycle: int = Field(ge=0)
    specialist_name: str = Field(min_length=1)
    kind: WorkspaceContributionKind
    content: str = Field(default="")
    payload: dict[str, Any] = Field(default_factory=dict)
    salience: float = Field(ge=0.0, le=1.0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class WorkspaceState(BaseModel):
    """Ephemeral blackboard working memory for one deliberative turn."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    cycle: int = Field(default=0, ge=0)
    max_cycles: int | None = Field(default=4, ge=1)
    goal: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    context: Any = None
    broadcast: WorkspaceContribution | None = None
    history: tuple[WorkspaceContribution, ...] = Field(default_factory=tuple)
    observations: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    tool_calls_count: int = Field(default=0, ge=0)
    converged: bool = False
    final_candidate: AssistantCandidate | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
