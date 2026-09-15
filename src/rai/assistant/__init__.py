"""Rich Assistant package for graph-memory conversation and reasoning."""

from .audit import (
    AssistantAuditEntry,
    AssistantAuditLedger,
    InMemoryAssistantAuditLedger,
    JsonlAssistantAuditLedger,
)
from .backends.deterministic import DeterministicAssistantBackend
from .backends.local import LocalAssistantBackend
from .context import AssistantContextBuilder
from .diagnostics import MemoryDiagnosticReport, MemoryStageDiagnostic, diagnose_memory
from .ports import (
    AssistantModelBackend,
    AssistantSessionSummary,
    MemoryGraphStore,
    MemoryQuery,
)
from .query import MemoryQueryResolver
from .records import (
    AnyAssistantRecord,
    AssistantCandidate,
    AssistantContextManifest,
    AssistantContextManifestItem,
    AssistantContextPackage,
    AssistantResponse,
    AssistantSessionId,
    ConversationTurn,
    InferenceRequest,
    MemoryOperation,
    MemoryOperationKind,
    MemoryProposal,
    MemoryRecord,
    MemoryRelation,
    MemoryRelationKind,
    parse_assistant_record,
)
from .service import AssistantService
from .store import SQLiteMemoryGraphStore

__all__ = [
    "AnyAssistantRecord",
    "AssistantAuditEntry",
    "AssistantAuditLedger",
    "AssistantCandidate",
    "AssistantContextBuilder",
    "AssistantContextManifest",
    "AssistantContextManifestItem",
    "AssistantContextPackage",
    "AssistantModelBackend",
    "AssistantResponse",
    "AssistantService",
    "AssistantSessionSummary",
    "AssistantSessionId",
    "ConversationTurn",
    "DeterministicAssistantBackend",
    "InMemoryAssistantAuditLedger",
    "InferenceRequest",
    "JsonlAssistantAuditLedger",
    "LocalAssistantBackend",
    "MemoryGraphStore",
    "MemoryDiagnosticReport",
    "MemoryOperation",
    "MemoryOperationKind",
    "MemoryProposal",
    "MemoryQuery",
    "MemoryQueryResolver",
    "MemoryRecord",
    "MemoryRelation",
    "MemoryRelationKind",
    "MemoryStageDiagnostic",
    "SQLiteMemoryGraphStore",
    "parse_assistant_record",
    "diagnose_memory",
]
