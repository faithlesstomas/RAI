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
from .extraction import (
    ExtractedClaimKind,
    ExtractedMemoryCandidate,
    MemoryExtractionOutput,
    SchemaConstrainedMemoryExtractor,
)
from .evidence import RichHistoryEvidenceProvider
from .evaluation import (
    RetrievalAnswerEvaluator,
    RetrievalChannelMeasurement,
    RetrievalChannel,
    RetrievalEvaluationCase,
    RetrievalEvaluationRun,
    evaluate_retrieval_floor,
)
from .ports import (
    AssistantModelBackend,
    AssistantEvidence,
    AssistantEvidenceProvider,
    AssistantSessionSummary,
    MemoryGraphStore,
    MemoryProposalExtractor,
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
from .summary import GroundedClaimSummaryProvider, validate_grounded_summary

__all__ = [
    "AnyAssistantRecord",
    "AssistantAuditEntry",
    "AssistantAuditLedger",
    "AssistantCandidate",
    "AssistantContextBuilder",
    "AssistantContextManifest",
    "AssistantContextManifestItem",
    "AssistantContextPackage",
    "AssistantEvidence",
    "AssistantEvidenceProvider",
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
    "MemoryProposalExtractor",
    "MemoryExtractionOutput",
    "ExtractedClaimKind",
    "ExtractedMemoryCandidate",
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
    "RichHistoryEvidenceProvider",
    "GroundedClaimSummaryProvider",
    "RetrievalAnswerEvaluator",
    "RetrievalChannel",
    "RetrievalChannelMeasurement",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationRun",
    "SQLiteMemoryGraphStore",
    "SchemaConstrainedMemoryExtractor",
    "parse_assistant_record",
    "diagnose_memory",
    "evaluate_retrieval_floor",
    "validate_grounded_summary",
]
