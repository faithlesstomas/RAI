# Architecture

## Runtime responsibilities

RAI is the durable, local boundary between Linux and replaceable cognitive
processors. It stores observations and state, mediates capabilities, enforces
policy and constructs bounded context packages. It does not delegate ownership
of the computer or durable state to an LLM SDK.

## Core contracts

The Stage 1 kernel will formalize four primary ports:

```python
class Collector(Protocol):
    async def events(self) -> AsyncIterator[Observation]: ...

class Capability(Protocol):
    async def invoke(self, request: ToolRequest) -> Result[ToolResult, ToolError]: ...

class LocalCognition(Protocol):
    async def analyze(self, task: LocalTask) -> Result[StructuredOutput, InferenceError]: ...

class AgentBackend(Protocol):
    async def execute(
        self,
        task: Task,
        context: ContextPackage,
        capabilities: CapabilitySet,
    ) -> Result[AgentResult, BackendError]: ...
```

Models do not write durable state directly. Their outputs are validated and
committed by runtime services with provenance.

## Data flow

```text
Linux event
   → Collector
   → Observation log
   → deterministic EpisodeBuilder
   → optional local structured analysis
   → state/memory with provenance
   → ContextBuilder
   → local response or AgentBackend escalation
   → verified result
```

## Relationship to GCAS and GAIA

GCAS can define portable cognitive records and contracts. GAIA can implement
higher-level planning, workspace and deliberation. RAI supplies perception,
execution, policy and local state for Linux. GAIA may be an `AgentBackend` or a
separate cognitive runtime consuming RAI through MCP.

## Transitional implementation

The current `ChatService` still calls Google Antigravity directly. That path is
compatibility code and must move behind `AgentBackend`. The current MCP router
and `TOOL_REGISTRY` duplicate capability metadata and will converge in Stage 1.
The `inference` package is experimental and is not yet connected to routing.

