# Architecture

## Runtime responsibilities

RAI is the durable, local boundary between Linux and replaceable cognitive
processors. It stores observations and state, mediates capabilities, enforces
policy and constructs bounded context packages. It does not delegate ownership
of the computer or durable state to an LLM SDK.

## Core contracts

The Stage 1 kernel defines immutable, versioned records in `rai.kernel.records`
and the following primary ports in `rai.kernel.ports`:

```python
class Collector(Protocol):
    async def start(self) -> Result[LifecycleState, ActionFailure]: ...
    def events(self, cancellation: CancellationToken) -> AsyncIterator[Result[Observation, ActionFailure]]: ...
    async def stop(self) -> Result[LifecycleState, ActionFailure]: ...

class Capability(Protocol):
    async def invoke(
        self, request: CapabilityRequest, cancellation: CancellationToken
    ) -> Result[ActionResult, ActionFailure]: ...

class LocalProcessor(Protocol):
    async def process(
        self, task: Task, context: ContextPackage,
        budget: InferenceBudget, cancellation: CancellationToken,
    ) -> Result[Claim, ActionFailure]: ...

class AgentBackend(Protocol):
    async def execute(
        self,
        task: Task,
        context: ContextPackage,
        capabilities: tuple[str, ...],
        budget: InferenceBudget,
        cancellation: CancellationToken,
    ) -> Result[ActionResult, ActionFailure]: ...
```

Models do not write durable state directly. Their outputs are validated and
committed by runtime services with provenance.

`CapabilityRegistry` is the only built-in operation catalog. CLI, REST, MCP and
the Antigravity compatibility adapter all resolve its descriptors, validate the
same JSON inputs and invoke `CapabilityService`. That service records a
deterministic `PolicyDecision`, obtains approval when required, invokes the
capability and writes exactly one terminal result to the audit ledger.

`ApplicationContainer` owns runtime services. `create_app()` creates an
independent FastAPI application and app-scoped MCP runtime; importing dependency
modules no longer creates history or model-registry singletons.

The Stage 2 event plane adds a transactional `EventJournal` between collectors
and processors. Authenticated loopback HTTP and a protected Unix socket share
one validation service. Independent cursors and acknowledgements make replay
explicit, while a deterministic subscriber demonstrates the complete path from
synthetic observation through policy to a durable terminal capability result.

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

For NCSI/J-lens integration, RAI hosts the optional neural implementation but
does not acquire cognitive-policy ownership. The J-lens provider runs as a
separately startable sidecar responsible for model and accelerator lifecycle,
activation access, lens artifacts and compact versioned events. GAIA consumes
those events through the NCSI contract and remains responsible for Cognitive
Objects, Workspace admission, Control, epistemic status and verification. The
[canonical cross-project plan](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/ncsi-jlens-integration.md)
owns the detailed milestones and acceptance gates.

## Transitional implementation

Google Antigravity is quarantined in `rai.backends.antigravity` behind the
public `AgentBackend` port. `rai.services.chat.ChatService` and the legacy CLI
are compatibility facades; SDK conversation internals do not enter kernel
records. New CLI capability commands are separated into command, transport and
rendering modules while the old command set lives in `cli_compatibility`.

The `inference` package remains experimental and is not yet connected to the
Stage 1 routing boundary.
