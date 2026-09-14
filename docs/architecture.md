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

The Stage 3 Rich History slice places a deterministic privacy firewall in front
of that journal. Bounded platform adapter records are dropped, reduced,
redacted or allowed before they become kernel observations. A failure-isolating
collector supervisor owns opt-in state, lock/emergency cancellation and health.
Deterministic fusion and episode construction retain source provenance in an
AES-GCM-encrypted local store; review, local questions and verified cascading
deletion share the authenticated control API. No model participates in this
path.

Native collection remains outside the daemon process. A bundled GNOME Shell
extension exports focus and workspace changes over a user-session D-Bus name;
the GNOME sidecar adds ScreenSaver lock and Mutter idle transitions. Separate
AT-SPI, focused-process and approved-root filesystem sidecars emit bounded JSON
lines. The supervisor launches these commands without a shell or credential
environment and the in-process collectors sanitize the records again before the
privacy firewall.

The first Stage 4.3 voice slice lives in `rai.speech`. It separates immutable
synthesis requests and backend metadata from a `SpeechSynthesizer` model port
and an `AudioPlayer` device port. A profile supplies an ordered backend policy;
selection intersects that policy with language, locality, latency, cost and
RAM/VRAM limits. The actuator produces a terminal result only after the player
returns a completion receipt and excludes both input text and PCM bytes from the
result. The deterministic backend and player exercise this lifecycle without a
model, network, microphone or speaker. A lazy Piper adapter and local
sounddevice player implement the same ports without import-time optional
dependency loading or implicit downloads. A registry adapter places the whole
operation behind `CapabilityService`, and the default registry exposes it to
the runtime and MCP. User-configurable profile/device loading and STT are still
planned.

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

## Planned assistant boundary

The target Rich Assistant is a bounded application service, not a second
cognitive kernel. One container-owned `AssistantService` will accept text,
voice and future desktop interactions and construct every model call from
explicit RAI state:

```text
ConversationTurn
   → interaction and privacy policy
   → bounded graph-memory retrieval
   → ContextPackage + ContextManifest
   → InferenceRequest
   → AssistantModelBackend + ReasoningStrategy
   → candidate AssistantResponse, memory proposals and capability proposals
   → validation and policy
   → one terminal response plus separately committed durable records
```

These records deliberately describe different things. A `ConversationTurn`
records what was said. An `InferenceRequest` bounds one model invocation. A
`Task` represents an explicit goal worth tracking or delegating. Consequently,
ordinary chat does not create a task. If a turn proposes an operating-system
action, the proposal still becomes a separate `CapabilityRequest` and passes
through the existing policy, approval, invocation and verification path.

The graph store, not a provider session, will own conversation and semantic
memory. Provider conversation IDs, KV caches, hidden states, Coconut recurrence
vectors and writable slots remain ephemeral backend state. The first replacement
slice and its failure boundaries are specified in the [assistant architecture](assistant-architecture.md).

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

The container now owns the experimental processor supervisor, all six bounded
task contracts and the optional policy-aware result cache. No daemon API,
assistant service or registered capability currently dispatches work to that
supervisor, and neither production text adapter has a repeatable live-model
acceptance test. It is therefore implemented infrastructure, but not yet a
complete Stage 1 routing or user-facing assistant path.
