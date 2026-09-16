# Rich Assistant architecture

## Status and scope

Rich Assistant delivers the Issue #34 graph-memory foundation and the first
user-facing Issue #35 slice. It provides transactional SQLite
graph storage (`SQLiteMemoryGraphStore`), two-phase interaction execution
(`AssistantService`), inspectable context packages with `ContextManifest`,
and direct local model inference (`LocalAssistantBackend`). The 8-step user-visible
acceptance scenario passes deterministically offline, and the restart/recall/
correction path has also been exercised with a pinned TinyLlama GGUF artifact.
Users can resume and inspect sessions, history, exact context packages, active
memory, operation traces and stage-specific diagnostics from CLI or API.

The assistant is a continuous, local-first interaction surface with explicit,
reconstructed context. It may conduct ordinary conversation, answer from RAI
memory and propose work. It is not a GCAS Workspace, an autonomous goal loop or
an alternative capability authority.

## Conversation is not a task

The assistant uses separate records for interaction, model execution and work:

| Concept | Responsibility |
|---|---|
| `ConversationTurn` | Immutable user or assistant utterance with session grouping, reply linkage, privacy class and provenance. |
| `AssistantResponse` | Exactly one terminal outcome for an accepted user turn, linked to the context and inference run that produced it. |
| `InferenceRequest` | Ephemeral or audit-retained envelope for one bounded backend invocation. |
| `MemoryProposal` | Untrusted candidate for retention, including its source, proposed kind, relations and privacy class. |
| `MemoryRecord` | Immutable, policy-admitted durable memory with provenance, validity and correction state. |
| `Task` | Explicit, trackable user goal or delegated unit of work. |
| `Claim` | Derived proposition that remains unverified until admitted or assessed by an owning policy/runtime. |
| `CapabilityRequest` | Proposed system action that must pass the common policy and verification path. |

A greeting, question, correction or conversational follow-up is only a
`ConversationTurn`. The service can create an `InferenceRequest` to answer it,
but it must not create a `Task`. A task requires explicit interaction semantics;
model intent classification can propose one but cannot create, authorize or
execute it silently.

## One-turn processing boundary

For every accepted turn, `AssistantService` performs one bounded pipeline:

```text
accept ConversationTurn
  → validate session, size, data class and cancellation state
  → persist the accepted turn
  → select a bounded recent conversation window
  → retrieve relevant allowed records from durable graph memory
  → build ContextPackage and ContextManifest
  → invoke one AssistantModelBackend through an InferenceRequest
  → validate the candidate response and proposals
  → deliver exactly one terminal AssistantResponse
  → commit separately approved turn, memory and audit records
```

Streaming chunks are transport events, not durable partial responses. A timeout,
cancellation, backend failure or invalid output produces one typed terminal
failure and never commits a generated assistant turn as successful.

The supported text surfaces are `rai assistant ask`, `rai assistant chat`,
`POST /api/v1/assistant/turn`, `POST /api/v1/assistant/stream` and the native
`/api/v1/assistant/ws` WebSocket. Read APIs expose sessions, turns, active
memories, exact context packages, operation traces and memory diagnostics. The default
standalone `rai`/`rai -p` path delegates to the same service. Legacy
provider-owned `/api/v1/run`, `/api/v1/stream` and `/ws/v1/chat` behavior is
disabled unless `legacy_chat.enabled` is explicitly set to `true`.

## State ownership

`AssistantSessionId` groups interactions and audit records; it is not model
state. Each inference starts from the explicit context package. Provider-side
conversation IDs and hidden history cannot be resumed as canonical state.
Durable assistant memory is scoped to the configured local profile rather than
to one conversation session. A profile selects model configuration, system
instruction and the memory namespace. A session ID selects only the bounded
recent-dialogue window. Cross-session retrieval still passes the current
client, privacy and data-class policy before a record can enter context.

The durable graph distinguishes interaction records from semantic memory.
Turn nodes use stable RAI IDs and `REPLIES_TO` edges. The current product slice
admits separately represented personal, project, system and conversation claims
from ordinary dialogue; explicit remember/forget requests are optional controls.
Qualified `DERIVED_FROM`, `SUPPORTS`, `CONTRADICTS`, `UPDATES` and `SUPERSEDES`
relations retain their evidence. A deterministic grounded-summary projection is
available only as a rebuildable evaluation channel and is not durable truth or a
default context route.
Model output is only a proposal: deterministic policy owns admission,
correction, expiry and deletion propagation.

An admitted user preference records that the user stated or selected the
preference; it is not a verified proposition about the external world. The
memory record retains its source-turn ID and remains distinct from the source
turn, so retention, correction and deletion can be applied independently and
audited through provenance edges.

Latent recurrence vectors, writable slots, KV caches, token scratchpads and raw
activations never become durable truth. They remain inside their backend or
sidecar and are represented externally only by bounded run metadata and approved
observer artifacts.

## Context composition

Every inference receives one finite context assembled from independently
selected layers:

```text
policy and assistant instructions
  + current ConversationTurn
  + bounded recent conversation window
  + bounded relevant durable graph memories
  + optional policy-approved Rich History references
```

The recent window and graph-memory retrieval are not interchangeable. The
window preserves local conversational coherence by following the active reply
chain and selecting recent turns. Durable retrieval selects current memories by
query relevance, graph relations, provenance, temporal validity, privacy and
supersession state, including memories created in another session. Each layer
has explicit item, character or token, privacy and latency budgets; the current
turn is retained before older context consumes the remaining size budget.

`ContextManifest` records recent-turn IDs and durable-memory IDs separately. It
also records exclusions, redactions, ranking reasons, policy and retrieval
versions and actual sizes. A test must therefore be able to prove that a reply
used graph memory rather than a hidden provider transcript or an oversized
conversation replay.

## First graph-memory vertical slice

The first implementation should deliver one narrow memory-backed behavior end
to end. A turn-only chat is useful scaffolding but does not satisfy this slice:

1. Define versioned fixtures for `ConversationTurn`, `InferenceRequest` and
   `AssistantResponse`, plus the minimal durable memory and provenance records,
   without changing the meaning of kernel `Task`.
2. Add a container-owned `AssistantService` and deterministic fake
   `AssistantModelBackend` that can return a response and bounded memory
   proposals for conformance tests.
3. Store user and assistant turn nodes plus `REPLIES_TO` edges in a minimal
   SQLite implementation of a backend-neutral graph-store protocol. Store an
   admitted preference as a separate node linked to its source turn with
   provenance, and represent a correction with `SUPERSEDES`.
4. For every request, select the recent reply-chain window and relevant current
   graph memories independently, then assemble one size- and privacy-bounded
   `ContextPackage` and retain its `ContextManifest`.
5. Reconstruct that context after daemon restart and across a new assistant
   session without provider conversation state or full-transcript replay.
6. Prove that ordinary conversation creates no `Task`, admits no assistant
   output as fact and creates no `CapabilityRequest`.
7. Cover duplicate delivery, cancellation, timeout, invalid backend output,
   poisoned or superseded retrieval, source deletion and provider-state leakage
   with deterministic tests.

This slice intentionally excludes proactive triggers, tool use, durable or
model-written lossy conversation summaries, vector retrieval, external model
APIs, Coconut recurrence and writable slots. It now includes governed free-form
candidate extraction, bitemporal claims, FTS5 raw-turn/claim retrieval, Rich
History evidence and the query-time grounded-summary evaluation baseline, while
keeping every model output outside the policy boundary.

## Local operation and observability

Use a chat-capable GGUF file through llama.cpp:

```bash
uv sync --extra inference-llama
uv run rai assistant chat --backend llama --model /path/to/model.gguf --show-context
```

`--show-context` prints the selected recent-turn IDs, durable-memory IDs,
exclusions, model checksum, prompt version, token usage, latency and generation
guard metadata. Each process reconstructs its prompt from those RAI-owned
records; no provider conversation ID is resumed. `RAI_DATA_DIR` selects an
isolated profile for A/B experiments.

The runtime has a bounded deterministic admission and grounding policy for
common personal facts, preferences, plans and optional explicit controls. A
separate local schema-constrained extractor can propose arbitrary facts,
preferences, plans, events, system state, relationships and conversation
commitments from natural user turns. Quoted, hearsay, hedged and malformed
candidates are retained as rejected operation evidence rather than silently
becoming memory. The local model still runs for every response, but critical
recall questions are answered from selected evidence. RAI records
`grounding_override=true` whenever that gate replaces model prose.

Context routing keeps recent interaction memory, FTS5/BM25 raw conversational
evidence, Rich History observations and admitted claims separate. Compact
memory above the configured sufficiency threshold avoids unnecessary raw
disclosure; otherwise the router falls back to source evidence. The manifest
records the route, alternatives, score, fallback, channel source IDs and exact
budget. Real-world validity and transaction-time intervals support current and
historical queries without rewriting old claims.

Every attempted mutation has a versioned `MemoryOperation`. Its trace records
the trigger, proposal/source span, policy outcome, before/after IDs and terminal
status. Replaying applied operations must equal the active SQLite projection.
`rai assistant diagnostics` reports extraction, admission, storage, update,
retrieval and answer-use stages separately. The M4 harness seeds a versioned
natural-conversation corpus with frozen admitted claims, invokes the same
`AssistantModelBackend` boundary as the product, and judges required/forbidden
answer content plus explicit abstention without model self-grading. Aggregates
retain missing energy and provider-cost measurements as missing rather than
silently treating them as zero. `rai assistant benchmark-memory` compares raw
turns, claims, grounded summaries, local dense feature hashing, weighted RRF and
authenticated bounded graph paths under equal retrieval/context budgets. Two
live manifests and the opt-in decision are recorded in
[the M4–M6 evaluation](assistant-m4-m6-evaluation.md).

`rai`, `rai -p`, `rai assistant ask` and `rai assistant chat` instantiate this
same service directly and do not require `rai serve`. The assistant HTTP routes
are an alternative transport over a server-owned instance of the same service.
Only explicit `rai --connect` enters the quarantined compatibility client; it
does not define canonical assistant memory or session semantics.

The SQLite database and audit ledger use per-user directories/files (0700/0600)
and secure deletion. They are not yet encrypted independently of the user
account. `SECRET` and `BLOCKED` turns are rejected; `PRIVATE` memory cannot be
downgraded and is excluded from default retrieval. The slice performs whole-item
exclusion rather than field-level redaction because its preference records have
no mixed-class fields.

## First user test: remembered preference

The first user-visible test runs against an isolated local profile and one
approved local `AssistantModelBackend`. A deterministic backend runs the same
scenario first as the CI conformance floor; a live-model pass is recorded as
separate product evidence.

1. In session A, the user says: "Zapamiętaj, że w przykładach kodu preferuję
   Guile zamiast Pythona." The response may acknowledge the preference, but
   success is determined by the graph: a preference memory is admitted
   separately from the turn and linked to it by provenance.
2. End the client and restart the RAI daemon. Start session B so its recent
   conversation window does not contain the original statement.
3. Ask: "W jakim języku powinieneś pokazywać mi przykłady kodu?" The response
   must select Guile. Its `ContextManifest` must list the preference under
   durable graph memories and list no original statement under recent turns.
   The inference trace must independently confirm empty provider-owned history.
4. Correct the preference: "Zmień tę preferencję: używaj Pythona w przykładach
   kodu." The new memory must `SUPERSEDES` the Guile preference while the old
   record remains auditable and ineligible for current retrieval.
5. Restart again and repeat the question. The answer and manifest must use only
   the current Python preference. Deleting the correction's source must exercise
   the declared deletion policy, remove or invalidate its derived retrieval
   entry and never silently reactivate the superseded Guile preference.

The harness also asserts bounded context size, source and ranking metadata,
exactly one terminal response per accepted turn, no `Task` or capability
invocation and no factual promotion of assistant prose. Passing only the recall
answer is insufficient: the stored graph and manifest are acceptance evidence.

## Relationship to existing components

The current `ProcessorSupervisor` remains useful for schema-constrained local
analysis and may support memory extraction or routing. It is not itself the chat
backend because it returns a `Claim`, not a conversational streaming response.
The future `AssistantModelBackend` may wrap compatible local engines while
preserving its separate response, streaming and usage contract.

All proposed actions continue through `CapabilityService`. Rich History remains
an independent, privacy-filtered evidence source. GAIA or another cognitive
runtime may later implement `AgentBackend` for delegated tasks, but it does not
own RAI assistant sessions or durable graph memory.

GAIA's implemented conversational memory is a behavioral reference for the
split between recent dialogue and relevant structured memory, restart recall,
epistemically neutral retrieval, invalidation and supersession. RAI should reuse
those acceptance ideas, not import GAIA's `Goal`, Global Workspace or Cognitive
Object lifecycle into ordinary conversation. A future GCAS adapter maps RAI
records at the boundary while RAI retains memory, privacy and context ownership.
