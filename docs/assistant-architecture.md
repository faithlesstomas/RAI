# Planned Rich Assistant architecture

## Status and scope

Rich Assistant is planned Stage 4.7 work. The current tree still exposes a
provider-owned `ChatService` compatibility facade and a flat SQLite conversation
history; neither defines the target API or memory semantics. This document
freezes the boundary for the first replacement slice before implementation.

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
  → retrieve a bounded set of allowed graph records
  → build ContextPackage and ContextManifest
  → invoke one AssistantModelBackend through an InferenceRequest
  → validate the candidate response and proposals
  → deliver exactly one terminal AssistantResponse
  → commit separately approved turn, memory and audit records
```

Streaming chunks are transport events, not durable partial responses. A timeout,
cancellation, backend failure or invalid output produces one typed terminal
failure and never commits a generated assistant turn as successful.

## State ownership

`AssistantSessionId` groups interactions and audit records; it is not model
state. Each inference starts from the explicit context package. Provider-side
conversation IDs and hidden history cannot be resumed as canonical state.

The durable graph will distinguish interaction records from semantic memory.
Initial turn nodes use stable RAI IDs and `REPLIES_TO` edges. Later slices may
admit preferences, claims, summaries, corrections and relations such as
`DERIVED_FROM`, `SUPPORTS`, `CONTRADICTS`, `ABOUT` and `SUPERSEDES`. Model output
is only a proposal: deterministic policy owns admission, correction, expiry and
deletion propagation.

Latent recurrence vectors, writable slots, KV caches, token scratchpads and raw
activations never become durable truth. They remain inside their backend or
sidecar and are represented externally only by bounded run metadata and approved
observer artifacts.

## First vertical slice

The first implementation should deliver one narrow behavior end to end:

1. Define versioned fixtures for `ConversationTurn`, `InferenceRequest` and
   `AssistantResponse` without changing the meaning of kernel `Task`.
2. Add a container-owned `AssistantService` and deterministic fake
   `AssistantModelBackend`.
3. Store user and assistant turn nodes plus `REPLIES_TO` edges in a minimal
   SQLite implementation of a backend-neutral graph-store protocol.
4. Reconstruct an explicit, size- and privacy-bounded context after daemon
   restart and retain its `ContextManifest`.
5. Prove that a normal chat exchange creates no `Task`, no `Claim` admitted as
   fact and no `CapabilityRequest`.
6. Cover duplicate delivery, cancellation, timeout, invalid backend output,
   deletion and provider-state leakage with deterministic tests.

This slice intentionally excludes proactive triggers, tool use, semantic-memory
promotion, vector retrieval, external model APIs, Coconut recurrence and
writable slots. Those features build on the same service and graph boundaries
after the direct conversational path is reliable.

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
