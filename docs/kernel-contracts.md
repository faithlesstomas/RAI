# Embodiment kernel contracts

Stage 1 defines a provider-neutral boundary between Linux devices, durable RAI
state and replaceable cognitive backends. The canonical language-neutral schema
is [`schemas/rai.kernel.v1.schema.json`](../schemas/rai.kernel.v1.schema.json).
Positive and negative conformance fixtures are published under
`tests/fixtures/kernel/v1/`.

## Versioning and compatibility

Every exchanged record contains `record_type`, `schema_version`, `record_id`,
an absolute `timestamp`, a structured `producer` identity and an optional
`correlation_id`. Schema versions use SemVer independently of the RAI package
version.

Readers accept later minor and patch releases within major version 1 and ignore
no unknown fields: senders must place new optional information in a new schema
revision, while receivers reject undeclared fields. A reader rejects malformed
versions and every unsupported major explicitly. Breaking field, meaning or
validation changes require a new major schema and a documented migration.

Derived records use `ProvenanceReference` entries. A reference identifies the
source record, its type and schema version, the producing component and the
relation to the derived record. Free-form provenance strings are not part of
the contract.

## GCAS and GAIA mapping

The mapping is semantic and does not import GAIA implementation modules:

| RAI record | GCAS/GAIA concept | Boundary rule |
|---|---|---|
| `Observation` | unverified Observation Cognitive Object | RAI supplies evidence and provenance; GAIA owns admission and epistemic status. |
| `Claim` | derived claim or belief candidate | It remains unverified unless the cognitive runtime promotes it. |
| `Task` | goal/task input | RAI carries the requested objective but does not own deliberation. |
| `ContextPackage` | bounded workspace input | Only manifest-listed, policy-approved content crosses the boundary. |
| `CapabilityRequest` | Action request | RAI policy and capability validation remain authoritative. |
| `ActionResult` / `ActionFailure` | terminal Action outcome | Exactly one typed terminal outcome is returned for a request. |
| `PolicyDecision` | authorization evidence | It is an RAI decision and cannot be overridden by a model. |

Adapters may translate these JSON records into GCAS wire objects. They must not
expose private backend SDK fields or move cognitive-control semantics into the
RAI kernel.

## Task and conversation boundary

The current version 1 kernel schema contains `Task` because local processors and
external agent backends need a bounded objective and a stable request identity.
It is not a chat-message type and does not imply planning, autonomy or tool
authority. In the transitional implementation, `LocalProcessor.process()` also
uses it as the execution envelope for one inference operation.

The planned assistant API separates those meanings:

| Planned record | Meaning | Persistence rule |
|---|---|---|
| `ConversationTurn` | Something a user or assistant said in an interaction. | An immutable interaction record; it is not automatically semantic memory. |
| `InferenceRequest` | One bounded request to a replaceable model backend. | Audit metadata may be retained; provider conversation state is never canonical memory. |
| `AssistantResponse` | The terminal response for one accepted turn. | Linked to its turn, context manifest and backend/run metadata. |
| `Task` | An explicit, trackable goal or delegated unit of work. | Created only through explicit interaction semantics and never inferred as authorization. |

Therefore a greeting, question, correction or conversational follow-up does not
become a `Task` merely because the assistant invokes a model to answer it. A
conversation may result in a proposed task, claim or capability request, but
each follows its own validation and policy boundary. These assistant records are
roadmap contracts and are not part of `rai.kernel.v1` until their schemas and
conformance fixtures land.
