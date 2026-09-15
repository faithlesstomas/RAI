# Assistant memory, GCAS and GWT-inspired processing

## Purpose and status

This note compares the post-Issue-#34 assistant memory plan with the GCAS 0.3
draft and GAIA's reference implementation. It records compatibility options;
it does not commit RAI to implementing GCAS, claim GCAS conformance or make
Global Workspace processing part of the supported assistant runtime.

The comparison uses the language- and implementation-independent
[GCAS 0.3 draft](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/gcas.md),
its operational Workspace semantics and GAIA's
[minimal cognitive cycle](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/gcas-core-cycle.md)
and
[conversation process](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/gcas-conversation.md)
as references. GAIA is non-normative: RAI should preserve GCAS semantics at an
adapter boundary rather than import GAIA implementation modules.

## Conclusion

The planned evidence-first SQLite architecture is a good possible host for the
GCAS object, justification and memory layers. It does not by itself implement
a Global Workspace or a conforming cognitive process. `ContextPackage` is a
transient projection selected for one inference; a GCAS Workspace is a bounded
recurrent process of candidate collection, eligibility, competition,
admission, broadcast, processor reaction and release.

RAI can therefore remain useful and complete without GCAS while preserving two
optional extension paths:

1. a small GWT-inspired context-selection experiment inside the assistant;
2. a full external GCAS cognitive runtime connected through a versioned adapter.

## GCAS state cross-check

GCAS models a process state as `S = <O, J, U, W, G, P, M, B, X>`. The planned
RAI correspondence is:

| GCAS state | RAI correspondence | Assessment |
|---|---|---|
| `O`: versioned Cognitive Objects | `ConversationTurn`, `MemoryProposal`, `MemoryRecord`, kernel `Observation`, `Claim`, `ActionResult` | Partial. Records are typed and immutable, but they do not yet share the complete Cognitive Object lifecycle and metadata. |
| `J`: justification graph | provenance plus planned `SUPPORTS`, `CONTRADICTS`, `UPDATES`, `SUPERSEDES` and source-span relations | Strong planned alignment, provided relations themselves remain provenance-qualified. |
| `U`: uncertainty projection | planned modality, source confidence and conflict state | Insufficient for GCAS 0.3. GCAS requires versioned, scoped `UncertaintyAssessment` objects with declared semantics and calibration. |
| `W`: bounded Workspace | bounded context assembly and `ContextManifest` | Input foundations exist, but one-shot retrieval and prompt assembly are not Workspace competition or broadcast. |
| `G`: Goals | explicit kernel `Task`; ordinary assistant turns deliberately create no task | Absent from ordinary chat by design. An optional GCAS adapter may create an internal Question/Goal without converting the RAI turn into a task or authorization. |
| `P`: cognitive processors | memory retriever, model backend, policy and future consolidator | Partial. GCAS-Core additionally requires explicit Generative and Deliberative roles and a recurrent processor protocol. |
| `M`: memory stores and indexes | raw evidence, bitemporal claims, semantic graph, FTS5, optional vectors and summaries | Strong planned alignment. |
| `B`: resource and failure budgets | `InferenceBudget`, usage ledger and bounded context/model execution | Strong alignment; a recurrent controller would also need progress, round and failure budgets. |
| `X`: authoritative environment observations | Rich History observations, capability results and current request | Strong alignment when fresh observations remain authoritative over retrieved memory. |

## Existing semantic alignment

The memory plan already follows several important GCAS rules:

- generated content is a proposal and not an admitted fact;
- conversation transcript, episodic evidence and semantic memory are separate;
- retrieval is epistemically neutral and must pass a guarded merge;
- persistent claims retain source provenance and temporal validity;
- contradictions and supersession do not silently erase history;
- prompts are bounded, transient projections rather than canonical state;
- capabilities, persistence and deletion remain behind RAI policy;
- exactly one terminal assistant response is committed for one accepted turn.

The planned M1–M7 work can supply most of the durable substrate needed by a
future GCAS adapter without making GCAS a dependency of the SQLite schema.

## Missing elements for a conformance claim

An assistant using the planned memory architecture must not be described as
GCAS-Core 0.3 conformant until it additionally demonstrates:

- a common Cognitive Object lifecycle independent of epistemic status;
- separate epistemic, verification and provenance axes;
- versioned uncertainty assessments with a declared calculus, scope and
  calibration limits;
- a bounded Workspace with observable
  `COLLECT -> ELIGIBILITY -> SCORE -> ADMIT -> BROADCAST -> REACT -> RELEASE`
  semantics;
- at least Generative and Deliberative processors, with shared failure sources
  declared if one physical model fills multiple roles;
- recurrent Cognitive Control with progress, interruption, failure and
  termination budgets;
- Goal acceptance contracts and an independent verification path;
- procedural, prospective and metacognitive memory where the claimed scope
  requires them;
- a governed Action/Result transaction integrated with, but unable to bypass,
  RAI capabilities and policy.

The most material gap is uncertainty, not storage technology. A scalar model
confidence or retrieval score cannot substitute for GCAS uncertainty and must
not be confused with scheduling priority, correctness or verification.

## Optional GWT-inspired context processing

The lowest-risk experiment fits after the reproducible retrieval floor and
adaptive context router are available. Retrieval channels submit discrete
context candidates; an optional bounded selector then performs one or a few
explicit focus rounds before `ContextPackage` is assembled:

```text
current turn and optional internal Question
  -> recent/raw/claim/graph retrieval candidates
  -> policy and provenance eligibility
  -> comparable scheduling features
  -> bounded competition and admission
  -> broadcast to subscribed context processors
  -> reactions or no response
  -> release or one further bounded round
  -> ContextPackage plus WorkspaceRoundManifest
```

Candidate scheduling metadata should keep at least these quantities separate:

- goal or query relevance;
- novelty or surprise;
- conflict and information-gap signals;
- urgency and risk;
- expected information gain;
- token, latency and compute cost;
- a reference to uncertainty, or an explicit `UNKNOWN` assessment.

Retrieval score remains evidence about discoverability, not truth. Scheduling
priority remains a control quantity, not confidence. Workspace admission and
broadcast must not promote epistemic or verification status.

This experiment should introduce an inspectable `WorkspaceRoundManifest`
containing all candidates, eligibility exclusions, score components, admitted
IDs, tie-break decisions, broadcasts, processor reactions, releases, budgets
and causal parents. It must not write memory or invoke a capability directly.

## Evaluation gate

Compare the GWT-inspired selector with the M5 adaptive router using identical
memory, model, prompt/output limits and compute budgets. At minimum evaluate:

- source and claim selection precision/recall;
- answer grounding and correct abstention;
- stale, conflicting and poisoned-memory selection;
- context size and useful evidence density;
- model calls, rounds, latency, energy and token cost;
- deterministic replay and exactly-once terminal behavior;
- non-progress, oscillation and budget-exhaustion rates.

Required ablations are: adaptive router without Workspace, competition without
broadcast, one round versus multiple rounds, and removal of individual score
features. A GWT label is not evidence of benefit. The experiment advances only
if it improves a declared RAI workload under matched budgets.

## Symbolic Workspace versus neural GW hypotheses

Two research questions must remain separate:

1. **Operational Workspace:** Does explicit candidate competition and broadcast
   improve context selection or deliberation? This can be tested after M4/M5
   with ordinary records and processors; it does not require J-lens, latent
   recurrence or neural slots.
2. **Neural GW-like dynamics:** Can ignition-, broadcast- or workspace-like
   behavior be identified in activations or writable slots? This belongs after
   read-only interpretability experiments and requires causal intervention and
   matched-compute controls before any architectural interpretation.

A positive result in either experiment does not establish the other and does
not establish biological Global Workspace Theory.

## Full GCAS integration boundary

If a later experiment needs a recurrent GCAS process, it should use a dedicated
`CognitiveRuntimeBackend` or the existing external-agent boundary rather than
turning `AssistantModelBackend` into a hidden controller.

RAI would supply policy-approved observations, candidate memories, budgets,
model execution and capability mediation. The cognitive runtime would own its
internal Questions, Goals, Workspace rounds, processor scheduling, epistemic
proposals and termination logic. It would return a terminal response plus
memory or capability proposals. RAI would still authorize persistence and
effects and would retain canonical assistant sessions, source evidence and
deletion semantics.

An internal GCAS Goal created to answer one turn is not a RAI kernel `Task` and
grants no tool authority. Only an explicit RAI capability request may cross the
execution boundary.

## Decision rule

Keep direct and adaptive-router conversation as the default. Treat a one-round
GWT-inspired selector as a falsifiable context-processing experiment and a full
GCAS controller as an optional external runtime. Do not add either to the
supported profile until it demonstrates benefit over the simpler path without
weakening evidence, privacy, deletion, budget or terminal-response invariants.

