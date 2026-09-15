# Assistant memory development plan

## Decision

RAI develops its memory architecture against a local SQLite reference
implementation. Graphiti/FalkorDB, AtomSpace and other stores remain optional
adapters. RAI owns the records, lifecycle operations, policy, provenance,
retrieval budgets and evaluation protocol; an adapter cannot become the source
of truth for those semantics.

The architectural centre is an auditable evidence and operation log, not a
particular graph database. The temporal graph, FTS index, embeddings and
summaries are derived, rebuildable projections over admitted records.

```text
policy-approved source turns and episodes
  -> untrusted extraction proposals
  -> deterministic validation and admission
  -> immutable evidence plus memory-operation log
  -> SQLite projections: temporal claims, graph, FTS5, optional vectors
  -> adaptive retrieval and evidence reconstruction
  -> bounded ContextPackage plus ContextManifest
  -> replaceable model backend
```

This plan starts after the bounded personal-memory MVP delivered by Issue #34.
It does not redefine that completed slice as general semantic memory.

## Memory layers

1. **Interaction memory** preserves the bounded conversational reply chain. It
   exists for local coherence and is not durable semantic truth.
2. **Raw episodic evidence** contains approved turns and Rich History episodes.
   It is immutable until retention or deletion policy removes it.
3. **Claim memory** contains admitted statements with source spans, modality,
   scope, confidence and temporal validity. A claim is not necessarily a fact.
4. **Semantic graph** connects claims, entities and episodes with
   provenance-qualified relations such as `ABOUT`, `SUPPORTS`, `CONTRADICTS`,
   `UPDATES` and `SUPERSEDES`.
5. **Derived summaries and communities** are disposable acceleration
   structures. They never acquire more authority than their sources.
6. **Working memory** is a request- or task-scoped selection of the layers
   above. It expires independently and is not silently promoted.

Every layer has an explicit owner, retention rule, privacy class and deletion
path. A selected item remains attributable to the operation and evidence that
created it.

## Delivery stages

### M1 — memory operations and diagnostic evaluation

Status: `[x]` for the SQLite reference MVP. Broader benchmark corpora remain
part of the later reproducible evaluation work.

Define versioned operations for `remember`, `forget`, `update`, `supersede`,
`reflect` and `reconstruct`. Each operation records its trigger, target, scope,
precondition, state transition, policy decision and evidence.

Build a diagnostic harness which distinguishes failures in extraction,
admission, storage, update, retrieval and answer use. Preserve the Issue #34
deterministic scenario as the conformance floor and add free-form conversational
fixtures inspired by MemOps, HaluMem and MemFail.

Exit evidence:

- every state mutation has one typed operation trace;
- replay produces the same current memory projection;
- stage-specific failures cannot be hidden by a correct final answer;
- no regex or model output can directly commit durable memory.

### M2 — evidence-preserving candidate extraction

Status: `[x]` for the local extraction/admission contract. Deterministic
Polish/English controls and a separate schema-constrained local SLM extractor
both emit untrusted proposals with exact source spans. Malformed, unsupported,
uncertain, quoted and privacy-ineligible proposals fail closed and extraction
failures are recorded independently from admission.

Replace the narrow Issue #34 recognizers with a proposal pipeline that can find
candidate preferences, personal statements, events, plans, corrections and
external-world claims in ordinary conversation. Start with deterministic
parsers as the control and add a schema-constrained local SLM extractor as an
untrusted proposal source.

The proposal must preserve exact source spans and distinguish assertion,
preference, intent, question, quotation, hearsay, uncertainty and negation.
Admission remains deterministic wherever possible; ambiguous conflicts are
held for review or passed to a bounded resolver without overwriting evidence.

Exit evidence:

- useful facts are proposed without an explicit `remember` trigger;
- hedged or quoted text is not promoted to a confident first-party fact;
- malformed, unsupported or privacy-ineligible proposals fail closed;
- extraction quality and admission quality are reported separately.

### M3 — bitemporal claim graph in SQLite

Status: `[x]` for the SQLite reference. Claims carry real-world validity and
transaction-time intervals; current and historical retrieval use both.
Corrections preserve qualified `CONTRADICTS` and `SUPERSEDES` relations, and
source deletion removes the raw FTS projection and dependent graph state.

Add separate transaction and real-world validity intervals. Preserve previous
states for historical questions while current-state queries exclude expired or
superseded claims. Give relations their own provenance and eligibility instead
of trusting topology merely because their endpoint records are valid.

Use normalized SQLite tables and foreign keys for authoritative records, FTS5
for lexical projections and recursive CTEs for bounded graph paths. Keep schema
and domain records independent of SQLite-specific identifiers.

Exit evidence:

- current and `as of` queries return different, correct states;
- corrections retain history and never reactivate an older value accidentally;
- deleting a source invalidates every dependent claim, edge and index entry;
- a relation with missing or rejected provenance cannot affect retrieval.

### M4 — reproducible retrieval floor

Status: `[/]`. Cross-session raw user turns and durable claims use FTS5/BM25
projections, while Rich History episodes are available through a bounded,
privacy-aware evidence provider. The reproducible runner compares raw turns and
claims with identical retrieval and character budgets and reports retrieval
quality, abstention, latency and context size. Summary comparison, answer
utilization and energy integration remain open, so this stage is not complete.

Implement raw-turn and raw-episode retrieval with FTS5/BM25 and query-driven
pruning. Compare it against claim retrieval and summary retrieval before adding
embedding or graph complexity. The harness fixes corpus, answerer, prompt,
retrieval budget, context budget and judge protocol.

Exit evidence:

- raw, claim and summary variants run under identical budgets;
- recall, precision, answer quality, abstention, latency, energy and context
  size are reported together;
- retrieval failure is distinguishable from model-utilization failure;
- a more complex variant is not enabled unless it beats the simple floor on a
  declared target workload.

### M5 — adaptive context router and verified fallback

Status: `[/]`. The context builder now chooses compact claim memory when its
configurable sufficiency threshold is met, otherwise falls back to raw turns
and approved external evidence or records `no_evidence`. The manifest records
candidate/rejected routes, score, fallback and evidence budget. Comparative
short/long-history evaluation and verified derived write-back remain open.

Select recent/full context for short histories when it is cheaper and at least
as accurate. For longer histories, route through lexical or semantic memory.
When a claim or summary is insufficient, escalate to raw supporting evidence.
Write verified findings back only as derived claims linked to their sources.

Thresholds are configuration and experiment results, not constants copied from
one benchmark. `ContextManifest` records the chosen route, rejected routes,
sufficiency signal, fallbacks and final evidence budget.

Exit evidence:

- the router beats or matches an always-memory policy across short and long
  histories under a fixed budget;
- a load-bearing answer cannot rely only on a lossy summary;
- insufficient evidence produces fallback or abstention, not fabrication;
- route selection is deterministic for fixed inputs and policy versions.

### M6 — hybrid and graph retrieval

Add a local dense channel and evaluate RRF against lexical retrieval. Evaluate
cross-encoder reranking only after measuring its local resource cost. Add
bounded temporal paths and personalized PageRank last.

Construct traversal input from an authenticated, policy-eligible subgraph.
Filtering only the final result is insufficient because poisoned topology can
change which otherwise trusted facts are selected.

Exit evidence:

- lexical, dense, fused and graph variants use the same returned-context budget;
- graph paths expose their supporting node and edge IDs in the manifest;
- injected untrusted edges cannot change trusted selection;
- quality gains survive multiple local models and are not judge-specific.

### M7 — scope, personalization and consolidation

Status: `[/]`. Profile, domain and purpose isolation is enforced for claims,
raw-turn retrieval and recent context; manifests expose the selected scope.
Negative tests cover unrelated-domain and wrong-purpose retrieval. Broader
leakage/sycophancy evaluation and consolidation remain open.

Partition durable memory by user, profile, domain and purpose. Retrieve personal
context only when the request and policy require it. Measure cross-domain
leakage and memory-induced sycophancy as safety failures.

Introduce consolidation and community summaries only for corpora where the M4
and M6 baselines demonstrate a need. Community membership and summaries remain
rebuildable projections with source coverage and expiry metadata.

Exit evidence:

- unrelated domains do not receive personal claims by default;
- personalization cannot override evidence or calibrated uncertainty;
- consolidation preserves modality, conflicts and source coverage;
- deleting all sources makes the derived summary ineligible.

### M8 — optional adapters and ADR

Implement `GraphitiMemoryGraphStore` only against the stable conformance suite.
Graphiti extraction is a proposal source, not an admission authority. Compare
SQLite and Graphiti with identical data, model, retrieval budget and deletion
requirements. Apply the same rule to later AtomSpace experiments.

Record an ADR after the benchmark. A candidate may become a supported profile
only if its measurable benefit exceeds its service, memory, backup, privacy and
failure-recovery cost without weakening RAI invariants.

Exit evidence:

- the common lifecycle and context conformance suite passes unchanged;
- export and deletion cover adapter-owned derived state;
- adapter unavailability does not corrupt canonical memory;
- the ADR includes negative results and reproducible run manifests.

## Evaluation matrix

Every meaningful experiment records at least:

| Dimension | Required comparison |
|---|---|
| History | short, medium and long; current and historical questions |
| Write representation | raw evidence, extracted claims, summaries |
| Retrieval | recent/full, BM25, dense, RRF, bounded graph path |
| Operation | remember, update, supersede, forget, reflect, reconstruct |
| Evidence quality | direct, hedged, hearsay, conflicting, poisoned, deleted |
| Scope | same domain, unrelated domain, private/ineligible source |
| Model role | extractor, resolver, answerer and judge recorded separately |
| Cost | ingest and query tokens, context size, latency, memory and energy |

Use RAI-owned deterministic fixtures first, then selected subsets or adapted
scenarios from LongMemEval, MemOps, HaluMem, MemFail and conversational-memory
benchmarks. Published scores are not directly comparable unless ingestion,
retrieval budgets, answer model and judge protocol are controlled.

## Definition of done for general assistant memory

General memory is not complete when the assistant merely recalls a planted
fact. It is complete only when the implementation and manifests prove all of
the following:

1. Natural conversation produces typed candidates without special trigger
   phrases, while policy remains the sole admission authority.
2. Source wording, modality, uncertainty, scope and temporal validity survive
   extraction, consolidation and retrieval.
3. Current, corrected and historical queries select the appropriate state with
   source evidence.
4. Retrieval is adaptive, independently bounded per channel and auditable from
   the context manifest.
5. Insufficient memory causes raw-evidence fallback or explicit abstention.
6. Poisoned records or graph topology cannot alter the eligible subgraph or
   smuggle protected data into context.
7. Deletion and forgetting remove or invalidate source data and every derived
   projection according to policy, without resurrecting superseded state.
8. Domain boundaries prevent irrelevant personalization and are covered by
   leakage and sycophancy tests.
9. Operation-level metrics identify where hallucination entered the lifecycle;
   final-answer accuracy alone is not the acceptance criterion.
10. SQLite passes the full suite locally. Optional adapters pass the same suite
    and demonstrate value under equal budgets before becoming supported.

## Research basis

Architecture and evidence preservation:

- [Zep](https://arxiv.org/abs/2501.13956) motivates bi-temporal fact validity,
  episodic/entity/community projections and incremental graph maintenance.
- [SuperLocalMemory 4.0](https://arxiv.org/abs/2608.08253) motivates governed,
  local-first SQLite storage and multi-channel retrieval.
- [TierMem](https://arxiv.org/abs/2602.17913) motivates an immutable raw tier,
  compact acceleration tiers and evidence fallback.
- [TRACE](https://arxiv.org/abs/2607.00339) and
  [SodaMem](https://arxiv.org/abs/2608.08055) motivate temporal evidence graphs,
  explicit update/conflict relations and source-grounded reconstruction.
- [LatticeMind](https://arxiv.org/abs/2608.08236) motivates cheap symbolic
  conflict checks before bounded model-assisted reconciliation.

Safety and memory lifecycle:

- [Manufactured Confidence](https://arxiv.org/abs/2606.29279) motivates
  preservation of hedging, hearsay and source modality across summaries.
- [Selection Integrity](https://arxiv.org/abs/2606.12290) motivates computing
  graph selection over an authenticated subgraph rather than filtering only
  returned records.
- [MemOps](https://arxiv.org/abs/2607.12893),
  [HaluMem](https://arxiv.org/abs/2511.03506) and
  [MemFail](https://arxiv.org/abs/2605.26667) motivate operation- and
  stage-specific evaluation instead of answer-only scoring.
- [Control-Plane Placement Shapes Forgetting](https://arxiv.org/abs/2606.15903)
  motivates evaluating deletion, canonicalization and mutation-time reasoning
  separately from recall.
- [Mitigating Over-Personalization](https://arxiv.org/abs/2608.08300) motivates
  domain-scoped context and explicit leakage/sycophancy tests.

Retrieval and evaluation protocol:

- [Diagnosing Retrieval vs. Utilization](https://arxiv.org/abs/2603.02473)
  motivates a strong raw-chunk retrieval baseline before lossy write-time
  transformation.
- [Beyond Memory Leaderboards](https://arxiv.org/abs/2607.16848) motivates
  equal retrieval budgets and full disclosure of ingestion and judge protocol.
- [Ground Truth First](https://arxiv.org/abs/2607.21962) and
  [Convomem](https://arxiv.org/abs/2511.10523) motivate adaptive routing because
  full or recent context can outperform memory systems at shorter horizons.

## Research interpretation

The cited 2025–2026 publications are recent preprints and should be treated as
testable design hypotheses. RAI should reproduce the relevant findings on its
own workloads. In particular, reported adapter rankings can be distorted by
retrieval volume, self-grading, excluded ingestion cost or benchmark label
quality. The roadmap therefore adopts their failure modes and experimental
controls before adopting their architectural complexity.
