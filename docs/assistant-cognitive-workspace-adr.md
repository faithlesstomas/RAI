# ADR 0002: Modular Interaction Controllers and Architectural Placement of Global Workspace (GWT) Deliberation

- **Status:** Accepted
- **Date:** 2026-10-14
- **Context:** Issue #52, Roadmap milestone ROADMAP.md:1520–1528, Issue #50 / MR !36, Issue #34, Issue #35
- **Deciders:** Rich AI Core Team

---

## 1. Context and Problem Statement

Rich AI (RAI) provides a local-first, secure integration layer between AI assistants and the Linux desktop. User requests to the assistant span a wide spectrum of cognitive complexity:
1. **Direct conversational queries and factual Q&A** (e.g. system status, greetings, questions answered from durable memory).
2. **Bounded multi-step actions and tool workflows** (e.g. discovering installed desktop apps, web research, file inspection).
3. **Complex deliberative and multi-perspective problem solving** (e.g. cross-checking conflicting claims, synthesizing hypotheses across disparate memory clusters, or detecting latent semantic drift).

Historically, `AssistantService` implemented an implicit, monolithic turn loop inside `_accept_turn_locked`. While this sufficed for early M1–M6 milestones, it created architectural friction:
- Single-turn direct generation and multi-step agent loops were conflated in the same code path.
- Experimental deliberative architectures—specifically Global Workspace Theory (GWT) blackboard models, specialist competition, and latent space neural probing (`rai.latent.v1` / `inference-jlens`)—risked polluting the core service with experimental state machines and hidden background token consumption.
- As mandated in `ROADMAP.md:1525`:
  > *"Require an ADR after the observation-only experiments to decide whether a useful GW-like process belongs in an optional RAI research module, an external cognitive runtime or nowhere. It must not emerge implicitly inside `AssistantService`."*

This Architecture Decision Record formalizes the modular controller design and evaluates the permanent architectural placement of Global Workspace deliberation within the RAI ecosystem.

---

## 2. Decision Drivers

1. **Fail-Closed Security & Zero Ambient Authority (Non-Negotiable):**
   No cognitive engine, blackboard, or specialist must possess direct capability execution authority or direct raw database access. All capability invocations must route through `CapabilityService` and `PolicyEngine` with deterministic policy checks, risk classification, and audit logging.
2. **Resource & Budget Predictability:**
   Local Linux desktop AI operates under constrained compute and memory budgets (e.g. 0.8B–4B parameter models on CPU or integrated GPUs). Deliberation loops that multiply inference invocations ($N_{\text{specialists}} \times N_{\text{cycles}}$) must never run unbounded or silently inflate background inference costs (`background_remote_tokens = 0`).
3. **Clean Protocol Separation:**
   The core `AssistantService` pipeline must remain predictable, deterministic, and easily auditable. Execution strategies must conform to a clear, replaceable interface.
4. **Epistemic Integrity & Ephemeral Cognition:**
   Working memory buffers, blackboard states, and attention weights must remain strictly ephemeral. Only validated, policy-admitted `MemoryRecord`s and `ActionResult`s may persist in durable SQLite graph memory.
5. **Zero Overhead for Optional Extras:**
   Deliberative specialists and neural sidecar probes (`JLensProbe`) must impose zero runtime or import overhead when their optional dependencies (`inference-jlens`) are disabled or absent.

---

## 3. Considered Options

### Option 1: Integrate GWT Deliberation Directly into Core `AssistantService`
- **Description:** Replace or expand `AssistantService._accept_turn_locked` with an internal blackboard loop that runs specialists on every user turn.
- **Drawbacks:**
  - Violates the principle of least complexity: simple turns (greetings, volume changes) would incur multi-specialist deliberation overhead.
  - Substantially increases token consumption (3x–5x) and latency on constrained desktop hardware.
  - Tightly couples the core assistant service to an experimental cognitive research architecture.
- **Outcome:** **Rejected.**

### Option 2: Delegate All Deliberation to External Runtimes Only
- **Description:** Keep RAI strictly as an execution harness (capability server + SQLite memory store); all multi-step and deliberative processing must be driven by external agent runtimes (such as GAIA, AutoGen, or LangGraph).
- **Drawbacks:**
  - Prevents RAI from providing a standalone, out-of-the-box local assistant capable of autonomous multi-step research and self-grounding.
  - Forces users to configure external network services or external frameworks for local desktop workflows.
- **Outcome:** **Rejected as sole solution.**

### Option 3: Modular `InteractionController` Protocol with Optional Research Module (Selected)
- **Description:**
  1. Extract an explicit `@runtime_checkable class InteractionController(Protocol)` in `rai.assistant.ports`.
  2. Implement `DirectController` as the default, lightweight single-turn execution strategy.
  3. Implement `BoundedStepController` for universal, finite-state-machine-supervised multi-step workflows driven by `SKILLS.md`.
  4. Implement `WorkspaceEngine` in an **isolated, optional research package** (`rai.assistant.cognitive`), never implicit in `AssistantService`.
  5. Support pluggable external cognitive runtimes (such as GCAS/GAIA adapters) conforming to the same protocol.
- **Outcome:** **Accepted.**

---

## 4. Decision Outcome

We adopt **Option 3**. The architecture is structured as follows:

```
+-------------------------------------------------------------------------+
|                           AssistantService                              |
|   (Turn validation, Session lock, Durable Memory, Audit commit, API)    |
+-------------------------------------------------------------------------+
                                    |
                                    v
                  <<InteractionController Protocol>>
                    process_turn(turn, context, budget, token)
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
        v                           v                           v
+------------------+     +----------------------+     +-------------------+
| DirectController |     | BoundedStepController|     |  WorkspaceEngine  |
|  (Default Core)  |     |  (Skills & Workflows)|     | (Optional Research|
|                  |     |                      |     |  Cognitive Module)|
| - 1-turn direct  |     | - Generic FSM loop   |     | - GWT blackboard  |
| - Fast, low cost |     | - PLANNING->COMPLETE |     | - Competing specs |
| - Abstention     |     | - SKILLS.md driven   |     | - Latent J-lens   |
+------------------+     +----------------------+     +-------------------+
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    v
                     +-----------------------------+
                     |      CapabilityService      |
                     |      & PolicyEngine         |
                     +-----------------------------+
```

### 4.1 Role and Placement of Each Controller

1. **`DirectController` (`rai.assistant.controllers.direct`):**
   - **Status:** Core default.
   - **Responsibility:** Single-turn conversational exchanges, direct answers, single-turn tool calls with immediate synthesis, and deterministic memory abstention.
   - **Cost profile:** Minimal latency (sub-millisecond test overhead, single model generation pass) and minimal token consumption.

2. **`BoundedStepController` (`rai.actions.workflow`):**
   - **Status:** Core agentic loop.
   - **Responsibility:** Multi-step autonomous research and desktop tasks. Governed by a universal finite state machine:
     `PLANNING` $\rightarrow$ `EVALUATING` $\rightarrow$ `EXECUTING` $\rightarrow$ `OBSERVING` $\rightarrow$ `SYNTHESIZING` $\rightarrow$ `COMPLETE`.
   - **Supervision:** Hard bounds on `max_steps`, `max_tool_calls`, token budgets, and wall-clock timeouts. Integrates with the `SkillRegistry` (`SKILLS.md`).

3. **`WorkspaceEngine` (`rai.assistant.cognitive.workspace`):**
   - **Status:** **Optional research module** (configured via `assistant.interaction_controller: "workspace"`).
   - **Responsibility:** Multi-perspective cognitive deliberation based on Global Workspace Theory (GWT).
   - **Architecture:**
     - **Specialists:** `RetrieverSpecialist`, `ActionReasonerSpecialist`, `AnswerDraftSpecialist`, `GroundingEvaluationSpecialist`.
     - **Salience Arbiter:** Evaluates contributions based on specialist confidence, novelty, and critique penalties.
     - **Broadcast Buffer:** Winning contribution is broadcast globally to all specialists for the next cycle.
     - **Latent Probe (`JLensProbe`):** Monitors semantic drift in hidden representations (`rai.latent.v1`) during deliberation cycles prior to token emission.
   - **Placement Rule:** Must remain strictly isolated within `rai.assistant.cognitive`. It must **never** be enabled as the unconfigured default or instantiate background threads outside an active user turn.

---

## 5. Comparative Empirical Evaluation

The comparative benchmark (`rai.assistant.cognitive.evaluation`, saved at `docs/evaluation/assistant-interaction-controllers-benchmark.json`) evaluated all three controllers under identical token budgets (4096 tokens) and timeout limits (30s) across four representative desktop scenarios:
1. `action_system_discovery`: Discovering installed system desktop applications.
2. `multi_turn_context_retention`: Recalling specific project metadata across turns.
3. `hallucination_abstention_vault`: Enforcing grounding and abstaining on unverified claims.
4. `conversational_fidelity`: General conversational assistant capabilities.

### Summary Metrics Table

| Controller | Role | Task Completion Rate | Token Overhead (Avg) | Latency Overhead (ms) | Grounding & Abstention | Architectural Complexity |
|---|---|---|---|---|---|---|
| **`DirectController`** | Core Default | **100%** (Single-turn / Factual) | **55.0 tokens** (1.0x baseline) | **0.04 ms** (1.0x baseline) | Deterministic Fail-Closed | Low (Single pass) |
| **`BoundedStepController`** | Core Agentic | **100%** (Multi-step Action) | **68.8 tokens** (1.25x baseline) | **0.23 ms** (5.8x baseline) | Verified Action Results | Medium (FSM loop) |
| **`WorkspaceEngine`** | Optional Research | **100%** (Specialist Deliberation) | **178.8 tokens** (3.25x baseline) | **0.36 ms** (9.0x baseline) | Multi-specialist Critique & Latent Probing | High (Blackboard + Arbiter) |

### Key Empirical Takeaways:
- **`DirectController`** is optimal for everyday Linux desktop assistance, chat, and single-turn commands. It delivers answers with lowest latency and minimal token consumption.
- **`BoundedStepController`** is the right fit for multi-step agentic workflows, capability discovery, and file/web inspection tasks under explicit step and tool limits.
- **`WorkspaceEngine`** exhibits higher token overhead and latency due to parallel specialist competition per cycle. However, it uniquely enables multi-perspective critique, hallucination filtering, and latent semantic drift monitoring. Consequently, **it belongs in an optional research module rather than core default runtime**.

---

## 6. Strict Compliance Invariants

1. **Zero Capability Execution Authority:**
   `WorkspaceEngine`, specialists, and controllers have zero ambient capability permissions. They can only return `ActionProposal`s or invoke `CapabilityService.invoke(...)` subject to `PolicyEngine` approval.
2. **Ephemeral Working Memory:**
   Contributions, salience scores, and workspace history exist only during the lifetime of `process_turn`. They are discarded upon turn completion and are never written to `SQLiteMemoryGraphStore` directly.
3. **Fail-Closed Abstention:**
   If evidence is absent and grounding fails, controllers must return the canonical abstention response rather than confabulating.
4. **Zero Overhead for Optional Extras:**
   `JLensProbe` checks for neural inspector availability; if absent or disabled, probe evaluation is a zero-cost no-op.
5. **Configurability:**
   Users and administrators can explicitly select the controller via `assistant.interaction_controller`: `"direct"`, `"bounded_step"`, or `"workspace"`.

---

## 7. Related Documents and Standards

- [docs/assistant-architecture.md](assistant-architecture.md) — Core Rich Assistant architecture.
- [docs/assistant-memory-gcas-crosscheck.md](assistant-memory-gcas-crosscheck.md) — Cross-check with GCAS 0.3 specification.
- `ROADMAP.md` (lines 1520–1528) — Cognitive experiments milestone.
- [docs/evaluation/assistant-interaction-controllers-benchmark.json](evaluation/assistant-interaction-controllers-benchmark.json) — Frozen comparative benchmark data.
