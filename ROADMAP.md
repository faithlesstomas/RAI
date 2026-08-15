# Rich AI roadmap

This document is the source of truth for the evolution of **Rich AI — A
Local-First Agent Runtime for Linux**.

## Product boundary

RAI owns:

- Linux perception and normalized observations,
- durable local state and memory,
- capability discovery and invocation,
- policy, authorization, sandboxing and audit,
- context packaging and privacy boundaries,
- routing between cheap local cognition and external agent backends,
- local MCP/HTTP/Unix-socket interfaces.

RAI does not own a universal reasoning loop. GAIA, Codex, Claude, Gemini,
Antigravity and future harnesses are consumers or implementations of an
`AgentBackend` contract. GCAS may define shared cognitive schemas; RAI provides
their Linux embodiment.

## Engineering invariants

1. Importing `rai` has no process-wide side effects.
2. Runtime state is not stored only in an LLM conversation.
3. Privileged execution is fail-closed and policy mediated.
4. One typed capability registry feeds MCP, the daemon and local backends.
5. Every derived memory can retain provenance to source observations.
6. Local models receive bounded, schema-constrained tasks.
7. External backends receive a minimal `ContextPackage`, not ambient access.
8. A phase is complete only when its acceptance tests pass.

## Stage 0 — trustworthy baseline

Purpose: make the existing daemon safe enough to evolve and deterministic
enough to test. This stage supersedes the previous claims that desktop and
security layers were already complete.

- [x] Reframe active documentation around the local-first runtime vision.
- [x] Make package imports side-effect free; remove global SDK/Popen patches.
- [x] Adopt XDG configuration, data, cache and runtime paths with test overrides.
- [x] Refuse command execution when Bubblewrap/Guix isolation is unavailable.
- [x] Probe Bubblewrap operability rather than only checking for its binary.
- [x] Protect control APIs with a per-user token and constant-time comparison.
- [x] Disable cross-origin browser access by default.
- [x] Isolate tests from the user's real config, history and cache.
- [x] Replace thread-dependent SQLite access with deterministic backend-neutral
  history persistence.
- [x] Add hard pytest timeouts and blocking critical lint jobs in CI.
- [x] Move heavyweight inference stacks to optional dependency groups.
- [x] Ignore local credentials, model weights and local harness state.

Operator security notice: rotate any real credentials that were stored as loose
workspace files. This cannot be automated or proven by repository code; ignored
files only prevent accidental future additions and do not undo prior disclosure.

Acceptance gate:

```text
full pytest completes within its timeout
critical ruff and pylint -E pass
no silent unsandboxed fallback exists
unauthenticated /api and WebSocket control requests are rejected
documentation does not describe Antigravity as the architectural core
```

## Stage 1 — runtime kernel and contracts

Deliver one internal invocation path shared by all interfaces.

- Define immutable `Observation`, `Claim`, `Task`, `ContextPackage`,
  `ToolRequest`, `ToolResult` and `Episode` records.
- Introduce `Capability`, `Collector`, `LocalCognition` and `AgentBackend`
  protocols.
- Replace `TOOL_REGISTRY` and MCP's duplicate dispatcher with a typed
  `CapabilityRegistry`.
- Introduce an explicit `PolicyEngine` and structured risk decisions.
- Replace module-level services with an application container/app factory.
- Split the CLI into commands, client transport, rendering and compatibility
  backend modules.
- Quarantine Antigravity behind `AntigravityBackend`; do not use SDK private
  fields in the runtime kernel.

Acceptance slice: one capability is invoked through CLI, REST and MCP with the
same validation, policy decision and result envelope.

## Stage 2 — Linux perception and episodes

Start with metadata rather than continuous screenshots or key logging.

- Add collector lifecycle and append-only observation storage.
- Implement active application/window observation for GNOME.
- Add bounded AT-SPI semantic events, foreground process and project/Git
  context.
- Redact password/secret fields before persistence.
- Implement retention, pause, per-application exclusion and deletion controls.
- Build episodes deterministically from time, idle, application and project
  boundaries.
- Preserve provenance from episode fields to source observations.

Acceptance slice: Firefox → terminal → editor activity becomes a reviewable
local episode without sending data over the network.

## Stage 3 — bounded local cognition

- Select one supported local execution path first: llama.cpp or Ollama.
- Remove/freeze non-functional IREE and ONNX entries until they have owners and
  conformance tests.
- Add schema-constrained classification, extraction, summarization, salience
  and risk tasks.
- Run blocking inference outside the main event loop.
- Record latency, tokens, memory use, confidence and schema failures.
- Build a small versioned evaluation corpus before model-specific tuning.

Acceptance slice: a 2–4B-class local model converts an episode into validated
structured output and fails safely on invalid output.

## Stage 4 — hybrid router

- Route using deterministic rules first, local classification second and policy
  last.
- Support `LOCAL`, `ASK`, `ESCALATE` and `DENY` decisions.
- Include privacy class, capability set, cost/latency budget, confidence and
  verification requirements in every routing decision.
- Build minimal task-specific context packages from durable state.
- Verify important results against tool output or observations before state is
  updated.

Acceptance slice: “what was I working on?” stays local, while a complex coding
task is packaged and escalated with an explicit capability boundary.

## Stage 5 — external agent backends and GAIA

- Stabilize `AgentBackend` conformance tests.
- Add one end-to-end harness backend before multiplying integrations.
- Integrate GAIA either as a backend or as a cognitive runtime consuming RAI's
  MCP/capability API.
- Keep backend conversation identifiers as adapter-owned metadata.
- Add cancellation, usage accounting, retry and evidence return contracts.
- Retain Antigravity only if it passes the same contracts without process-wide
  patching.

## Stage 6 — user experience and ecosystem

- Activity/history review and deletion UI.
- Native GNOME approval and status surface.
- Emacs client.
- COSMIC parity based on actual platform APIs.
- Guile dashboard and experimental web UI.
- TTS and accessibility improvements.

These features remain intentionally behind the runtime kernel and the first
vertical slice.

## Technical debt carried into Stage 1

- `cli.py` and `config_manager.py` are oversized and mix policy, I/O and UI.
- MCP and `TOOL_REGISTRY` describe overlapping tool catalogs.
- Antigravity compatibility still owns the current chat execution path.
- local inference protocols are disconnected from the daemon and lack tests.
- IREE is a stub and the ONNX factory references an absent implementation.
- full style linting contains legacy violations; critical lint is blocking now.
- existing GitLab issues #8 and #9 remain relevant to lazy loading and blocking
  local inference. Issues #2 and #6 require reproduction against the new
  contracts before implementation.

## Definition of done for roadmap work

A checkbox requires code or documentation in the current tree, relevant tests,
an explicit failure mode, and updated user-facing documentation. A prototype,
an unused protocol or a mocked unit test alone does not complete a subsystem.
