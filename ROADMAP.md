# Rich AI roadmap

This document is the source of truth for the evolution of **Rich AI — a
Linux-first local and edge embodiment runtime**. RAI connects operating systems,
devices, sensors and policy-controlled actions to GCAS/GAIA and other replaceable
cognitive backends. It is not a second cognitive kernel or another monolithic
chat orchestrator.

## Product boundary

RAI owns:

- Linux-first local and edge perception and normalized observations,
- device identity, status, discovery and capability advertisement,
- durable local observation, action-result and audit state,
- capability discovery and invocation,
- policy, authorization, sandboxing and audit,
- context packaging and privacy boundaries,
- deterministic safety reflexes and routing between bounded local processing and
  external cognitive backends,
- local MCP/HTTP/Unix-socket interfaces and authenticated network transports for
  remote device agents.

RAI does not own a universal reasoning loop, Global Workspace, Cognitive
Control, goal verification or open-ended deliberation. GAIA is the reference
GCAS cognitive runtime and owns those semantics. RAI may perform bounded,
schema-constrained inference and deterministic local safety reactions, but it
must not silently become a second cognitive process controller.

GCAS defines the portable cognitive and provenance contracts. RAI provides their
Linux and edge embodiment: it produces observations, mediates actions and
returns typed results. Codex, Claude, Gemini, Antigravity and future harnesses
remain replaceable `AgentBackend` implementations. A remote Android, VR or
wearable client is a thin `DeviceAgent`, not necessarily a full RAI installation.

## Deployment boundary

- A full RAI runtime runs on a Linux workstation, server or capable edge node
  such as a Raspberry Pi.
- Local collectors, processors and actuators communicate with RAI over in-process
  ports or Unix domain sockets.
- Separate physical devices communicate over authenticated network transports;
  Unix domain sockets never cross a host boundary.
- Constrained or platform-managed devices use a thin agent SDK that implements
  the same wire contracts while respecting platform lifecycle and permission
  models.
- Raw media remains local by default. Cognitive backends receive normalized
  observations or explicit, policy-approved `MediaReference` values.

## Engineering invariants

1. Importing `rai` has no process-wide side effects.
2. Runtime state is not stored only in an LLM conversation.
3. Privileged execution is fail-closed and policy mediated.
4. One typed capability registry feeds MCP, the daemon and local backends.
5. Every derived memory can retain provenance to source observations.
6. Local models receive bounded, schema-constrained tasks.
7. External backends receive a minimal `ContextPackage`, not ambient access.
8. One cognitive process has one explicit controller; RAI never competes with
   GAIA's Workspace or Control.
9. Network delivery is replayable and idempotent; reconnects never silently lose
   or duplicate an accepted action.
10. A phase is complete only when its acceptance tests pass.

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

## Stage 1 — embodiment kernel and portable contracts

Deliver one internal invocation path shared by all interfaces.

- Define immutable `DeviceDescriptor`, `Observation`, `MediaReference`,
  `Episode`, `Claim`, `Task`, `ContextPackage`, `CapabilityRequest`,
  `ActionResult` and `ActionFailure` records.
- Introduce `Capability`, `Collector`, `Actuator`, `LocalProcessor`,
  `DeviceAgent` and `AgentBackend` protocols.
- Publish versioned, language-neutral JSON schemas and conformance fixtures for
  observations, capabilities, actions and results. Keep the GCAS/GAIA mapping
  explicit and test it in both projects.
- Replace `TOOL_REGISTRY` and MCP's duplicate dispatcher with a typed
  `CapabilityRegistry`.
- Introduce an explicit `PolicyEngine` and structured risk decisions.
- Replace module-level services with an application container/app factory.
- Add a synthetic collector and actuator as deterministic reference adapters.
- Split the CLI into commands, client transport, rendering and compatibility
  backend modules.
- Quarantine Antigravity behind `AntigravityBackend`; do not use SDK private
  fields in the runtime kernel.

Acceptance slice: one synthetic observation is validated and stored, and one
capability is invoked through CLI, REST and MCP with the same schema validation,
policy decision and result envelope.

## Stage 2 — first RAI ↔ GAIA embodiment slice

Prove the cross-project ownership boundary before expanding either side.

- Add append-only observation and action-result storage with replay cursors,
  acknowledgements and event-ID deduplication.
- Add authenticated observation ingest and subscription APIs over HTTP and Unix
  sockets; use TCP-based transports between physical devices.
- Implement a GAIA adapter that converts a validated RAI `Observation` into an
  unverified Observation Cognitive Object with complete provenance.
- Accept bounded GAIA `Action` requests through the same capability registry and
  return typed `ActionResult` or `ActionFailure` records.
- Keep the generic sensor protocol separate from the NCSI/J-lens neural-state
  protocol; neither contract is an alias for the other.
- Add one minimal local actuator, preferably Piper TTS or a deterministic test
  substitute, to prove the return path.
- Preserve exactly-once cognitive terminal semantics while using idempotent,
  at-least-once transport delivery.

Acceptance slice: synthetic `person_present` observation → RAI store → GAIA
Observation CO → policy-approved `speech.synthesize` action → durable result.

## Stage 3 — local perception and deterministic episodes

Start with metadata rather than continuous screenshots or key logging.

- Add production collector lifecycle and connect every collector to the Stage 2
  observation store.
- Implement active application/window observation for GNOME.
- Add bounded AT-SPI semantic events, foreground process and project/Git
  context.
- Add opt-in camera, microphone/VAD and device-status collectors behind the same
  contracts; do not persist continuous raw media by default.
- Redact password/secret fields before persistence.
- Implement retention, pause, per-application and per-device exclusion, consent
  and deletion controls.
- Build episodes deterministically from time, idle, application and project
  boundaries. Episodes segment evidence; they are not a second reasoning loop.
- Preserve provenance from episode fields to source observations.

Acceptance slice: Firefox → terminal → editor activity and one opt-in sensory
event become reviewable local episodes without sending raw media over the
network.

## Stage 4 — bounded edge processing and routing

Local processing classifies or transforms observations. It must not acquire
Workspace, Control, goal-verification or open-ended planning responsibilities.

- Select one supported local execution path first: llama.cpp or Ollama.
- Remove/freeze non-functional IREE and ONNX entries until they have owners and
  conformance tests.
- Add separately startable vision, STT and TTS processor adapters. Treat Hailo,
  Whisper and Piper as optional device capabilities with explicit resource and
  failure reporting, not imports required by the core daemon.
- [/] **Optional NCSI/J-lens neural sidecar** — A separately startable
  Transformers process that owns model/accelerator lifecycle, residual-stream
  hooks, versioned J-lens artifacts, compact neural observations, cancellation,
  and typed failures. Keep raw tensors inside the sidecar and keep PyTorch/J-lens
  dependencies out of the default RAI installation. RAI provides the neural
  runtime; GAIA retains Workspace, Control, epistemic, and verification
  semantics. The read-only server, Transformers engine, artifact fitter/loader,
  shared fixtures and unit tests exist; live acceptance, terminal request-ID
  replay protection, unique lens selection and GAIA transport remain open.
  Detailed milestone status is tracked only in the
  [canonical cross-project integration plan](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/ncsi-jlens-integration.md).
- Add schema-constrained classification, extraction, summarization, salience
  and risk tasks.
- Run blocking inference outside the main event loop.
- Record latency, tokens, memory/accelerator use, confidence, concurrency and
  schema failures.
- Build a small versioned evaluation corpus before model-specific tuning.
- Route using deterministic rules first, local classification second and policy
  last.
- Support `LOCAL`, `ASK`, `ESCALATE` and `DENY` decisions.
- Include privacy class, capability set, cost/latency budget, confidence and
  verification requirements in every routing decision.
- Build minimal task-specific context packages from durable state.
- Verify important results against tool output or observations before state is
  updated.

Acceptance slice: a bounded local processor converts an episode or utterance
into validated structured output; deterministic policy keeps an allowed task
local or escalates it to GAIA, and processor failure cannot terminate the core
runtime or masquerade as a successful observation.

## Stage 5 — multi-device operation and external backends

- Define device registration, heartbeat, status, capability advertisement and
  revocation contracts.
- Add bounded offline spooling, reconnect/replay, backpressure and key rotation.
- Provide a Linux reference `DeviceAgent`; define an SDK boundary for later
  Android, VR and wearable clients without requiring the full Python daemon on
  those platforms.
- Add authenticated network deployment profiles and document mTLS or an
  equivalent device-identity boundary before LAN exposure.
- Stabilize `AgentBackend` conformance tests.
- Add one end-to-end harness backend before multiplying integrations.
- Harden the Stage 2 GAIA bridge as the reference cognitive-runtime integration.
  Neural-state streaming continues to use the separately versioned NCSI sidecar
  contract rather than coupling GAIA to RAI implementation modules.
- Keep backend conversation identifiers as adapter-owned metadata.
- Add cancellation, usage accounting, retry and evidence return contracts.
- Retain Antigravity only if it passes the same contracts without process-wide
  patching.

Acceptance slice: a Raspberry Pi or simulated remote Linux agent disconnects,
buffers bounded observations, reconnects without duplication, and executes one
GAIA-requested capability with a verified action-result chain.

## Stage 6 — user experience and ecosystem

- Activity/history review and deletion UI.
- Device, permission, connectivity and local-model status UI.
- Native GNOME approval and status surface.
- Emacs client.
- COSMIC parity based on actual platform APIs.
- Guile dashboard and experimental web UI.
- TTS, audio-duplex and accessibility improvements beyond the minimal Stage 2
  actuator.

These features remain intentionally behind the runtime kernel and the first
vertical slice.

## Legacy chat retirement gates

Legacy removal is incremental and follows working replacement slices rather
than a big-bang rewrite.

1. The application container and typed registry become the only capability
   invocation path.
2. The Stage 2 Observation → GAIA → Action → Result slice passes cross-project
   acceptance tests.
3. CLI, REST and MCP use the same runtime services and policy decisions.
4. `ChatService`, agent chains and provider-specific configuration move behind
   an explicitly named compatibility package and stop defining durable state.
5. Antigravity becomes an optional `AgentBackend`, disabled in the default
   installation.
6. Legacy chain endpoints receive a documented deprecation and migration path.
7. Compatibility code that does not pass the common backend and security
   contracts is removed before 1.0.

## Technical debt carried into Stage 1

- `cli.py` and `config_manager.py` are oversized and mix policy, I/O and UI.
- MCP and `TOOL_REGISTRY` describe overlapping tool catalogs.
- Antigravity compatibility still owns the current chat execution path.
- Conversation history still has more implementation weight than normalized
  observation and action-result state.
- local inference protocols are disconnected from the daemon and lack tests.
- IREE is a stub and the ONNX factory references an absent implementation.
- Device identity, reconnect/replay and backpressure contracts do not yet exist.
- full style linting contains legacy violations; critical lint is blocking now.
- existing GitLab issues #8 and #9 remain relevant to lazy loading and blocking
  local inference. Issues #2 and #6 require reproduction against the new
  contracts before implementation.

## Definition of done for roadmap work

A checkbox requires code or documentation in the current tree, relevant tests,
an explicit failure mode, and updated user-facing documentation. A prototype,
an unused protocol or a mocked unit test alone does not complete a subsystem.
