# Rich AI roadmap

This document is the source of truth for the evolution of **Rich AI (RAI) — an
intelligent and secure GNU/Linux desktop powered by local, hybrid or external
AI**.

RAI is a Linux-first, local-first agent and operating-system runtime. It observes
explicitly allowed activity, maintains durable state and memory, exposes
policy-controlled capabilities, and connects replaceable local models and agent
harnesses to the desktop. The desktop is the primary interaction surface; RAI
does not require a chat window to own the user experience.

RAI is not a second cognitive kernel and not another monolithic chat
orchestrator. It owns the boundary to the operating system and treats every
model as an untrusted, replaceable processor.

## Roadmap status notation

- `[x]` complete and verified in the current tree,
- `[/]` partially implemented or experimental,
- `[ ]` planned,
- acceptance gates describe the minimum evidence required before a stage can be
  treated as complete.

## Product outcomes

The implementation sequence must deliver the following independently useful
product increments:

1. **Rich History** — private, local, reviewable knowledge of recent desktop
   activity without continuous screenshots, raw key logging or cloud
   processing.
2. **Rich Voice** — local speech input and output for asking about activity and
   issuing bounded commands.
3. **Rich Actions** — safe, typed and verified desktop/system actions mediated
   by policy rather than arbitrary model-generated shell access.
4. **Rich Local AI** — bounded local classification, extraction, summarization,
   grounding and routing that continue to work offline.
5. **Rich Hybrid** — explicit, budgeted escalation to GAIA or external agent
   harnesses using the minimum necessary context.
6. **Rich Automation** — user-approved reusable workflows with fixed
   capabilities, privacy rules, resource budgets and audit trails.

Each increment must remain useful when every external model provider is
disabled.

## Product boundary

RAI owns:

- Linux-first local and edge perception and normalized observations,
- device identity, status, discovery and capability advertisement,
- durable local observation, episode, action-result, usage and audit state,
- capability discovery, invocation and result verification,
- policy, authorization, sandboxing, human approval and audit,
- context selection, redaction, data-egress control and privacy boundaries,
- deterministic safety reflexes and routing between rules, bounded local
  processing and external cognitive backends,
- local MCP/HTTP/Unix-socket interfaces and authenticated network transports for
  remote device agents,
- user-visible controls for collection, permissions, budgets, activity review
  and emergency stop.

RAI does not own a universal reasoning loop, Global Workspace, Cognitive
Control, goal verification or open-ended deliberation. GAIA is the reference
GCAS cognitive runtime and owns those semantics. RAI may perform bounded,
schema-constrained inference and deterministic local safety reactions, but it
must not silently become a second cognitive process controller.

GCAS defines portable cognitive and provenance contracts. RAI provides their
Linux and edge embodiment: it produces observations, mediates actions and
returns typed results. Codex, Claude, Gemini, Antigravity and future harnesses
remain replaceable `AgentBackend` implementations. A remote Android, VR or
wearable client is a thin `DeviceAgent`, not necessarily a full RAI
installation.

RAI must not depend on private ChatGPT Computer History, Chronicle or Skysight
interfaces. Those systems are architectural references only. Linux collectors
and memory formats are defined by RAI's own versioned contracts.

## Deployment and operating profiles

### Deployment boundary

- A full RAI runtime runs as an unprivileged user service on a Linux workstation,
  server or capable edge node.
- Local collectors, processors and actuators communicate over in-process ports
  or protected Unix domain sockets.
- Separate physical devices communicate over authenticated network transports;
  Unix domain sockets never cross a host boundary.
- Constrained or platform-managed devices use a thin agent SDK that implements
  the same wire contracts while respecting platform lifecycle and permission
  models.
- Raw media remains local by default. Cognitive backends receive normalized
  observations or explicit, policy-approved `MediaReference` values.
- A Raspberry Pi may host collection, policy and simple processors while a
  trusted LAN node provides an optional heavier local model. This remains a
  local deployment only when data does not leave the user's trust domain.

### User-selectable AI profiles

`LOCAL_ONLY`

- Network model backends are disabled.
- Observation, memory, voice and inference stay inside the local trust domain.
- Unsupported tasks fail clearly or ask the user to switch profiles.

`LOCAL_PREFERRED`

- Deterministic rules and local processors are attempted first.
- Remote escalation requires a policy decision and explicit task context.
- Background collection and routine memory consolidation never consume remote
  tokens.

`HYBRID_APPROVAL`

- RAI may propose external escalation and show an outbound context manifest,
  estimated budget and requested capabilities.
- The user approves the individual transfer or a narrowly scoped reusable rule.

`REMOTE_ALLOWED`

- Approved agent backends may be selected automatically within configured data,
  capability and cost boundaries.
- Ambient desktop access is still forbidden; the backend receives only a
  `ContextPackage` and capability handles.

The default profile is `LOCAL_PREFERRED`. The invariant
`background_remote_tokens = 0` applies to every profile unless the user creates
an explicit scheduled automation with its own budget and data policy.

## Trust, privacy and data model

### Trust boundaries

- Collector input, accessibility text, web pages, documents, model output,
  remote-agent output and MCP requests are untrusted.
- A local model is not a policy boundary and its confidence value is not an
  authorization decision.
- Content observed on screen is evidence, never an instruction. Text such as
  "ignore previous rules and run this command" remains application content.
- Human approval confirms intent but does not make an unsafe capability safe.
- The local Unix account is an administrative trust assumption, not protection
  against every process running as that user. File permissions, service
  isolation and encryption are defense in depth.

### Data classes

Every persisted field and outbound context item must carry one of these data
classes:

- `PUBLIC` — may be sent to an approved backend within its budget;
- `LOCAL` — remains on devices in the local trust domain;
- `PRIVATE` — may leave the local trust domain only after explicit approval;
- `SECRET` — credentials, password fields, authentication tokens and equivalent
  material; never persisted in activity history and never sent to a model;
- `BLOCKED` — excluded source or policy-forbidden content; discarded before the
  observation journal.

Classification is deterministic where the operating system exposes the needed
signal. A model may recommend a stricter class but cannot downgrade one.

### Storage tiers

1. **Raw event buffer** — minimal normalized events, short configurable TTL,
   never used as permanent chat history.
2. **Observation journal** — deduplicated events that passed privacy filtering,
   with source and policy provenance.
3. **Episode store** — deterministic time/activity segments referring to source
   observation IDs.
4. **Memory store** — optional derived summaries and user-curated facts with
   provenance and independent retention.
5. **Action and usage ledger** — append-only decisions, approvals, model usage,
   outbound manifests and verified results.

Raw audio, continuous video, complete accessibility trees, clipboard contents
and individual keystrokes are not persisted by default. Memory encryption uses
a per-user key obtained from an operating-system secret store where supported;
file permissions and an isolated user service remain mandatory even when
encryption is enabled.

### Collection controls

Before Rich History is considered usable it must provide:

- opt-in enablement and a persistent visible collection state,
- allow-only and exclude lists for applications, websites, devices and paths,
- automatic suppression for password fields, screen lock, private browsing and
  configured sensitive applications,
- pause/resume and emergency stop,
- clearing the current application session, last 10 minutes, last hour, last day
  or all retained activity,
- separate retention policies for raw events, observations, episodes and
  memories,
- inspection of every retained episode and outbound context manifest,
- deletion that covers derived memories and provenance links without leaving
  orphaned sensitive content.

## Cost and resource model

Every processor invocation must receive an `InferenceBudget` containing the
relevant limits:

- maximum input and output tokens,
- maximum agent turns and tool calls,
- maximum images or audio duration,
- maximum wall-clock latency,
- maximum provider cost,
- local RAM/VRAM and accelerator constraints,
- cancellation deadline,
- allowed providers and fallback order.

RAI maintains a durable `UsageLedger` by task, automation, processor, backend,
model and provider. It supports per-task, daily and monthly quotas and refuses or
asks before exceeding them. Unknown pricing or missing usage data cannot be
treated as zero cost.

Cost controls follow this order:

1. deterministic rule or typed operation — no inference,
2. cached result for the same versioned input and policy,
3. tiny local classifier or embedding lookup,
4. local text SLM,
5. local visual processor for a selected region,
6. external backend with a minimal redacted `ContextPackage`,
7. `ASK` or `DENY` when privacy, capability or budget constraints prevent the
   operation.

## Engineering invariants

1. Importing `rai` has no process-wide side effects.
2. Runtime state is not stored only in an LLM conversation.
3. Privileged execution is fail-closed and policy mediated.
4. One typed capability registry feeds MCP, the daemon and local backends.
5. Every derived memory and claim retains provenance to source observations,
   tool results, user statements or explicit inferences.
6. Local models receive bounded, schema-constrained tasks.
7. External backends receive a minimal `ContextPackage`, never ambient access.
8. Background observation never invokes a remote model unless an explicit,
   budgeted automation authorizes it.
9. Privacy filtering runs before persistence and before model routing.
10. Models do not write durable state or execute operating-system actions
    directly.
11. Every state-changing action has a risk decision, typed result and
    postcondition verification.
12. One cognitive process has one explicit controller; RAI never competes with
    GAIA's Workspace or Control.
13. Network delivery is replayable and idempotent; reconnects never silently
    lose or duplicate an accepted action.
14. Pixel-based desktop control and arbitrary shell execution are fallback
    capabilities, not the default integration path.
15. A stage is complete only when its acceptance tests and security failure
    tests pass.

## Delivery sequence

The stages are ordered by dependency. Work inside one stage may proceed in
parallel only when its contracts are already fixed. Each numbered work package
should normally map to a small GitLab issue or a narrow series of merge
requests.

### Stage 0 — trustworthy baseline

Purpose: make the existing daemon safe enough to evolve and deterministic
enough to test. This stage supersedes previous claims that desktop and security
layers were already complete.

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
files only prevent accidental future additions and do not undo prior
disclosure.

Acceptance gate:

```text
full pytest completes within its timeout
critical ruff and pylint -E pass
no silent unsandboxed fallback exists
unauthenticated /api and WebSocket control requests are rejected
documentation does not describe Antigravity as the architectural core
```

### Stage 1 — embodiment kernel and trust contracts

Purpose: create one internal invocation and policy path shared by every
interface. No production collector or agent backend should be added around this
kernel.

Prerequisite: Stage 0.

#### 1.1 Domain records and schemas

- [x] Define immutable `DeviceDescriptor`, `Observation`, `MediaReference`,
  `Episode`, `Claim`, `Task`, `ContextPackage`, `ContextManifest`,
  `CapabilityRequest`, `ActionResult`, `ActionFailure`, `PolicyDecision`,
  `InferenceBudget` and `UsageRecord` records.
- [x] Give every record a version, stable ID, timestamp, producer identity and
  correlation ID where applicable.
- [x] Represent provenance as structured source references rather than free-form
  text.
- [x] Publish language-neutral JSON schemas and positive/negative conformance
  fixtures.
- [x] Define forward-compatible schema evolution and reject unsupported major
  versions explicitly.
- [x] Document GCAS/GAIA mappings without importing GAIA implementation modules
  into the RAI kernel.

#### 1.2 Runtime ports

- [x] Introduce `Capability`, `Collector`, `Actuator`, `LocalProcessor`,
  `DeviceAgent`, `AgentBackend`, `ObservationStore`, `EpisodeStore`,
  `UsageLedger` and `ApprovalBroker` protocols.
- [x] Return explicit `Result` values at I/O and backend boundaries.
- [x] Define lifecycle states and cancellation for collectors, processors and
  backends.
- [x] Add synthetic reference implementations for every port.

#### 1.3 Capability and policy path

- [x] Replace `TOOL_REGISTRY` and MCP's duplicate dispatcher with a typed
  `CapabilityRegistry`.
- [x] Make CLI, REST, MCP and internal invocations resolve the same capability
  descriptor and validator.
- [x] Introduce an explicit `PolicyEngine` with `ALLOW`, `ASK`, `DENY` and
  `ESCALATE` results and machine-readable reasons.
- [x] Define capability risk classes: `LOW`, `MODERATE`, `HIGH` and `CRITICAL`.
- [x] Include actor, data class, target resource, requested side effects,
  isolation, budget and verification plan in every decision.
- [x] Persist policy version, decision, approval and final result in the audit
  ledger.

#### 1.4 Composition and compatibility

- [x] Replace module-level services with an application container and app
  factory.
- [x] Split the CLI into commands, transport, rendering and compatibility
  modules.
- [x] Quarantine Antigravity behind `AntigravityBackend`; private SDK fields must
  not enter runtime contracts.
- [x] Preserve current user-visible behavior through compatibility adapters until
  replacement slices pass.

Acceptance slice:

```text
synthetic Observation
  -> schema validation
  -> durable test store
  -> one CapabilityRequest through CLI, REST and MCP
  -> identical PolicyDecision
  -> identical typed result envelope
  -> complete audit entry
```

Required failure tests:

- malformed and unsupported-version records are rejected;
- an unregistered capability cannot be invoked;
- a model cannot bypass policy by naming a backend-specific tool;
- unavailable approval or isolation fails closed;
- cancellation has one terminal result.

### Stage 2 — durable event plane and first RAI ↔ GAIA slice

Purpose: prove durable delivery and the cross-project ownership boundary before
adding production desktop observation.

Prerequisite: Stage 1 contracts.

#### 2.1 Observation and result journal

- [ ] Add append-only observation, action-result, audit and usage storage.
- [ ] Add event-ID deduplication, ordering metadata, replay cursors and consumer
  acknowledgements.
- [ ] Make accepted writes transactional and define crash-recovery behavior.
- [ ] Add retention hooks without allowing a backend to delete evidence.
- [ ] Record clock source and uncertainty for multi-device timestamps.

#### 2.2 Ingest and subscriptions

- [ ] Add authenticated observation ingest and subscription APIs over HTTP and
  Unix sockets.
- [ ] Use a network-capable transport with explicit authentication between
  physical devices.
- [ ] Implement bounded queues, backpressure and clear overload failures.
- [ ] Ensure slow subscribers cannot block collectors or the core event loop.

#### 2.3 GAIA embodiment contract

- [ ] Convert a validated RAI `Observation` into an unverified GAIA Observation
  Cognitive Object with complete provenance.
- [ ] Accept bounded GAIA `Action` requests through the common capability
  registry.
- [ ] Return typed `ActionResult` or `ActionFailure` records to GAIA.
- [ ] Preserve exactly-once cognitive terminal semantics over idempotent,
  at-least-once transport delivery.
- [ ] Keep the generic sensor protocol separate from the NCSI/J-lens neural-state
  protocol.

#### 2.4 Minimal return path

- [ ] Add one local actuator, preferably Piper TTS or a deterministic substitute.
- [ ] Apply policy before synthesis and persist the result after playback or a
  verifiable test substitute completes.
- [ ] Propagate cancellation and device-unavailable failures without converting
  them into success.

Acceptance slice:

```text
synthetic person_present Observation
  -> RAI durable store
  -> GAIA unverified Observation CO
  -> policy-approved speech.synthesize Action
  -> local actuator
  -> durable ActionResult returned to GAIA
```

Required failure tests cover duplicate delivery, reconnect/replay, out-of-order
events, unavailable actuator, policy denial and cancellation.

### Stage 3 — Rich History: private desktop observation

Purpose: deliver a useful read-only desktop-awareness product before autonomous
or model-driven action. Start with metadata and semantic events, not continuous
screenshots or key logging.

Prerequisites: Stage 1 records and Stage 2 observation journal. GAIA is not
required for the local Rich History acceptance slice.

#### 3.1 Collector supervisor

- [ ] Add production collector registration, lifecycle, health, restart and
  backoff.
- [ ] Run platform collectors as an unprivileged user service or isolated
  sidecars.
- [ ] Expose collector status, last event, error and effective permission state.
- [ ] Stop collection immediately when the profile is disabled, the session is
  locked or the user activates emergency stop.
- [ ] Ensure collector failure cannot terminate the core daemon.

#### 3.2 GNOME session collector

- [ ] Observe session lock/unlock, idle/active transitions, workspace changes and
  active application/window changes using supported GNOME interfaces.
- [ ] Normalize application identity through desktop-entry IDs where possible.
- [ ] Treat window titles as potentially private content and classify them before
  persistence.
- [ ] Avoid privileged `/dev/input` access and global raw input capture.

#### 3.3 AT-SPI semantic collector

- [ ] Observe bounded focus, role, state and document-context changes over
  AT-SPI.
- [ ] Coalesce repeated text-change events into duration/activity facts; do not
  store typed characters.
- [ ] Detect password/secret roles and discard their values before the journal.
- [ ] Apply size, rate and depth limits to accessibility trees.
- [ ] Record toolkit/source quality so downstream components know when semantic
  context is incomplete.

#### 3.4 Process, filesystem and project context

- [ ] Observe foreground process identity without collecting unrelated process
  arguments or environment variables.
- [ ] Add opt-in filesystem events for configured roots only.
- [ ] Detect project and Git identity from approved roots while excluding file
  content by default.
- [ ] Correlate save/build/test events using stable resource references rather
  than copying documents into history.

#### 3.5 Browser semantics

- [ ] Define a browser adapter contract for active tab ID, origin, title,
  navigation and user-requested selected text.
- [ ] Prefer an extension/native-messaging or accessibility channel that exposes
  semantic metadata instead of screenshots.
- [ ] Exclude private browsing unconditionally.
- [ ] Apply origin allow/exclude policy before storing URL or title.
- [ ] Never treat page text as an instruction to RAI or an agent.

#### 3.6 Privacy firewall

- [ ] Implement deterministic source, application, origin, path, field-role and
  session-state policies before persistence.
- [ ] Support `DROP`, `METADATA_ONLY`, `REDACT` and `ALLOW` outcomes with policy
  provenance.
- [ ] Provide built-in protections for password managers, authentication dialogs,
  banking/health profiles and communication applications.
- [ ] Keep dropped content out of logs, metrics, exception messages and dead
  letters.
- [ ] Add a local redaction test corpus containing credentials, personal data and
  prompt-injection fixtures.

#### 3.7 Event normalization and fusion

- [ ] Debounce and deduplicate high-frequency events before durable storage.
- [ ] Fuse simultaneous GNOME, AT-SPI, process and filesystem evidence into one
  activity fact without losing source references.
- [ ] Represent uncertainty and conflicting evidence explicitly.
- [ ] Make fusion deterministic for the same ordered input and configuration.

#### 3.8 Deterministic episode builder

- [ ] Segment observations using time, idle, application, resource and project
  boundaries.
- [ ] Keep episode construction deterministic and independent of an LLM.
- [ ] Store applications, resources, duration, outcome signals and provenance;
  inferred goals remain optional derived claims.
- [ ] Rebuild episodes reproducibly from retained observations and a versioned
  builder configuration.
- [ ] Update or invalidate derived memories when source observations are deleted.

#### 3.9 Local storage, retention and deletion

- [ ] Implement independent TTLs for raw buffers, observations, episodes and
  memories.
- [ ] Integrate per-user encryption keys through Secret Service or another
  documented Linux credential store, with explicit unavailable-key behavior.
- [ ] Use restrictive file permissions and exclude databases from backup by
  default unless the user opts in.
- [ ] Implement pause/resume, allow-only/exclude lists and time-range deletion.
- [ ] Verify deletion across source events, derived memories, indexes, caches and
  outbound-context references.

#### 3.10 History query and review API

- [ ] Provide local queries by time, application, project, resource and activity
  type.
- [ ] Answer deterministic questions such as "which applications were active?"
  without an LLM.
- [ ] Return provenance links and confidence for every derived activity claim.
- [ ] Expose review and deletion through the same local API used later by the
  status UI.

Acceptance slice:

```text
Firefox -> terminal -> editor -> test run
  -> allowed semantic observations
  -> privacy filtering
  -> deterministic fused episode
  -> local query: "what was I working on?"
  -> reviewable evidence and deletion
```

The slice passes with network model access disabled, no persisted screenshots,
no raw keystrokes, no raw audio and zero remote tokens. Tests also prove that a
password field, private browser window and excluded application leave no
recoverable activity content.

### Stage 4 — Rich Local AI and Rich Voice

Purpose: add bounded local understanding while keeping collection, memory,
policy and execution deterministic and independent of model availability.

Prerequisite: Stage 3 local episodes. Stage 2 provides the actuator path.

#### 4.1 Processor supervisor

- [ ] Select one supported local text execution path first: llama.cpp or Ollama.
- [ ] Define processor discovery, model metadata, health, load/unload, concurrency,
  cancellation and resource reporting.
- [ ] Run blocking inference outside the daemon event loop.
- [ ] Unload large models after configurable idle periods on constrained devices.
- [ ] Report unavailable RAM/VRAM/accelerator capacity as a typed failure.
- [ ] Remove or freeze non-functional IREE and ONNX entries until they have owners
  and conformance tests.

#### 4.2 Bounded local tasks

- [ ] Add schema-constrained intent classification, entity extraction, episode
  summarization, salience estimation, privacy-risk elevation and routing hints.
- [ ] Give each task a fixed schema, token/resource budget and failure policy.
- [ ] Validate outputs before they may become derived claims.
- [ ] Prevent local processors from writing state or invoking capabilities
  directly.
- [ ] Cache results by model/artifact version, normalized input and policy version.

#### 4.3 Local voice loop

- [ ] Start with push-to-talk and local VAD/STT; add an optional local wake word
  only after false-activation evaluation.
- [ ] Reuse `speech.synthesize` for local TTS responses.
- [ ] Do not persist raw audio by default; retain transcript only under the active
  history policy.
- [ ] Include transcript confidence, language and timing provenance.
- [ ] Ask for clarification rather than acting on low-confidence or ambiguous
  transcriptions.
- [ ] Do not allow voice alone to approve `HIGH` or `CRITICAL` actions.

#### 4.4 Visual fallback

- [ ] Define a separately startable visual processor and `MediaReference`
  lifecycle.
- [ ] Request a selected window or region through XDG Desktop Portal/PipeWire;
  never bypass Wayland permissions.
- [ ] Invoke vision only when semantic adapters are insufficient for the active
  user task.
- [ ] Crop and redact locally before inference and delete ephemeral captures after
  the task unless the user explicitly retains them.
- [ ] Treat OCR and visual grounding as uncertain observations requiring
  verification before action.
- [ ] Keep continuous visual monitoring outside the default product scope.

#### 4.5 Routing and evaluation

- [ ] Route through deterministic rules first, local classification second and
  policy last.
- [ ] Support `LOCAL`, `ASK`, `ESCALATE` and `DENY` routing decisions.
- [ ] Include privacy class, capability set, budget, confidence and verification
  requirements in every decision.
- [ ] Build a versioned evaluation corpus covering desktop intent, Polish and
  English voice, episode summaries, false wakeups, prompt injection and model
  abstention.
- [ ] Measure task accuracy, schema failures, latency, energy, tokens, RAM/VRAM,
  cold start and cancellation latency on representative laptop and edge classes.
- [ ] Choose model size and quantization from evidence; do not make a particular
  2–4B model part of the architecture contract.

#### 4.6 Optional NCSI/J-lens neural sidecar

- [/] Maintain a separately startable Transformers process that owns model and
  accelerator lifecycle, residual-stream hooks, versioned J-lens artifacts,
  compact neural observations, cancellation and typed failures.
- [x] The read-only server, Transformers engine, artifact fitter/loader, shared
  fixtures and unit tests exist.
- [x] Duplicate lens IDs are rejected and terminal request IDs cannot be reused
  during one sidecar process lifetime.
- [ ] Complete live acceptance, bounded or restart-persistent replay protection
  and GAIA transport.
- [ ] Keep raw tensors inside the sidecar and PyTorch/J-lens dependencies outside
  the default installation.

RAI provides the optional neural runtime; GAIA retains Workspace, Control,
epistemic and verification semantics. Detailed milestone status is tracked only
in the [canonical cross-project integration plan](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/ncsi-jlens-integration.md).

Acceptance slices:

1. A local processor converts an episode into validated structured output while
   offline; processor failure does not affect the original episode.
2. A Polish or English push-to-talk request asks about recent activity and is
   answered locally through TTS without retaining raw audio.
3. An accessibility-poor test application triggers one consented, cropped visual
   inference; its result is marked uncertain and the capture is removed.

Required failure tests cover malformed output, model timeout, cancellation,
out-of-memory, unavailable accelerator, low-confidence speech and denied screen
capture.

### Stage 5 — Rich Actions: safe desktop and system control

Purpose: turn read-only awareness into useful assistance without giving a model
ambient shell, keyboard or pointer authority.

Prerequisites: Stage 1 policy/capability path and Stage 3 observations. Stage 4
models and voice improve intent handling but are not required for deterministic
actions.

#### 5.1 Capability catalog

- [ ] Implement a minimal versioned catalog:
  `application.list`, `application.launch`, `file.search`, `document.open`,
  `browser.search`, `browser.open_result`, `browser.read_page`,
  `system.volume.get`, `system.volume.set`, `process.inspect` and
  `shell.run_sandboxed`.
- [ ] Use stable resource/result IDs so follow-ups such as "open the first result"
  do not depend on a model repeating a path or URL.
- [ ] Declare inputs, outputs, side effects, risk class, required isolation,
  verification and compensation for each capability.
- [ ] Prefer D-Bus, application APIs, desktop entries, XDG Portals and AT-SPI in
  that order before pointer/keyboard simulation.

#### 5.2 Resource authority

- [ ] Replace raw paths, URLs, window coordinates and PIDs in model-facing calls
  with scoped handles where practical.
- [ ] Bind handles to user, task, expiry, allowed operations and source policy.
- [ ] Reject stale, substituted or broadened handles.
- [ ] Resolve the final target again immediately before a state-changing action.

#### 5.3 Risk and approval policy

- [ ] Allow `LOW` read-only actions automatically under the selected profile.
- [ ] Permit reusable approval rules only for narrowly parameterized `MODERATE`
  actions.
- [ ] Require action-time approval for `HIGH` actions such as external data
  transfer or editing user documents.
- [ ] Deny `CRITICAL` actions by default; require explicit administrative policy
  for destructive, privilege-changing or credential-related operations.
- [ ] Present exact targets, side effects, data egress and rollback limitations in
  approval prompts.

#### 5.4 Execution and verification

- [ ] Make requests idempotent where possible and use request IDs to suppress
  duplicate execution.
- [ ] Define preconditions and postconditions for every state-changing
  capability.
- [ ] Verify effects from operating-system or application state rather than model
  narration.
- [ ] Return `SUCCEEDED`, `FAILED`, `PARTIAL`, `CANCELLED` or `UNKNOWN` with
  evidence.
- [ ] Add compensation/undo only where it is well defined; never imply rollback
  for irreversible actions.
- [ ] Keep arbitrary shell execution inside verified Bubblewrap/Guix isolation
  with network and write mounts disabled unless individually approved.

#### 5.5 GUI fallback

- [ ] Define `ui.inspect`, `ui.activate` and bounded text-entry capabilities over
  semantic element references.
- [ ] Use pointer coordinates only after target grounding and immediately recheck
  window identity and geometry.
- [ ] Require stronger confirmation for password dialogs, external publication,
  purchases, deletion and privilege prompts.
- [ ] Stop on unexpected dialogs, focus changes or unverifiable outcomes.

Acceptance scenarios:

1. "Launch application X" resolves a desktop entry, applies policy, launches it
   and verifies the process/window.
2. "Find document Y and open the second result" searches allowed roots, returns
   stable result IDs and opens the chosen document through a portal or registered
   application.
3. "Search the web for Z; open the first result; read the page" preserves result
   identity, treats page content as untrusted and performs no unrelated action.
4. A document containing prompt injection cannot cause tool invocation or policy
   changes.

Each scenario must pass through CLI and MCP using the same capability schema,
policy decision, approval behavior and verified result.

### Stage 6 — Rich Hybrid: external agents, ACP and budget control

Purpose: allow difficult reasoning and coding tasks to use GAIA or external
agents without transferring ownership of memory, policy or the Linux desktop.

Prerequisites: Stages 1, 3 and 5. A backend cannot be production-enabled until
usage accounting, cancellation and data-egress auditing work.

#### 6.1 Context construction and egress

- [ ] Build task-specific `ContextPackage` values from durable state through
  deterministic retrieval and policy filtering.
- [ ] Include a `ContextManifest` listing sources, data classes, redactions,
  approximate size, intended recipient, retention expectation and reason for
  transfer.
- [ ] Prefer claims, summaries and stable resource handles over raw files,
  screenshots or full activity history.
- [ ] Preview the manifest in `HYBRID_APPROVAL` mode before transmission.
- [ ] Persist the manifest, approval and actual transmitted-size/usage metadata.
- [ ] Prevent a backend from requesting broader historical context without a new
  policy decision.

#### 6.2 Token and cost governor

- [ ] Enforce `InferenceBudget` before and during every backend request.
- [ ] Add per-task, automation, model, provider, daily and monthly limits.
- [ ] Count retries, cached-token billing, tool turns, image inputs and partial
  streamed responses where reported by the provider.
- [ ] Treat missing or unverifiable usage/pricing as unknown and apply the
  configured conservative limit.
- [ ] Cancel at the deadline or budget boundary and retain the partial evidence
  without treating it as success.
- [ ] Provide usage reports and alerts without leaking prompt content into
  telemetry.

Suggested safe initial defaults:

```yaml
profile: LOCAL_PREFERRED
background_remote_tokens: 0
remote_on_ambiguous_input: ask
per_task:
  max_agent_turns: 3
  max_images: 1
  max_tool_calls: 8
on_missing_usage: ask
on_budget_exceeded: cancel
```

Token and currency amounts remain operator configuration because models and
prices change independently of RAI releases.

#### 6.3 AgentBackend conformance

- [ ] Stabilize `AgentBackend` lifecycle, streaming, cancellation, usage,
  retryability, evidence and failure contracts.
- [ ] Keep backend conversation/session IDs as adapter-owned metadata.
- [ ] Add GAIA as the reference cognitive-runtime integration.
- [ ] Add one end-to-end external harness backend before multiplying providers.
- [ ] Retain Antigravity only if it passes the common contracts without
  process-wide patching or private runtime coupling.
- [ ] Make backend removal or outage preserve local history and capability state.

#### 6.4 ACP and MCP roles

- [ ] Add an optional ACP client adapter after the base `AgentBackend` contract is
  stable.
- [ ] Map ACP session creation, prompt streaming, cancellation, plans and
  permission requests into RAI records without making ACP a security boundary.
- [ ] Expose approved RAI capabilities to agents through MCP or direct typed
  adapters; MCP remains behind the agent while ACP manages the agent session.
- [ ] Re-evaluate protocol-version compatibility at implementation time and keep
  protocol negotiation explicit.
- [ ] Add conformance fixtures that prove an ACP agent cannot bypass RAI policy or
  obtain ambient desktop context.

#### 6.5 Hybrid routing

- [ ] Route deterministic tasks directly to capabilities without spending model
  tokens.
- [ ] Route bounded language/perception tasks to local processors when their
  evaluation envelope covers the request.
- [ ] Escalate open-ended planning, complex coding or research only within data,
  cost, latency and capability policy.
- [ ] Allow the user to pin or exclude providers for a task or data class.
- [ ] Use model confidence only as routing evidence; policy remains deterministic.
- [ ] Re-verify external claims and requested actions against local tools and
  observations before committing state.

Acceptance slice:

```text
user asks for a complex task
  -> local router marks it out of local scope
  -> ContextPackage and outbound manifest are built
  -> policy/budget decision and optional approval
  -> GAIA or one external AgentBackend executes
  -> agent uses only advertised RAI capabilities
  -> cancellation/usage/evidence are returned
  -> local postconditions verify the result
```

The same task in `LOCAL_ONLY` must remain local or fail clearly. Passive Rich
History operation must still report zero external tokens.

### Stage 7 — edge and multi-device operation

Purpose: extend the same observation, policy and capability model to constrained
Linux devices and trusted local networks.

Prerequisite: stable Stage 1 wire contracts and Stage 2 replay semantics.

#### 7.1 Device identity and lifecycle

- [ ] Define registration, enrollment, heartbeat, status, capability
  advertisement, key rotation and revocation.
- [ ] Separate human-readable device names from cryptographic identity.
- [ ] Make permission and privacy policy device-specific.
- [ ] Expose clock quality, power state, accelerator availability and connectivity
  as device observations.

#### 7.2 Transport resilience

- [ ] Add bounded offline spooling, reconnect/replay, event deduplication and
  backpressure.
- [ ] Prevent unbounded event retention on constrained storage.
- [ ] Apply privacy filtering on the source device before spooling or transport.
- [ ] Document mTLS or an equivalent device-identity boundary before LAN
  exposure.
- [ ] Fail closed when identity, policy version or transport security cannot be
  verified.

#### 7.3 Reference DeviceAgent

- [ ] Provide a minimal Linux `DeviceAgent` SDK and reference implementation.
- [ ] Support collection, capability advertisement and verified bounded actions
  without requiring the full Python daemon.
- [ ] Run wake word/VAD and simple classification on-device where practical.
- [ ] Allow heavier local VLM/SLM processing on an explicitly trusted LAN node
  under the same `ContextManifest` and budget rules.
- [ ] Define an SDK boundary for later Android, VR and wearable clients without
  assuming unrestricted background execution.

Acceptance slice:

```text
Raspberry Pi or simulated DeviceAgent
  -> records allowed observations while disconnected
  -> reconnects without loss or duplication
  -> advertises current capabilities
  -> receives one policy-approved GAIA action
  -> returns a verified result
```

Revoked, cloned or over-quota devices must be rejected without dropping audit
evidence.

### Stage 8 — Rich Automation, security UX and ecosystem

Purpose: make the runtime understandable and controllable without turning a chat
window into the primary desktop interface.

Prerequisites: the underlying history, policy, action and budget APIs from
Stages 3–6.

#### 8.1 Minimal security surface

- [/] Provide a native GNOME status surface for `OFF`, `PAUSED`, `OBSERVING`,
  `THINKING`, `ASKING_REMOTE` and `ACTING` states. Existing desktop/HITL dialogs
  are partial prototypes.
- [ ] Show active microphone, screen-capture and external-backend use distinctly.
- [ ] Add immediate pause and emergency stop independent of model availability.
- [ ] Show the active AI profile, provider, local-model status and remaining
  budget.
- [ ] Make approval prompts accessible from keyboard and assistive technologies.

#### 8.2 History, privacy and usage review

- [/] Connect the experimental NiceGUI history view to the real
  observation/episode stores and deletion policy.
- [ ] Show episode evidence, source applications and derived claims separately.
- [ ] Expose allow-only/exclude rules and retention settings.
- [ ] Display every outbound `ContextManifest`, recipient, approval and reported
  usage.
- [ ] Support selective deletion and export without exposing `SECRET` or dropped
  content.

#### 8.3 Workflow learning and automation

- [ ] Detect recurring event/capability patterns locally without automatically
  enabling them.
- [ ] Generate a reviewable workflow specification with fixed inputs,
  capabilities, data classes, schedule, confirmation points and budget.
- [ ] Require explicit activation and make every automation independently
  pausable and revocable.
- [ ] Reject workflows that require ambient shell/desktop authority.
- [ ] Store versioned executions and verify postconditions on every run.
- [ ] Disable an automation after repeated unexpected, partial or unverifiable
  outcomes.

#### 8.4 Client and desktop ecosystem

- [/] Complete the Emacs client migration to Stage 1 capability contracts.
- [/] Reach COSMIC parity through actual platform APIs and acceptance coverage;
  the current adapter is partial.
- [/] Migrate the Guile dashboard and experimental web UI to the common runtime
  contracts or retire them.
- [ ] Add browser integration only after origin permissions, private-mode
  exclusion and prompt-injection tests pass.
- [ ] Extend voice and accessibility beyond the minimal local loop while
  preserving the same approval and privacy rules.

Acceptance slice:

The user can see when RAI observes, reasons, contacts a provider or acts; inspect
and delete history; review transmitted context and cost; stop the runtime; and
approve one recurring workflow whose future executions remain within their
declared capabilities and budget.

### Stage 9 — production hardening and 1.0 readiness

Purpose: prove that the complete local/hybrid desktop runtime is operable,
recoverable and secure enough for non-development use.

#### 9.1 Security verification

- [ ] Maintain a threat model covering collectors, accessibility APIs, browser
  content, model supply chain, prompt injection, local IPC, remote backends,
  capabilities, storage and devices.
- [ ] Add adversarial fixtures for indirect prompt injection, target substitution,
  approval spoofing, replay, symlink/path races and malicious accessibility
  trees.
- [ ] Fuzz public schemas and capability validation.
- [ ] Perform an independent security review before declaring 1.0.
- [ ] Document residual risks and safe deployment profiles.

#### 9.2 Reliability and recovery

- [ ] Define service-level objectives for daemon availability, collection loss,
  action verification, cancellation latency and budget enforcement.
- [ ] Test crash recovery, database migration, partial deletion, corrupted model
  output and interrupted upgrades.
- [ ] Provide backup/export and restore rules that preserve encryption and
  provenance without silently restoring deleted activity.
- [ ] Make every optional backend and processor removable without breaking local
  history or deterministic actions.

#### 9.3 Performance and hardware profiles

- [ ] Publish measured workstation, laptop and Raspberry Pi reference profiles.
- [ ] Bound idle CPU, RAM, disk writes and battery impact for passive collection.
- [ ] Measure local STT/SLM/VLM cold and warm paths and document automatic unload
  behavior.
- [ ] Provide graceful degradation from visual to semantic to deterministic
  operation as resources disappear.

#### 9.4 Packaging and operations

- [ ] Ship systemd user units, XDG integration and least-privilege packaging.
- [ ] Document portal, AT-SPI, browser and desktop-specific permission setup.
- [ ] Provide schema/database migration and rollback procedures.
- [ ] Define supported Linux distributions, desktops, model runtimes and protocol
  versions.
- [ ] Publish an operator guide for local-only, hybrid, trusted-LAN and external
  deployments.
- [/] Keep Sphinx, MyST Markdown and Furo as the single publishable documentation
  system; the base configuration and GitLab Pages job exist, while complete
  navigation and warning-free coverage of root project documents remain open.
- [ ] Export and archive the release OpenAPI schema so HTTP API documentation can
  be reviewed and compared without running the daemon.
- [ ] Publish versioned documentation from immutable SemVer tags and retain a
  clearly labeled development version from `main`.
- [ ] Make documentation warnings, broken internal links and version consistency
  release-blocking CI checks.

1.0 acceptance gate:

```text
Rich History works locally with zero remote tokens
Rich Voice handles bounded requests without retaining raw audio
Rich Actions cannot bypass policy and verifies consequential results
Rich Hybrid exposes every transfer and enforces cancellation and budgets
prompt injection cannot directly invoke capabilities or alter policy
history deletion removes source and derived data according to policy
external providers and local models are replaceable without losing runtime state
critical tests, security tests, coverage and lint pass in CI
```

## Cross-stage reference scenarios

These scenarios are maintained from the first supporting stage onward and become
release-level regression tests.

### Scenario A — resume recent work

1. RAI observes allowed Firefox, terminal and editor events.
2. It creates one deterministic project episode locally.
3. The user asks, "What was I working on before the break?"
4. RAI answers locally with provenance and no network request.
5. Deleting the episode removes its derived memory and retrieval entry.

### Scenario B — find and open a document

1. The user asks by voice or text to find document Y.
2. RAI searches only configured roots and returns stable result IDs.
3. The user selects a result by ordinal or metadata.
4. RAI opens it through an approved application/portal and verifies the result.
5. No document content is sent externally unless separately authorized.

### Scenario C — search and read the web

1. RAI performs a user-requested search and returns stable result IDs.
2. "Open the first result" uses the existing ID, not a regenerated URL.
3. "Read this page" retrieves bounded content and labels it untrusted.
4. Page text cannot change instructions, policy or capabilities.
5. Any external summarizer receives a visible, budgeted context manifest.

### Scenario D — hybrid coding task

1. Local history identifies the active project and recent failing test without
   copying unrelated files.
2. A local processor classifies the requested repair as out of scope.
3. RAI builds a minimal `ContextPackage` with approved files and evidence.
4. GAIA or an external harness receives only the package and scoped capabilities.
5. Commands run in the configured sandbox, changes remain reviewable and tests
   provide postcondition evidence.
6. Usage and transmitted context appear in the local audit view.

### Scenario E — privacy boundary

1. A password manager, private browser window or excluded chat application is
   focused.
2. The privacy firewall drops content before the observation journal.
3. No exception, metric, memory, prompt, cache or outbound manifest contains the
   protected value.
4. A later model request cannot retrieve what was never persisted.

## Legacy chat retirement gates

Legacy removal is incremental and follows working replacement slices rather
than a big-bang rewrite.

1. The application container and typed registry become the only capability
   invocation path.
2. The Stage 2 Observation → GAIA → Action → Result slice passes cross-project
   acceptance tests.
3. CLI, REST and MCP use the same runtime services and policy decisions.
4. Rich History stores normalized observations and episodes independently of
   conversation history.
5. `ChatService`, agent chains and provider-specific configuration move behind
   an explicitly named compatibility package and stop defining durable state.
6. Antigravity becomes an optional `AgentBackend`, disabled in the default
   installation.
7. Legacy chain endpoints receive a documented deprecation and migration path.
8. Compatibility code that does not pass the common backend and security
   contracts is removed before 1.0.

## Technical debt carried beyond Stage 1

- `cli_compatibility.py` and `config_manager.py` remain oversized compatibility
  modules and still mix some I/O and UI concerns.
- Antigravity compatibility still owns the legacy chat execution path, although
  it is isolated behind the provider-neutral `AgentBackend` contract.
- Conversation history still has more implementation weight than normalized
  observation and action-result state.
- Local inference protocols are disconnected from the daemon and lack tests.
- IREE is a stub and the ONNX factory references an absent implementation.
- Device identity, reconnect/replay and backpressure contracts do not yet exist.
- Full style linting contains legacy violations; critical lint is blocking now.
- Existing GitLab issues #8 and #9 remain relevant to lazy loading and blocking
  local inference. Issues #2 and #6 require reproduction against the new
  contracts before implementation.

## Issue and merge-request template for roadmap work

Every implementation issue should state:

- stage and work-package ID,
- user-visible outcome,
- contract or trust boundary affected,
- input/output schemas and versioning impact,
- explicit allowed and forbidden behavior,
- resource and token budget,
- expected failure, cancellation and recovery behavior,
- unit, conformance, integration and security tests,
- telemetry/audit evidence that does not expose private content,
- documentation and migration changes.

A merge request should complete one reviewable vertical behavior. Adding a
protocol, dependency or unused adapter without connecting it to a tested slice
does not advance the roadmap checkbox.

## Definition of done for roadmap work

A checkbox requires code or documentation in the current tree, relevant tests,
an explicit failure mode and updated user-facing documentation. A prototype, an
unused protocol, a mocked happy-path unit test or a model demonstration alone
does not complete a subsystem.

For collection, inference, action or backend work, "done" additionally means:

- privacy classification and retention are defined,
- cancellation and resource limits are tested,
- model/backend absence has a safe behavior,
- audit and provenance are complete,
- no new provider-specific assumption leaks into the runtime kernel,
- the stage acceptance slice still works in `LOCAL_ONLY` where applicable.
