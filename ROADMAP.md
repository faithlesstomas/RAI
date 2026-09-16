# Rich AI roadmap

This document is the source of truth for the evolution of **Rich AI (RAI) — an
intelligent and secure GNU/Linux desktop powered by local, hybrid or external
AI**.

RAI is the secure, Linux-first integration layer between AI assistants and the
operating system. It observes explicitly allowed activity, maintains durable
local state and memory, exposes policy-controlled capabilities, and connects
replaceable local models and agent harnesses to the desktop. The desktop is the
primary interaction surface; RAI does not require a chat window to own the user
experience.

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
5. **Rich Assistant** — a continuous, local-first assistant with durable graph
   memory and a bounded context reconstructed for every interaction rather than
   inherited from a model-provider conversation.
6. **Rich Hybrid AI** — policy-routed use of local models and external model
   APIs with explicit context-egress, token, cost, latency and compute budgets.
7. **Rich Agent Interop** — optional, separately packaged connections to
   external cognitive runtimes and agent harnesses through bounded context and
   capability contracts.
8. **Rich Automation** — user-approved reusable workflows with fixed
   capabilities, privacy rules, resource budgets and audit trails.

Rich History, Voice, Actions, Local AI and Assistant must remain useful when
every external model provider and agent runtime is disabled. Rich Hybrid AI and
Agent Interop are optional extensions; their absence or outage must not degrade
local memory, policy, audit or previously accepted work.

## Product boundary

RAI owns:

- Linux-first local and edge perception and normalized observations,
- device identity, status, discovery and capability advertisement,
- durable local observation, episode, action-result, usage and audit state,
- capability discovery, invocation and result verification,
- policy, authorization, sandboxing, human approval and audit,
- context selection, redaction, data-egress control and privacy boundaries,
- provider-neutral assistant sessions, graph-memory retrieval and bounded
  context construction for local-first user interaction,
- deterministic safety reflexes and routing between rules, bounded local
  processing and external cognitive backends,
- local MCP/HTTP/Unix-socket interfaces and authenticated network transports for
  remote device agents,
- user-visible controls for collection, permissions, budgets, activity review
  and emergency stop.

RAI does not own a universal reasoning loop, Global Workspace, Cognitive
Control, goal verification or open-ended deliberation. An optional external
cognitive runtime may own those semantics. RAI may perform bounded,
schema-constrained inference and deterministic local safety reactions, but it
must not silently become an open-ended cognitive process controller.

RAI may own a bounded assistant interaction loop: accept one user or approved
proactive event, reconstruct one finite context from durable memory, invoke one
replaceable reasoning strategy within a budget, validate its candidate outputs
and commit only policy-approved records. This loop is not a GCAS Workspace and
does not grant the model ownership of memory, tools, goals or execution.

Hybrid model inference and external-agent interoperability are separate ports.
An `AssistantModelBackend` performs one bounded model inference locally or
through an external API. An optional `AgentBackend` delegates a bounded task to
an external cognitive runtime or agent harness. Both receive explicit context,
privacy and resource budgets and report usage to the common ledger, but agent
session semantics do not leak into the assistant core.

RAI owns its versioned embodiment, capability, policy and provenance contracts.
Those contracts must support a useful standalone product and must not import or
require an external cognitive runtime. GCAS compatibility is an optional
semantic mapping, not the source of RAI's runtime types. Local and external LLM
providers remain replaceable `AssistantModelBackend` implementations. External
cognitive runtimes and agent harnesses remain optional, separately packaged
`AgentBackend` implementations; named integrations belong in adapters rather
than the core. J-lens is an optional neural-inspection sidecar, not a core
dependency or release gate. A remote Android, VR or wearable client is a thin
`DeviceAgent`, not necessarily a full RAI installation.

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

This accounting applies uniformly to local processors, local or external
`AssistantModelBackend` implementations and optional `AgentBackend` runtimes.
Local execution may have zero provider price, but its tokens, wall time, energy
when measurable, RAM/VRAM and accelerator occupancy are still budgeted usage.

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
11. Provider conversation IDs, KV caches and hidden states are ephemeral
    backend metadata, never the source of truth for assistant memory.
12. Every assistant inference receives a finite, versioned context reconstructed
    from policy-approved durable records; a raw transcript is not replayed by
    default.
13. Local models, external model APIs and agent harnesses use the same mandatory
    inference-budget and usage-ledger boundary.
14. Every state-changing action has a risk decision, typed result and
    postcondition verification.
15. One cognitive process has one explicit controller; an optional RAI research
    controller and an external cognitive runtime cannot co-own the same process.
16. Network delivery is replayable and idempotent; reconnects never silently
    lose or duplicate an accepted action.
17. Pixel-based desktop control and arbitrary shell execution are fallback
    capabilities, not the default integration path.
18. A stage is complete only when its acceptance tests and security failure
    tests pass.
19. An invalidated, expired or superseded fact cannot be admitted into an active
    ContextPackage without an explicit historical or temporal provenance flag.
20. A derived claim or summary cannot silently increase source certainty,
    erase modality or survive without an authorized path to raw evidence.
21. Remember, forget, update, supersede, reflect and reconstruct are explicit
    memory operations with typed inputs, state transitions and audit evidence.
22. Retrieval and adapter comparisons use the same context budget, corpus,
    answerer and evaluation protocol; unconstrained recall is not evidence of a
    better memory architecture.
23. Graph ranking runs only over a policy-eligible, provenance-qualified
    subgraph. Untrusted topology cannot influence selection of trusted facts.
24. Durable memory is scoped by user, profile and domain. Context assembly does
    not leak personal claims into an unrelated domain merely because they are
    relevant by embedding similarity.

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
- [x] Replace the compatibility calculator's Python evaluator with a bounded
  arithmetic parser and make the reviewed Ruff security profile blocking.
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

### Stage 2 — durable local event plane

Purpose: give RAI a provider-neutral, replayable event backbone before adding
production desktop observation. This stage connects the Stage 1 records,
policy and capability path without requiring an external cognitive runtime.

Prerequisite: Stage 1 contracts.

#### 2.1 Ordered event journal

- [x] Define one versioned `EventJournal` port and envelope for `Observation`,
  `ActionResult`, `ActionFailure`, audit and usage events without weakening the
  schemas of the enclosed records.
- [x] Assign a durable monotonic sequence to every accepted event and expose
  bounded ordered reads from an opaque replay cursor.
- [x] Make `record_id` idempotent: an identical retry returns the original
  sequence, while reuse with different content fails as a typed conflict.
- [x] Make accepted appends transactional and durable before returning success;
  document recovery after interruption at every commit boundary.
- [x] Persist independent consumer positions and acknowledgements so processing
  can resume without silently losing an accepted event.
- [x] Add retention hooks without allowing a backend to delete evidence.
- [x] Record clock source and uncertainty for multi-device timestamps.

The existing one-record-per-file observation store may be used as migration
input or a simple projection, but it does not satisfy this journal contract
until ordering, conflict detection, replay and acknowledgements are present.

#### 2.2 Local ingest, replay and subscriptions

- [x] Add authenticated, schema-validated event ingest over loopback HTTP and a
  protected Unix-domain socket.
- [x] Add bounded replay and subscription APIs that start from an explicit
  cursor and return the next durable cursor with every batch.
- [x] Validate record type, version, producer, size and data class before the
  journal accepts an event.
- [x] Persist consumer acknowledgements only after their processing result is
  durable.
- [x] Implement bounded queues, backpressure and clear overload failures.
- [x] Ensure slow subscribers cannot block collectors or the core event loop.
- [x] Keep trusted-LAN authentication, offline device spooling and cross-device
  reconnect behavior in Stage 7; Stage 2 exposes no unauthenticated network
  listener.

#### 2.3 Provider-neutral dispatch and return path

- [x] Add a local deterministic subscriber that maps one accepted synthetic
  observation to a bounded `CapabilityRequest` without model reasoning.
- [x] Dispatch the request exclusively through the common capability registry,
  `PolicyEngine`, cancellation and audit path.
- [x] Use a deterministic test actuator and persist its typed `ActionResult` or
  `ActionFailure` before acknowledging the source event.
- [x] Guarantee one durable terminal result for a request across retries,
  cancellation and process restart.
- [x] Keep real speech synthesis in Stage 4 and production desktop/system
  actuators in Stage 5; the Stage 2 actuator is a conformance fixture, not a
  user-facing integration.

Acceptance slice:

```text
synthetic person_present Observation
  -> authenticated local ingest
  -> ordered durable EventJournal
  -> replay from an explicit cursor
  -> deterministic local subscriber
  -> CapabilityRequest through the common PolicyEngine
  -> deterministic test actuator
  -> durable ActionResult
  -> consumer acknowledgement
```

Required failure tests cover identical duplicate delivery, conflicting ID reuse,
out-of-order source timestamps, restart/replay before and after acknowledgement,
malformed and oversized events, unauthorized ingest, slow-consumer backpressure,
unavailable actuator, policy denial and cancellation. The slice must pass with
GAIA, GCAS compatibility, Antigravity, J-lens and every external model disabled.

### Release gate A — `rich-ai` developer preview

Purpose: publish an installable preview containing both the Stage 1 kernel and a
working Stage 2 event plane without presenting later desktop capabilities as a
finished product. GAIA, GCAS and J-lens remain outside this release gate.

Prerequisite: the Stage 2 provider-neutral acceptance slice and its required
failure tests pass in CI.

- [x] Select `rich-ai` as the Python distribution identity while retaining the
  `rai` repository, import namespace, CLI command, configuration and protocol
  names.
- [x] Add PyPI metadata, project links, classifiers and explicit optional
  dependency groups.
- [x] Keep Antigravity and J-lens outside the base dependency set, and make a
  base `import rai` work without their SDKs.
- [x] Build both wheel and source distribution in CI, validate them with Twine,
  and smoke-test the wheel in a clean virtual environment.
- [x] Add isolated GitLab OIDC jobs for TestPyPI and PyPI; publishing credentials
  must not be stored as long-lived CI variables.
- [x] Complete the Stage 2 provider-neutral acceptance slice before running a
  version-producing `release` job.
- [/] Configure `rich-ai` trusted publishers on TestPyPI and PyPI for this
  GitLab project, `.gitlab-ci.yml`, and the `testpypi`/`pypi` environments.
  OIDC publishing is verified; protection of the production environment and
  release tags in GitLab remains open.
- [x] Publish a TestPyPI candidate and verify installation, `import rai`,
  `rai --version`, capability listing, local event ingest/replay and package
  links from a clean machine.
- [x] Publish the first immutable PyPI developer preview, `0.5.0a1`; do not
  rebuild or reuse that version or the already released `0.3.1` version.

Acceptance gate:

```text
python -m pip install --pre rich-ai succeeds in a clean environment
import rai and rai --version report the same version as PyPI metadata
rai capability list and the Stage 2 local event slice work without GAIA,
Antigravity or J-lens installed
wheel and sdist pass twine check and contain no dependency on PyPI project rai
or direct VCS/URL dependency
the tag, GitLab release and PyPI artifacts identify the same immutable commit
```

Post-preview documentation follow-up (not blocking Release gate A):

- [ ] Split warning-as-error documentation validation from GitLab Pages
  deployment so merge requests and `main` verify docs without replacing the
  public release site.
- [ ] Deploy the public Pages site from an immutable release tag only after the
  corresponding `publish_pypi` job succeeds.
- [ ] Add browsable documentation versions (for example `/latest/`,
  `/0.4.0a3/` and `/0.4.0/`) with an explicit version selector, using either a
  Sphinx multi-version build or GitLab Pages versioned deployments.

### Stage 3 — Rich History: private desktop observation

Purpose: deliver a useful read-only desktop-awareness product before autonomous
or model-driven action. Start with metadata and semantic events, not continuous
screenshots or key logging.

Status: `[/]` — the provider-neutral pipeline, privacy boundary, encrypted
storage, deterministic episode builder and local review API pass automated
acceptance tests. Live GNOME validation is complete; production validation of
AT-SPI, browser integration and the complete cross-application scenario remains
open and is tracked separately below.

Prerequisites: Stage 1 records and the Stage 2 event journal and local
subscriptions. GAIA is not required for the local Rich History acceptance
slice.

#### 3.1 Collector supervisor

- [x] Add production collector registration, lifecycle, health, restart and
  backoff.
- [x] Run platform collectors as daemon-managed unprivileged sidecars. Packaged
  systemd user-service integration remains a Stage 9 concern.
- [x] Expose collector status, last event, error and effective permission state.
- [x] Stop collection immediately when the profile is disabled, the session is
  locked or the user activates emergency stop.
- [x] Ensure collector failure cannot terminate the core daemon.

#### 3.2 GNOME session collector

- [x] Observe session lock/unlock, idle/active transitions, workspace changes and
  active application/window changes using supported GNOME interfaces.
- [x] Normalize application identity through desktop-entry IDs where possible.
- [x] Treat window titles as potentially private content and classify them before
  persistence.
- [x] Avoid privileged `/dev/input` access and global raw input capture.

#### 3.3 AT-SPI semantic collector

- [/] Observe bounded focus, role, state and document-context changes over
  AT-SPI.
- [/] Coalesce repeated text-change events into duration/activity facts; do not
  store typed characters.
- [/] Detect password/secret roles and discard their values before the journal.
- [/] Apply size, rate and depth limits to accessibility trees.
- [/] Record toolkit/source quality so downstream components know when semantic
  context is incomplete.

#### 3.4 Process, filesystem and project context

- [x] Observe foreground process identity without collecting unrelated process
  arguments or environment variables.
- [x] Add opt-in filesystem events for configured roots only.
- [x] Detect project and Git identity from approved roots while excluding file
  content by default.
- [x] Correlate save/build/test events using stable resource references rather
  than copying documents into history.

#### 3.5 Browser semantics

- [x] Define a browser adapter contract for active tab ID, origin, title,
  navigation and user-requested selected text.
- [/] Implement an extension/native-messaging or accessibility producer that
  exposes semantic metadata instead of screenshots; the adapter contract and
  privacy boundary already exist.
- [x] Exclude private browsing unconditionally.
- [x] Apply origin allow/exclude policy before storing URL or title.
- [x] Never treat page text as an instruction to RAI or an agent.

#### 3.6 Privacy firewall

- [x] Implement deterministic source, application, origin, path, field-role and
  session-state policies before persistence.
- [x] Support `DROP`, `METADATA_ONLY`, `REDACT` and `ALLOW` outcomes with policy
  provenance.
- [x] Provide built-in protections for password managers, authentication dialogs,
  banking/health profiles and communication applications.
- [x] Keep dropped content out of logs, metrics, exception messages and dead
  letters.
- [x] Add a local redaction test corpus containing credentials, personal data and
  prompt-injection fixtures.

#### 3.7 Event normalization and fusion

- [x] Debounce and deduplicate high-frequency events before durable storage.
- [x] Fuse simultaneous GNOME, AT-SPI, process and filesystem evidence into one
  activity fact without losing source references.
- [x] Represent uncertainty and conflicting evidence explicitly.
- [x] Make fusion deterministic for the same ordered input and configuration.

#### 3.8 Deterministic episode builder

- [x] Segment observations using time, idle, application, resource and project
  boundaries.
- [x] Keep episode construction deterministic and independent of an LLM.
- [x] Store applications, resources, duration, outcome signals and provenance;
  inferred goals remain optional derived claims.
- [x] Rebuild episodes reproducibly from retained observations and a versioned
  builder configuration.
- [x] Update or invalidate derived memories when source observations are deleted.

#### 3.9 Local storage, retention and deletion

- [x] Implement independent TTLs for raw buffers, observations, episodes and
  memories.
- [x] Integrate per-user encryption keys through Secret Service or another
  documented Linux credential store, with explicit unavailable-key behavior.
- [x] Use restrictive file permissions and exclude databases from backup by
  default unless the user opts in.
- [x] Implement pause/resume, allow-only/exclude lists and time-range deletion.
- [x] Verify deletion across source events, derived memories, indexes, caches and
  outbound-context references.

#### 3.10 History query and review API

- [x] Provide local queries by time, application, project, resource and activity
  type.
- [x] Answer deterministic questions such as "which applications were active?"
  without an LLM.
- [x] Return provenance links and confidence for every derived activity claim.
- [x] Expose review and deletion through the same local API used later by the
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

The automated acceptance slice constructs bounded `SourceEvent` records at the
collector boundary. It verifies the provider-neutral pipeline but does not by
itself prove that every supported desktop integration works in a real user
session.

Production validation gate (still open):

- [x] Capture lock, idle, workspace and active-window transitions from the
  packaged GNOME extension in a supported live GNOME session; validated on
  GNOME Shell 50.1 with extension version 2 in GitLab issue #10 and MR !16.
- [ ] Capture bounded focus, document and coalesced text-activity events from a
  live AT-SPI accessibility bus without retaining entered text.
- [ ] Deliver browser navigation metadata through a packaged browser producer
  and prove that private-mode activity is absent at the RAI ingest boundary.
- [ ] Run the Firefox -> terminal -> editor -> test-run acceptance scenario using
  production collectors, inspect its evidence and verify deletion from a clean
  user profile.

### Stage 4 — Rich Local AI and Rich Voice

Purpose: add bounded local understanding while keeping collection, memory,
policy and execution deterministic and independent of model availability.

Prerequisite: Stage 3 local episodes. Stage 1 provides the policy-controlled
actuator boundary and Stage 2 provides durable input and result delivery.

#### 4.1 Processor supervisor

Status: `[/]` — the lifecycle-managed supervisor and its safety boundaries are
implemented and tested, but live backend acceptance, operational discovery and
host-capacity reporting remain open.

- [/] Select one supported local text execution path first. Ollama is the
  configured default and both Ollama and llama.cpp adapters have contract tests,
  but neither has a repeatable live-model acceptance test in the current tree.
- [/] Define processor discovery, model metadata, health, load/unload,
  concurrency, cancellation and resource reporting. The protocols and lifecycle
  are implemented; discovery currently reports importable adapters rather than
  daemon/model readiness, and `ModelMetadata` is not populated by the engines.
- [x] Run blocking inference outside the daemon event loop.
- [x] Unload large models after configurable idle periods on constrained devices.
- [/] Report unavailable RAM/VRAM/accelerator capacity as a typed failure. Known
  engine requirements are checked against task budgets, but physical host and
  accelerator availability is not probed yet.
- [x] Remove or freeze non-functional IREE and ONNX entries until they have owners
  and conformance tests.

#### 4.2 Bounded local tasks

Status: `[x]` — all six versioned task contracts and the persistent,
policy-aware bounded-result cache are implemented. Cache reuse remains disabled
until an operator supplies an immutable model artifact version.

- [x] Add schema-constrained intent classification, entity extraction, episode
  summarization, salience estimation, privacy-risk elevation and routing hints.
- [x] Give each task a fixed schema, token/resource budget and failure policy.
- [x] Validate outputs before they may become derived claims.
- [x] Prevent local processors from writing state or invoking capabilities
  directly.
- [x] Cache results by model/artifact version, normalized input and policy version.

#### 4.3 Local voice loop

Status: `[/]` — provider-neutral synthesis contracts, configurable
`realtime_local`, `quality_local` and `premium_remote` profiles, privacy-aware
backend selection, a lazy local Piper adapter, sounddevice playback, capability
service composition, default-registry/MCP exposure, Markdown normalization and
deterministic substitutes are implemented. User-configurable profile/device
loading and legacy-facade removal remain in GitLab #22; PTT/STT, additional
backends and voice enrollment are tracked by #23–#25 under umbrella #21.

- [ ] Start with push-to-talk and local VAD/STT; add an optional local wake word
  only after false-activation evaluation.
- [/] Implement `speech.synthesize` as a policy-controlled local `Actuator`.
  The Piper implementation is registered in the default capability service and
  exposed through MCP; user-configurable profile/device loading and
  legacy-facade removal remain open.
- [x] Persist the typed result only after playback or its verifiable test
  substitute completes; device-unavailable and cancellation remain failures.
- [/] Do not persist raw audio by default; retain transcript only under the active
  history policy. Synthesized PCM is absent from terminal results, while STT and
  transcript retention are not implemented yet.
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
- [x] Provide the versioned HTTP/NDJSON-over-UDS transport consumed by GAIA's
  production NCSI adapter without importing either project's implementation
  modules into the other.
- [ ] Complete repeatable RAI live-model acceptance, authentication, bounded or
  restart-persistent replay protection and the remaining disconnect, OOM,
  timeout and cancellation failure gates.
- [x] Keep raw tensors inside the sidecar and PyTorch/J-lens dependencies outside
  the default installation.

RAI provides the optional neural runtime; GAIA retains Workspace, Control,
epistemic and verification semantics. Detailed milestone status is tracked only
in the [canonical cross-project integration plan](https://gitlab.com/tk-lab1/ai/gaia/-/blob/main/docs/ncsi-jlens-integration.md).

#### 4.7 Rich Assistant: graph-memory conversation and reasoning laboratory

Status: `[/]` — the local graph-memory foundation plus a usable CLI,
resumable/inspectable chat, exact context history and authenticated native
REST/WebSocket surface are implemented. SQLite memory operations, replay and
stage diagnostics form the Issue #35 M1 floor. Schema-constrained free-form
extraction, bitemporal claims, FTS5 raw-turn/claim retrieval, adaptive context
routing, Rich History evidence grounding and an equal-budget raw/claim/grounded-
summary evaluation runner, a versioned representative corpus and a product-
backend answer evaluator with an independent deterministic judge are
implemented. M4–M6 now add two recorded live-model manifests, explicitly scoped
energy readings, short/medium/long adaptive-router comparison, verified derived
write-back, local dense and weighted-RRF baselines, and authenticated bounded
graph traversal. The more complex retrieval channels remain opt-in after a
negative benchmark result. Persistent consolidation and optional adapters
remain planned.

Purpose: provide a continuous, local-first desktop assistant whose durable
memory and context policy belong to RAI while its LLM and reasoning strategy are
replaceable. This is a bounded interaction runtime, not an implementation of
GCAS, a Global Workspace or an autonomous goal loop.

The core request path is:

```text
explicit ConversationTurn or policy-approved proactive trigger
  -> privacy and interaction policy
  -> graph-memory retrieval
  -> finite ContextPackage plus ContextManifest
  -> bounded InferenceRequest
  -> replaceable AssistantModelBackend and ReasoningStrategy
  -> candidate response, memory proposals and capability proposals
  -> validation and policy
  -> one delivered response and separately committed durable records
```

**Assistant service and API replacement**

- [x] Introduce one container-owned `AssistantService` as the application
  boundary for text, voice and future desktop clients. Do not implement a
  second orchestration path per transport.
- [x] Define distinct immutable contracts for `ConversationTurn`, terminal
  `AssistantResponse` and the bounded `InferenceRequest` sent to a model
  backend. A normal conversational turn is not a `Task` merely because it
  requires inference.
- [x] Reserve `Task` for an explicit, trackable user goal or delegated unit of
  work. Conversation may lead to a proposed task, but neither a model response
  nor intent classification may silently create or authorize one.
- [x] Replace `ChatService`, legacy agent chains and the Antigravity-owned chat
  endpoint rather than preserving their behavior. CLI, REST or WebSocket chat
  surfaces that remain may change schema and semantics before 1.0.
- [x] Treat an `AssistantSessionId` only as interaction and audit grouping. Each
  model request starts from the explicit `ContextPackage`; provider conversation
  IDs and implicit server-side history are forbidden as state.
- [x] Define streaming, cancellation, deadlines, interruption and exactly-once
  terminal delivery independently of a model provider.

**Durable graph memory and reconstructed context**

- [x] Define immutable assistant memory records and provenance edges for user
  statements, preferences, conversation turns, assistant claims, observation
  and episode references, summaries, corrections, contradictions and
  supersession. Use RAI-native records; GCAS mappings remain optional adapters.
- [x] Keep the review/audit transcript logically separate from semantic memory.
  Assistant output is an unverified claim and never becomes a fact merely
  because it was generated or displayed.
- [x] Define a `MemoryGraphStore` protocol and a backend-independent schema
  before selecting a graph engine. Storage-specific identifiers, queries and
  executable rules must not leak into assistant-domain records.
- [x] Represent at least two logical timescales: a small, volatile working graph
  for the active interaction or explicit task and durable episodic/semantic
  graph memory. Promotion, consolidation, expiry and eviction are explicit,
  auditable operations.
- [x] Implement a lightweight local reference adapter, initially as a SQLite
  graph projection with stable node, hyperedge and provenance IDs, so tests and
  the default desktop profile require no additional database service.
- [/] Make SQLite the authoritative evidence and memory-operation store before
  adding learned extraction or alternative graph engines:
  - [x] retain immutable, policy-approved source turns and reference approved
    Rich History episodes through source IDs and observation provenance;
  - [x] store `MemoryProposal` evidence, admission decision and resulting state transition
    separately rather than treating extraction output as durable truth;
  - [/] preserve source spans, speaker, channel, modality, confidence, privacy,
    domain scope and model/policy versions for every derived claim;
  - [x] make the active MVP projection rebuildable from admitted records and
    operation logs;
  - [/] add approved Rich History evidence; complete policy/model-version fields
    remain open.
- [x] Extend `SQLiteMemoryGraphStore` with bi-temporal claims and relations:
  - [x] record transaction time (`recorded_at`, `expired_at`) separately from
    real-world validity (`valid_from`, `valid_until`);
  - [x] preserve obsolete facts for explicitly historical queries while excluding
    them from current-state context;
  - [x] represent `SUPPORTS`, `CONTRADICTS`, `UPDATES` and `SUPERSEDES` with their
    own provenance and policy eligibility;
  - [x] bind derived facts to source turns and preserve episode observation IDs.
- [x] Establish a simple, reproducible retrieval floor before graph traversal:
  - [x] index raw turns and claims with SQLite FTS5/BM25 and retrieve approved
    Rich History episodes through a bounded lexical evidence provider without
    duplicating their sensitive persisted payload into assistant storage;
  - [x] add query-driven pruning and independently bounded recent, raw-evidence and
    semantic-memory channels;
  - [x] compare raw chunks, extracted facts, summaries, dense retrieval, weighted
    RRF and bounded graph paths using the same retrieval and context budgets;
    the runner separates retrieval from answer-utilization failures. A versioned corpus
    covers personal, project, system, commitment, correction, abstention and
    privacy-isolation cases; the answerer uses the product backend contract and
    an independent deterministic judge. Two live local-model manifests record
    sensor-scoped energy and the complete evaluation configuration.
- [x] Add a tiered context router which selects recent/full context for short
  histories and escalates from summaries or claims to raw evidence when the
  selected tier is insufficient. Record the route, sufficiency decision and
  fallbacks in `ContextManifest`; tune thresholds empirically per model and
  workload rather than treating published thresholds as constants. The
  comparison covers short, medium and long histories against an always-memory
  baseline, and verified write-back is source-covered and provenance-linked.
- [ ] After the retrieval floor and adaptive router are reproducible, evaluate
  optional GWT-inspired context processing: one or a few bounded rounds of
  candidate eligibility, competition, admission, broadcast and release before
  `ContextPackage` assembly. Compare it with the router under identical memory,
  model and compute budgets; treat it as an attention/orchestration experiment,
  not a memory tier or a claim of GCAS conformance. See
  [Assistant memory, GCAS and GWT-inspired processing](docs/assistant-memory-gcas-crosscheck.md).
- [x] Add multi-channel retrieval incrementally:
  - [x] start with lexical FTS5/BM25 plus temporal and policy filters;
  - [x] benchmark a local dense embedding channel and Weighted Reciprocal Rank
    Fusion (RRF) before enabling either by default;
  - [x] defer a local cross-encoder because the cheaper hybrid channels show no
    quality gain that could justify added latency, memory and energy cost;
  - [x] add bounded graph traversal over the provenance-qualified subgraph with
    adversarial selection-integrity tests. Personalized PageRank remains a later
    experiment only if the bounded path baseline demonstrates a need.
- [ ] Add community summaries only after representative corpus-size benchmarks
  show that raw/claim retrieval and bounded graph paths are insufficient.
  Treat the community graph as a derived projection; evaluate incremental LPA
  against batch clustering without assuming constant update complexity.
- [ ] Benchmark an optional [FalkorDB](https://www.falkordb.com/) /
  [Graphiti](https://github.com/getzep/graphiti) adapter
  (`GraphitiMemoryGraphStore`) after the SQLite conformance floor is stable:
  - keep RAI records and operation semantics canonical; Graphiti entity and
    relation extraction remains an untrusted proposal source;
  - enforce the Privacy Firewall, `InferenceBudget`, `UsageLedger`, source-span
    provenance and the same retrieval budget used by SQLite baselines;
  - buffer model-assisted ingestion at episode boundaries or explicitly
    selected high-salience turns instead of requiring synchronous extraction
    for every message;
  - do not promote Graphiti to a supported default unless it beats the SQLite
    floor on quality and operational cost without weakening deletion, audit or
    selection-integrity guarantees.
- [ ] Research an [OpenCog AtomSpace](https://github.com/opencog/atomspace) memory adapter
  and a separate [Hyperon/MeTTa](https://github.com/trueagi-io/hyperon-experimental)
  reasoning sidecar:
  - Do not force Cypher/LPG abstractions onto AtomSpace; express bi-temporal validity,
    hyperedges and contradiction resolution as native MeTTa term-rewriting rules and
    Probabilistic Logic Networks (PLN) inside the isolated reasoning sidecar.
  - Ensure symbolic hypergraph evaluation runs outside the trusted RAI daemon and cannot
    bypass data classification, budgeting or policy gates.
- [ ] Record the storage, retrieval and reasoning choices in an ADR based on
  representative RAI workloads. SQLite, FalkorDB and AtomSpace are replaceable storage
  candidates; Hyperon/MeTTa is a separate reasoning experiment. None is an automatic runtime
  dependency.

The staged experiment design, evidence model and acceptance matrix are detailed
in [Assistant memory development plan](docs/assistant-memory-roadmap.md).

Primary memory and temporal graph references:

- Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan and Daniel Chalef,
  [*Zep: A Temporal Knowledge Graph Architecture for Agent Memory*](https://arxiv.org/abs/2501.13956),
  arXiv:2501.13956, 2025. Introduces bi-temporal validity windows (T, T'), 3-tier
  subgraphs (Ge, Gs, Gc), incremental community updates via Label Propagation, and
  parametric Cypher templates.
- Varun Pratap Bhardwaj, Garima Singh and Arun Pratap Bhardwaj,
  [*SuperLocalMemory 4.0: The Governed Memory Operating System for AI Agents*](https://arxiv.org/abs/2608.08253),
  arXiv:2608.08253, 2026. Demonstrates a governed, local-first memory OS over managed SQLite
  stores, multi-channel RRF retrieval, temporal candidate filtering, and strict falsifiable
  reliability invariants.
- Bernal Jiménez Gutiérrez et al.,
  [*HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*](https://arxiv.org/abs/2405.14831),
  arXiv:2405.14831, accepted to NeurIPS 2024. Introduces personalized PageRank (PPR) over
  knowledge graphs for zero-token multi-hop associative retrieval.
- Darren Edge et al.,
  [*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130),
  arXiv:2404.16130, 2024. Foundational architecture for hierarchical community summarization.
- [*TierMem: From Lossy to Verified -- The Sufficiency Principle for Agent Memory*](https://arxiv.org/abs/2602.17913),
  arXiv:2602.17913, 2026. Preserves an immutable raw-evidence tier and escalates
  from compact memory when it cannot answer the current query sufficiently.
- Heng Zhou et al.,
  [*LatticeMind: A Conflict-Aware Memory Primitive for Multi-Agent Systems*](https://arxiv.org/abs/2608.08236),
  arXiv:2608.08236, 2026. Evaluates explicit conflict status and deterministic
  write-time checks before model-assisted reconciliation.
- [*Manufactured Confidence: How Memory Summarization Can Create False Certainty in LLM Agents*](https://arxiv.org/abs/2606.29279),
  arXiv:2606.29279, 2026. Shows that lossy summaries can erase hedging and
  provenance, turning uncertain source material into confidently reused claims.
- [*Selection Integrity for LLM Graph Memory*](https://arxiv.org/abs/2606.12290),
  arXiv:2606.12290, 2026. Shows that untrusted graph topology can corrupt
  selection even when retrieved fact records themselves have valid provenance.
- [*Diagnosing Retrieval vs. Utilization in Long-Term Conversational Memory*](https://arxiv.org/abs/2603.02473),
  arXiv:2603.02473, 2026. Finds retrieval choice more consequential than several
  lossy write-time transformations and motivates a strong raw-chunk baseline.
- [*Beyond Memory Leaderboards: A Reproducible Protocol for Full-Text Scientific Recall*](https://arxiv.org/abs/2607.16848),
  arXiv:2607.16848, 2026. Demonstrates that ingestion granularity, raw-text
  preservation, modality and retrieval budget can invert adapter rankings.
- [x] Build every context from the current request, selected recent interaction
  records and relevant graph memories; reserve approved Rich History references
  for a later slice. Apply
  explicit token, character, item, privacy and latency budgets.
- [x] Persist a `ContextManifest` containing selected source IDs, exclusions,
  redactions, ranking reasons, policy/model versions and actual size so a reply
  can be reproduced and audited without retaining an opaque provider prompt.
- [x] Add deterministic consolidation, expiry, correction and deletion
  propagation. A model may propose memory candidates but cannot commit, delete
  or lower their privacy classification.

**Replaceable models and reasoning strategies**

- [x] Define a provider-neutral `AssistantModelBackend` and separate
  `ReasoningStrategy` contract. Report capabilities such as streaming, hidden
  states, KV cache, latent recurrence, J-lens and deterministic seeding instead
  of branching core code by provider name.
- [ ] Implement comparable `DIRECT`, `TOKEN_SCRATCHPAD` and
  `LATENT_RECURRENCE` strategies. Token scratchpads and latent state are
  ephemeral research data and are not displayed or persisted by default.
- [ ] Start latent recurrence with a fixed iteration count and hard compute/time
  ceilings. Add convergence-, confidence- or learned-halting policies only
  after fixed-step baselines and failure behavior are measured.
- [ ] Return a typed unsupported-capability result when a backend cannot expose
  hidden states or accept recurrent embeddings; do not silently emulate latent
  recurrence with tokens.
- [ ] Record a versioned run manifest with model, tokenizer, quantization,
  strategy, iteration policy, random seed, budgets, context manifest and
  observer artifacts.
- [ ] Apply `InferenceBudget` and `UsageLedger` to every strategy and backend.
  Count input/output tokens, latent iterations, retries, time and local compute;
  external APIs additionally report or conservatively estimate monetary cost.

**Coconut-style continuous latent inference**

`LATENT_RECURRENCE` specifically means a
[Coconut-style](https://arxiv.org/abs/2412.06769) chain of continuous thought,
not token recurrence, read-only J-lens observation or activation steering. At a
latent position the final hidden-state vector for the preceding position is fed
back as the next input embedding without the language-model-head -> token ->
embedding round trip. The recurrent vector, KV cache and raw activations remain
inside the neural process.

Primary references:

- Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston
  and Yuandong Tian,
  [*Training Large Language Models to Reason in a Continuous Latent
  Space*](https://arxiv.org/abs/2412.06769), arXiv:2412.06769, accepted to COLM
  2025. The paper introduces Coconut (Chain of Continuous Thought) and the
  direct hidden-state-to-input-embedding recurrence used by this roadmap.
- [Official Meta FAIR Coconut implementation](https://github.com/facebookresearch/coconut),
  including the staged curriculum-training and evaluation configurations used
  to reproduce the paper's GSM8K, ProntoQA and ProsQA experiments.

This execution mode requires a checkpoint trained for continuous thoughts. The
ability of a generic Transformers model to return hidden states or accept
`inputs_embeds` is not evidence that the checkpoint supports Coconut. An
untrained or incompatible checkpoint must return a typed unsupported result
rather than running an unvalidated feedback loop.

The existing separately startable neural sidecar becomes the single owner of
the model, accelerator, KV cache and latent vectors for both direct and Coconut
generation. Do not load a second copy of the same model in another daemon only
to provide latent inference. Inside that boundary, keep the transport contract,
the direct Transformers engine, a dedicated `CoconutEngine` and optional
read-only observers separable.

The sidecar exposes a transport-independent, versioned execution contract,
initially named `rai.latent.v1`, over protected Unix-domain-socket HTTP with
bounded streaming. Its minimum surface is:

```text
GET  /api/v1/latent/capabilities
GET  /api/v1/latent/models
POST /api/v1/latent/generate
POST /api/v1/latent/requests/{request-id}/cancel
```

The event union distinguishes lifecycle, hidden computation and visible output
without exporting the latent vector:

```text
GenerationStarted
LatentStepCompleted
NeuralStateObserved       # optional bounded observer output
TokenDelta
GenerationCompleted | GenerationFailed
```

RAI integrates this surface through an `AssistantModelBackend`; GAIA and other
processes use protocol clients rather than importing the engine. The caller
chooses the reasoning mode, fixed latent-step budget and optional observer. The
sidecar executes one bounded inner numerical recurrence and enforces stricter
local ceilings. When GAIA is the caller, GAIA retains ownership of the outer
Workspace and Cognitive Control process; RAI does not create a competing
cognitive controller.

- [ ] Record an ADR that freezes the first Coconut execution semantics: latent
  start/end markers, the exact returned hidden vector, final normalization,
  attention mask, position IDs, KV-cache updates and the transition back to
  ordinary token decoding.
- [ ] Define a versioned Coconut artifact manifest containing immutable base and
  trained checkpoint revisions, tokenizer and special-token IDs, training
  recipe, supported latent-step range, dtype/quantization compatibility and
  checksums.
- [ ] Extend the neural sidecar with a dedicated `CoconutEngine`, fixed-step
  execution, hard iteration/time/RAM/VRAM limits, cancellation between forward
  passes, typed incompatibility and no raw-tensor egress.
- [ ] Freeze `rai.latent.v1` conformance fixtures for capabilities, requests,
  streaming events, cancellation, terminal replay and failure taxonomy before
  implementing RAI, GAIA or third-party clients.
- [ ] Add a RAI assistant adapter and a separate GAIA protocol adapter. Both send
  explicit context and budgets; neither receives latent vectors, owns model
  state or bypasses the caller's memory, epistemic or capability policy.
- [ ] Keep Coconut curriculum training and checkpoint production in an explicit
  offline workflow. The serving sidecar loads verified artifacts and never
  starts training or downloads a model as an inference side effect.
- [ ] Compare `DIRECT`, `TOKEN_SCRATCHPAD` and `COCONUT_FIXED` on the same
  Coconut-compatible checkpoint, `ContextPackage`, seed and declared output and
  compute budgets. Start with fixed latent counts including zero as the direct
  control, and record answer/verifier quality, latency, energy, RAM/VRAM,
  stability, failure and cancellation behavior.
- [ ] Run every fixed-step comparison with observers disabled and with one
  read-only observer enabled. Observer mode must leave deterministic output
  unchanged and must not alter recurrence depth, halting or policy.
- [ ] Consider learned or confidence/convergence-based latent halting only after
  the fixed-step baselines are reproducible. Treat activation steering,
  ablation and patching as a separate post-observation intervention gate.

**Writable neural workspace slots**

Treat writable slots as another optional inference architecture for this
assistant, independent of GCAS, GAIA or any other cognitive runtime. The module
tests whether a small fixed-capacity neural workspace improves reasoning and use
of graph-retrieved context at a matched token and compute budget. It does not
define symbolic slot roles, a global-workspace controller or an alternative
durable memory system.

For a request or bounded inference session, let
`W[t] in R^(K x d_slot)` be a bank of `K` writable vectors. The model reads from
that bank while processing its residual stream and produces gated, bounded
updates for the next step or context segment. The residual stream carries the
current computation through model depth; slots carry selected state through
inference time. Slot identity is therefore not a token position, graph-node ID
or claim that a vector has a human-readable role.

Keep the graph store as the assistant's durable, auditable memory. A
`ContextPackage` may initialize or condition slots, and visible model output may
propose an ordinary memory update through the existing validation path, but the
slot tensor itself is ephemeral research state. Version 1 resets it at every
request boundary and never persists it between conversations. Longer-lived
neural state requires a separate privacy, isolation, deletion and contamination
review.

Implement the experiment as a `SlotWorkspaceEngine` beside the direct and
`CoconutEngine` implementations in the existing neural sidecar. The sidecar
owns slot tensors, accelerator state and any recurrent cache. Core assistant
code selects a declared `ReasoningStrategy` and sees only typed events, bounded
metrics and visible output. It must not import model-specific slot classes or
receive raw tensors.

Expose the engine through a separate versioned `rai.slots.v1` contract rather
than adding slot lifecycle semantics to `rai.latent.v1`. The initial surface is
request-scoped:

```text
GET  /api/v1/slots/capabilities
POST /api/v1/slots/generate
POST /api/v1/slots/requests/{request-id}/cancel
```

The request declares the architecture, slot count, update count, observer and
budgets. Streaming may report `SlotStepCompleted` and aggregate occupancy,
update-norm, attention-entropy or routing metrics, but never slot vectors. An
opaque internal handle and monotonically increasing version prevent stale
writes; neither is a durable memory identifier.

Evaluate three increasingly invasive variants instead of treating "slots" as
one mechanism:

1. `SLOT_TOKENS`: special memory/register tokens passed between bounded context
   segments. This is the cheapest implementation baseline.
2. `FAM_FEEDBACK`: selected hidden representations from one block become
   attention-accessible working memory for the next block without introducing
   a new learned cross-attention module.
3. `SLOT_CROSS_ATTN`: a separate bank read through cross-attention and updated
   by a gated writer/router. This is the target architecture discussed here,
   but it changes the model computation and requires adaptation training.

Relevant precedents are
[Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881), which passes
trained memory tokens between segments;
[TransformerFAM](https://arxiv.org/abs/2404.09173), which feeds latent
representations back as working memory without adding weights;
[Hymba](https://arxiv.org/abs/2411.13676), whose released models use learned
meta tokens; and
[MemoryLLM](https://arxiv.org/abs/2402.04624), which provides a much larger
self-updatable latent memory pool. They are comparison points, not evidence
that an ordinary causal-LM checkpoint already implements the proposed slot
semantics.

Training is an explicit experimental stage, not an inference side effect:

- Protocol, lifecycle, isolation and observer tests need no training. Released
  compatible checkpoints may also be evaluated unchanged as external
  baselines.
- `SLOT_TOKENS` and `FAM_FEEDBACK` may reuse pretrained weights, but require
  continued or task fine-tuning before their memory behavior can be interpreted
  as useful. A wrapper that merely recycles hidden states is a negative control.
- `SLOT_CROSS_ATTN` introduces slot initialization, read and write behavior and
  therefore requires at least parameter-efficient adaptation of those modules;
  full continued pretraining is considered only after the frozen-backbone or
  LoRA/adapter experiment passes its gate.
- Keep all dataset creation, training and checkpoint publication in a separate
  offline workflow. The sidecar loads a pinned, verified artifact manifest and
  never trains or downloads weights during serving.

- [ ] Record a slot-architecture ADR covering tensor shapes, insertion layers,
  initialization, read attention, writer/router, gating, normalization,
  detach/backpropagation policy, segment boundaries, reset and cancellation.
- [ ] Freeze `rai.slots.v1` schemas and conformance fixtures, including
  capability negotiation, unsupported checkpoints, stale handles, exactly one
  terminal event and proof that no state leaks between requests.
- [ ] Implement `SLOT_TOKENS` first and benchmark fixed `K` values such as 8,
  16 and 32 against `DIRECT` and `TOKEN_SCRATCHPAD` before adding new model
  modules.
- [ ] Add `FAM_FEEDBACK` as the first pretrained-checkpoint adaptation and
  measure zero-shot recycling as a negative control versus a reproducible
  parameter-efficient fine-tune.
- [ ] Implement `SLOT_CROSS_ATTN` with a frozen-backbone adapter experiment
  first. Train slot initializers, cross-attention and gated writer/router on
  next-token plus synthetic retention, overwrite, conflict and multi-step
  reasoning tasks.
- [ ] Evaluate every variant on the same base checkpoint, `ContextPackage`,
  graph-retrieval results, seed and output/compute budgets. Record answer and
  verifier quality, retrieval use, contradiction handling, latency, energy,
  RAM/VRAM, slot utilization, stability and cancellation behavior.
- [ ] Require causal ablations: shuffled or zeroed slots, frozen updates,
  read-only slots and equivalent extra context tokens. Reject a claimed memory
  benefit when a simpler token or compute-matched baseline explains it.
- [ ] Compare Coconut and slot recurrence separately. Attempt a combined
  `COCONUT_SLOTS` strategy only after both `COCONUT_FIXED` and at least one slot
  variant independently pass their acceptance gates.
- [ ] Keep semantic slot labels and durable cross-request neural state out of
  the first implementation. Consider them only after stable slot utilization
  and a measurable assistant-level benefit are reproduced.

**Read-only interpretability first**

- [ ] Generalize the NCSI/J-lens integration behind an
  `InterpretabilityObserver` with a no-op implementation and optional J-lens,
  logit-lens, probe or sparse-autoencoder adapters.
- [ ] Keep residual-stream tensors and latent slots inside the neural sidecar.
  Persist only bounded observations or explicitly approved research artifacts.
- [ ] Separate observation from intervention. Enabling a read-only observer must
  not change context selection, sampling parameters, halting or capability
  policy; steering requires a future, separately reviewed experiment contract.
- [ ] Compare strategies on identical model checkpoints, contexts, seeds and
  output/compute budgets. Measure answer quality, recall, contradiction
  handling, false-memory adoption, context efficiency, latency, energy and
  observer perturbation.

**Post-observation research gate: GW-like coordination and uncertainty**

- [ ] Use the read-only latent-recurrence, J-lens and slot experiments to test
  whether stable broadcast-, competition-, ignition- or workspace-like dynamics
  can be operationally identified. Do not assume in advance that they implement
  Global Workspace Theory.
- [ ] Compare an explicit GCAS Workspace mapping with alternative GWT-inspired
  and non-workspace controllers using falsifiable tasks and the same memory,
  model and compute budgets. Record negative results and ambiguous mappings.
- [ ] Define bounded neural/slot observations that can be related to working-
  graph nodes without treating a decoded label as the latent state's literal
  meaning or automatically creating a durable belief.
- [ ] Add an explicit uncertainty model for memories, retrieved claims and
  proposed conclusions. Evaluate a Bayesian evidence-update baseline with
  provenance, source reliability, temporal validity, contradiction and
  calibration; a model's self-reported confidence is evidence, not probability
  or authorization.
- [ ] If intervention is justified, place the experimental workspace/controller
  behind a separately startable protocol with no direct persistence or
  capability access. It emits proposals and observations through the same
  validation and policy path as every model.
- [ ] Require an ADR after the observation-only experiments to decide whether a
  useful GW-like process belongs in an optional RAI research module, an external
  cognitive runtime or nowhere. It must not emerge implicitly inside
  `AssistantService`.

**Continuous local assistance without continuous LLM inference**

- [ ] Feed the assistant from privacy-filtered Rich History observations and
  deterministic episodes. Collection may be continuous and opt-in; LLM
  inference is event-, schedule- or user-triggered and separately budgeted.
- [ ] Perform routine consolidation locally. Preserve
  `background_remote_tokens = 0` unless the user creates an explicit automation
  with its own data and cost policy.
- [ ] Require opt-in scopes, rate limits, quiet hours, deduplication and a visible
  reason for every proactive interruption. Merely observing a salient event is
  not permission to notify or act.
- [ ] Allow the model to emit typed response, memory and capability proposals
  only. All operating-system actions continue through `CapabilityService`,
  policy, approval and postcondition verification.

Implementation order:

1. Freeze conformance fixtures for `ConversationTurn`, `AssistantResponse` and
   `InferenceRequest` and minimal durable memory records. Implement
   `AssistantService`, a deterministic fake backend and a SQLite graph store.
   Store turn nodes and `REPLIES_TO` edges, admit an explicit user preference as
   a separate provenance-linked memory, and reconstruct every context from both
   a bounded recent reply-chain window and relevant current graph memories.
2. Complete the first user-visible memory test with one real local backend:
   remember a preference, restart into a session whose recent window excludes
   the source turn, retrieve the preference through graph memory, supersede it
   after a correction and prove the new value survives another restart. Retain
   a `ContextManifest` that distinguishes recent-turn and durable-memory inputs;
   do not call a turn-only chat an assistant MVP.
3. Connect policy-approved Rich History episodes and local text/voice clients.
4. Add Direct and token-scratchpad comparison;
   external model APIs arrive through the separate Stage 6 hybrid module.
5. Extend the neural sidecar with the versioned `rai.latent.v1` surface, a
   Coconut-compatible fixed-step engine and read-only observers; add
   reproducible experiment manifests and evaluations.
6. Add the independent `rai.slots.v1` surface and evaluate `SLOT_TOKENS`, then
   trained `FAM_FEEDBACK` and `SLOT_CROSS_ATTN`; do not combine them with
   Coconut until their separate ablations pass.
7. Benchmark optional Neo4j and AtomSpace memory adapters, the Hyperon/MeTTa
   reasoning sidecar and GraphRAG-style retrieval, then record the relevant
   ADRs.
8. Remove the obsolete chat path, add opt-in proactive triggers and only after
   the observation gate consider GW-like coordination or approved capability
   proposals.

Acceptance slices:

1. A local processor converts an episode into validated structured output while
   offline; processor failure does not affect the original episode.
2. A Polish or English push-to-talk request asks about recent activity and is
   answered locally through TTS without retaining raw audio.
3. An accessibility-poor test application triggers one consented, cropped visual
   inference; its result is marked uncertain and the capture is removed.
4. After a daemon restart, the assistant answers from selected graph memories
   and source references while the model backend receives no implicit provider
   conversation or full-transcript replay.
5. Correcting an older user statement supersedes it; the next context excludes
   the obsolete value, and deletion removes both source and derived retrieval
   entries.
6. Direct, token-scratchpad and `COCONUT_FIXED` runs consume the same recorded
   `ContextPackage`; a Coconut-incompatible checkpoint or unsupported strategy
   fails explicitly and does not change durable memory.
7. With a fixed seed and deterministic backend, enabling a read-only observer
   leaves the delivered response unchanged and stores no raw hidden-state tensor
   by default.
8. A request-scoped slot run resets its state on completion or cancellation,
   exports no raw vector and cannot affect the next request; its result is
   compared with a token- and compute-matched no-slot control.
9. A cancelled or interrupted turn emits at most one terminal result, commits no
   partial assistant claim as fact and cannot invoke a capability directly.
10. The same memory/context conformance suite passes against the SQLite reference
   adapter and at least one isolated candidate adapter before an alternative
   graph engine can become a supported profile.
11. A contradicted or weakly supported claim retains its evidence and calibrated
    uncertainty state; neither retrieval score nor model confidence silently
    promotes it to fact.

Required failure tests cover malformed output, model timeout, cancellation,
out-of-memory, unavailable accelerator, low-confidence speech, denied screen
capture, poisoned retrieved content, stale/superseded memory, observer failure,
incompatible Coconut artifacts, latent timeout/non-convergence, rejected
executable graph rules, unavailable optional graph backends, incompatible slot
artifacts, stale slot versions, cross-request slot leakage, slot-update
instability and attempted direct capability invocation.

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

### Stage 6 — Rich Hybrid AI and optional agent interoperability

Purpose: combine local inference with external model APIs and, independently,
allow difficult delegated tasks to use optional external cognitive runtimes or
agent harnesses without transferring ownership of memory, policy or the Linux
desktop.

Prerequisites: Stages 1, 3 and 5. A backend cannot be production-enabled until
usage accounting, cancellation and data-egress auditing work.

Hybrid LLM inference and delegated agent execution are separate flows and
modules. Both preserve RAI's product boundary:

```text
assistant turn
  -> deterministic trigger/router
  -> policy-filtered ContextPackage
  -> local or external AssistantModelBackend
  -> validated response and memory/capability proposals

bounded delegated task
  -> policy-filtered ContextPackage
  -> optional, separately packaged AgentBackend
  -> typed CapabilityRequest
  -> shared CapabilityService and PolicyEngine
  -> durable ActionResult or ActionFailure
```

This coordination is not a universal reasoning loop inside RAI. The selected
agent harness owns its internal reasoning. When it serves `AssistantService`,
RAI still owns canonical assistant memory and interaction semantics; backend
session metadata cannot replace them. RAI also owns observation, context
release, capability authority, verification and durable evidence.

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

#### 6.3 External model API backends

- [ ] Implement external LLM APIs through `AssistantModelBackend`, not
  `AgentBackend`. A model backend performs one bounded inference and receives no
  ambient tools, desktop access or ownership of assistant memory.
- [ ] Send the same versioned `ContextPackage` and reasoning-strategy contract to
  local and external models where capabilities permit. Unsupported latent or
  observer features fail explicitly rather than changing strategy silently.
- [ ] Disable provider-side conversation persistence when possible and never use
  a remote conversation ID as context. Record provider retention expectations in
  the outbound `ContextManifest`.
- [ ] Reconcile reported prompt, cached, completion and reasoning tokens with
  local estimates; treat absent or inconsistent usage as unknown and apply the
  conservative budget policy.
- [ ] Keep provider SDKs, credentials, retry rules and response normalization in
  optional adapters. Removing one provider must not change memory or public
  assistant-domain records.

#### 6.4 AgentBackend conformance

- [ ] Stabilize `AgentBackend` lifecycle, streaming, cancellation, usage,
  retryability, evidence and failure contracts.
- [ ] Package agent interoperability separately from assistant model backends;
  installing an external LLM API adapter must not install ACP, agent tools or an
  agent harness.
- [ ] Keep backend conversation/session IDs as adapter-owned metadata and never
  use them as the source of assistant memory or context.
- [ ] Add GAIA as the reference cognitive-runtime integration.
- [ ] Add one end-to-end external harness backend before multiplying providers.
- [ ] Retain Antigravity only if it passes the common contracts without
  process-wide patching or private runtime coupling.
- [ ] Make backend removal or outage preserve local history and capability state.

#### 6.5 Optional GAIA embodiment adapter

- [ ] Implement GAIA strictly as an adapter over the public Stage 2 event and
  Stage 1 capability contracts; do not import GAIA implementation modules into
  the RAI kernel.
- [ ] Convert a validated RAI `Observation` into an unverified GAIA Observation
  Cognitive Object with complete provenance and data-class metadata.
- [ ] Accept bounded GAIA `Action` requests only through the common capability
  registry, policy, approval and audit path.
- [ ] Return typed `ActionResult` or `ActionFailure` events through the durable
  journal and preserve one cognitive terminal result over retryable delivery.
- [ ] Keep GAIA/GCAS identifiers and session state inside the adapter rather than
  adding them to provider-neutral RAI records.
- [ ] Keep the generic observation channel separate from the optional
  NCSI/J-lens neural-state protocol.

The GAIA round-trip is an additional cross-project conformance test, not a
prerequisite for the Stage 2 event plane, Rich History or local-only operation.

#### 6.6 ACP and MCP roles

- [x] Expose the base typed capability registry through authenticated local MCP;
  MCP calls use the shared validation, policy, approval and audit path.
- [ ] Add an optional ACP client adapter after the base `AgentBackend` contract is
  stable.
- [ ] Map ACP session creation, prompt streaming, cancellation, plans and
  permission requests into RAI records without making ACP a security boundary.
- [ ] Integrate external agents with approved RAI capabilities through MCP or
  direct typed adapters; MCP remains behind the agent while ACP manages the
  agent session.
- [ ] Re-evaluate protocol-version compatibility at implementation time and keep
  protocol negotiation explicit.
- [ ] Add conformance fixtures that prove an ACP agent cannot bypass RAI policy or
  obtain ambient desktop context.

#### 6.7 Hybrid routing

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

Acceptance slices:

```text
assistant turn exceeds the selected local model envelope
  -> minimal ContextPackage and outbound manifest are built
  -> provider and token/cost budget policy is applied
  -> one external AssistantModelBackend inference runs without tools
  -> response proposals and exact or conservatively estimated usage return
  -> assistant memory remains local and provider-independent

user asks for a complex task
  -> local router marks it out of local scope
  -> ContextPackage and outbound manifest are built
  -> policy/budget decision and optional approval
  -> one optional external AgentBackend executes
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
  -> receives one policy-approved RAI capability request
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

#### 9.4 System packaging and operations

Python developer previews are delivered by Release gate A. This package covers
operating-system integration and production operations rather than postponing
all distribution work until Stage 9.

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
Rich Assistant reconstructs bounded context from durable graph memory
Rich Actions cannot bypass policy and verifies consequential results
Rich Hybrid AI exposes every model transfer and enforces token/cost budgets
Rich Agent Interop remains optional and cannot bypass capability policy
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

### Scenario F — durable assistant conversation

1. The user states a project preference, ends the client and restarts RAI.
2. A later question retrieves that preference from graph memory with provenance;
   no model-provider conversation is resumed.
3. The user corrects the preference and the prior record is marked superseded
   rather than silently overwritten.
4. Direct and latent-recurrence backends receive the same bounded context and
   produce separate, reproducible run manifests.
5. An optional J-lens observer records bounded neural observations without
   changing the response or acquiring authority over memory and capabilities.

## Legacy chat replacement and removal gates

No supported consumer currently depends on the legacy chat endpoint. RAI is
pre-1.0, so the replacement may intentionally break its CLI, REST, WebSocket,
configuration and persistence semantics. Do not build a compatibility facade or
migration layer without a concrete consumer requirement.

1. Freeze tests for the new `AssistantService`, graph-memory and reconstructed-
   context semantics before deleting the old implementation.
2. Make the application container and typed registry the only capability path;
   assistant model output remains a proposal rather than an invocation.
3. Replace legacy chat handlers with the new service or remove unused transports
   outright. Update OpenAPI, CLI help, configuration and tests in the same
   breaking change.
4. Remove provider-owned conversation resume, legacy agent chains and the simple
   transcript store once their replacement acceptance slice passes restart and
   cancellation tests.
5. Retain Antigravity only if a current research or product need justifies a
   conforming backend that accepts explicit `ContextPackage` values and cannot
   invoke tools outside `CapabilityService`; otherwise remove it.
6. Keep audit transcripts, Rich History and semantic graph memory as distinct
   stores with explicit retention and deletion behavior.
7. Document development-database reset or one narrow migration only when needed
   for current test data; there is no general pre-1.0 compatibility promise.
8. Remove all obsolete compatibility code before treating Rich Assistant as a
   completed product outcome.

## Technical debt carried beyond Stage 1

- `cli_compatibility.py` and `config_manager.py` remain oversized compatibility
  modules and still mix some I/O and UI concerns.
- Antigravity compatibility still owns the legacy chat execution path and is
  scheduled for incompatible replacement by `AssistantService`, not long-term
  preservation behind another facade.
- Conversation history is still a flat transcript and has more implementation
  weight than normalized observation, graph memory and action-result state.
- The processor supervisor and bounded-task contracts are container-owned and
  tested, but no daemon API or registered capability dispatches work to them and
  neither production adapter has a repeatable live-model acceptance test.
- IREE and ONNX are intentionally frozen behind typed failures until they have
  owners, runtime availability and conformance tests.
- Device identity and cross-device reconnect/spooling contracts do not yet
  exist. Local event replay and bounded backpressure are already implemented in
  Stage 2.
- Full style linting contains legacy violations; critical lint is blocking now.
- GitLab issues #8, #9, #16 and #17 were resolved by the Stage 4.1 supervisor
  and Stage 4.2 bounded-result work. Issues #2 and #6 require reproduction
  against the new contracts before implementation.

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
