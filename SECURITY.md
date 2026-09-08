# RAI security model

RAI gives models and agent harnesses access to Linux capabilities. Model output,
collector input, remote backend output and MCP requests are therefore untrusted.

## Trust boundaries

- The local Unix user and their protected XDG directories are trusted.
- Agent backends, prompts, retrieved content and generated commands are not.
- A sandbox reduces host authority but is not treated as a perfect security
  boundary against kernel vulnerabilities.
- HITL approval confirms user intent; it does not make a dangerous command safe.

## Control API authentication

All `/api/*` HTTP endpoints and the chat WebSocket require a per-user token. RAI
uses `RAI_API_TOKEN` when set, otherwise it creates a mode-`0600` token in its
runtime directory. Trusted local CLI clients read the same token.

`RAI_DISABLE_AUTH=1` disables the boundary and must only be used in isolated
tests. The daemon binds to loopback by default. Binding to a non-loopback address
requires transport security and access controls outside the current Stage 0
scope.

CORS is disabled by default. Set a comma-separated `RAI_CORS_ORIGINS` only for
trusted local frontends.

Stage 2 event ingest, replay and acknowledgements use the same token boundary.
The additional Unix-domain socket is created inside the protected XDG runtime
directory with mode `0600`; it is not a network listener. Event size, replay
batch and subscriber queue bounds fail closed with typed errors. `SECRET` and
`BLOCKED` observations are rejected before the durable journal.

## Command execution

- Bubblewrap is preferred, Guix is the fallback.
- Presence of the `bwrap` binary is not sufficient; RAI probes whether user
  namespaces actually work.
- If no supported sandbox is operational, execution is refused.
- Network access is disabled unless explicitly requested and approved.
- The workspace is mounted read-only and a dedicated output directory is the
  only normal write mount.

Static command patterns are defense in depth, not the primary boundary. Future
work may retire them after equivalent negative coverage exists. The structured
`PolicyEngine` is authoritative for typed capability invocation: it validates
the actor, data class, target, declared side effects, isolation, budget and
verification plan before returning `ALLOW`, `ASK`, `DENY` or `ESCALATE`.

The compatibility calculator does not execute Python expressions. It parses a
small allowlisted arithmetic syntax and bounds expression length, AST size,
collection cardinality, numeric magnitude and exponent size. Attribute access,
imports, comprehensions and calls outside the explicit function allowlist are
rejected. The reviewed Ruff security profile is a blocking CI check for this
and other Python security boundaries.

`SECRET` and `BLOCKED` capability requests are denied. Critical-risk requests
are denied by the Stage 1 policy, moderate/high-risk requests require approval,
and unavailable approval or required isolation fails closed. A model-supplied
backend/private tool name has no authority unless it resolves in the common
`CapabilityRegistry`.

Every resolved request writes a decision event and one typed terminal event to
the protected XDG data audit ledger. A ledger failure prevents invocation. The
audit includes the policy version, approval identifier where applicable and the
final `ActionResult` or `ActionFailure`.

Google Antigravity SDK access is confined to `rai.backends.antigravity`. Its
private conversation fields are compatibility implementation details and never
appear in public runtime records.

## Voice data and synthesis

Speech text and audio are data, not control authority. The Stage 4.3 synthesis
boundary keeps generated PCM in memory and returns only backend/model identity,
language and a playback receipt. Neither the input text nor PCM bytes are copied
into its terminal result. Real player and backend implementations must preserve
that rule and report sanitized typed failures.

The local profiles do not contain remote fallbacks. Selecting a remote backend
requires a profile that permits remote execution, an explicitly authorized
request and trusted runtime composition with remote synthesis enabled. The
default actuator composition keeps remote execution disabled, so an
`allow_remote` argument supplied by an untrusted caller is insufficient.
`SECRET` and `BLOCKED` text is never eligible for remote synthesis. Once the
actuator is connected to the runtime, every invocation must additionally pass
through `CapabilityService` so destination, disclosure, cost and side effects
receive the normal policy decision and audit record.

Microphone capture and STT are not implemented yet. Their planned boundary is
push-to-talk, no raw-audio persistence by default and policy-controlled
transcript retention. Voice input alone will not approve high- or critical-risk
actions.

## Credentials and private data

Never store tokens or model-provider keys in the repository. Use environment
variables, a protected `.env`, desktop keyring integration, or another secret
store. If a real credential is ever written to a loose workspace file, rotate it
even if Git reports that the file was untracked.

Rich History collection is opt-in. Its pre-persistence firewall drops password
and secret roles, private browser sessions, excluded applications, origins and
paths; sensitive profiles become metadata-only and configured patterns are
redacted. Dropped values are not logged. Raw input, typed characters,
screenshots, audio, process arguments and document content are not collected.
GNOME, AT-SPI, foreground-process and filesystem producers run as separate
unprivileged sidecars launched without a shell. Their environment allowlist
omits RAI, API and model credentials, stderr is discarded, and the daemon
validates and sanitizes every bounded JSON record again. The filesystem
producer requires explicit non-root paths, does not follow symlinks and reads
names and mtimes without opening file contents. The process producer reads only
the focused PID's `/proc/<pid>/exe` basename; it never reads `cmdline` or
`environ`.

Complete history records are authenticated-encrypted with a per-user AES-256-GCM
key obtained through Secret Service. If the key is unavailable, history fails
closed. Time-range deletion covers the event journal, observations, episodes,
derived-memory links and outbound-context references and reports whether any
reference remains. Database metadata needed for local filtering (timestamps,
application, project, resource and activity type) is not field-encrypted; it is
protected by the local Unix account, mode-`0600` files and the isolated user
service. This leakage is a documented residual risk.
Private observation bodies are not duplicated into the plaintext event journal;
the journal retains only their source, kind, record identifier and privacy
decision. Browser origin/URL disagreement and remote or policy-excluded
`file://` resources fail closed. Sidecar stdout buffering is capped before JSON
parsing, and timestamps beyond the permitted local future skew are rejected.
Privacy deletion enables SQLite secure-delete semantics and truncates WAL data
after the encrypted store and event journal commit their removals.

## Reporting vulnerabilities

Do not include secrets or exploit details in a public issue. Contact the project
maintainer privately, then coordinate a disclosure and credential rotation plan.
