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

## Assistant memory and model output

The supported local assistant reconstructs every inference from a bounded
`ContextPackage`; provider conversation IDs are not canonical state. The old
provider-owned CLI path is no longer the standalone default, and its HTTP/SSE/
WebSocket endpoints return `410 Gone` unless `legacy_chat.enabled` is explicitly
set for compatibility.

The backend keeps the configured system instruction in the sole `system`
message. Retrieved memories, episodic records, grounded summaries and external
evidence are rendered as explicitly labelled, untrusted user-role data. Stored
turns cannot introduce a `system` role. This role separation is defense in
depth against persisted prompt injection; retrieved text never gains instruction
authority merely because it was selected as context.

Assistant output and memory proposals are untrusted. Only deterministic policy
admits the current MVP's bounded personal attributes, preferences, plans and
explicit memory requests. It requires an exact source span, rejects quoted,
hearsay and hedged candidates from automatic admission, preserves provenance to
the user turn and prevents a proposal from lowering its source privacy class.
`SECRET` and `BLOCKED` turns are rejected. Default retrieval allows `PUBLIC` and
`LOCAL` records only and revalidates profile, temporal validity, privacy and
source existence even when a storage adapter returns a candidate. Selected and
excluded IDs are recorded in the context manifest.

Remember, supersede, reject and forget attempts are append-only typed operation
records. Source deletion securely erases the source-derived memory payload and
graph edges; the operation log retains only opaque IDs needed to prove that an
obsolete memory is not reactivated.

Assistant SQLite and JSONL files are created in mode-`0700` directories with
mode-`0600` files; SQLite secure deletion is enabled. Unlike Rich History, this
first assistant store is not independently encrypted. Its threat boundary is
therefore the protected local Unix account. Use an isolated `RAI_DATA_DIR` for
experiments and remove that profile when its retention is no longer wanted.

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
`SECRET` and `BLOCKED` text is never eligible for remote synthesis. The default
runtime registers the actuator through `CapabilityService`, so every CLI, REST
or MCP invocation receives the normal destination, disclosure, cost and side
effect policy decision and audit record.

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

The optional bounded-inference result cache is disposable local acceleration,
not durable memory. Its key stores no prompt or normalized input, and reuse is
isolated by the current policy, model artifact, task contract, prompt,
classification, approved source identity and caller constraints. Cached JSON is
validated again before a new provenance-bearing claim is created. The cache
database has mode `0600`, a bounded capacity and TTL, but its validated derived
payload is not field-encrypted. This is a documented residual risk under the
trusted local Unix-account boundary. Operators who do not want derived private
data in the XDG cache must set `local_ai.result_cache.enabled` to `false` and
remove the disposable cache database.

## Reporting vulnerabilities

Do not include secrets or exploit details in a public issue. Contact the project
maintainer privately, then coordinate a disclosure and credential rotation plan.
