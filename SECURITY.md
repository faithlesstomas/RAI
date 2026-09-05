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

## Credentials and private data

Never store tokens or model-provider keys in the repository. Use environment
variables, a protected `.env`, desktop keyring integration, or another secret
store. If a real credential is ever written to a loose workspace file, rotate it
even if Git reports that the file was untracked.

The planned activity collector must provide pause, exclusion, retention,
redaction and deletion controls before it is considered usable.

## Reporting vulnerabilities

Do not include secrets or exploit details in a public issue. Contact the project
maintainer privately, then coordinate a disclosure and credential rotation plan.
