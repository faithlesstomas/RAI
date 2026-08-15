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

## Command execution

- Bubblewrap is preferred, Guix is the fallback.
- Presence of the `bwrap` binary is not sufficient; RAI probes whether user
  namespaces actually work.
- If no supported sandbox is operational, execution is refused.
- Network access is disabled unless explicitly requested and approved.
- The workspace is mounted read-only and a dedicated output directory is the
  only normal write mount.

Static command patterns are defense in depth, not the primary boundary. Future
work will replace ad-hoc risk checks with a structured `PolicyEngine`.

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

