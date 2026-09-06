# Rich History

Rich History is RAI's local, read-only desktop activity pipeline. It accepts
bounded semantic metadata from GNOME, AT-SPI, foreground-process, approved-root
filesystem and browser bridges. It does not capture screenshots, raw input,
typed characters, audio, process arguments, environment variables or document
contents.

## Privacy boundary

Collection is opt-in and disabled initially. `pause`, session lock and emergency
stop cancel every supervised collector immediately. Before an event can enter
the Stage 2 journal, the deterministic privacy firewall evaluates session,
application, browser origin, filesystem path and accessibility field role. Its
versioned decision is one of `DROP`, `METADATA_ONLY`, `REDACT` or `ALLOW` and is
included with persisted evidence.

Password/secret fields, private browser windows, excluded applications,
disallowed origins and paths are dropped. Communication applications and
banking, health or authentication origins are metadata-only. Credential,
email-address and payment-number patterns are redacted. Dropped values are not
included in errors or logs. Web and accessibility text remains untrusted data;
the history pipeline cannot invoke a capability.

The complete records are encrypted with AES-256-GCM. The default key provider
obtains a per-user key through Secret Service (`secret-tool`). If Secret Service
is absent, locked or refuses the key, Rich History fails closed and persists
nothing. The database and its parent directory use restrictive permissions and
a `CACHEDIR.TAG` excludes the history directory from backup by default. An
explicit embedded deployment can inject another `KeyProvider`.

## Deterministic processing

Repeated events are debounced, simultaneous evidence is fused with all source
IDs retained, and conflicts lower confidence and remain explicit. The
versioned episode builder splits activity on idle/time and project boundaries.
It requires no model and consumes zero remote tokens. Deleting a time range
removes source observations from both history storage and the event journal,
cascades through episodes, derived memories and outbound-context references,
then reproducibly rebuilds remaining episodes.

## Local API

All routes use the same authentication boundary as other `/api/*` endpoints:

- `POST /api/v1/activity/collection` — `pause`, `resume`, `lock`, `unlock`,
  `emergency_stop` or `clear_emergency_stop`;
- `GET /api/v1/activity/collectors` — lifecycle, permission, last-event, error
  and restart status;
- `POST /api/v1/activity/observations` — validated semantic adapter input;
- `GET /api/v1/activity/episodes` — time/application/project/resource/activity
  filters and reviewable provenance;
- `GET /api/v1/activity/answer?question=what_was_i_working_on` — deterministic
  summary without an LLM;
- `DELETE /api/v1/activity/episodes` — time-range deletion with residual-link
  verification.

GNOME shell, AT-SPI and browser integrations connect to the bounded adapter
contracts in `rai.history.collectors`. They should run unprivileged in the user
session and emit only the normalized fields described by `SourceEvent`.
Production bridges are registered explicitly as command arrays and run as
separate processes without a shell or inherited credential variables:

```yaml
rich_history:
  enabled: true
  collectors:
    gnome:
      command: ["rai-gnome-history-sidecar"]
      permission: GRANTED
    atspi:
      command: ["rai-atspi-history-sidecar"]
      permission: GRANTED
    filesystem:
      command: ["rai-filesystem-history-sidecar", "/home/me/projects"]
      roots: ["/home/me/projects"]
```

Each sidecar writes one bounded `SourceEvent` JSON object per stdout line.
Unexpected source identities, oversized or invalid records and non-zero exits
are isolated by the supervisor and reflected only as typed health state; raw
stderr and untrusted event content never enter daemon logs.
