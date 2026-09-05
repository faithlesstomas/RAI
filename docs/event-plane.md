# Durable local event plane

Stage 2 connects collectors and the Stage 1 capability path through a local,
provider-neutral journal. It does not require GAIA, GCAS, Antigravity, J-lens,
network model access or model weights.

## Journal contract

`SQLiteEventJournal` stores schema-validated `Observation`, `PolicyDecision`,
`ActionResult`, `ActionFailure` and `UsageRecord` values in envelopes described
by [`schemas/rai.events.v1.schema.json`](../schemas/rai.events.v1.schema.json).
`PolicyDecision` envelopes are the journal's typed audit events; detailed
capability audit entries remain in the append-only audit ledger.
The enclosed Stage 1 record is unchanged. Each committed envelope has a durable
monotonic sequence, opaque next cursor, acceptance timestamp, clock source,
clock uncertainty and canonical-content SHA-256.

Every envelope has a `data_class`. It is inherited from observations and policy
decisions. Event types without an embedded class, including terminal and usage
records, must supply it explicitly; omission and any mismatch fail closed.

An identical retry of a `record_id` returns the original envelope. Reusing that
ID for different canonical content returns `CONFLICT`. Source timestamps do not
control journal order. `SECRET` and `BLOCKED` observations are rejected before
persistence. The default encoded event limit is 256 KiB and replay pages are
limited to 100 records.

SQLite uses WAL mode, foreign-key checks and `synchronous=FULL`. Append and ACK
return only after commit. An interruption before commit leaves no accepted
event or checkpoint; after commit, retry recovers the original sequence.
Consumer ACK positions are independent and monotonic. Retention requests are
durable hooks: this backend does not delete journal evidence.

## Local transports

The authenticated HTTP API is:

- `POST /api/v1/events` — ingest one kernel event;
- `GET /api/v1/events?cursor=...&limit=...` — bounded ordered replay;
- `GET /api/v1/events/subscriptions/{consumer_id}` — bounded pull subscription
  from the consumer's durable position;
- `POST /api/v1/events/consumers/{consumer_id}/ack` — advance that position.

The daemon also starts a mode-`0600` Unix socket at the XDG runtime path
`events-v1.sock`. It accepts newline-delimited JSON `ingest`, `replay` and `ack`
operations. Both transports require the same per-user API token and use the
same validation service. RAI opens no Stage 2 trusted-LAN listener.

In-process subscribers have bounded queues. A slow consumer receives a typed
`OVERLOADED` failure and is disconnected without blocking appenders or other
consumers.

## Deterministic acceptance subscriber

The local Stage 2 subscriber maps only the synthetic `person_present`
observation to the conformance capability `test.echo`. It constructs a bounded
`CapabilityRequest`, invokes it through `CapabilityRegistry`, `PolicyEngine`,
cancellation and audit, then commits the policy decision and exactly one typed
terminal result before acknowledging the source cursor. It is a conformance
fixture, not a production desktop actuator.
