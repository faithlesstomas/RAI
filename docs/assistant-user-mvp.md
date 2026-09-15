# Rich Assistant user MVP

The supported assistant is one RAI-owned runtime shared by CLI, REST and native
WebSocket clients. Conversation history, semantic memory and exact model context
remain local and do not depend on a provider conversation ID.

## CLI workflow

Start or resume an interactive conversation:

```bash
uv run rai assistant chat --backend llama --model /path/to/chat-model.gguf \
  --session-id work
```

Inside chat, `/history`, `/context`, `/memories`, `/operations` and
`/diagnostics` inspect persisted state. Ordinary conversation is the primary
memory input: with a local model backend, a separate schema-constrained pass
proposes typed claims from the complete user turn and deterministic policy
decides whether they may be stored. `/remember TEXT` and `/forget TEXT` remain
optional explicit controls, not a prerequisite for useful memory. Reusing
`--session-id work` resumes its reply chain; memory is profile-scoped and can
therefore be recalled in a different session.

Inspection commands do not load the model:

```bash
uv run rai assistant sessions
uv run rai assistant history --session-id work
uv run rai assistant context --session-id work
uv run rai assistant memories
uv run rai assistant operations
uv run rai assistant diagnostics
```

## REST clients

Run `uv run rai serve`, then authenticate in the same way as other `/api/*`
control endpoints. The assistant surface is:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/assistant/turn` | complete one turn |
| `POST` | `/api/v1/assistant/stream` | deliver a committed response as SSE |
| `GET` | `/api/v1/assistant/sessions` | discover resumable sessions |
| `GET` | `/api/v1/assistant/sessions/{id}/turns` | inspect chat history |
| `GET` | `/api/v1/assistant/sessions/{id}/context` | inspect latest exact context |
| `GET` | `/api/v1/assistant/contexts/{manifest_id}` | inspect context by manifest |
| `GET` | `/api/v1/assistant/memories` | inspect active memory |
| `GET` | `/api/v1/assistant/memory-operations` | inspect mutation/admission trace |
| `GET` | `/api/v1/assistant/diagnostics/memory` | check projection integrity |

A turn request is provider-neutral:

```json
{
  "prompt": "Mój ulubiony kolor jest zielony.",
  "session_id": "work",
  "request_id": "client-generated-id",
  "data_class": "LOCAL"
}
```

Reuse a `request_id` only to retry that exact request. The terminal result
contains the assistant turn, context manifest, admitted memory IDs and memory
operation IDs.

## Native WebSocket

Connect to `/api/v1/assistant/ws`, supplying the API token through the normal
authorization header or `?token=` query parameter. Optional `?session_id=work`
resumes a known session. The server first sends:

```json
{"type": "session", "session_id": "work"}
```

Send the same JSON object accepted by `POST /turn`. A successful reply is:

```json
{"type": "response", "payload": {"session_id": "work", "content": "..."}}
```

Validation and service failures use `{"type":"error","error":{...}}` and do
not create an alternative persistence path.

## How memory is assembled

RAI keeps three distinct kinds of context instead of flattening them into one
model-written summary:

1. the recent reply chain for conversational continuity;
2. raw, policy-eligible evidence recovered from earlier user turns with
   FTS5/BM25 or from Rich History episodes;
3. admitted claims with exact source spans, modality, confidence, scope,
   provenance and separate real-world and transaction-time intervals.

For each answer the adaptive router first checks whether a compact claim is
sufficient. If it is not, it falls back to raw evidence. If neither route has
support, the answerer is instructed to abstain. The persisted context manifest
records the selected route, rejected routes, sufficiency score, fallback,
ranking reasons, source IDs and character budget.

Corrections retain their earlier state for historical queries and create both
`CONTRADICTS` and `SUPERSEDES` relations. Ambiguous conflicts cannot silently
replace a claim. Deleting a source removes its raw retrieval entry and derived
state without reactivating an older value.

## Current boundary

This is an evidence-first general-memory increment, not human-like memory.
Quoted, hearsay, uncertain, malformed and source-less candidates fail closed.
The current retrieval floor is lexical; local dense retrieval, equal-budget
benchmarking, domain-aware consolidation and optional graph-store adapters are
still Issue #35 work. Interactive latency depends on the selected local model
and hardware; deterministic mode exists for conformance tests, not as a chat
model.
