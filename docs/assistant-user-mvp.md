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
`/diagnostics` inspect persisted state. `/remember TEXT` explicitly stores a
fact. `/forget TEXT` removes matching active memory and `/forget all` clears the
active profile projection. Reusing `--session-id work` resumes its reply chain;
memory is profile-scoped and can therefore be recalled in a different session.

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

## Current boundary

This is a useful personal-memory MVP, not general human-like memory. Natural
attribute, preference and plan patterns are deliberately conservative. Quoted,
hearsay, uncertain and malformed candidates fail closed. FTS/dense retrieval,
bitemporal claims, Rich History ingestion and model-assisted extraction remain
future Issue #35 milestones. Interactive latency depends on the selected local
model and hardware; deterministic mode exists for conformance tests, not as a
chat model.
