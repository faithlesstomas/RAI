# Rich AI (RAI)

> **Pronunciation:** /raɪ/ — like *rye* or *sky*; in Polish it sounds close to
> *raj* (English: *paradise*).

> **The secure, local-first integration layer between AI assistants and Linux.**

Rich AI lets AI assistants and agents work with a Linux system without giving
them ambient, unrestricted access to it. Models can propose what should happen;
RAI owns what they may observe, which typed capabilities they may invoke, when
the user must approve an action, how it is isolated, and how its result is
verified and audited.

RAI is neither another agent framework nor a model server. It is the durable
system boundary underneath replaceable local models, cloud assistants and agent
harnesses: one place for Linux perception, private state and memory, capability
policy, human approval and action provenance.

The project is in an architectural transition. The daemon, CLI, MCP gateway,
desktop adapters, history store, sandbox and HITL foundations exist today.
Provider-neutral kernel records, runtime ports, capability policy and audit
contracts are implemented. Perception collection and deterministic episode
building are implemented by the opt-in Rich History slice. A lifecycle-managed
local processor, six schema-constrained task contracts and a policy-aware result
cache are also implemented, although live-model acceptance and runtime routing
remain open. The RAI-native assistant, graph memory and hybrid routing are
roadmap work. GAIA, GCAS compatibility, Google Antigravity, Coconut-style latent
inference, writable slots and J-lens are optional integrations or research
paths; none is required by the core product.

The repository and import namespace remain `rai`. The Python distribution is
published on PyPI as `rich-ai`, so installation and imports intentionally
use different names: `pip install rich-ai`, then `import rai` or run `rai`.

## Design principles

- **Local first:** observation, routine classification, state and memory stay on
  the machine by default.
- **State outside the model:** an LLM context is a disposable view, not the
  runtime's memory or source of truth.
- **Provenance over prompting:** claims should point to observations, tool
  results, user statements, memories or explicit inferences.
- **Capabilities, not arbitrary authority:** every system action passes through
  one capability registry and one policy boundary.
- **Fail closed:** unavailable isolation or missing authorization disables a
  privileged action; it never silently falls back to host execution.
- **Provider independence:** local inference, GAIA, Codex, Claude, Gemini and
  other harnesses integrate through optional backend contracts; disabling all
  of them must not break the local runtime.
- **Vertical slices:** new architecture must be demonstrated end-to-end before
  broad refactors or additional user interfaces are added.

## Target architecture

```text
 CLI · Voice · GNOME · Emacs · future desktop clients
                         │
              MCP / HTTP / Unix socket
                         │
┌────────────────────────▼─────────────────────────┐
│                 RAI Local Runtime                │
│                                                 │
│ AssistantService (planned)                      │
│ ConversationTurn ─ Context Builder ─ Graph Mem. │
│          │               │                      │
│          │       AssistantModelBackend          │
│          │        local · external API          │
│          │                                      │
│ Capability Registry ─ Policy ─ HITL ─ Audit     │
│          │                                      │
│          ├── desktop / shell / files / process  │
│          │                                      │
│ Event Bus ─ Episode Builder ─ State & Memory    │
│     ▲                 │                         │
│ Collectors       Local Cognition                │
│ GNOME/AT-SPI     classify/extract/summarize     │
│                       │                         │
│              Task & Escalation Router           │
└────────────────────────┬─────────────────────────┘
                         │ AgentBackend
                 GAIA · external harnesses
```

See [docs/architecture.md](docs/architecture.md) for component boundaries and
[ROADMAP.md](ROADMAP.md) for the implementation sequence.

## Current capabilities

- FastAPI daemon and interactive/local CLI.
- MCP, REST and WebSocket interfaces.
- immutable versioned kernel records and a published JSON Schema.
- one typed capability registry shared by CLI, REST, MCP and internal backends.
- deterministic `ALLOW`, `ASK`, `DENY` and `ESCALATE` policy decisions with an
  append-only local audit ledger.
- agent template and conversation-session separation.
- SQLite conversation history independent of the active backend.
- GNOME and COSMIC adapters for the currently implemented desktop operations.
- Bubblewrap or Guix command isolation with fail-closed fallback.
- HITL approval broker.
- per-user token authentication for `/api/*` and WebSocket control interfaces.
- opt-in Rich History with isolated GNOME/AT-SPI/process/filesystem collectors,
  pre-persistence privacy filtering, encrypted episodes, provenance queries and
  verified deletion.
- experimental Antigravity chat compatibility.
- container-owned local processor supervision, six bounded task contracts and a
  persistent policy-aware cache; Ollama and llama.cpp still require repeatable
  live-model acceptance.
- policy-controlled local `speech.synthesize` playback through Piper, exposed by
  the default capability registry and MCP when a local voice is provisioned.
- a testable Rich Assistant MVP with RAI-owned SQLite graph memory, bounded
  reconstructed context, llama.cpp/Ollama backends and inspectable manifests.

This list is deliberately narrower than the target architecture. In particular,
the graph-memory MVP currently admits one narrow class of explicit code-language
preferences; general semantic-memory extraction and hybrid model routing remain
planned. The old provider-owned `ChatService` is disabled by default and remains
only as an opt-in compatibility facade.

## Installation

The base `rich-ai` package can be installed from PyPI (pre-releases included):

```bash
python -m pip install --pre rich-ai
```

To install from source for development (Python 3.10 or newer and `uv` are required):

```bash
git clone https://gitlab.com/tk-lab1/ai/rai.git
cd rai
uv sync --extra dev
```

Optional integrations are installed explicitly:

```bash
uv sync --extra gnome-tools
uv sync --extra antigravity
uv sync --extra inference-llama
uv sync --extra inference-onnx
uv sync --extra inference-jlens --group jlens-reference
uv sync --extra tts
```

The `jlens-reference` group is development-only because the pinned upstream
reference implementation is not published on PyPI. It is deliberately absent
from `rich-ai` package metadata.

Model weights do not belong in the repository. Store them under an XDG data or
cache directory and keep only a reproducible manifest/checksum in version
control.

### Try the local graph-memory assistant

Install the llama.cpp extra and pass a chat-capable GGUF model:

```bash
uv sync --extra inference-llama
uv run rai assistant chat --backend llama --model /path/to/model.gguf --show-context
```

When the current working tree contains `models/*.gguf`, `--backend` and
`--model` may be omitted; RAI prefers a filename containing `chat`. The same
settings can be persisted in `$XDG_CONFIG_HOME/rai/config.json`:

```json
{
  "assistant": {
    "backend": "llama",
    "model": "/path/to/model.gguf",
    "context_window": 2048,
    "max_output_tokens": 256,
    "temperature": 0.2
  }
}
```

The active agent profile is also a supported configuration source. Its
`backend`, `model`, `ollama_host` and `system` fields are used when no explicit
assistant or CLI override exists. For example, with an active Ollama profile,
both commands below select the same graph-memory runtime:

```bash
uv run rai
uv run rai assistant chat
```

Use `--backend ollama --model MODEL_NAME` for an explicit locally running
Ollama model. `--profile NAME` selects both a configuration profile and its
durable-memory scope; `--session-id ID` selects only the bounded recent-dialogue
window. Inside interactive chat, use `/memories`, `/context`, `/session` and
`/help` to inspect the runtime.

There is no silent fake fallback: missing weights, missing optional dependencies
and unavailable daemons produce a typed error. Memory is stored under
`$XDG_DATA_HOME/rai/assistant/`; use an isolated `RAI_DATA_DIR` when comparing
experimental runs. See the [assistant operating and acceptance guide](docs/assistant-live-acceptance.md).

Piper voices are discovered under
`$XDG_DATA_HOME/rai/piper_voices/<voice-id>/` (or the equivalent default XDG
data directory). Each voice needs a matching `.onnx` and `.onnx.json` pair;
synthesis never downloads models during an invocation. See
[docs/local-voice.md](docs/local-voice.md) for profiles, defaults and current
limitations.

## Running RAI

```bash
# Standalone graph-memory assistant (same runtime as `rai assistant chat`)
uv run rai

# List policy-controlled kernel capabilities
uv run rai capability list

# Invoke a complete versioned CapabilityRequest record
uv run rai capability invoke '{...}'

# Local daemon (loopback by default)
uv run rai serve

# Explicit legacy compatibility client connected to the daemon
uv run rai --connect
```

The daemon protects control endpoints with a per-user token. By default it is
created with mode `0600` at:

```text
$XDG_RUNTIME_DIR/rai/api-token
```

If `XDG_RUNTIME_DIR` is unavailable, RAI uses its XDG cache directory. Set
`RAI_API_TOKEN` to supply a token explicitly. `RAI_DISABLE_AUTH=1` is intended
only for isolated tests and local development.

Configuration and runtime data follow the XDG base-directory convention:

```text
$XDG_CONFIG_HOME/rai   agent templates and settings
$XDG_DATA_HOME/rai     persistent history and data
$XDG_CACHE_HOME/rai    disposable sandbox/cache data
$XDG_RUNTIME_DIR/rai   socket/token runtime state
```

## Documentation

The versioned documentation is published with GitLab Pages at
[tk-lab1.gitlab.io/ai/rai/](https://tk-lab1.gitlab.io/ai/rai/).
The source documentation lives in [`docs/`](docs/), including
the [architecture](docs/architecture.md), [Rich History](docs/rich-history.md),
[local voice](docs/local-voice.md), [assistant
architecture](docs/assistant-architecture.md), [live acceptance evidence](docs/assistant-live-acceptance.md) and [embodiment kernel
contracts](docs/kernel-contracts.md). The language-neutral Stage 1 contract is
also available as a [JSON Schema](schemas/rai.kernel.v1.schema.json).

Install the documentation dependencies and build the Sphinx site locally:

```bash
uv sync --extra docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html/`. For a live-reloading
preview while editing documentation, run:

```bash
uv run sphinx-autobuild docs docs/_build/html
```

The GitLab Pages job rebuilds the same site from `main`. When the daemon is
running, its generated HTTP API reference is available at `/docs`, `/redoc` and
`/openapi.json`.

## Security model

RAI treats model output and remote agent output as untrusted. Shell and Python
execution is refused when no supported sandbox is operational. Network access
and selected high-risk operations require HITL approval. CORS is disabled by
default; trusted browser origins must be listed in `RAI_CORS_ORIGINS`.

This baseline does not make arbitrary generated code safe under every kernel or
desktop configuration. Read [SECURITY.md](SECURITY.md) before exposing the daemon
outside a single-user workstation.

## Verification

```bash
uv run pytest --timeout=30 --cov=src/rai --cov-report=term
uv run ruff check src tests --select E9,F63,F7,F82
uv run pylint -E src/rai
```

Full style linting is tracked separately while legacy modules are decomposed:

```bash
uv run ruff check src tests
```

## ❤️ Support this project

RAI is an independent open-source project developed with passion for the Linux
and AI communities. If you find it useful, you can support its continued
development, research and maintenance:

- [GitHub Sponsors](https://github.com/sponsors/faithlesstomas)
- [Ko-fi](https://ko-fi.com/faithlesstomas)
- Cryptocurrency / Base profile:
  [@faith4.base.eth](https://base.app/profile/faith4)

## Project status and contribution

RAI is experimental pre-1.0 software. The next product proof is a
provider-neutral, memory-backed conversation slice: immutable user turn →
bounded recent conversation window plus relevant durable graph memories →
explicit `ContextPackage` and `ContextManifest` → replaceable model backend →
terminal response and provenance-linked SQLite graph records. The first user
test must recall and then supersede a preference across daemon restarts without
provider-owned history. An ordinary conversation must not create a tracked
`Task` or invoke a capability.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a GitLab issue or merge
request. The project is licensed under the [Apache-2.0 License](LICENSE).
