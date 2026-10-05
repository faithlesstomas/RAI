# Configuration and Typer CLI

This pre-1.0 breaking change resolves #6 and implements the core refactor in #46. Configuration parsing and
persistence are shared by CLI and server. The retired Click compatibility CLI
and its `ChatService` client have been removed. The native assistant, capabilities,
benchmarks and neural sidecar commands use modular Typer adapters.

## Files and precedence

RAI accepts YAML (`.yaml`, `.yml`) and JSON through the same typed schema.
The preferred default is `$XDG_CONFIG_HOME/rai/config.yaml`. `RAI_CONFIG_DIR`
overrides that directory. An existing `config.json` is still readable. If more
than one default configuration file exists, startup fails and asks for an explicit
selection; files are never silently merged.

Use `rai --config /path/settings.yaml ...` or `RAI_CONFIG_FILE` to select a file.
An explicitly selected missing file is an error. With no default file, RAI uses
in-memory defaults without creating files. A selected configuration reads legacy
`agents.yaml` only from its own directory, and only if it has no inline `agents`.
Reads never rewrite files, rename tools, or change model versions.

Model-setting precedence, highest first:

1. Explicit command options.
2. `RAI_ASSISTANT_BACKEND`, `RAI_ASSISTANT_MODEL`, `LEMONADE_HOST`, `LEMONADE_API_KEY`.
3. `assistant` settings.
4. `local_ai` settings (also used by the local processor supervisor).
5. Selected `agents` profile.
6. Runtime defaults; the existing auto backend/GGUF discovery applies when unset.

The shared defaults are 4096 output tokens, an 8192-token model context window,
and a 32000-character context selection limit. `assistant ask` and `assistant chat`
accept `--max-output-tokens` and `--context-window` to override these settings for
one invocation. Both must be positive; `--thinking-budget` accepts zero or more.
The character limit is a selection bound, not an exact token count. Models with
smaller contexts need explicit lower limits.

A profile is model configuration and memory scope. `--session-id` identifies a
conversation, independently of the profile. Explicit `false` overrides inherited
`true`. Unknown keys and wrong scalar types are errors, including strings used
instead of booleans. Malformed files and duplicate keys never fall back silently.

```yaml
schema_version: 1
active_agent: work
agents:
  work:
    backend: lemonade
    model: Qwen3.5-4B-GGUF
    system: Answer using the capabilities and evidence actually available.
assistant:
  max_output_tokens: 4096
  context_window: 8192
  enable_thinking: false
rich_history:
  enabled: false
```

`actions.allowed_file_roots` configures file listing, search and document/directory
opening. `actions.browser_endpoint` selects the isolated browser adapter endpoint.
See [Linux actions](linux-actions.md) for capabilities and verification limits. Workspace
retention and client leases remain tracked separately in #48.

## Inspection and editing

```bash
uv run rai config validate
uv run rai config show --effective
uv run rai config show --effective --profile work
uv run rai config set assistant.enable_thinking false
uv run rai config set assistant.max_output_tokens 4096
uv run rai config set actions.allowed_file_roots '["/home/user/Documents"]'
uv run rai profile list
uv run rai profile use work
uv run rai assistant chat --profile work
```

`config show --effective` reports merged model settings and their sources, without
loading a model. Automatic backend selection and model artifact discovery are
performed later by the runtime. Inside chat, `/config` shows the same resolution
with that invocation's overrides; changes to persisted settings require restarting
the conversation. Credential fields are redacted in configuration displays and
validation errors omit input values. `config set` parses JSON values; other input
is plain text. Schema validation runs before writing.

Writes use a private temporary file, fsync and atomic replacement, with mode 0600.
Session IDs and provider-resume mappings are stored separately under
`$XDG_STATE_HOME/rai` (override: `RAI_STATE_DIR`), scoped to the selected configuration
path. Saving settings preserves legacy session state before removing it from the
settings file. Conversation memory and audit storage are unchanged.

## Explicit migration

```bash
uv run rai config migrate ~/.config/rai/config.json --output /tmp/rai-settings.yaml
uv run rai --config /tmp/rai-settings.yaml config validate
uv run rai --config /tmp/rai-settings.yaml config show --effective
```

Migration copies inline profiles or sibling `agents.yaml`, accepts the historical
`sessions` profile name, and preserves session state. It never overwrites the
source or an existing destination. Unsupported obsolete keys are reported for
manual review rather than silently dropped. Keep the old files until the new
configuration has been reviewed. Choose the new file explicitly, or move the old
configuration outside the default discovery names. YAML writes normalize formatting
and do not preserve comments.

## CLI changes and approvals

| Previous invocation | Supported invocation |
| --- | --- |
| `rai` to start chat | `rai assistant chat` (`rai` now shows help) |
| `rai -p TEXT` | `rai assistant ask TEXT` |
| `rai serve` | `rai server serve` |
| `rai config` or profile-only display | `rai config show --effective` |
| `rai agent` for profile selection | `rai profile list/show/use` |
| `rai --connect` | Removed; use the native assistant REST API for remote clients |

Typer supplies shell completion (`--install-completion` / `--show-completion`).
Backend completion uses the runtime catalog; profiles and model names come from
configuration. Model-name completion does not query an inference server. Runtime plugin discovery,
live server model discovery and streaming Markdown rendering remain in #46.
Memory inspection stays under `assistant` (`sessions`, `history`, `context` and
chat slash commands); there is no separate top-level `memory` or `system` group.

`assistant ask/chat --data-class PUBLIC|LOCAL|PRIVATE` classifies new turns only.
LOCAL remains the default. The remote native Antigravity backend reviews an
outbound context manifest and asks for approval, with no automatic conversion of
stored LOCAL data to PUBLIC. LOCAL, SECRET and BLOCKED context cannot leave the
machine. The final backend firewall still validates the full outbound manifest.
Approval is bound to the exact context and persisted with its terminal result.

Capability actions use a terminal approval broker bound to the policy decision.
Noninteractive stdin, rejection, timeout and cancellation do not grant approval.
There is no global `--yes` bypass. External clients continue to use the existing
server approval flow.

## Remaining legacy removal (#43)

This MR removes `cli_compatibility.py`, the legacy CLI processors/slash handlers,
and provider-owned resume from the CLI path. Obsolete tests specific to those
removed internals are replaced by Typer/configuration contracts; native assistant
memory, restart, cancellation and transport tests remain.

The opt-in `/api/v1/run`, `/api/v1/stream`, `/ws/v1/chat` execution endpoints,
`ChatService`, legacy
Antigravity backend, transcript/trajectory helpers still have concrete consumers. Their removal, client migration and OpenAPI changes remain in #43.
The unmaintained Textual TUI, configuration screen, model-tool detection helper
and unused legacy configuration UI functions have been removed. Future graphical
and editor clients (GNOME, Emacs and other environments) use the server API.
`config_manager.py` retains adapters for the remaining server consumers; parsing and persistence
have moved to `rai.configuration`. This change does not close #43 or #48.
