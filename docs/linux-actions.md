# Linux actions (implementation in progress)

Issue [#38](https://gitlab.com/tk-lab1/ai/rai/-/issues/38) tracks the acceptance
of this catalog. The presence of an adapter does not establish completion of
its desktop acceptance gate.

## Understanding and execution

With the local assistant backend, a local text engine produces a validated
intent: an action, a clarification question with choices, or no action.
Recognition does not require command prefixes. Model output cannot include an
approval or executable command. Invalid output fails without executing an action.
An ambiguous intent produces a question; application discovery can also ask the
user to distinguish installed applications.

The runtime resolves choices to opaque resource handles bound to the actor,
task, operation, data classification and expiry. It checks resource identity
again before execution. All catalog actions use `CapabilityService` policy and
approval contracts. The action service adds durable request reservations and
terminal-result replay: an interrupted reservation is reported as unknown rather
than repeated automatically. A successful adapter call alone is insufficient;
application launch checks process identity, volume changes read back audio state,
and document opening checks a process file descriptor for the selected file.

`schemas/rai.actions.v1.schema.json` exports proposals, internal resource records
and terminal results. Resource targets belong in trusted runtime storage; clients
receive opaque handles. This is an additive feature for a future minor release;
it does not change the existing major wire-contract version.

## Configuration

The runtime accepts an `actions` section in `$XDG_CONFIG_HOME/rai/config.json`:

```json
{
  "actions": {
    "allowed_file_roots": ["/absolute/path/to/documents"],
    "browser_endpoint": "http://127.0.0.1:9222"
  }
}
```

Restart the conversation after changing configuration. `file.access` reports
the configured document roots and current editing limitations. Opening an
absolute document path first resolves it within these roots, then uses the
same handle and approval path as a selected search result.

File search has no allowed roots by default. It searches supported document
filenames, excludes symlinks and executable files, and bounds traversal and
results. It does not search document content.

Browser open/read requires an explicitly configured Chromium debugging endpoint
on IPv4 loopback. Use a dedicated browser profile; do not expose a personal
profile through a debugging port. RAI does not start or configure this browser.
Search uses the optional `ddgs` metasearch backend from the `tools` extra.
Public queries may be sent to the search services selected by that backend. Browser actions require
PUBLIC data classification; LOCAL context is not silently reclassified to make a
network request succeed. Page text and search results remain untrusted content.
Opening or reading a result uses a handle returned by search, not a model-supplied
URL or script. Missing configuration produces `BROWSER_NOT_CONFIGURED`.

Application launch requires a graphical session, installed desktop entries,
GIO and the system Python GI bindings. Volume control uses `pactl` with
PulseAudio or PipeWire compatibility. Process inspection returns current-user
identity/state metadata, excluding command arguments, environment and memory.

## Catalog and limits

| Capability | Input purpose | Verification |
| --- | --- | --- |
| `application.list` | Installed application query | Desktop-entry discovery |
| `application.launch` | Application handle | PID, executable and start time |
| `file.access` | Document access scope | Runtime configuration |
| `file.search` | Document filename query | Allowed-root metadata |
| `document.open` | Document handle | Open file descriptor identity |
| `browser.search` | Public web query | Search response, issued result handles |
| `browser.open_result` | Search-result handle | Observed browser URL |
| `browser.read_page` | Search-result handle | Matching open tab and URL |
| `system.volume.get` | Task scope | Audio sink state |
| `system.volume.set` | Sink handle and 0–100 percentage | Sink identity and volume readback |
| `process.inspect` | Process query or handle | Owner and process start time |

Each runtime instance admits at most four concurrent catalog actions. The durable
execution store retains up to 100,000 request identities and refuses new identities
when full; it does not expire old identities and risk repeating their effects.
Requests are limited to 16 KiB and serialized terminal records to 64 KiB.

Catalog execution has a 20-second deadline and approval waits a 60-second
deadline. Resource handles expire after five minutes. Browser search returns at
most five results and page extraction returns at most 16,000 characters. Fixed
program output is limited to 256 KiB. Cancellation after execution begins can
produce `UNKNOWN`; this deliberately makes no claim that the effect was undone.

A desktop application that delegates work or closes its document descriptor
immediately may not provide the evidence these adapters require. In that case,
RAI reports an unverified result instead of asserting success. Full local-model,
transport-parity and desktop acceptance evidence is still being completed.

## Recorded launch acceptance

On 2026-10-04 a GNOME graphical session completed the Polish request
“Otworzysz mi kalkulator?” using the local Lemonade backend and
`Qwen3.5-4B-GGUF`. The model received bounded installed-application names and
IDs and selected `org.gnome.Calculator.desktop`. An isolated acceptance runtime
used temporary handle/execution databases and an approval broker restricted to
that exact application and the `application-launch` effect. This broker tests the
execution boundary; it does not constitute acceptance of the interactive approval
UI.

The terminal result was `SUCCEEDED`, with process PID, start time and
`/usr/bin/gnome-calculator` identity independently observed through the adapter.
The scenario used zero remote model tokens. No personal document, history or
browser content was included. Separate Lemonade intent probes recognized English
launch, Polish negation (`no_action`) and an underspecified request (`clarify`
with choices). These observations establish the tested launch path only, not the
acceptance of the rest of the catalog.

An additional adapter acceptance run on 2026-10-04 launched headless Chrome with
a temporary, isolated profile and a loopback debugging endpoint. `browser.open`
verified the URL of a local synthetic HTML page; `browser.read` returned its title
and text with `untrusted_content=true`. The page deliberately contained an
instruction-like sentence; extraction returned it as data. This adapter test does
not by itself prove downstream model resistance to page injection. The temporary
browser and fixture server were terminated after the run.

On 2026-10-04 the `ddgs` search adapter returned five valid results for the
synthetic public query “Python official documentation”, including python.org
and docs.python.org. The legacy `duckduckgo-search` client had returned an empty
list for that probe; the new browser catalog uses `ddgs` instead.

## Audit recovery

Terminal execution evidence and its audit envelope (including approval identity)
are committed together to a SQLite outbox. If ledger delivery fails, retries
attempt audit delivery before exposing the stored result; they never repeat the
system effect. JSONL terminal delivery is idempotent across a restart, including
when the ledger append succeeded but its outbox acknowledgement was lost.
Pre-outbox results with a policy decision cannot prove delivery and return
`AUDIT_STATE_UNVERIFIED` instead of an unaudited success.

Conversation capability descriptions list individual registered actions only
when a local action executor is connected. They explicitly qualify availability
by configuration, backend health, policy and approval.
