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

The runtime accepts an `actions` section in JSON or YAML configuration (see
[configuration](configuration.md)). The following JSON is equivalent in either format:

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
| `file.list` | Absolute allowed directory path | Direct metadata listing; no symlink traversal |
| `file.search` | Document filename query | Allowed-root metadata |
| `document.open` | File/directory handle and optional application handle | File descriptor identity or FileManager1 location and process identity |
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


## Directory and application selection

Example requests with the Lemonade backend:

- `Wylistuj zawartość katalogu /home/user/Documents`
- `Otwórz katalog /home/user/Documents w aplikacji Pliki`
- `Otwórz plik /home/user/Documents/note.txt w aplikacji Text Editor`

Paths must fall inside `actions.allowed_file_roots`. `file.list` returns at most
30 direct child names/types, with a truncation flag; it does not read file contents
or traverse symlinks. File search still selects supported non-executable document
types; an absolute directory path can also be resolved for opening. Arbitrary
source/binary files are not promoted to openable documents by directory listing.

A selected application is resolved from installed desktop entries into a second
actor/task/classification-bound handle. The approval preview names both the path
and application. Both resources are revalidated before GIO activation. Missing,
ambiguous or changed applications never silently fall back to the desktop default.
Directory confirmation checks the requested URI in FileManager1 and identifies its
bus-owner process; a chosen application's executable must match. Unsupported
file managers, cold-start delays, apps that close document descriptors immediately,
or unavailable evidence produce `UNKNOWN`, even if a window may have opened.

Recent intent dialogue is restricted to the session, profile and permitted data
classification. Action outputs are returned directly; automatic model synthesis
of action results is disabled until those results can pass through bounded context
selection, provenance and outbound approval. Authority handles are not appended to
model prompts after manifest construction.


## Configuration/CLI integration acceptance (2026-10-05)

With isolated configuration, state and handle databases, a real Lemonade
`Gemma-4-E4B-it-GGUF` session through `rai assistant ask` listed a synthetic
allowed directory and opened a new subdirectory in `org.gnome.Nautilus.desktop`.
The terminal approval displayed both the directory and application; entering
`y` led to a verified success using FileManager1 location/process evidence.
No private files were used; no remote model tokens were sent.

A separate HTTP JSON-RPC MCP client connected to a local RAI server with a
fresh authentication token. It listed the synthetic directory, exercised denial
through the approval API (`DENIED`), then approved a separate request and observed
`SUCCEEDED` for directory opening. Approval previews named the exact resources.
This is an external-process transport run, not an in-process transport fixture.
The test server was terminated afterward. The approvals were driven by the test
operator, not inferred from model output.

The same local model selected Text Editor for a Polish document-opening intent
when MIME metadata was included in its bounded catalog. This is intent evidence,
not a verified end-to-end Text Editor document-opening test. In one probe the
model retained sentence-final punctuation in an unquoted path; quoting paths is
recommended, and the runtime never substitutes a different path automatically.
Other desktop applications and document types retain their verification limits.
