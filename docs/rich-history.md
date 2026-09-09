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
disallowed origins and paths are dropped. Browser-declared origins must agree
with URL hosts, origin policy covers subdomains, and local `file://` resources
from accessibility adapters pass through the same path policy as filesystem
events. Communication applications and banking, health or authentication
origins are metadata-only. Credential, email-address and payment-number
patterns are redacted across titles, URLs, paths, resource IDs and semantic
payloads. Dropped values are not included in errors or logs. Web and
accessibility text remains untrusted data; the history pipeline cannot invoke a
capability.

The complete records are encrypted with AES-256-GCM. The default key provider
obtains a per-user key through Secret Service (`secret-tool`). If Secret Service
is absent, locked or refuses the key, Rich History fails closed and persists
nothing. The database and its parent directory use restrictive permissions and
a `CACHEDIR.TAG` excludes the history directory from backup by default. An
explicit embedded deployment can inject another `KeyProvider`.

The shared event journal receives only a metadata projection for `PRIVATE`
observations; private titles, URLs, paths, selected text and semantic payloads
remain exclusively in the encrypted history store. A bounded in-memory raw
event buffer supports short-term processing and is cleared on pause, lock and
emergency stop.

## Deterministic processing

Repeated events are debounced, simultaneous evidence is fused with all source
IDs retained, and conflicts lower confidence and remain explicit. The
versioned episode builder splits activity on idle/time and project boundaries.
It requires no model and consumes zero remote tokens. Deleting a time range
removes source observations from both history storage and the event journal,
cascades through episodes, derived memories and outbound-context references,
then reproducibly rebuilds remaining episodes.
Retention is enforced immediately when collection starts and periodically
thereafter. Observation expiry is coordinated with journal deletion and episode
rebuilding; raw-buffer, episode and memory TTLs are then applied independently.
Explicit and retention-driven deletion use SQLite secure-delete behavior and
truncate WAL data after rebuilding to remove stale plaintext metadata pages.

## Local API

All routes use the same authentication boundary as other `/api/*` endpoints:

- `POST /api/v1/activity/collection` — `pause`, `resume`, `lock`, `unlock`,
  `emergency_stop` or `clear_emergency_stop`;
- `GET /api/v1/activity/collectors` — lifecycle, permission, last-event, error
  and restart status, plus global collection and retention health;
- `POST /api/v1/activity/observations` — validated semantic adapter input;
- `GET /api/v1/activity/episodes` — time/application/project/resource/activity
  filters and reviewable provenance;
- `GET /api/v1/activity/answer?question=what_was_i_working_on` — deterministic
  summary without an LLM;
- `DELETE /api/v1/activity/episodes` — time-range deletion or
  `current_application_session`, `last_10_minutes`, `last_hour`, `last_day` and
  `all` presets, with residual-link verification.

Install and enable the bundled GNOME Shell extension before enabling the GNOME
and foreground-process collectors. The packaged extension declares GNOME Shell
45 through 50 compatibility:

```bash
uv sync --extra gnome-tools
rai-history-install-gnome-extension
gnome-extensions enable rai-history@tk-lab1
```

Log out and back in when GNOME requests it. The extension uses supported Shell
APIs and exports active-window, process ID and workspace changes over the user
session bus. `rai-history-gnome` combines those events with the supported GNOME
ScreenSaver and Mutter IdleMonitor interfaces. `rai-history-atspi` subscribes to
bounded focus/state/document events and coalesces text changes without reading
their values. `rai-history-process` resolves only the focused PID's executable
basename. `rai-history-filesystem` polls names and mtimes below explicit roots
without opening file content.

## Live GNOME validation

Run the opt-in smoke test from a graphical GNOME terminal after installing the
extension and logging in again:

```bash
RAI_REQUIRE_LIVE_GNOME=1 uv run pytest \
  tests/test_rich_history.py -k live_gnome_sidecar -q
```

Without `RAI_REQUIRE_LIVE_GNOME`, the test skips when GNOME, the user-session
D-Bus or the packaged extension is unavailable. With the variable set, a
missing prerequisite fails the designated desktop-validation run. The test
starts the packaged sidecar, validates its bounded initial lock, idle,
workspace and active-window snapshot, rejects path-like application identities,
and proves that a live private title reaches neither the plaintext event journal
nor the encrypted database as plaintext.

Capture manual transitions without printing window titles or process IDs:

```bash
timeout 120s rai-history-gnome \
  | jq --unbuffered 'del(.title, .resource_id)'
```

While that command runs, focus another application, change workspace, remain
idle for at least 60 seconds, then lock and unlock the session. A successful run
contains `active_window`, `workspace_changed`, `idle`, `active`,
`session_locked` and `session_unlocked`. Exit status 124 is expected when
`timeout` ends a healthy long-running sidecar.

Record the tested platform and inspect the daemon-managed collectors:

```bash
gnome-shell --version
. /etc/os-release && printf '%s %s\n' "$NAME" "$VERSION_ID"
gnome-extensions info rai-history@tk-lab1
curl --fail --silent http://127.0.0.1:8228/api/v1/activity/collectors | jq
```

Add the normal API authorization header to `curl` unless authentication was
explicitly disabled for a loopback-only development service. The acceptance
record must contain versions and event kinds, never raw titles, command lines,
environment variables, credentials or screenshots.

Production bridges are registered in `$XDG_CONFIG_HOME/rai/config.json` as
command arrays and run as separate processes without a shell or inherited
credential variables:

```json
{
  "rich_history": {
    "enabled": true,
    "retention_interval_seconds": 300,
    "retention_days": {
      "raw": 0.006944,
      "observations": 30,
      "episodes": 90,
      "memories": 365
    },
    "privacy": {
      "allowed_applications": ["firefox.desktop", "code.desktop", "org.gnome.terminal.desktop"],
      "excluded_origins": ["private.example"],
      "allowed_paths": ["/home/me/projects"]
    },
    "collectors": {
      "gnome": {
        "command": ["rai-history-gnome"],
        "permission": "GRANTED"
      },
      "atspi": {
        "command": ["rai-history-atspi"],
        "permission": "GRANTED"
      },
      "process": {
        "command": ["rai-history-process"],
        "permission": "GRANTED"
      },
      "filesystem": {
        "command": ["rai-history-filesystem", "/home/me/projects"],
        "roots": ["/home/me/projects"],
        "permission": "GRANTED"
      }
    }
  }
}
```

Each sidecar writes one bounded `SourceEvent` JSON object per stdout line.
Unexpected source identities, oversized or invalid records and non-zero exits
are isolated by the supervisor and reflected only as typed health state; raw
stderr and untrusted event content never enter daemon logs.
