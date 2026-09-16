# RAI for Emacs

This directory contains an experimental, dependency-free Emacs client for the
RAI-owned Assistant API. The client intentionally isolates the evolving HTTP
contract in `rai-transport.el`; chat, editor context and local Emacs Lisp
facilities do not depend directly on wire paths or response keys.

## Install and connect

Add this directory to `load-path` and load `rai`:

```elisp
(add-to-list 'load-path "/path/to/rai/clients/emacs")
(require 'rai)
(setq rai-base-url "http://127.0.0.1:8000")
```

Run the daemon with `uv run rai serve`. Authentication is mandatory for the
Assistant endpoints. The client looks for `RAI_API_TOKEN`, then the protected
runtime token file created by RAI, and finally an `auth-source` entry named
`rai@rai-local`. For example, an encrypted `~/.authinfo.gpg` entry may contain:

```text
machine rai-local login rai password TOKEN
```

Do not put the token in `init.el` or in the URL. Use `M-x rai-ping` to check the
daemon and `M-x rai-chat` to open the client. In the chat buffer:

| Key | Command |
|---|---|
| `a` | ask without editor context |
| `c` | ask with explicit bounded context |
| `g` | inspect the exact server-side context of the last answer |
| `s` | list resumable sessions |
| `m` | inspect active Assistant memories |
| `k` | disconnect the current request |

The base client currently uses complete REST turns. This is deliberate: the
response adapter and endpoint map provide a small migration seam while the
Assistant protocol is under active development. Streaming can be added behind
the same boundary once its event and cancellation contract stabilizes.

## Editor state monitor

`M-x rai-context-monitor-mode` enables an opt-in, local monitor. It retains a
small in-memory ring containing buffer focus, save and a limited allowlist of
command names. It never records individual keys, minibuffer input, clipboard
data or buffer text, and it does not transmit events by default.

`M-x rai-context-show-current` previews exactly what an explicit contextual
request would include. `M-x rai-ask-with-context` includes the active region,
or the defun at point, subject to a character budget. Sensitive modes, remote
files, credential files and `.env` files fail closed. Customize the `rai-context`
group to make the policy stricter for a particular setup.

The monitor is useful for conversational continuity across coding, terminals,
documentation, compilation and navigation, but it is not yet a proactive event
collector. A future daemon integration should use a versioned editor-event
schema, privacy classification, bounded retention and the normal RAI policy
path rather than sending raw hook events to a model.

## Emacs Lisp and REPL access

`M-x rai-open-elisp-repl` opens local IELM. It does not grant the Assistant
access. `M-x rai-emacs-describe-symbol` provides read-only documentation
inspection.

`M-x rai-elisp-eval` is disabled by default. When explicitly enabled with
`rai-elisp-evaluation-enabled`, every expression still requires local user
confirmation and its result remains local. Arbitrary Elisp has all permissions
of the Emacs process, so server- or model-originated text must never be passed
to it automatically.

The recommended future automation surface is a typed capability set such as
`emacs.inspect`, `emacs.open`, `emacs.invoke-command` and `emacs.apply-edits`,
with argument schemas, command allowlists, approval and audit. A raw remote REPL
should not be exposed as an RAI capability.

## Development

Run ERT tests and byte compilation with:

```bash
make -C clients/emacs test
make -C clients/emacs compile
```

The package supports Emacs 28.1 and newer and has no third-party Elisp
dependencies.
