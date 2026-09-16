# Emacs client

The experimental RAI Emacs package provides a provider-neutral Assistant chat,
inspectable server context, a bounded opt-in editor-state monitor and guarded
local Emacs Lisp facilities. Installation, privacy behavior, commands and the
development workflow are documented in the
[client README](../clients/emacs/README.md).

The package is a client of the existing RAI runtime. It does not own durable
memory, model sessions, policy or capability authority. Its HTTP endpoint map
and response adapter form a compatibility boundary while the Assistant module
is evolving.

Editor events remain local to Emacs in this first slice. They are metadata-only
and enter an Assistant turn only when the user explicitly invokes a contextual
request. Continuous daemon ingestion requires a future versioned contract and
must pass the same privacy, retention and audit controls as other RAI
collectors.

Arbitrary remote Emacs Lisp evaluation is intentionally not exposed. Typed,
allowlisted Emacs capabilities should be added through the common RAI
capability registry before any model-driven Emacs automation is enabled.
