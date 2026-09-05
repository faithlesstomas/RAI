# Changelog

## [Unreleased]

### Added
- Added reproducible wheel and source-distribution validation, a clean-wheel
  smoke test, Python 3.10-3.12 CI coverage, an explicit alpha-preview release
  job, and GitLab OIDC trusted-publishing jobs for TestPyPI and PyPI.

### Changed
- Selected `rich-ai` as the PyPI distribution name while retaining `rai` for
  the repository, Python namespace, CLI, configuration and protocol identity.
- Repositioned RAI as the secure, local-first integration layer between AI
  assistants and Linux, with GAIA, GCAS compatibility, Antigravity and J-lens
  explicitly outside the standalone product's required core.
- Moved Google Antigravity and GitLab tooling out of the base installation and
  corrected optional extras so they cannot resolve to the unrelated PyPI
  project named `rai`.

## [0.3.1] - 2026-09-04

### Fixed
- Made GitLab CI install locked dependencies through `uv`, kept Pages
  independent of the manual release job, removed the unused Sphinx static path
  and updated Antigravity conversation fixtures for the current SDK contract.

## [0.3.0] - 2026-09-04

### Changed
- Reframed RAI as a local-first agent runtime for Linux and made Antigravity a
  transitional compatibility backend.
- Expanded the implementation roadmap for Rich History, local voice and AI,
  safe desktop actions, hybrid backends, token budgets and privacy controls.
- Documented the SemVer release, testing and Sphinx/GitLab Pages workflow.
- Updated the semantic-release configuration for current version stamping,
  pre-1.0 development and GitLab `vX.Y.Z` releases.
- Adopted XDG-compliant configuration, data, cache and runtime directories.
- Made conversation history independent of thread-based database helpers.
- Moved heavyweight inference dependencies into optional extras.

### Security
- Removed package-import monkeypatches of Antigravity and `subprocess.Popen`.
- Changed sandbox selection to fail closed when Bubblewrap/Guix is unavailable.
- Added per-user token authentication and disabled CORS by default.
- Added ignore rules for loose credential files and local model weights.

### Added
- Added the Stage 1 embodiment kernel: immutable versioned domain records,
  structured provenance, language-neutral JSON Schema and conformance fixtures.
- Added provider-neutral runtime ports, cancellation/lifecycle contracts and
  deterministic synthetic reference implementations.
- Added a typed capability registry, four-outcome policy engine, shared
  CLI/REST/MCP invocation envelope and durable capability audit ledger.
- Added an application container and FastAPI app factory.
- `--debug` flag to enable debug logging for the application and the `python-gitlab` library.
- `GitlabTools` to interact with the GitLab API.
- Debug logging to `GitlabTools` methods.

### Fixed
- Made `rai --version` use the package version instead of a stale hard-coded
  value.
- `TypeError: 'Function' object is not callable` by correctly implementing `GitlabTools` as a `Toolkit`.
- `requests.exceptions.ChunkedEncodingError: Response ended prematurely` in `list_projects` by using an iterator.
- Issue with loading `GITLAB_BASE_URL` from `.env` file by stripping quotes and trailing slashes.
- Logic for checking required environment variables for `GitlabTools`.
