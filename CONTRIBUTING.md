# Contributing to Rich AI

RAI is evolving into a local-first agent runtime for Linux. Contributions should
strengthen a complete runtime boundary or vertical slice instead of introducing
another provider-specific orchestration layer.

Read [ROADMAP.md](ROADMAP.md), [docs/architecture.md](docs/architecture.md) and
[SECURITY.md](SECURITY.md) before changing runtime or execution code.

## Development setup

Use `uv` and the project virtual environment:

```bash
git clone https://gitlab.com/tk-lab1/ai/rai.git
cd rai
uv sync --extra dev
```

The `dev` extra includes the test, lint and optional general-purpose tool
dependencies needed for the documented Pylint command to resolve lazy tool
imports consistently with CI.

Install optional platform or inference dependencies only when needed:

```bash
uv sync --extra gnome-tools
uv sync --extra inference-llama
```

Do not commit model weights, API keys, access tokens, local databases or local
harness state. If a credential reaches a workspace file, rotate it; adding the
file to `.gitignore` is not a substitute for rotation.

## Development rules

- Prefer `Protocol` boundaries, immutable records and explicit `Result` values
  at I/O/backend boundaries.
- Keep operating-system state and durable memory outside LLM contexts.
- Treat model and remote-agent output as untrusted input.
- Security controls must fail closed.
- Avoid package-import side effects, global monkeypatches and hidden singleton
  construction.
- Add one backend through a conformance contract rather than branching core
  logic by provider name.
- Use English commit messages and documentation/code identifiers.

## Commits and semantic versioning

RAI follows [Semantic Versioning 2.0.0](https://semver.org/). The public API
includes the Python package, CLI behavior, REST/OpenAPI endpoints, MCP
capabilities, versioned JSON/wire schemas and documented `DeviceAgent` and
`AgentBackend` contracts.

Every change must identify its release impact:

- `PATCH` (`x.y.Z`) fixes incorrect behavior without breaking the public API;
- `MINOR` (`x.Y.0`) adds backward-compatible functionality or deprecates a
  public API;
- `MAJOR` (`X.0.0`) introduces an incompatible public API change;
- pre-release suffixes such as `-alpha.1`, `-beta.1` and `-rc.1` may be used for
  unstable release candidates.

Version `0.y.z` denotes initial development. With the current
`allow_zero_version = true` and `major_on_zero = false` semantic-release policy,
a breaking change before 1.0 advances the minor version instead of declaring
the API stable. The 1.0 release must explicitly define the supported public API
and change this policy as part of its release preparation.

Use Conventional Commit subjects so `python-semantic-release` can determine the
next version:

```text
fix(scope): backward-compatible bug fix            # PATCH
feat(scope): backward-compatible functionality     # MINOR
feat(scope)!: incompatible public API change        # MAJOR

BREAKING CHANGE: describe the migration requirement
```

`docs:`, `test:`, `refactor:`, `perf:`, `build:`, `ci:` and `chore:` describe
non-feature work and normally do not trigger a release. Use `fix(docs):` when a
documentation correction changes the supported public contract, and use
`feat!:`/`fix!:` plus a `BREAKING CHANGE:` footer whenever the corresponding
change is incompatible.

Do not update versions or create release tags in ordinary feature/fix merge
requests. Each commit declares its semantic impact, while the release pipeline
performs one coherent version update after all selected changes reach `main`.
This avoids duplicate tags and intermediate versions for incomplete work.

The canonical release version is the unprefixed SemVer value stored in both:

```text
pyproject.toml                 [project].version
src/rai/__init__.py            __version__
```

`python-semantic-release` updates both locations. Runtime surfaces, including
`rai --version`, FastAPI metadata and generated Sphinx documentation, must read
the package `__version__` instead of maintaining another literal. Git release
tags use the corresponding annotated name `vX.Y.Z`; the `v` belongs to the tag,
not to the package version.

## Unit and integration tests

Every behavior change requires tests at the lowest useful level:

- pure domain and policy logic gets deterministic unit tests;
- protocols, schemas and backends get positive and negative conformance tests;
- collector, capability, storage and transport changes get integration tests
  with isolated XDG directories and synthetic adapters;
- privacy and security boundaries get explicit denial, redaction, timeout,
  cancellation and fail-closed tests;
- bug fixes include a regression test that fails before the fix;
- changes to a public schema include compatibility fixtures for the previous
  supported version or an explicitly documented breaking migration.

Tests must not require the developer's real desktop session, credentials,
history, network accounts or model weights. Platform integration tests should
use a deterministic substitute or skip with a precise reason when the actual
platform capability is unavailable. A mocked happy path alone does not complete
a roadmap acceptance slice.

The distribution metadata, runtime and CLI version are covered by
`tests/test_version.py`. Keep that test green whenever release configuration or
package metadata changes.

## Verification

Before opening a merge request:

```bash
uv run pytest --timeout=30 --cov=src/rai --cov-report=term
uv run ruff check src tests --select E9,F63,F7,F82
uv run pylint -E src/rai
```

Run full style diagnostics as well and avoid adding new violations:

```bash
uv run ruff check src tests
```

## Documentation system

RAI uses **Sphinx with MyST Markdown and the Furo theme**. This stack is already
configured in `docs/conf.py`, accepts the project's existing `.md` files,
generates Python API reference pages and is published by the GitLab Pages job.
Do not introduce MkDocs or a second documentation source tree unless an accepted
architecture decision demonstrates a capability that the current stack cannot
provide.

FastAPI's `/docs`, `/redoc` and `/openapi.json` describe the live HTTP API. They
do not replace the versioned project documentation for installation,
architecture, security, collectors, capabilities, protocols, deployment,
operations or release notes. Endpoint descriptions and Pydantic models must be
complete enough for OpenAPI to remain the canonical HTTP schema; the Sphinx site
explains how and why to use that API.

Documentation ownership is divided as follows:

- `README.md` — project overview and first successful local run;
- `ROADMAP.md` — planned work, status and acceptance gates;
- `CONTRIBUTING.md` — development, testing, documentation and release process;
- `SECURITY.md` — trust model, supported controls and vulnerability reporting;
- `CHANGELOG.md` — released user-visible changes and the current `Unreleased`
  section;
- `docs/` — publishable guides, architecture, protocols, operations and
  generated Python API reference;
- FastAPI OpenAPI — generated HTTP endpoint and data-model reference.

When behavior, configuration, a public contract or an operator workflow
changes, update the relevant documentation in the same merge request. Planned
behavior belongs in the roadmap, not in present-tense user documentation.
Security-sensitive behavior must also update `SECURITY.md` and include residual
limitations.

Install and build the documentation locally with:

```bash
uv sync --extra docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

For live preview while editing:

```bash
uv run sphinx-autobuild docs docs/_build/html
```

Warnings fail the documentation build locally and in CI. Keep links relative
where possible, include new pages in `docs/index.md`, and never commit
`docs/_build/` or generated `public/` output. GitLab Pages publishes the static
HTML generated from `main`; release documentation therefore remains rebuildable
from the corresponding immutable Git tag.

## Release process

Releases are created from a clean, reviewed `main` branch by the manual
`publish_release` GitLab CI job. The normal process is:

1. Confirm all intended merge requests are merged and their Conventional Commit
   messages express the correct SemVer impact.
2. Confirm `CHANGELOG.md` describes user-visible, compatibility, security and
   migration changes.
3. Run the complete tests, critical lint and warning-as-error documentation
   build documented above.
4. Preview the calculated release without modifying the repository:

   ```bash
   semantic-release --noop version
   ```

5. Review the proposed version and release notes. Correct commit metadata or the
   changelog before releasing; do not compensate for a wrong classification by
   manually choosing an arbitrary version.
6. Run the manual `publish_release` job for the exact green commit on `main`.
   The job updates `pyproject.toml` and `src/rai/__init__.py`, builds the
   changelog and package, creates the release commit and `vX.Y.Z` tag, pushes
   them and publishes the GitLab release.
7. Verify the tag points at the release commit and that the following agree:

   ```bash
   git describe --tags --exact-match
   uv run rai --version
   uv build
   ```

   Also verify the built package metadata, FastAPI version, release artifacts,
   release notes and documentation site.

8. Start the next development cycle with an empty or updated `Unreleased`
   section as produced by the release tooling.

Never move, replace or reuse a published tag, and never modify artifacts for an
existing version. A release error is corrected by a new SemVer release.

If the automated release job is unavailable, use a maintainer-approved
break-glass procedure: select the SemVer version, update both canonical version
locations and `CHANGELOG.md` in one `chore(release): X.Y.Z` commit, build and
verify artifacts, create the annotated `vX.Y.Z` tag, and push the commit and tag
together. Do not combine manual and semantic-release flows for the same
release.

## GitLab workflow

Use `glab` for issues, merge requests and CI inspection. An issue should state:

- the user-visible or architectural outcome,
- the boundary/contract affected,
- failure and security behavior,
- acceptance tests,
- documentation that must change.

Keep merge requests narrow enough to review, but complete enough to move the
selected vertical slice end-to-end.
